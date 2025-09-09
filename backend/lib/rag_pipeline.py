"""
RAG (Retrieval Augmented Generation) pipeline for medical queries
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .ai_query_classifier import AIQueryClassifier
from .embeddings import EmbeddingService
from .knowledge_graph import KnowledgeGraphService


class RAGQuery:
    """RAG query configuration"""

    def __init__(
        self,
        query: str,
        drugs: Optional[List[str]] = None,
        symptoms: Optional[List[str]] = None,
        current_medications: Optional[List[str]] = None,
        max_results: int = 10,
        include_vector: bool = True,
        include_graph: bool = True,
    ):
        self.query = query
        self.drugs = drugs or []
        self.symptoms = symptoms or []
        self.current_medications = current_medications or []
        self.max_results = max_results
        self.include_vector = include_vector
        self.include_graph = include_graph


class QueryAnalysis:
    """Query analysis results"""

    def __init__(
        self,
        extracted_drugs: List[str],
        extracted_symptoms: List[str],
        query_type: str,
        confidence: float,
    ):
        self.extracted_drugs = extracted_drugs
        self.extracted_symptoms = extracted_symptoms
        self.query_type = query_type
        self.confidence = confidence


class RetrievalContext:
    """Context retrieved from various sources"""

    def __init__(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: Dict[str, Any],
        query_analysis: QueryAnalysis,
        routing_info: Optional[Dict[str, Any]] = None,
    ):
        self.vector_results = vector_results
        self.graph_results = graph_results
        self.query_analysis = query_analysis
        self.routing_info = routing_info or {}
        self.conversation_context = {}


class RAGPipeline:
    """Main RAG pipeline for medical query processing"""

    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()
        self.knowledge_graph_service = KnowledgeGraphService(db)
        self.ai_classifier = AIQueryClassifier()

    async def analyze_query(
        self, query: str, routing_info: Dict[str, Any]
    ) -> QueryAnalysis:
        """Create QueryAnalysis from AI classifier results (deprecated fallback analysis)"""

        # Use AI classifier results as primary source
        extracted_drugs = routing_info.get("extracted_drugs", [])
        query_intent = routing_info.get("intent", "general_medical")
        confidence = routing_info.get("confidence", 0.5)

        # Legacy symptom extraction (kept for backward compatibility)
        query_lower = query.lower()
        symptom_keywords = [
            "pain",
            "headache",
            "fever",
            "nausea",
            "hypertension",
            "diabetes",
            "depression",
            "anxiety",
            "infection",
            "inflammation",
            "allergy",
            "high blood pressure",
            "disorder",
            "syndrome",
            "disease",
        ]
        extracted_symptoms = [
            symptom for symptom in symptom_keywords if symptom in query_lower
        ]

        return QueryAnalysis(
            extracted_drugs=extracted_drugs,
            extracted_symptoms=extracted_symptoms,
            query_type=query_intent,
            confidence=confidence,
        )

    async def retrieve_relevant_context(self, rag_query: RAGQuery) -> RetrievalContext:
        """Retrieve relevant context using AI-powered query routing"""

        # Get AI classifier routing info first
        routing_info = self.ai_classifier.route_query(
            rag_query.query, drugs=rag_query.drugs + rag_query.current_medications
        )

        # Create analysis from AI classifier results
        analysis = await self.analyze_query(rag_query.query, routing_info)

        # Use all available drug information (AI classifier already includes provided drugs)
        all_drugs = list(set(routing_info["extracted_drugs"]))

        # Vector search results with intelligent source filtering
        vector_results = []
        if rag_query.include_vector:
            source_filter = None
            if (
                routing_info["use_focused_search"]
                and routing_info["recommended_sources"]
            ):
                source_filter = routing_info["recommended_sources"]

            vector_results = await self.embedding_service.search_similar(
                self.db,
                rag_query.query,
                limit=rag_query.max_results,
                min_similarity=0.6,
                source_filter=source_filter,
            )

        # Knowledge graph results using existing service
        graph_results = {}
        if rag_query.include_graph and all_drugs:
            graph_results = self.knowledge_graph_service.get_comprehensive_drug_info(
                all_drugs[0], all_drugs[1:] if len(all_drugs) > 1 else []
            )

        return RetrievalContext(
            vector_results=vector_results,
            graph_results=graph_results,
            query_analysis=analysis,
            routing_info=routing_info,
        )

    async def enhanced_retrieve(
        self, query: str, options: Dict[str, Any] = None
    ) -> RetrievalContext:
        """Enhanced retrieval with various options"""
        if options is None:
            options = {}

        rag_query = RAGQuery(
            query=query,
            drugs=options.get("drugs"),
            current_medications=options.get("currentMedications"),
            max_results=options.get("maxVectorResults", 15),
            include_vector=True,
            include_graph=True,
        )

        # Extract conversation context
        conversation_context = options.get("conversationContext", {})

        # Enhance drugs and medications with conversation history
        if conversation_context.get("mentioned_drugs"):
            enhanced_drugs = list(
                set(
                    [
                        *(rag_query.drugs or []),
                        *(conversation_context.get("mentioned_drugs", [])),
                    ]
                )
            )
            rag_query.drugs = enhanced_drugs

        retrieval_context = await self.retrieve_relevant_context(rag_query)

        # Set conversation context
        retrieval_context.conversation_context = conversation_context

        # Filter and rank results based on focus
        focus_on = options.get("focusOn")
        if focus_on:
            retrieval_context.vector_results = self._filter_by_focus(
                retrieval_context.vector_results, focus_on
            )

        return retrieval_context

    def _filter_by_focus(
        self, results: List[Dict[str, Any]], focus: str
    ) -> List[Dict[str, Any]]:
        """Filter results based on focus area"""
        focus_keywords = {
            "interactions": [
                "interact",
                "combination",
                "together",
                "synergistic",
                "antagonistic",
            ],
            "contraindications": [
                "contraindication",
                "avoid",
                "should not",
                "dangerous",
                "toxic",
            ],
            "general": [],
        }

        if focus == "general":
            return results

        keywords = focus_keywords.get(focus, [])
        return [
            result
            for result in results
            if any(keyword in result["content"].lower() for keyword in keywords)
        ]

    def format_context_for_llm(self, context: RetrievalContext) -> str:
        """Format context for LLM consumption"""
        formatted_context = ""

        # Add AI classification information
        if context.routing_info:
            formatted_context += "## AI Query Classification:\n"
            formatted_context += (
                f"- Detected Intent: {context.routing_info.get('intent', 'unknown')}\n"
            )
            formatted_context += (
                f"- AI Confidence: {context.routing_info.get('confidence', 0):.2f}\n"
            )
            formatted_context += f"- Search Strategy: {'Focused search' if context.routing_info.get('use_focused_search') else 'Broad search'}\n"
            if context.routing_info.get("recommended_sources"):
                formatted_context += f"- Data Sources Used: {', '.join(context.routing_info['recommended_sources'])}\n"
            formatted_context += f"- AI Reasoning: {'; '.join(context.routing_info.get('reasoning', []))}\n\n"

        # Add vector search results
        if context.vector_results:
            formatted_context += "## Relevant Medical Information (Vector Search):\n"
            for i, result in enumerate(context.vector_results[:5]):
                formatted_context += (
                    f"### Result {i + 1} (Similarity: {result['similarity']:.3f}):\n"
                )
                formatted_context += f"**Source:** {result['source']}\n"
                formatted_context += f"**Content:** {result['content']}\n\n"

        # Add knowledge graph results
        drug_info = context.graph_results.get("drugInfo", [])
        if drug_info:
            formatted_context += "## Drug Information (from drug_metadata table):\n"
            for drug in drug_info:
                formatted_context += f"- **{drug['name']}** ({drug['type']}): ID {drug['id']}, Source: {drug['source']}\n"
            formatted_context += "\n"

        # Add product stage information
        product_stage = context.graph_results.get("productStage")
        if product_stage:
            formatted_context += (
                "## Product Stage & Availability (from drug_product_stages table):\n"
            )
            formatted_context += f"- **Drug:** {product_stage['drugName']}\n"
            formatted_context += (
                f"- **Current Stage:** {product_stage['productStage']}\n"
            )
            formatted_context += (
                f"- **Stage Description:** {product_stage['description']}\n"
            )
            formatted_context += f"- **Stage Code:** {product_stage['stageCode']}\n\n"

        # Add warnings (prioritized by critical keywords in content, limited to top 5)
        warnings = context.graph_results.get("warnings", [])
        if warnings:
            # Sort by critical keywords in the description/content
            def prioritize_warning(warning):
                description = warning.get("description", "").lower()
                # High priority keywords
                if any(
                    word in description
                    for word in [
                        "contraindicated",
                        "avoid",
                        "dangerous",
                        "severe",
                        "fatal",
                        "toxic",
                    ]
                ):
                    return 0
                # Medium priority keywords
                elif any(
                    word in description
                    for word in ["caution", "monitor", "adjust", "increase", "decrease"]
                ):
                    return 1
                # Default priority by description length (more detailed = more important)
                else:
                    return 2 - min(
                        len(description) / 100, 1
                    )  # Longer descriptions get higher priority

            sorted_warnings = sorted(warnings, key=prioritize_warning)
            top_warnings = sorted_warnings[:5]

            formatted_context += "## Important Warnings (from drug_relations table):\n"
            for warning in top_warnings:
                formatted_context += (
                    f"- **{warning['type'].upper()}**: {warning['description']}\n"
                )
                formatted_context += f"  *Source: {warning['source']}*\n"

            # Add summary if more warnings exist
            if len(warnings) > 5:
                formatted_context += (
                    f"- *... and {len(warnings) - 5} additional warnings available*\n"
                )
            formatted_context += "\n"

        # Add food interactions (limited to top 8 most relevant)
        food_interactions = context.graph_results.get("foodInteractions", [])
        if food_interactions:
            # Prioritize interactions with more specific/detailed descriptions
            sorted_interactions = sorted(
                food_interactions, key=lambda x: len(x["interaction"]), reverse=True
            )
            top_interactions = sorted_interactions[:8]

            formatted_context += (
                "## Food Interactions (from food_interactions table):\n"
            )
            for interaction in top_interactions:
                formatted_context += (
                    f"- **{interaction['drugName']}**: {interaction['interaction']}\n"
                )
                formatted_context += f"  *Source: {interaction['source']}*\n"

            # Add summary if more interactions exist
            if len(food_interactions) > 8:
                formatted_context += f"- *... and {len(food_interactions) - 8} additional food interactions available*\n"
            formatted_context += "\n"

        # Add drug-drug interactions (limited to top 6 most critical)
        drug_drug_interactions = context.graph_results.get("drugDrugInteractions", [])
        if drug_drug_interactions:
            # Prioritize by interaction severity keywords, then by description length
            def prioritize_drug_interaction(interaction):
                interaction_text = interaction["interaction"].lower()
                # High priority keywords
                if any(
                    word in interaction_text
                    for word in [
                        "contraindicated",
                        "avoid",
                        "dangerous",
                        "severe",
                        "fatal",
                    ]
                ):
                    return 0
                # Medium priority keywords
                elif any(
                    word in interaction_text
                    for word in ["caution", "monitor", "increase", "decrease", "adjust"]
                ):
                    return 1
                # Default priority by description length
                else:
                    return 2

            sorted_interactions = sorted(
                drug_drug_interactions, key=prioritize_drug_interaction
            )
            top_interactions = sorted_interactions[:6]

            formatted_context += (
                "## Drug-Drug Interactions (from drug_relations table):\n"
            )
            for interaction in top_interactions:
                formatted_context += f"- **{interaction['drug1Name']} + {interaction['drug2Name']}**: {interaction['interaction']}\n"
                if interaction.get("interactionType"):
                    formatted_context += f"  *Type: {interaction['interactionType']}*\n"

            # Add summary if more interactions exist
            if len(drug_drug_interactions) > 6:
                formatted_context += f"- *... and {len(drug_drug_interactions) - 6} additional drug interactions available*\n"
            formatted_context += "\n"

        # Add dosage information (limited to top 8 most relevant formulations with availability status)
        dosage_info = context.graph_results.get("dosage", [])
        if dosage_info:
            # Separate available vs withdrawn products
            available_dosages = [
                d for d in dosage_info if not d.get("isWithdrawn", False)
            ]
            withdrawn_dosages = [d for d in dosage_info if d.get("isWithdrawn", False)]

            # Prioritize available products first, then withdrawn
            prioritized_dosages = []
            seen_forms = set()

            # First: add available products with unique dosage forms/strengths
            for dosage in available_dosages:
                form_strength = (
                    f"{dosage.get('dosageForm', 'N/A')}_{dosage.get('strength', 'N/A')}"
                )
                if form_strength not in seen_forms or len(prioritized_dosages) < 6:
                    prioritized_dosages.append(dosage)
                    seen_forms.add(form_strength)
                    if len(prioritized_dosages) >= 6:
                        break

            # Then: add withdrawn products to show historical context (up to 2)
            for dosage in withdrawn_dosages:
                if len(prioritized_dosages) >= 8:
                    break
                form_strength = (
                    f"{dosage.get('dosageForm', 'N/A')}_{dosage.get('strength', 'N/A')}"
                )
                if form_strength not in seen_forms:
                    prioritized_dosages.append(dosage)
                    seen_forms.add(form_strength)

            formatted_context += "## Dosage & Availability Information (from drug_dosage + drug_product_stages tables):\n"

            for dosage in prioritized_dosages:
                availability_status = dosage.get("availabilityStatus", "Status unknown")
                is_withdrawn = dosage.get("isWithdrawn", False)

                # Mark withdrawn products clearly
                status_indicator = "⚠️ WITHDRAWN" if is_withdrawn else "✅ Available"

                formatted_context += f"- **{dosage['productName']}** ({status_indicator}) - {availability_status}\n"
                if dosage.get("dosageForm"):
                    formatted_context += f"  - Form: {dosage['dosageForm']}\n"
                if dosage.get("strength"):
                    formatted_context += f"  - Strength: {dosage['strength']}\n"
                if dosage.get("route"):
                    formatted_context += f"  - Route: {dosage['route']}\n"
                if dosage.get("manufacturer"):
                    formatted_context += f"  - Manufacturer: {dosage['manufacturer']}\n"
                if is_withdrawn and dosage.get("stageDescription"):
                    formatted_context += (
                        f"  - **Withdrawal Reason**: {dosage['stageDescription']}\n"
                    )

            # Add summary with availability breakdown
            if len(dosage_info) > 8:
                available_count = len(available_dosages)
                withdrawn_count = len(withdrawn_dosages)
                formatted_context += f"- *... {available_count} available formulations, {withdrawn_count} withdrawn products in total*\n"
            formatted_context += "\n"

        return formatted_context
