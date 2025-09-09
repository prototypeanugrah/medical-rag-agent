"""
Medical RAG Agent - Main AI agent for processing medical queries
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from sqlalchemy.orm import Session

from ..models.database import ChatMessage, ChatSession
from .error_handling import AIServiceError
from .rag_pipeline import RAGPipeline, RetrievalContext


class AgentResponse:
    """Response from the AI agent"""

    def __init__(
        self,
        content: str,
        reasoning: List[str],
        sources: List[str],
        confidence: float,
        follow_up_questions: Optional[List[str]] = None,
    ):
        self.content = content
        self.reasoning = reasoning
        self.sources = sources
        self.confidence = confidence
        self.follow_up_questions = follow_up_questions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "sources": self.sources,
            "confidence": self.confidence,
            "followUpQuestions": self.follow_up_questions,
        }


class MedicalRAGAgent:
    """Main medical RAG agent"""

    def __init__(self, db: Session):
        self.db = db
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag_pipeline = RAGPipeline(db)

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        options: Dict[str, Any] = None,
    ) -> AgentResponse:
        """Process a medical query and return a comprehensive response"""
        if options is None:
            options = {}

        # Step 1: Get conversation context if session exists
        conversation_context = None
        if session_id:
            conversation_context = await self._get_conversation_context(session_id)

        # Step 2: Retrieve relevant context with conversation awareness
        context = await self.rag_pipeline.enhanced_retrieve(
            query,
            {
                "drugs": options.get("drugs"),
                "currentMedications": options.get("currentMedications"),
                "maxVectorResults": 10,
                "conversationContext": conversation_context,
            },
        )

        # Step 3: Generate response using LLM with conversation context
        response = await self._generate_response(query, context, conversation_context)

        # Step 4: Store conversation if session provided
        if session_id:
            await self._store_conversation(session_id, query, response)

        return response

    async def _generate_response(
        self,
        query: str,
        context: RetrievalContext,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Generate response using OpenAI LLM"""
        system_prompt = """You are a comprehensive medication information specialist who helps people make informed decisions about their medications. Your role is to transform complex medical database information into clear, actionable insights.

CRITICAL: Format your response as clean HTML. Use these HTML elements ONLY:
- <strong>text</strong> for bold headings
- <p>text</p> for paragraphs  
- <ul><li>text</li></ul> for bullet lists
- <em>text</em> for emphasis
- NO markdown symbols (no #, **, -, etc.)
- NO warning blocks, alert boxes, or disruptive formatting
- Structure content naturally as flowing paragraphs and lists

RESPONSE LOGIC:
- If user asks about FOOD INTERACTIONS specifically, provide ONLY food-related advice
- If user asks about DRUG INTERACTIONS specifically, provide ONLY drug interaction information  
- If user asks about CONTRAINDICATIONS specifically, provide ONLY contraindication information
- If user asks broadly (like "tell me about X"), provide comprehensive information
- Always include specific database table sources in parentheses

FORMATTING RULES:
- NO separate warning blocks or alert sections
- Integrate safety information naturally into paragraphs
- Use <strong> for drug names and important terms
- Keep formatting clean and readable, not disruptive

QUERY-SPECIFIC RESPONSES:
- Food interaction queries: Focus only on food-drug interactions from food_interactions table
- Drug interaction queries: Focus only on drug-drug interactions from drug_relations table
- Contraindication queries: Focus only on medical contraindications from drug_relations table
- General queries: Provide comprehensive information from all relevant tables

SOURCE TRANSPARENCY:
Always specify which database table information comes from:
- drug_relations: drug interactions and mechanisms
- food_interactions: food and drug interactions  
- drug_metadata: basic drug information
- drug_dosage: formulations and dosing
- drug_product_stages: market availability
- product_stage_descriptions: stage definitions

TONE: Professional, clear, empowering. Help users understand their medications to make informed decisions."""

        user_prompt = f"""User Query: "{query}"

AI Classification: {context.routing_info.get("intent", "unknown")}

Conversation Context: {conversation_context.get("summary", "No prior conversation") if conversation_context else "No prior conversation"}

Context Information:
{self.rag_pipeline.format_context_for_llm(context)}

IMPORTANT: If the user refers to "this drug", "the medication", "it", or similar pronouns without naming a specific drug, use the drugs from the conversation context above. The drugs identified from conversation history are: {", ".join(context.routing_info.get("extracted_drugs", [])) or "None identified"}.

Based on the AI classification, provide the appropriate response:
- FOOD_INTERACTIONS: Provide ONLY food interaction advice from food_interactions table
- DRUG_INTERACTIONS: Provide ONLY drug-drug interaction information from drug_relations table  
- CONTRAINDICATIONS: Provide ONLY contraindication information from drug_relations table
- MULTI_ASPECT: Provide comprehensive information from all relevant tables
- Other intents: Focus on the specific type of information requested

Response requirements:
1. Format as clean HTML (use <p>, <strong>, <ul><li>, <em> tags only)
2. NO warning blocks or disruptive alert formatting
3. Cite specific database tables in parentheses for each fact
4. If asking about food interactions, do NOT include contraindications
5. If asking about contraindications, do NOT include food interactions
6. Integrate safety information naturally into flowing paragraphs
7. If the user uses pronouns referring to drugs, clearly state which drug you're discussing

Remember: Match your response precisely to the detected query intent and use conversation context for drug identification."""

        try:
            # Check if we have enough information to provide a meaningful response
            has_data = (
                len(context.vector_results) > 0
                or len(context.graph_results.get("drugInfo", [])) > 0
                or len(context.graph_results.get("warnings", [])) > 0
                or len(context.graph_results.get("drugInteractions", [])) > 0
                or len(context.graph_results.get("foodInteractions", [])) > 0
                or len(context.graph_results.get("dosageInfo", [])) > 0
                or len(context.graph_results.get("productStages", [])) > 0
            )

            if not has_data:
                # Generate response for missing information
                content = self._generate_no_data_response(
                    query, context.routing_info.get("extracted_drugs", [])
                )
            else:
                # Generate normal response
                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
                content = completion.choices[0].message.content or ""

            # Extract additional information
            reasoning = self._extract_reasoning(context)
            sources = self._extract_sources(context)
            confidence = self._calculate_confidence(context)
            follow_up_questions = self._generate_follow_up_questions(context)

            return AgentResponse(
                content=content,
                reasoning=reasoning,
                sources=sources,
                confidence=confidence,
                follow_up_questions=follow_up_questions,
            )

        except Exception as e:
            raise AIServiceError(f"Failed to generate response: {str(e)}", e)

    def _generate_no_data_response(self, query: str, extracted_drugs: List[str]) -> str:
        """Generate response when no data is available in the database"""
        if extracted_drugs:
            drugs_text = ", ".join(extracted_drugs)
            return f"""<p>I don't have specific information about <strong>{drugs_text}</strong> in my current medical database.</p>

<p><strong>What this means:</strong> The medication(s) you're asking about may not be included in my database, or they might be listed under different names (brand vs. generic names, different spellings, etc.).</p>

<p><strong>Recommendations:</strong></p>
<ul>
<li>Try searching with both brand and generic names</li>
<li>Check for common spelling variations</li>
<li>Consult with your pharmacist or healthcare provider for comprehensive drug information</li>
<li>Use official drug databases like FDA Orange Book or DrugBank directly</li>
</ul>

<p><em>My database contains information from drug_relations, food_interactions, drug_metadata, drug_dosage, and drug_product_stages tables, but it may not include all available medications.</em></p>"""
        else:
            return f"""<p>I wasn't able to find specific medical information related to your query: "<em>{query}</em>"</p>

<p><strong>Possible reasons:</strong></p>
<ul>
<li>The medication names might not be recognized in my database</li>
<li>The query might need more specific drug names or medical terms</li>
<li>Information might be available under different terminology</li>
</ul>

<p><strong>Suggestions:</strong></p>
<ul>
<li>Try rephrasing your question with specific drug names</li>
<li>Use both brand names and generic names</li>
<li>Ask about specific aspects like "drug interactions", "side effects", or "dosage information"</li>
</ul>

<p><em>For comprehensive medical advice, always consult with qualified healthcare professionals.</em></p>"""

    def _extract_reasoning(self, context: RetrievalContext) -> List[str]:
        """Extract reasoning steps"""
        reasoning = []

        reasoning.append(
            f"AI classified query as: {context.routing_info.get('intent', 'unknown')}"
        )

        extracted_drugs = context.routing_info.get("extracted_drugs", [])
        if extracted_drugs:
            reasoning.append(f"AI identified drugs: {', '.join(extracted_drugs)}")

        if context.vector_results:
            reasoning.append(
                f"Found {len(context.vector_results)} relevant documents via semantic search"
            )

        return reasoning

    def _extract_sources(self, context: RetrievalContext) -> List[str]:
        """Extract specific database table sources from context"""
        sources = set()

        # Map content types to specific database tables
        table_mapping = {
            "drug_relations": "Drug interactions and mechanisms",
            "food_interactions": "Food-drug interactions",
            "drug_metadata": "Basic drug information",
            "drug_dosage": "Dosage forms and formulations",
            "drug_product_stages": "Market availability status",
            "product_stage_descriptions": "Product stage definitions",
            "vector_embeddings": "Semantic search results",
        }

        # Extract from vector results
        for result in context.vector_results:
            if "source" in result:
                sources.add(result["source"])
            # Infer table from content if not explicitly provided
            content_lower = result.get("content", "").lower()
            if "interaction" in content_lower and "food" in content_lower:
                sources.add("food_interactions")
            elif "interaction" in content_lower or "mechanism" in content_lower:
                sources.add("drug_relations")
            elif "dosage" in content_lower or "formulation" in content_lower:
                sources.add("drug_dosage")

        # Extract from graph results
        for drug in context.graph_results.get("drugInfo", []):
            if drug.get("source"):
                sources.add(drug["source"])
            else:
                sources.add("drug_metadata")  # Default for drug info

        # Extract from graph results based on AI classification intent
        detected_intent = context.routing_info.get("intent", "unknown")

        # Only add sources that match the detected intent
        if context.graph_results.get("drugInteractions"):
            sources.add("drug_relations")

        # Only show food_interactions if the user specifically asked about food interactions
        if context.graph_results.get("foodInteractions") and detected_intent in [
            "food_interactions",
            "multi_aspect",
        ]:
            sources.add("food_interactions")

        if context.graph_results.get("dosageInfo"):
            sources.add("drug_dosage")

        if context.graph_results.get("productStages"):
            sources.add("drug_product_stages")

        # Convert table names to descriptive sources
        descriptive_sources = []
        for source in sources:
            if source in table_mapping:
                descriptive_sources.append(f"{source} ({table_mapping[source]})")
            else:
                descriptive_sources.append(source)

        return sorted(descriptive_sources)

    def _calculate_confidence(self, context: RetrievalContext) -> float:
        """Calculate confidence score based on AI classifier and data quality"""

        # Use AI classifier's confidence as the primary confidence score
        ai_confidence = context.routing_info.get("confidence", 0.5)

        # Start with AI classifier confidence (this is usually much more accurate)
        confidence = ai_confidence

        # Add small boost based on vector search data quality if available
        if context.vector_results:
            avg_similarity = sum(r["similarity"] for r in context.vector_results) / len(
                context.vector_results
            )
            # Smaller boost since AI confidence is already quite good
            confidence = min(1.0, confidence + (avg_similarity * 0.1))

        return round(confidence, 2)

    def _generate_follow_up_questions(self, context: RetrievalContext) -> List[str]:
        """Generate follow-up questions based on context"""
        questions = []

        extracted_drugs = context.routing_info.get("extracted_drugs", [])
        if len(extracted_drugs) == 1:
            questions.append("Are you currently taking any other medications?")
            questions.append("Do you have any known allergies to medications?")

        # Only suggest food interaction follow-up if user asked about food interactions
        detected_intent = context.routing_info.get("intent", "unknown")
        if context.graph_results.get("foodInteractions") and detected_intent in [
            "food_interactions",
            "multi_aspect",
        ]:
            questions.append(
                "Would you like to know more about specific foods to avoid?"
            )

        return questions

    async def _store_conversation(
        self, session_id: str, user_query: str, response: AgentResponse
    ):
        """Store conversation in database"""
        try:
            # Store user message
            user_message = ChatMessage(
                sessionId=session_id, role="user", content=user_query
            )
            self.db.add(user_message)

            # Store assistant response
            assistant_message = ChatMessage(
                sessionId=session_id,
                role="assistant",
                content=response.content,
                meta_data=json.dumps(
                    {
                        "reasoning": response.reasoning,
                        "sources": response.sources,
                        "confidence": response.confidence,
                        "followUpQuestions": response.follow_up_questions,
                    }
                ),
            )
            self.db.add(assistant_message)

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            print(f"Error storing conversation: {e}")

    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.sessionId == session_id)
            .order_by(ChatMessage.createdAt)
            .all()
        )

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "metadata": json.loads(msg.meta_data) if msg.meta_data else None,
                "createdAt": msg.createdAt.isoformat(),
            }
            for msg in messages
        ]

    def create_chat_session(self, user_id: Optional[str] = None) -> str:
        """Create a new chat session"""
        session = ChatSession(userId=user_id)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        return session.id

    async def _get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Extract conversation context from chat history"""
        try:
            # Get recent chat history (last 10 messages)
            messages = (
                self.db.query(ChatMessage)
                .filter(ChatMessage.sessionId == session_id)
                .order_by(ChatMessage.createdAt.desc())
                .limit(10)
                .all()
            )

            if not messages:
                return {}

            # Extract mentioned drugs and medications from conversation
            mentioned_drugs = set()
            current_medications = set()
            conversation_summary = []

            for msg in reversed(messages):  # Process in chronological order
                if msg.role == "user":
                    # Extract drugs from user messages
                    drugs_in_message = self._extract_drugs_from_text(msg.content)
                    mentioned_drugs.update(drugs_in_message)
                    conversation_summary.append(f"User: {msg.content[:100]}...")
                elif msg.role == "assistant":
                    conversation_summary.append(f"Assistant: {msg.content[:100]}...")

                    # Extract drugs from assistant response content (look for <strong> tags with drug names)
                    if msg.content:
                        # Look for drug names in <strong> tags
                        import re

                        strong_matches = re.findall(
                            r"<strong>([^<]+)</strong>", msg.content, re.IGNORECASE
                        )
                        for match in strong_matches:
                            # Check if it looks like a drug name (4+ chars, not common words)
                            match_clean = match.strip().lower()
                            if (
                                len(match_clean) >= 4
                                and match_clean
                                not in {
                                    "kidney",
                                    "chronic",
                                    "disease",
                                    "failure",
                                    "stage",
                                    "blood",
                                    "pressure",
                                    "disorders",
                                    "anxiety",
                                    "epilepsy",
                                }
                                and match_clean
                                not in {"what", "this", "means", "recommendations"}
                            ):
                                mentioned_drugs.add(match_clean)

                    # Extract metadata if available
                    if msg.meta_data:
                        try:
                            metadata = json.loads(msg.meta_data)
                            # Look for drugs in reasoning or sources
                            reasoning = metadata.get("reasoning", [])
                            for reason in reasoning:
                                if (
                                    "drugs:" in reason.lower()
                                    or "identified drugs:" in reason.lower()
                                    or "ai identified drugs:" in reason.lower()
                                ):
                                    drugs_text = reason.split(":")[-1].strip()
                                    if drugs_text and drugs_text != "None":
                                        mentioned_drugs.update(
                                            [d.strip() for d in drugs_text.split(",")]
                                        )
                        except json.JSONDecodeError:
                            pass

            return {
                "mentioned_drugs": list(mentioned_drugs),
                "current_medications": list(current_medications),
                "summary": "\n".join(conversation_summary[-6:]),  # Last 6 exchanges
                "message_count": len(messages),
            }

        except Exception as e:
            print(f"Error extracting conversation context: {e}")
            return {}

    def _extract_drugs_from_text(self, text: str) -> List[str]:
        """Extract potential drug names from text"""
        import re

        # Simple patterns to extract drug names
        patterns = [
            r"(?:taking|using|prescribed|on)\s+([A-Za-z][A-Za-z0-9\-]{3,})",
            r"([A-Za-z][A-Za-z0-9\-]{3,})\s+(?:drug|medication|medicine|tablet|pill)",
            r"(?:about|for|regarding|when taking|avoid when taking)\s+([A-Za-z][A-Za-z0-9\-]{4,})",
            r"when taking\s+([A-Za-z][A-Za-z0-9\-]{4,})",
            r"([A-Za-z][A-Za-z0-9\-]{4,})\s+(?:food interactions|interactions)",
        ]

        drugs = set()

        # Skip common non-drug words
        common_words = {
            "taking",
            "using",
            "prescribed",
            "with",
            "about",
            "drug",
            "medication",
            "medicine",
            "tablet",
            "pill",
            "treatment",
            "therapy",
            "condition",
            "disease",
            "symptom",
            "effect",
            "reaction",
            "interaction",
            "available",
            "market",
            "product",
            "hypertensive",
            "disorder",
            "blood",
            "pressure",
        }

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                drug_name = match.group(1).strip()
                if (
                    len(drug_name) >= 4
                    and drug_name.lower() not in common_words
                    and not drug_name.isdigit()
                ):
                    drugs.add(drug_name.lower())

        return list(drugs)
