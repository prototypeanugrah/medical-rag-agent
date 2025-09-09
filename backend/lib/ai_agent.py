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
        warnings: List[Dict[str, Any]],
        sources: List[str],
        confidence: float,
        follow_up_questions: Optional[List[str]] = None,
    ):
        self.content = content
        self.reasoning = reasoning
        self.warnings = warnings
        self.sources = sources
        self.confidence = confidence
        self.follow_up_questions = follow_up_questions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
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
        system_prompt = """You are a comprehensive medication information specialist who helps people make informed decisions about their medications. Your role is to transform complex medical database information into clear, actionable insights for people who want to truly understand their medications.

YOUR APPROACH:
- Assume the user wants comprehensive understanding, not just basic answers
- Always explain WHY things matter, not just WHAT to do
- Present information in layers: immediate answer, then deeper context
- Use your vast database to show connections and patterns they wouldn't find elsewhere
- Always specify which database table information comes from for transparency

CORE STRENGTHS:
- You have access to 1.7M+ drug relationships that reveal hidden connections
- You can cross-reference interactions across multiple data sources simultaneously  
- You provide the "complete picture" that users can't get from generic AI or basic drug websites
- You understand medication context - not just isolated drug facts

RESPONSE STRUCTURE:
1. **Direct Answer** - Address their specific question first
2. **Comprehensive Context** - Show related information from your database they should know about
3. **Cross-References** - Connect information across different database tables
4. **Source Transparency** - Always cite which database table (drug_relations, food_interactions, drug_metadata, drug_dosage, drug_product_stages, product_stage_descriptions)
5. **Decision Support** - Help them understand implications for their specific situation
6. **Healthcare Provider Discussion Points** - Specific questions to bring up with their doctor/pharmacist

INFORMATION DEPTH & PRIORITIZATION:
- Provide comprehensive information by default - these users want to understand fully
- Information is intelligently prioritized based on content analysis:
  * Warnings with keywords like "contraindicated", "avoid", "dangerous" appear first
  * Drug interactions prioritized by critical keywords and description detail
  * Food interactions sorted by description comprehensiveness
  * Dosage forms prioritized by uniqueness (different forms/strengths shown first)
- When there are many data points (100+ drug relations), the system shows the most important ones and summarizes the rest
- Explain mechanisms of action when relevant (from drug_relations table)
- Show interaction patterns across multiple drugs (cross-reference drug_relations with food_interactions)
- Include availability and formulation details (from drug_product_stages and drug_dosage tables)
- Reveal connections in your 1.7M+ drug relationship database that aren't obvious
- Acknowledge when additional information is available beyond what's presented

TONE & STYLE:
- Respectful of their desire for comprehensive information
- Never patronizing or oversimplified
- Acknowledge complexity while making it digestible
- Empowering - help them become informed participants in their healthcare
- Safety-conscious but not fear-mongering
- Assume they can handle detailed medical information

Remember: These users WANT comprehensive information. They're not looking for simple answers - they want to understand their medications deeply so they can make informed decisions and have productive conversations with their healthcare providers."""

        user_prompt = f"""User Query: "{query}"

Context Information:
{self.rag_pipeline.format_context_for_llm(context)}

Please provide a comprehensive response that leverages your extensive medical database. Remember:
- This user wants to understand their medication thoroughly, not just get basic instructions
- The information provided has been intelligently prioritized - most critical items first
- If you see "additional X available" notes, acknowledge there's more data and explain the prioritization
- Show connections and patterns from your 1.7M+ drug relationships that aren't obvious
- Cross-reference information across multiple database tables to provide complete context
- Always specify which database table each piece of information comes from
- Explain the WHY behind medical recommendations, not just the WHAT
- Include related information they should know about, even if not directly asked
- Provide decision support to help them make informed choices
- Give them specific points to discuss with their healthcare provider

Format your response in layers: direct answer first, then comprehensive context with prioritization explanation, then cross-references and decision support."""

        try:
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
            warnings = self._extract_warnings(context)
            reasoning = self._extract_reasoning(context)
            sources = self._extract_sources(context)
            confidence = self._calculate_confidence(context)
            follow_up_questions = self._generate_follow_up_questions(context)

            return AgentResponse(
                content=content,
                reasoning=reasoning,
                warnings=warnings,
                sources=sources,
                confidence=confidence,
                follow_up_questions=follow_up_questions,
            )

        except Exception as e:
            raise AIServiceError(f"Failed to generate response: {str(e)}", e)

    def _extract_warnings(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Extract warnings from context"""
        warnings = []

        # Extract from knowledge graph warnings
        for warning in context.graph_results.get("warnings", []):
            warnings.append(
                {
                    "type": warning["type"],
                    "severity": warning["severity"],
                    "message": warning["description"],
                }
            )

        # Extract critical terms from vector results
        for result in context.vector_results:
            content = result["content"].lower()
            if any(term in content for term in ["contraindicated", "avoid"]):
                warnings.append(
                    {
                        "type": "contraindication",
                        "severity": "high",
                        "message": f"Critical interaction found: {result['content'][:200]}...",
                    }
                )

        return warnings

    def _extract_reasoning(self, context: RetrievalContext) -> List[str]:
        """Extract reasoning steps"""
        reasoning = []

        reasoning.append(f"Query analyzed as: {context.query_analysis.query_type}")

        if context.query_analysis.extracted_drugs:
            reasoning.append(
                f"Identified drugs: {', '.join(context.query_analysis.extracted_drugs)}"
            )

        if context.vector_results:
            reasoning.append(
                f"Found {len(context.vector_results)} relevant documents via semantic search"
            )

        warnings_count = len(context.graph_results.get("warnings", []))
        if warnings_count > 0:
            reasoning.append(
                f"Identified {warnings_count} safety warnings from knowledge graph"
            )

        return reasoning

    def _extract_sources(self, context: RetrievalContext) -> List[str]:
        """Extract sources from context"""
        sources = set()

        for result in context.vector_results:
            sources.add(result["source"])

        for drug in context.graph_results.get("drugInfo", []):
            sources.add(drug["source"])

        for warning in context.graph_results.get("warnings", []):
            if warning.get("source"):
                for src in warning["source"].split(", "):
                    sources.add(src)

        return list(sources)

    def _calculate_confidence(self, context: RetrievalContext) -> float:
        """Calculate confidence score"""
        confidence = context.query_analysis.confidence

        # Boost confidence based on available data
        if context.vector_results:
            avg_similarity = sum(r["similarity"] for r in context.vector_results) / len(
                context.vector_results
            )
            confidence = min(1.0, confidence + (avg_similarity * 0.3))

        if context.graph_results.get("warnings"):
            confidence = min(1.0, confidence + 0.2)

        return round(confidence, 2)

    def _generate_follow_up_questions(self, context: RetrievalContext) -> List[str]:
        """Generate follow-up questions based on context"""
        questions = []

        if len(context.query_analysis.extracted_drugs) == 1:
            questions.append("Are you currently taking any other medications?")
            questions.append("Do you have any known allergies to medications?")

        if context.graph_results.get("foodInteractions"):
            questions.append(
                "Would you like to know more about specific foods to avoid?"
            )

        high_severity_warnings = [
            w
            for w in context.graph_results.get("warnings", [])
            if w.get("severity") == "high"
        ]
        if high_severity_warnings:
            questions.append(
                "Have you discussed these potential interactions with your doctor?"
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
                        "warnings": response.warnings,
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
            r"(?:about|for)\s+([A-Za-z][A-Za-z0-9\-]{4,})",
        ]

        drugs = set()
        text_lower = text.lower()

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
