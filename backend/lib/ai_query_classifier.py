"""
AI-powered Query Classification for Medical RAG System
Uses a small language model to intelligently classify user queries and determine appropriate data sources
"""

import json
import os
import re
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class QueryIntent(Enum):
    """Query intent categories"""

    PRODUCT_AVAILABILITY = "product_availability"
    FOOD_INTERACTIONS = "food_interactions"
    DRUG_INTERACTIONS = "drug_interactions"
    SIDE_EFFECTS = "side_effects"
    CONTRAINDICATIONS = "contraindications"
    DRUG_INFORMATION = "drug_information"
    TREATMENT_INDICATION = "treatment_indication"
    DOSAGE_ADMINISTRATION = "dosage_administration"
    GENERAL_MEDICAL = "general_medical"
    MULTI_ASPECT = "multi_aspect"


class SourceMapping:
    """Maps query intents to appropriate data sources"""

    INTENT_TO_SOURCES = {
        QueryIntent.PRODUCT_AVAILABILITY: [
            "product_stages",
            "stage_descriptions",
            "drug_metadata",
        ],
        QueryIntent.FOOD_INTERACTIONS: ["food_interactions"],
        QueryIntent.DRUG_INTERACTIONS: ["drug_relations"],
        QueryIntent.SIDE_EFFECTS: ["drug_relations"],
        QueryIntent.CONTRAINDICATIONS: ["drug_relations"],
        QueryIntent.DRUG_INFORMATION: ["drug_metadata", "drug_relations"],
        QueryIntent.TREATMENT_INDICATION: ["drug_relations"],
        QueryIntent.DOSAGE_ADMINISTRATION: ["drug_dosage", "drug_metadata"],
        QueryIntent.GENERAL_MEDICAL: ["drug_relations", "drug_metadata"],
        QueryIntent.MULTI_ASPECT: None,  # Use all sources
    }


class AIQueryClassifier:
    """AI-powered query classifier using OpenAI's models"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.drug_patterns = self._build_drug_patterns()

    def _build_drug_patterns(self) -> List[str]:
        """Build patterns to extract drug names from queries"""
        return [
            r"(?:drug|medication|medicine|pill|tablet|capsule)\s+([A-Za-z][A-Za-z0-9\-]{2,})",
            r"(?:taking|using|prescribed|on)\s+([A-Za-z][A-Za-z0-9\-]{3,})",
            r"([A-Za-z][A-Za-z0-9\-]{3,})\s+(?:drug|medication|medicine|tablet|pill)",
            r"(?:^|\s)([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)?)\s+(?:and|with|or|for|\?|$)",
            r"(?:with|for|about)\s+([A-Za-z][A-Za-z0-9\-]{4,})",
        ]

    def extract_drugs(self, query: str) -> List[str]:
        """Extract drug names from the query using regex patterns"""
        drugs = set()

        for pattern in self.drug_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                drug_name = match.group(1).strip()
                if self._is_valid_drug_name(drug_name):
                    drugs.add(drug_name.lower())

        return list(drugs)

    def _is_valid_drug_name(self, name: str) -> bool:
        """Check if extracted name is likely a valid drug name"""
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
            "capsule",
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
            "brand",
            "generic",
            "approved",
        }

        return (
            len(name) >= 4
            and name.lower() not in common_words
            and not name.isdigit()
            and not all(c in ".-_" for c in name)
        )

    def classify_query_intent(self, query: str) -> Dict[str, any]:
        """Use AI to classify the query intent and confidence"""

        system_prompt = """You are an expert medical query classifier. Your job is to analyze user queries about medications and classify them into specific intents.

            Available Intent Categories:
            1. product_availability - Questions about what products/medications are available, marketed, approved, or in development
            2. food_interactions - Questions about food interactions, dietary restrictions, or what to eat/avoid with medications
            3. drug_interactions - Questions about combining medications, drug-drug interactions, or taking multiple drugs together
            4. side_effects - Questions about adverse effects, reactions, or symptoms caused by medications
            5. contraindications - Questions about when NOT to take medications, dangerous conditions, or warnings
            6. drug_information - General information about what a drug is, how it works, or its properties
            7. treatment_indication - Questions about what conditions a drug treats or is used for
            8. dosage_administration - Questions about dosage forms, strengths, routes of administration, how much to take, or how to take medications
            9. general_medical - Vague medical questions that don't fit other categories
            10. multi_aspect - Queries asking for comprehensive information covering multiple aspects

            Instructions:
            - Analyze the user query carefully
            - Choose the SINGLE most appropriate intent category
            - If the query clearly asks for multiple types of information, choose "multi_aspect"
            - Provide a confidence score between 0.0 and 1.0
            - Extract any drug names mentioned in the query

            Response format (JSON only):
            {
                "intent": "intent_name",
                "confidence": 0.85,
                "reasoning": "Brief explanation of why this intent was chosen",
                "extracted_drugs": ["drug1", "drug2"]
            }
        """

        user_prompt = f"""Classify this medical query:

            Query: "{query}"

            Analyze the query and respond with the classification in JSON format.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                result = json.loads(content)

                # Validate the response
                if "intent" not in result:
                    raise ValueError("Missing 'intent' in response")

                # Ensure confidence is within bounds
                confidence = min(1.0, max(0.0, result.get("confidence", 0.5)))

                # Validate intent
                intent_name = result["intent"]
                try:
                    intent = QueryIntent(intent_name)
                except ValueError:
                    # Fallback to general_medical for invalid intents
                    intent = QueryIntent.GENERAL_MEDICAL
                    confidence = 0.3

                return {
                    "intent": intent.value,
                    "confidence": confidence,
                    "reasoning": result.get("reasoning", "AI classification"),
                    "extracted_drugs": result.get("extracted_drugs", []),
                }

            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "intent": QueryIntent.GENERAL_MEDICAL.value,
                    "confidence": 0.3,
                    "reasoning": "Failed to parse AI response",
                    "extracted_drugs": [],
                }

        except Exception as e:
            print(f"AI classification error: {e}")
            # Fallback to regex-based extraction for drugs
            extracted_drugs = self.extract_drugs(query)
            return {
                "intent": QueryIntent.GENERAL_MEDICAL.value,
                "confidence": 0.3,
                "reasoning": f"AI service error: {str(e)}",
                "extracted_drugs": extracted_drugs,
            }

    def route_query(
        self, query: str, drugs: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """Route query to appropriate data sources using AI classification"""

        # Get AI classification
        ai_result = self.classify_query_intent(query)

        # Combine AI-extracted drugs with provided drugs
        all_drugs = list(set((drugs or []) + ai_result["extracted_drugs"]))

        # Get intent and determine sources
        intent = ai_result["intent"]
        confidence = ai_result["confidence"]

        # Get appropriate sources based on intent
        try:
            intent_enum = QueryIntent(intent)
            recommended_sources = SourceMapping.INTENT_TO_SOURCES.get(intent_enum)
        except ValueError:
            # Invalid intent, use all sources
            recommended_sources = None

        # Determine if we should use focused search
        # Use focused search for high confidence single-intent queries
        use_focused_search = (
            confidence >= 0.7
            and intent != QueryIntent.MULTI_ASPECT.value
            and recommended_sources is not None
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "extracted_drugs": all_drugs,
            "recommended_sources": recommended_sources,
            "use_focused_search": use_focused_search,
            "reasoning": [
                f"AI classified query as: {intent} (confidence: {confidence:.2f})",
                ai_result["reasoning"],
                f"Identified drugs: {', '.join(all_drugs) or 'None'}",
                f"Strategy: {'Focused search' if use_focused_search else 'Broad search across all sources'}",
                f"Data sources: {recommended_sources or 'All available sources'}",
            ],
        }


# Example usage and testing
if __name__ == "__main__":
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        exit(1)

    classifier = AIQueryClassifier()

    # Test queries
    test_queries = [
        "What products are available in the market for aspirin?",
        "What food interactions will happen if I take warfarin?",
        "Can I take ibuprofen and acetaminophen together?",
        "What are the side effects of metformin?",
        "Is aspirin contraindicated for people with asthma?",
        "What is lisinopril used to treat?",
        "How much aspirin should I take daily?",
        "Tell me about warfarin: interactions, side effects, and available products",
    ]

    print("🤖 AI-Powered Query Classification Test")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 80)

        result = classifier.route_query(query)

        # Color coding for confidence levels
        confidence = result["confidence"]
        if confidence >= 0.8:
            confidence_color = "🟢"  # High confidence
        elif confidence >= 0.6:
            confidence_color = "🟡"  # Medium confidence
        else:
            confidence_color = "🔴"  # Low confidence

        print(f"    Intent: {result['intent']}")
        print(f"    Confidence: {confidence_color} {confidence:.2f}")
        print(f"    Extracted Drugs: {result['extracted_drugs'] or 'None'}")
        print(
            f"    Focused Search: {'✅ Yes' if result['use_focused_search'] else '❌ No (broad search)'}"
        )

        if result["recommended_sources"]:
            print(f"    Data Sources: {', '.join(result['recommended_sources'])}")
        else:
            print("    Data Sources: All sources (comprehensive search)")

        print("    Reasoning:")
        for reason in result["reasoning"]:
            print(f"      • {reason}")
