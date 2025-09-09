"""
Knowledge graph service for drug information and interaction analysis
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from ..models.database import (
    DrugDosage,
    DrugProductStage,
    DrugRelation,
    FoodInteraction,
    ProductStageDescription,
)
from ..models.schemas import (
    DrugInfo,
    InteractionSeverity,
    InteractionType,
    InteractionWarning,
)


class KnowledgeGraphService:
    """Service for querying the medical knowledge graph"""

    def __init__(self, db: Session):
        self.db = db

    def find_drug_by_name(self, drug_name: str) -> List[DrugInfo]:
        """Find drugs by name from the knowledge graph"""
        drug_relations = (
            self.db.query(DrugRelation)
            .filter(
                or_(
                    DrugRelation.xName.ilike(f"%{drug_name}%"),
                    DrugRelation.yName.ilike(f"%{drug_name}%"),
                )
            )
            .all()
        )

        drugs = []
        seen_drugs = set()

        for relation in drug_relations:
            # Check x entity
            if (
                drug_name.lower() in relation.xName.lower()
                and relation.xId not in seen_drugs
            ):
                drugs.append(
                    DrugInfo(
                        id=relation.xId,
                        name=relation.xName,
                        type=relation.xType,
                        source=relation.xSource,
                    )
                )
                seen_drugs.add(relation.xId)

            # Check y entity
            if (
                drug_name.lower() in relation.yName.lower()
                and relation.yId not in seen_drugs
            ):
                drugs.append(
                    DrugInfo(
                        id=relation.yId,
                        name=relation.yName,
                        type=relation.yType,
                        source=relation.ySource,
                    )
                )
                seen_drugs.add(relation.yId)

        return drugs
        
    def _is_product_withdrawn(self, product_stage: str) -> bool:
        """Check if a product stage indicates withdrawal/discontinuation"""
        withdrawn_stages = [
            "WITHDRAWN", "DISCONTINUED", "SUSPENDED", "CANCELLED",
            "TERMINATED", "STOPPED", "RECALLED", "BANNED"
        ]
        return product_stage.upper() in withdrawn_stages
        
    def _get_availability_status(self, product_stage: str) -> str:
        """Get human-readable availability status from product stage"""
        stage_upper = product_stage.upper()
        
        # Withdrawn/Discontinued products
        if stage_upper in ["WITHDRAWN", "DISCONTINUED", "SUSPENDED"]:
            return "No longer available"
        elif stage_upper in ["CANCELLED", "TERMINATED", "STOPPED"]:
            return "Development stopped"
        elif stage_upper in ["RECALLED", "BANNED"]:
            return "Recalled from market"
            
        # Available products
        elif stage_upper in ["APPROVED", "MARKETED", "LAUNCHED"]:
            return "Currently available"
        elif stage_upper in ["GENERIC_AVAILABLE", "OTC"]:
            return "Available (generic/OTC)"
            
        # Development stages
        elif "PHASE" in stage_upper or stage_upper in ["CLINICAL", "TRIAL"]:
            return "In clinical development"
        elif stage_upper in ["EXPERIMENTAL", "RESEARCH"]:
            return "In research phase"
            
        # Default
        else:
            return f"Status: {product_stage}"

    def get_drug_relations(self, drug_name: str) -> List[DrugRelation]:
        """Get all relations for a drug"""
        return (
            self.db.query(DrugRelation)
            .filter(
                or_(
                    DrugRelation.xName.ilike(f"%{drug_name}%"),
                    DrugRelation.yName.ilike(f"%{drug_name}%"),
                )
            )
            .all()
        )

    def get_food_interactions(self, drug_name: str) -> List[FoodInteraction]:
        """Get food interactions for a drug"""
        return (
            self.db.query(FoodInteraction)
            .filter(FoodInteraction.drugName.ilike(f"%{drug_name}%"))
            .all()
        )

    def get_contraindications(self, drug_name: str) -> List[DrugRelation]:
        """Get contraindications for a drug"""
        return (
            self.db.query(DrugRelation)
            .filter(
                and_(
                    DrugRelation.relation == "contraindication",
                    or_(
                        DrugRelation.xName.ilike(f"%{drug_name}%"),
                        DrugRelation.yName.ilike(f"%{drug_name}%"),
                    ),
                )
            )
            .all()
        )

    def get_drug_dosage_info(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get dosage information for a drug with product stage context"""
        dosages = (
            self.db.query(DrugDosage)
            .filter(DrugDosage.productName.ilike(f"%{drug_name}%"))
            .all()
        )
        
        # Get product stage for context
        product_stage_info = self.get_drug_product_stage(drug_name)
        
        # Convert to dict and add product stage context
        dosage_list = []
        for dosage in dosages:
            dosage_dict = {
                "id": dosage.id,
                "drugId": dosage.drugId,
                "productName": dosage.productName,
                "dosageForm": dosage.dosageForm,
                "route": dosage.route,
                "strength": dosage.strength,
                "manufacturer": dosage.manufacturer,
                "source_table": "drug_dosage"
            }
            
            # Add product stage context if available
            if product_stage_info:
                dosage_dict["productStage"] = product_stage_info["productStage"]
                dosage_dict["stageDescription"] = product_stage_info["description"]
                dosage_dict["isWithdrawn"] = self._is_product_withdrawn(product_stage_info["productStage"])
                dosage_dict["availabilityStatus"] = self._get_availability_status(product_stage_info["productStage"])
            else:
                dosage_dict["productStage"] = "UNKNOWN"
                dosage_dict["stageDescription"] = "Product stage information not available"
                dosage_dict["isWithdrawn"] = False
                dosage_dict["availabilityStatus"] = "Status Unknown"
                
            dosage_list.append(dosage_dict)
        
        return dosage_list

    def get_drug_product_stage(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get product stage information for a drug"""
        stage = (
            self.db.query(DrugProductStage)
            .filter(DrugProductStage.drugName.ilike(f"%{drug_name}%"))
            .first()
        )

        if not stage:
            return None

        # Get stage description
        description = (
            self.db.query(ProductStageDescription)
            .filter(ProductStageDescription.stageCode == stage.productStage)
            .first()
        )

        return {
            "drugName": stage.drugName,
            "productStage": stage.productStage,
            "description": description.description
            if description
            else "No description available",
            "stageCode": stage.productStage,
            "source_table": "drug_product_stages",
        }

    def get_drug_drug_interactions(
        self, drug_name: str, current_medications: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get drug-drug interactions from drug relations"""
        if current_medications is None:
            current_medications = []

        interactions = []

        # Find drug_drug relations involving the primary drug
        drug_relations = (
            self.db.query(DrugRelation)
            .filter(
                and_(
                    DrugRelation.relation == "drug_drug",
                    or_(
                        DrugRelation.xName.ilike(f"%{drug_name}%"),
                        DrugRelation.yName.ilike(f"%{drug_name}%"),
                    ),
                )
            )
            .all()
        )

        for relation in drug_relations:
            # Determine which drug is the primary and which is the interacting drug
            if drug_name.lower() in relation.xName.lower():
                drug1_name = relation.xName
                drug1_id = relation.xId
                drug2_name = relation.yName
                drug2_id = relation.yId
            else:
                drug1_name = relation.yName
                drug1_id = relation.yId
                drug2_name = relation.xName
                drug2_id = relation.xId

            interactions.append(
                {
                    "id": relation.id,
                    "drug1Name": drug1_name,
                    "drug1Id": drug1_id,
                    "drug2Name": drug2_name,
                    "drug2Id": drug2_id,
                    "interaction": relation.displayRelation,
                    "interactionType": relation.relationType,
                    "source": f"{relation.xSource}, {relation.ySource}",
                }
            )

        return interactions

    def analyze_drug_interactions(
        self, primary_drug: str, current_medications: List[str] = None
    ) -> List[InteractionWarning]:
        """Analyze drug interactions and return warnings"""
        if current_medications is None:
            current_medications = []

        warnings = []

        # Check food interactions
        food_interactions = self.get_food_interactions(primary_drug)
        for interaction in food_interactions:
            warnings.append(
                InteractionWarning(
                    type=InteractionType.FOOD,
                    severity=InteractionSeverity.MEDIUM,
                    description=interaction.interaction,
                    source=interaction.source,
                    relatedFoods=self._extract_food_items(interaction.interaction),
                )
            )

        # Check contraindications
        contraindications = self.get_contraindications(primary_drug)
        for contraindication in contraindications:
            related_disease = (
                contraindication.yName
                if primary_drug.lower() in contraindication.xName.lower()
                else contraindication.xName
            )

            warnings.append(
                InteractionWarning(
                    type=InteractionType.DISEASE_CONTRAINDICATION,
                    severity=InteractionSeverity.HIGH,
                    description=f"{primary_drug} is contraindicated with {related_disease}",
                    source=f"{contraindication.xSource}, {contraindication.ySource}",
                )
            )

        return warnings

    def _extract_food_items(self, interaction_text: str) -> List[str]:
        """Extract food items mentioned in interaction text"""
        common_foods = [
            "chamomile",
            "garlic",
            "ginger",
            "ginkgo",
            "ginseng",
            "alcohol",
            "grapefruit",
            "vitamin k",
            "dairy",
            "caffeine",
            "turmeric",
        ]

        return [food for food in common_foods if food in interaction_text.lower()]

    def _assess_interaction_severity(
        self, interaction: str, interaction_type: Optional[str] = None
    ) -> InteractionSeverity:
        """Assess the severity of an interaction"""
        high_risk_terms = [
            "contraindicated",
            "avoid",
            "dangerous",
            "severe",
            "fatal",
            "toxic",
        ]
        medium_risk_terms = ["caution", "monitor", "adjust", "reduce", "increase"]

        text = f"{interaction} {interaction_type or ''}".lower()

        if any(term in text for term in high_risk_terms):
            return InteractionSeverity.HIGH
        elif any(term in text for term in medium_risk_terms):
            return InteractionSeverity.MEDIUM
        else:
            return InteractionSeverity.LOW

    def get_comprehensive_drug_info(
        self, drug_name: str, current_medications: List[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive drug information"""
        if current_medications is None:
            current_medications = []

        drug_info = self.find_drug_by_name(drug_name)
        relations = self.get_drug_relations(drug_name)
        food_interactions = self.get_food_interactions(drug_name)
        contraindications = self.get_contraindications(drug_name)
        drug_drug_interactions = self.get_drug_drug_interactions(
            drug_name, current_medications
        )
        warnings = self.analyze_drug_interactions(drug_name, current_medications)
        dosage_info = self.get_drug_dosage_info(drug_name)
        product_stage = self.get_drug_product_stage(drug_name)

        return {
            "drugInfo": [
                {**drug.dict(), "source_table": "drug_metadata"} for drug in drug_info
            ],
            "productStage": product_stage,
            "relations": [
                {
                    "id": rel.id,
                    "relation": rel.relation,
                    "displayRelation": rel.displayRelation,
                    "xName": rel.xName,
                    "xType": rel.xType,
                    "yName": rel.yName,
                    "yType": rel.yType,
                    "source_table": "drug_relations",
                }
                for rel in relations
            ],
            "foodInteractions": [
                {
                    "id": fi.id,
                    "drugName": fi.drugName,
                    "interaction": fi.interaction,
                    "source": fi.source,
                    "source_table": "food_interactions",
                }
                for fi in food_interactions
            ],
            "drugDrugInteractions": [
                {**interaction, "source_table": "drug_relations"}
                for interaction in drug_drug_interactions
            ],
            "contraindications": [
                {
                    "id": contra.id,
                    "relation": contra.relation,
                    "xName": contra.xName,
                    "yName": contra.yName,
                    "source_table": "drug_relations",
                }
                for contra in contraindications
            ],
            "warnings": [
                {**warning.dict(), "source_table": "drug_relations"}
                for warning in warnings
            ],
            "dosage": dosage_info,  # Already formatted as dicts with all needed fields
        }
