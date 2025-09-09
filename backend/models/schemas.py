"""
Pydantic schemas for API validation and serialization
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


# Data schemas for ingestion
class DrugRelationData(BaseModel):
    relation: str
    display_relation: str
    x_index: int
    x_id: str
    x_type: str
    x_name: str
    x_source: str
    y_index: int
    y_id: str
    y_type: str
    y_name: str
    y_source: str
    relation_type: Optional[str] = None


class FoodInteractionData(BaseModel):
    drugName: str
    drugId: Optional[str] = None
    interaction: str
    source: str


class DrugMetadata(BaseModel):
    drugId: str
    name: Optional[str] = None
    type: Optional[str] = None
    products: List[str] = []


class DrugProductStage(BaseModel):
    drugName: str
    productStage: str


class DrugFoodInteraction(BaseModel):
    drugName: str
    interaction: str


class KnowledgeGraphRelation(BaseModel):
    relation: str
    display_relation: str
    x_index: int
    x_id: str
    x_type: str
    x_name: str
    x_source: str
    y_index: int
    y_id: str
    y_type: str
    y_name: str
    y_source: str
    relation_type: str


class KnowledgeGraphQuery(BaseModel):
    drugName: Optional[str] = None
    drugId: Optional[str] = None
    disease: Optional[str] = None
    symptom: Optional[str] = None
    interactionType: Optional[str] = None


class RetrievalResult(BaseModel):
    content: str
    source: str
    metadata: Any
    relevanceScore: float


class ChatMessage(BaseModel):
    id: str
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    metadata: Optional[Any] = None
    createdAt: datetime


class DrugInfo(BaseModel):
    id: str
    name: str
    type: str
    source: str


class InteractionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InteractionType(str, Enum):
    DRUG_DRUG = "drug-drug"
    FOOD = "food"
    DISEASE_CONTRAINDICATION = "disease-contraindication"


class InteractionWarning(BaseModel):
    type: InteractionType
    severity: InteractionSeverity
    description: str
    source: str
    relatedDrugs: Optional[List[str]] = None
    relatedFoods: Optional[List[str]] = None


# API Request/Response schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=2000)
    sessionId: Optional[str] = None
    drugs: Optional[List[str]] = Field(None, max_items=50)
    currentMedications: Optional[List[str]] = Field(None, max_items=50)
    userSymptoms: Optional[List[str]] = Field(None, max_items=20)


class ChatResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class IngestionRequest(BaseModel):
    type: str = Field(..., pattern="^(drug_relations|food_interactions)$")
    data: List[Union[DrugRelationData, FoodInteractionData]]
    reindex: Optional[bool] = False


class IngestionResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class SetupRequest(BaseModel):
    action: str = Field(..., pattern="^(initialize|clear|reindex|sample)$")


class SetupResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None


# Database model responses
class DrugRelationResponse(BaseModel):
    id: str
    relation: str
    displayRelation: str
    xIndex: int
    xId: str
    xType: str
    xName: str
    xSource: str
    yIndex: int
    yId: str
    yType: str
    yName: str
    ySource: str
    relationType: Optional[str]
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True


class ChatSessionResponse(BaseModel):
    id: str
    userId: Optional[str]
    createdAt: datetime
    updatedAt: datetime
    messages: List[ChatMessage] = []

    class Config:
        from_attributes = True
