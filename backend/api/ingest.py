"""
Data ingestion API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..lib.data_ingestion import DataIngestionService
from ..lib.database import get_db
from ..models.schemas import IngestionRequest, IngestionResponse

router = APIRouter()


@router.post("/", response_model=IngestionResponse)
async def ingest_data(
    ingestion_request: IngestionRequest, db: Session = Depends(get_db)
):
    """Ingest medical data"""
    try:
        ingestion_service = DataIngestionService(db)

        if ingestion_request.type == "drug_relations":
            # Type check and convert
            drug_relations = []
            for item in ingestion_request.data:
                if hasattr(item, "relation"):  # It's a DrugRelationData
                    drug_relations.append(item)
                else:
                    raise HTTPException(
                        status_code=400, detail="Invalid data format for drug_relations"
                    )

            result = await ingestion_service.ingest_drug_relations(drug_relations)

        elif ingestion_request.type == "food_interactions":
            # Type check and convert
            food_interactions = []
            for item in ingestion_request.data:
                if hasattr(item, "drugName") and hasattr(
                    item, "interaction"
                ):  # It's a FoodInteractionData
                    food_interactions.append(item)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid data format for food_interactions",
                    )

            result = await ingestion_service.ingest_food_interactions(food_interactions)

        else:
            raise HTTPException(status_code=400, detail="Invalid ingestion type")

        # Reindex embeddings if requested
        if ingestion_request.reindex:
            await ingestion_service.reindex_embeddings()

        return IngestionResponse(success=True, data=result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ingestion API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=IngestionResponse)
async def get_ingestion_stats(db: Session = Depends(get_db)):
    """Get ingestion statistics"""
    try:
        ingestion_service = DataIngestionService(db)
        stats = ingestion_service.get_ingestion_stats()

        return IngestionResponse(success=True, data=stats)

    except Exception as e:
        print(f"Ingestion stats API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
