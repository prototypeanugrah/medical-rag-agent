"""
Setup API endpoints
"""

import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..lib.data_ingestion import DataIngestionService
from ..lib.database import get_db
from ..models.schemas import SetupRequest, SetupResponse

router = APIRouter()


@router.post("/", response_model=SetupResponse)
async def setup_action(setup_request: SetupRequest, db: Session = Depends(get_db)):
    """Perform setup actions"""
    try:
        ingestion_service = DataIngestionService(db)

        if setup_request.action == "initialize":
            await ingestion_service.initialize_database()
            return SetupResponse(
                success=True,
                message="Database initialized with sample data and embeddings",
            )

        elif setup_request.action == "clear":
            ingestion_service.clear_all_data()
            return SetupResponse(success=True, message="All data cleared from database")

        elif setup_request.action == "reindex":
            await ingestion_service.reindex_embeddings()
            return SetupResponse(
                success=True, message="Embeddings reindexed successfully"
            )

        elif setup_request.action == "sample":
            await ingestion_service.create_sample_data()
            return SetupResponse(success=True, message="Sample data created")

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Use: initialize, clear, reindex, or sample",
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Setup API error: {e}")
        return SetupResponse(
            success=False, error="Setup failed", data={"details": str(e)}
        )


@router.get("/", response_model=SetupResponse)
async def get_setup_status(db: Session = Depends(get_db)):
    """Get setup status"""
    try:
        ingestion_service = DataIngestionService(db)
        stats = ingestion_service.get_ingestion_stats()

        return SetupResponse(
            success=True,
            data={
                "stats": stats,
                "status": "ready",
                "apiKeys": {"openai": bool(os.getenv("OPENAI_API_KEY"))},
            },
        )

    except Exception as e:
        print(f"Setup status API error: {e}")
        return SetupResponse(success=False, error="Failed to get status")
