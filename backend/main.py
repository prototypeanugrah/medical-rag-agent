"""
FastAPI backend for the Medical RAG Agent
"""

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.chat import router as chat_router
from .api.ingest import router as ingest_router
from .api.setup import router as setup_router
from .lib.database import init_db
from .lib.error_handling import handle_error

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Medical RAG Agent API",
    description="Backend API for Medical Knowledge Retrieval and Drug Interaction Checking",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(ingest_router, prefix="/api/ingest", tags=["ingestion"])
app.include_router(setup_router, prefix="/api/setup", tags=["setup"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    await init_db()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    error_response = handle_error(exc)
    return JSONResponse(
        status_code=error_response["status_code"],
        content={
            "success": False,
            "error": error_response["message"],
            "code": error_response["code"],
        },
    )


@app.get("/")
async def root():
    return {"message": "Medical RAG Agent API", "status": "running", "version": "1.0.0"}


@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "healthy", "service": "medical-rag-agent", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )
