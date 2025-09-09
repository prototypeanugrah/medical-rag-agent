"""
Chat API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..lib.ai_agent import MedicalRAGAgent
from ..lib.database import get_db
from ..lib.error_handling import (
    RateLimiter,
    ValidationError,
    sanitize_input,
    validate_query,
)
from ..models.schemas import ChatRequest, ChatResponse

router = APIRouter()

# Initialize rate limiter (10 requests per minute)
rate_limiter = RateLimiter(max_requests=10, window_ms=60000)


@router.post("/", response_model=ChatResponse)
async def chat(
    request: Request, chat_request: ChatRequest, db: Session = Depends(get_db)
):
    """Process a chat message"""
    try:
        # Get client IP for rate limiting
        client_ip = (
            request.headers.get("x-forwarded-for")
            or request.headers.get("x-real-ip")
            or str(request.client.host)
        )

        # Check rate limit
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Please try again later."
            )

        # Validate and sanitize input
        if not validate_query(chat_request.message):
            raise ValidationError("Invalid query format or length")

        sanitized_message = sanitize_input(chat_request.message)
        sanitized_drugs = [sanitize_input(drug) for drug in (chat_request.drugs or [])]
        sanitized_medications = [
            sanitize_input(med) for med in (chat_request.currentMedications or [])
        ]
        sanitized_symptoms = [
            sanitize_input(symptom) for symptom in (chat_request.userSymptoms or [])
        ]

        # Create agent
        agent = MedicalRAGAgent(db)

        # Create session if not provided
        session_id = chat_request.sessionId
        if not session_id:
            session_id = agent.create_chat_session()

        # Process the query
        response = await agent.process_query(
            sanitized_message,
            session_id,
            {
                "drugs": sanitized_drugs,
                "currentMedications": sanitized_medications,
                "userSymptoms": sanitized_symptoms,
            },
        )

        return ChatResponse(
            success=True, data={"response": response.to_dict(), "sessionId": session_id}
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=ChatResponse)
async def get_chat_history(sessionId: str, db: Session = Depends(get_db)):
    """Get chat history for a session"""
    try:
        if not sessionId:
            raise HTTPException(status_code=400, detail="Session ID required")

        agent = MedicalRAGAgent(db)
        history = agent.get_chat_history(sessionId)

        return ChatResponse(success=True, data={"history": history})

    except Exception as e:
        print(f"Chat history API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
