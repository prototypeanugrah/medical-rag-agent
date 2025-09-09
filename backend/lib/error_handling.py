"""
Error handling utilities for the Medical RAG Agent
"""

import re
import time
from collections import defaultdict
from typing import Dict, List, Optional


class AppError(Exception):
    """Base application error"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        is_operational: bool = True,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.is_operational = is_operational
        self.code = code


class ValidationError(AppError):
    """Validation error"""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(f"Validation Error: {message}", 400, True, "VALIDATION_ERROR")


class DatabaseError(AppError):
    """Database error"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"Database Error: {message}", 500, True, "DATABASE_ERROR")
        if original_error:
            self.__cause__ = original_error


class EmbeddingError(AppError):
    """Embedding service error"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Embedding Service Error: {message}", 500, True, "EMBEDDING_ERROR"
        )
        if original_error:
            self.__cause__ = original_error


class AIServiceError(AppError):
    """AI service error"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"AI Service Error: {message}", 500, True, "AI_SERVICE_ERROR")
        if original_error:
            self.__cause__ = original_error


class RateLimitError(AppError):
    """Rate limit error"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429, True, "RATE_LIMIT_ERROR")


def handle_error(error: Exception) -> Dict[str, any]:
    """Handle and format errors for API responses"""

    if isinstance(error, AppError):
        return {
            "message": str(error),
            "status_code": error.status_code,
            "code": error.code,
        }

    # Handle specific third-party errors
    error_message = str(error).lower()

    if "openai" in error_message:
        return {
            "message": "AI service temporarily unavailable",
            "status_code": 503,
            "code": "AI_SERVICE_UNAVAILABLE",
        }

    if "database" in error_message or "sqlalchemy" in error_message:
        return {
            "message": "Database service temporarily unavailable",
            "status_code": 503,
            "code": "DATABASE_UNAVAILABLE",
        }

    # Generic server error
    print(f"Unhandled error: {error}")
    return {
        "message": "Internal server error",
        "status_code": 500,
        "code": "INTERNAL_SERVER_ERROR",
    }


def validate_environment_variables():
    """Validate required environment variables"""
    import os

    required = ["OPENAI_API_KEY"]
    missing = []

    for key in required:
        if not os.getenv(key):
            missing.append(key)

    if missing:
        raise ValidationError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not input_str:
        return ""

    # Remove potentially harmful characters
    sanitized = input_str.replace("<", "").replace(">", "")
    sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"data:", "", sanitized, flags=re.IGNORECASE)

    # Trim and limit length
    return sanitized.strip()[:2000]


def validate_drug_name(drug_name: str) -> bool:
    """Validate drug name format"""
    if not drug_name:
        return False

    # Basic validation for drug names
    drug_name_pattern = r"^[a-zA-Z0-9\s\-\.]+$"
    return bool(re.match(drug_name_pattern, drug_name)) and 2 <= len(drug_name) <= 100


def validate_query(query: str) -> bool:
    """Validate user query"""
    if not query or not query.strip():
        return False

    trimmed = query.strip()
    return 3 <= len(trimmed) <= 2000


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, max_requests: int = 10, window_ms: int = 60000):
        self.max_requests = max_requests
        self.window_ms = window_ms
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier"""
        now = time.time() * 1000  # Convert to milliseconds
        request_times = self.requests[identifier]

        # Remove old requests outside the window
        valid_requests = [
            timestamp for timestamp in request_times if now - timestamp < self.window_ms
        ]

        if len(valid_requests) >= self.max_requests:
            return False

        # Add current request
        valid_requests.append(now)
        self.requests[identifier] = valid_requests

        return True

    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for the identifier"""
        now = time.time() * 1000
        request_times = self.requests[identifier]

        valid_requests = [
            timestamp for timestamp in request_times if now - timestamp < self.window_ms
        ]

        return max(0, self.max_requests - len(valid_requests))
