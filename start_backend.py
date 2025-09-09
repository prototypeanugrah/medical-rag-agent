#!/usr/bin/env python3
"""
Start script for the Python backend
"""

import os

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))

    print(f"🚀 Starting Medical RAG Agent Backend on port {port}")
    print(f"📍 API will be available at: http://localhost:{port}")
    print(f"📊 API docs will be available at: http://localhost:{port}/docs")
    print(f"🔗 Make sure frontend points to: http://localhost:{port}")

    # Start the server
    uvicorn.run(
        "backend.main:app", host="0.0.0.0", port=port, reload=True, log_level="info"
    )
