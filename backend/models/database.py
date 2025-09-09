"""
SQLAlchemy database models - converted from Prisma schema
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Import pgvector support
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

Base = declarative_base()


def generate_cuid():
    """Generate a CUID-like ID using UUID"""
    return str(uuid.uuid4())


class DrugRelation(Base):
    __tablename__ = "drug_relations"

    id = Column(String, primary_key=True, default=generate_cuid)
    relation = Column(String, nullable=False)
    displayRelation = Column("displayRelation", String, nullable=False)
    xIndex = Column("xIndex", Integer, nullable=False)
    xId = Column("xId", String, nullable=False)
    xType = Column("xType", String, nullable=False)
    xName = Column("xName", String, nullable=False)
    xSource = Column("xSource", String, nullable=False)
    yIndex = Column("yIndex", Integer, nullable=False)
    yId = Column("yId", String, nullable=False)
    yType = Column("yType", String, nullable=False)
    yName = Column("yName", String, nullable=False)
    ySource = Column("ySource", String, nullable=False)
    relationType = Column("relationType", String, nullable=True)
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DrugMetadata(Base):
    __tablename__ = "drug_metadata"

    id = Column(String, primary_key=True, default=generate_cuid)
    drugId = Column("drugId", String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    type = Column(String, nullable=True)
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DrugProductStage(Base):
    __tablename__ = "drug_product_stages"

    id = Column(String, primary_key=True, default=generate_cuid)
    drugName = Column("drugName", String, unique=True, nullable=False)
    productStage = Column("productStage", String, nullable=False)
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ProductStageDescription(Base):
    __tablename__ = "product_stage_descriptions"

    id = Column(String, primary_key=True, default=generate_cuid)
    stageCode = Column("stageCode", String, unique=True, nullable=False)
    description = Column(String, nullable=False)
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class FoodInteraction(Base):
    __tablename__ = "food_interactions"

    id = Column(String, primary_key=True, default=generate_cuid)
    drugName = Column("drugName", String, nullable=False)
    drugId = Column("drugId", String, nullable=True)
    interaction = Column(String, nullable=False)
    source = Column(String, nullable=False)
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DrugDosage(Base):
    __tablename__ = "drug_dosage"

    id = Column(String, primary_key=True, default=generate_cuid)
    drugId = Column("drugId", String, nullable=False)
    productName = Column("productName", String, nullable=True)
    dosageForm = Column("dosageForm", String, nullable=True)
    route = Column("route", String, nullable=True)
    strength = Column("strength", String, nullable=True)
    manufacturer = Column("manufacturer", String, nullable=True)
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updatedAt", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id = Column(String, primary_key=True, default=generate_cuid)
    content = Column(Text, nullable=False)
    
    # Use pgvector for PostgreSQL, fallback to LargeBinary for SQLite
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(512), nullable=False)  # 512-dimensional vector
    else:
        embedding = Column(
            LargeBinary, nullable=False
        )  # Binary vector data (FLOAT32 array) for SQLite
        
    content_hash = Column(
        "content_hash", String(64), nullable=False
    )  # SHA256 hash for idempotency
    meta_data = Column("metadata", Text, nullable=False)  # JSON string of metadata
    source = Column(String, nullable=False)
    sourceId = Column("sourceId", String, nullable=False)
    chunk_id = Column(
        "chunk_id", Integer, nullable=True, default=0
    )  # For chunked content
    token_count = Column(
        "token_count", Integer, nullable=True
    )  # Token count for monitoring
    createdAt = Column("createdAt", DateTime, default=datetime.utcnow)

    # Create indexes for performance
    __table_args__ = (
        Index("ix_content_hash", "content_hash"),
        Index("ix_source_sourceid", "source", "sourceId"),
        Index("ix_source_sourceid_chunk", "source", "sourceId", "chunk_id"),
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_cuid)
    userId = Column("user_id", String, nullable=True)
    createdAt = Column("created_at", DateTime, default=datetime.utcnow)
    updatedAt = Column(
        "updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationship
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_cuid)
    sessionId = Column(
        "session_id",
        String,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    meta_data = Column(
        "metadata", Text, nullable=True
    )  # JSON string for additional data
    createdAt = Column("created_at", DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("ChatSession", back_populates="messages")
