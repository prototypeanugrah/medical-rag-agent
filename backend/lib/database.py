"""
Database connection and session management
"""

import os
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from ..models.database import Base

# Database URL - supporting both SQLite and PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_rag.db")

# Configure async URL based on database type
if DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    # For PostgreSQL, we need to handle connection pooling and engine options differently
    engine_kwargs = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "echo": False
    }
    async_engine_kwargs = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "echo": False
    }
else:
    # SQLite configuration
    ASYNC_DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    engine_kwargs = {"echo": False}
    async_engine_kwargs = {"echo": False}

# Create engines
engine = create_engine(DATABASE_URL, **engine_kwargs)
async_engine = create_async_engine(ASYNC_DATABASE_URL, **async_engine_kwargs)

# Create sessionmakers
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        # Handle schema migration for vector_embeddings if needed
        await conn.run_sync(_check_and_upgrade_schema)
        await conn.run_sync(Base.metadata.create_all)


def _check_and_upgrade_schema(connection):
    """Check and upgrade schema if needed"""
    from sqlalchemy import inspect

    inspector = inspect(connection)

    # Check if vector_embeddings table exists and needs migration
    if "vector_embeddings" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("vector_embeddings")]
        needs_migration = (
            "content_hash" not in columns
            or "chunk_id" not in columns
            or "token_count" not in columns
        )

        if needs_migration:
            print("🔄 Upgrading vector_embeddings table schema...")
            # Drop and recreate the table with new schema
            Base.metadata.tables["vector_embeddings"].drop(connection, checkfirst=True)
            print("   • Old table dropped")


def init_db_sync():
    """Initialize database tables synchronously"""
    print("🔧 Ensuring all database tables exist...")

    # Check if we need to handle schema migration
    from sqlalchemy import inspect

    inspector = inspect(engine)

    # Check if vector_embeddings table exists and needs migration
    if "vector_embeddings" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("vector_embeddings")]
        needs_migration = (
            "content_hash" not in columns
            or "chunk_id" not in columns
            or "token_count" not in columns
        )

        if needs_migration:
            print("🔄 Upgrading vector_embeddings table schema...")
            # Drop and recreate the table with new schema
            Base.metadata.tables["vector_embeddings"].drop(engine, checkfirst=True)
            print("   • Old table dropped")

    # Create all tables (will create with proper schema)
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables verified/created")


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class DatabaseService:
    """Database service for common operations"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, model_instance):
        """Create a new record"""
        self.db.add(model_instance)
        self.db.commit()
        self.db.refresh(model_instance)
        return model_instance

    def get_by_id(self, model_class, id: str):
        """Get record by ID"""
        return self.db.query(model_class).filter(model_class.id == id).first()

    def get_all(self, model_class):
        """Get all records"""
        return self.db.query(model_class).all()

    def update(self, model_instance):
        """Update a record"""
        self.db.commit()
        self.db.refresh(model_instance)
        return model_instance

    def delete(self, model_instance):
        """Delete a record"""
        self.db.delete(model_instance)
        self.db.commit()
        return True


class AsyncDatabaseService:
    """Async database service for common operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, model_instance):
        """Create a new record"""
        self.db.add(model_instance)
        await self.db.commit()
        await self.db.refresh(model_instance)
        return model_instance

    async def get_by_id(self, model_class, id: str):
        """Get record by ID"""
        result = await self.db.execute(select(model_class).filter(model_class.id == id))
        return result.scalar_one_or_none()

    async def get_all(self, model_class):
        """Get all records"""
        result = await self.db.execute(select(model_class))
        return result.scalars().all()

    async def update(self, model_instance):
        """Update a record"""
        await self.db.commit()
        await self.db.refresh(model_instance)
        return model_instance

    async def delete(self, model_instance):
        """Delete a record"""
        await self.db.delete(model_instance)
        await self.db.commit()
        return True
