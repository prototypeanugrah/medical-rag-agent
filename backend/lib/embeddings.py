"""
Optimized embedding service for semantic search and indexing
- Async OpenAI client with concurrency control
- Token-aware batching for optimal throughput
- Binary vector storage for performance
- Idempotent operations with content hashing
- Robust retry logic with exponential backoff
"""

import asyncio
import hashlib
import json
import os
import struct
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

# Import pgvector functions for PostgreSQL
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

from ..models.database import (
    DrugDosage,
    DrugMetadata,
    DrugProductStage,
    DrugRelation,
    FoodInteraction,
    ProductStageDescription,
    VectorEmbedding,
    PGVECTOR_AVAILABLE as MODEL_PGVECTOR_AVAILABLE,
)
from .error_handling import EmbeddingError

# Load environment variables
load_dotenv()


class EmbeddingService:
    """Optimized service for creating and managing embeddings"""

    def __init__(
        self,
        concurrency_limit: int = 4,
        batch_size: int = 512,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        ollama_base_url: Optional[str] = None,
    ):
        # Determine if we're using PostgreSQL with pgvector
        self.use_pgvector = PGVECTOR_AVAILABLE and MODEL_PGVECTOR_AVAILABLE
        # Provider selection
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER", "openai")).lower()

        # Client/model configuration per provider
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or os.getenv(
                "OPENAI_EMBED_MODEL", "text-embedding-3-small"
            )
            # Reduced from 1536 for faster processing
            self.dimensions = dimensions or int(
                os.getenv("EMBEDDING_DIMENSIONS", "512")
            )
            self.http_client = None
        elif self.provider == "ollama":
            # Local Ollama HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=ollama_base_url
                or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
            )
            self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            # Let dimensions be inferred from the first response if not provided
            self.dimensions = dimensions or None
            self.client = None
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        self.batch_size = batch_size  # Configurable batch size
        self.token_budget = 100_000  # Max tokens per API request
        self.concurrency_limit = concurrency_limit

        # Initialize tiktoken encoder for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    def content_hash(self, text: str) -> str:
        """Generate SHA256 hash of content for idempotency"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limit"""
        if max_tokens is None:
            max_tokens = self.token_budget // 2  # Conservative limit per text

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def embedding_to_bytes(self, embedding: List[float]) -> bytes:
        """Convert embedding list to compact binary format (FLOAT32)"""
        return struct.pack(f"{len(embedding)}f", *embedding)

    def bytes_to_embedding(self, data: bytes) -> List[float]:
        """Convert binary data back to embedding list"""
        num_floats = len(data) // 4  # 4 bytes per float32
        return list(struct.unpack(f"{num_floats}f", data))

    def build_token_aware_batches(
        self, items: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> List[List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Build batches optimized for token count rather than just item count"""
        batches = []
        current_batch = []
        current_tokens = 0

        for content, source, source_id, metadata in items:
            # Truncate overly long content
            token_count = self.count_tokens(content)

            if token_count > self.token_budget:
                content = self.truncate_text(content, self.token_budget // 2)
                token_count = self.count_tokens(content)

            # Check if adding this item would exceed our budget
            if current_batch and current_tokens + token_count > self.token_budget:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append((content, source, source_id, metadata))
            current_tokens += token_count

            # Also respect the maximum batch size
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

        if current_batch:
            batches.append(current_batch)

        return batches

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30, jitter=5),
        stop=stop_after_attempt(8),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def create_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings with robust retry logic"""
        try:
            async with self.semaphore:
                if self.provider == "openai":
                    response = await self.client.embeddings.create(
                        model=self.model, input=texts, dimensions=self.dimensions
                    )
                    return [data.embedding for data in response.data]
                elif self.provider == "ollama":
                    # Per https://ollama.com/library/nomic-embed-text:v1.5 the embeddings
                    # endpoint is POST /api/embeddings with { model, prompt }
                    embeddings: List[List[float]] = []
                    for text in texts:
                        r = await self.http_client.post(
                            "/api/embeddings",
                            json={"model": self.model, "prompt": text},
                        )
                        r.raise_for_status()
                        data = r.json()
                        if "embedding" in data:
                            embeddings.append(data["embedding"])  # single vector
                        elif "embeddings" in data and isinstance(
                            data["embeddings"], list
                        ):
                            # some servers may return list even for single prompt
                            first = data["embeddings"][0] if data["embeddings"] else []
                            embeddings.append(first)
                        else:
                            raise ValueError(
                                "Ollama /api/embeddings response missing 'embedding(s)'"
                            )

                    # Infer dimensions if not set
                    if (
                        self.dimensions is None
                        and embeddings
                        and len(embeddings[0]) > 0
                    ):
                        self.dimensions = len(embeddings[0])
                    return embeddings
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"⚠️  API call failed: {str(e)}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            embeddings = await self.create_embeddings_with_retry([text])
            return embeddings[0]
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding: {str(e)}", e)

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        try:
            return await self.create_embeddings_with_retry(texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to create embeddings: {str(e)}", e)

    def get_existing_hashes(self, db: Session, source: str = None) -> set:
        """Get existing content hashes to avoid reprocessing"""
        query = select(VectorEmbedding.content_hash)
        if source:
            query = query.where(VectorEmbedding.source == source)
        result = db.execute(query).scalars().all()
        return set(result)

    def filter_new_items(
        self,
        db: Session,
        items: List[Tuple[str, str, str, Dict[str, Any]]],
        source: str = None,
    ) -> List[Tuple[str, str, str, Dict[str, Any], str, int]]:
        """Filter items to only include new ones based on content hash"""
        existing_hashes = self.get_existing_hashes(db, source)
        new_items = []

        for content, source_name, source_id, metadata in items:
            content_hash = self.content_hash(content)
            if content_hash not in existing_hashes:
                token_count = self.count_tokens(content)
                new_items.append(
                    (
                        content,
                        source_name,
                        source_id,
                        metadata,
                        content_hash,
                        token_count,
                    )
                )

        return new_items

    async def process_embedding_batch(
        self, db: Session, batch: List[Tuple[str, str, str, Dict[str, Any], str, int]]
    ) -> int:
        """Process a single batch of embeddings with bulk database operations"""
        if not batch:
            return 0

        texts = [item[0] for item in batch]

        # Create embeddings for the entire batch
        embeddings = await self.create_embeddings_with_retry(texts)

        # Prepare bulk insert data
        vector_embeddings = []
        for (
            content,
            source,
            source_id,
            metadata,
            content_hash,
            token_count,
        ), embedding in zip(batch, embeddings):
            # Attach embedding provenance metadata
            metadata_with_provider = dict(metadata or {})
            try:
                metadata_with_provider.update(
                    {
                        "_embedding": {
                            "provider": self.provider,
                            "model": self.model,
                            "dimensions": len(embedding),
                        }
                    }
                )
            except Exception:
                pass
            # Store embedding in appropriate format based on database type
            if self.use_pgvector:
                # For PostgreSQL with pgvector, store as array
                embedding_data = embedding
            else:
                # For SQLite, store as binary blob
                embedding_data = self.embedding_to_bytes(embedding)
                
            vector_embedding = VectorEmbedding(
                id=str(uuid.uuid4()),
                content=content,
                embedding=embedding_data,
                content_hash=content_hash,
                meta_data=json.dumps(metadata_with_provider or {}),
                source=source,
                sourceId=source_id,
                token_count=token_count,
            )
            vector_embeddings.append(vector_embedding)

        # Bulk insert with single commit
        db.add_all(vector_embeddings)
        db.commit()

        return len(vector_embeddings)

    async def store_embeddings_batch_optimized(
        self,
        db: Session,
        items: List[Tuple[str, str, str, Dict[str, Any]]],
        pbar: Optional[tqdm] = None,
        skip_existing: bool = True,
    ) -> Dict[str, int]:
        """Store embeddings in optimized batches with idempotency"""
        stats = {"processed": 0, "skipped": 0, "new": 0}

        # Filter out existing items if requested
        if skip_existing:
            source = items[0][1] if items else None
            filtered_items = self.filter_new_items(db, items, source)
            stats["skipped"] = len(items) - len(filtered_items)
            items_to_process = filtered_items
        else:
            items_to_process = [
                (
                    content,
                    source,
                    source_id,
                    metadata,
                    self.content_hash(content),
                    self.count_tokens(content),
                )
                for content, source, source_id, metadata in items
            ]

        if not items_to_process:
            if pbar:
                pbar.update(len(items))
            return stats

        # Build token-aware batches
        batches = []
        current_batch = []
        current_tokens = 0

        for item in items_to_process:
            content, source, source_id, metadata, content_hash, token_count = item

            # Check if adding this item would exceed our budget
            if current_batch and current_tokens + token_count > self.token_budget:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(item)
            current_tokens += token_count

            # Also respect the maximum batch size
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

        if current_batch:
            batches.append(current_batch)

        # Process batches with concurrency control
        total_processed = 0
        for i in range(0, len(batches), self.concurrency_limit):
            batch_group = batches[i : i + self.concurrency_limit]

            # Process batch group concurrently
            tasks = [self.process_embedding_batch(db, batch) for batch in batch_group]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"❌ Batch processing error: {result}")
                else:
                    total_processed += result

            # Update progress bar
            if pbar:
                items_in_group = sum(len(batch) for batch in batch_group)
                pbar.update(items_in_group)

        stats["processed"] = total_processed
        stats["new"] = total_processed

        return stats

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_np = np.array(a)
        b_np = np.array(b)

        dot_product = np.dot(a_np, b_np)
        magnitude_a = np.linalg.norm(a_np)
        magnitude_b = np.linalg.norm(b_np)

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    async def search_similar(
        self,
        db: Session,
        query_text: str,
        limit: int = 10,
        min_similarity: float = 0.7,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using pgvector for PostgreSQL or cosine similarity for SQLite"""
        query_embedding = await self.create_embedding(query_text)

        if self.use_pgvector:
            # Use pgvector for efficient similarity search with PostgreSQL
            return await self._search_similar_pgvector(
                db, query_embedding, limit, min_similarity, source_filter
            )
        else:
            # Fallback to manual cosine similarity for SQLite
            return await self._search_similar_cosine(
                db, query_embedding, limit, min_similarity, source_filter
            )
    
    async def _search_similar_pgvector(
        self,
        db: Session,
        query_embedding: List[float],
        limit: int,
        min_similarity: float,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """PostgreSQL pgvector similarity search"""
        # Build base query with cosine similarity
        query = select(
            VectorEmbedding.content,
            VectorEmbedding.source,
            VectorEmbedding.sourceId,
            VectorEmbedding.meta_data,
            VectorEmbedding.embedding.cosine_distance(query_embedding).label('distance')
        )
        
        if source_filter:
            query = query.where(VectorEmbedding.source.in_(source_filter))
        
        # Convert distance to similarity (1 - distance for cosine)
        # Filter by minimum similarity threshold
        max_distance = 1 - min_similarity
        query = query.where(VectorEmbedding.embedding.cosine_distance(query_embedding) <= max_distance)
        
        # Order by similarity (lowest distance = highest similarity) and limit
        query = query.order_by(VectorEmbedding.embedding.cosine_distance(query_embedding))
        query = query.limit(limit)
        
        results = db.execute(query).fetchall()
        
        similarities = []
        for row in results:
            similarity = 1 - row.distance  # Convert distance back to similarity
            similarities.append({
                "content": row.content,
                "source": row.source,
                "sourceId": row.sourceId,
                "metadata": json.loads(row.meta_data),
                "similarity": similarity,
            })
        
        return similarities
    
    async def _search_similar_cosine(
        self,
        db: Session,
        query_embedding: List[float],
        limit: int,
        min_similarity: float,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """SQLite manual cosine similarity search"""
        # Build query
        query = select(VectorEmbedding)
        if source_filter:
            query = query.where(VectorEmbedding.source.in_(source_filter))

        all_embeddings = db.execute(query).scalars().all()

        similarities = []
        for embedding_record in all_embeddings:
            stored_embedding = self.bytes_to_embedding(embedding_record.embedding)
            # Skip vectors with mismatched dimensions to avoid invalid comparisons
            if len(stored_embedding) != len(query_embedding):
                continue
            similarity = self.cosine_similarity(query_embedding, stored_embedding)

            if similarity >= min_similarity:
                similarities.append(
                    {
                        "content": embedding_record.content,
                        "source": embedding_record.source,
                        "sourceId": embedding_record.sourceId,
                        "metadata": json.loads(embedding_record.meta_data),
                        "similarity": similarity,
                    }
                )

        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:limit]

    async def index_drug_relations_with_progress(self, db: Session) -> None:
        """Index drug relations with optimized batched processing"""
        relations = db.query(DrugRelation).all()

        items = []
        for relation in relations:
            content = (
                f"{relation.relation} between {relation.xName} ({relation.xType}) "
                f"and {relation.yName} ({relation.yType}). "
                f"{relation.xName} is a {relation.xType} from {relation.xSource}. "
                f"{relation.yName} is a {relation.yType} from {relation.ySource}."
            )

            metadata = {
                "relation": relation.relation,
                "displayRelation": relation.displayRelation,
                "xName": relation.xName,
                "xType": relation.xType,
                "xSource": relation.xSource,
                "yName": relation.yName,
                "yType": relation.yType,
                "ySource": relation.ySource,
                "xId": relation.xId,
                "yId": relation.yId,
            }

            items.append((content, "drug_relations", relation.id, metadata))

        with tqdm(
            total=len(relations), desc="🧬 Drug Relations", unit="relations"
        ) as pbar:
            stats = await self.store_embeddings_batch_optimized(db, items, pbar)
            pbar.set_postfix(stats)

    async def index_food_interactions_with_progress(self, db: Session) -> None:
        """Index food interactions with optimized batched processing"""
        interactions = db.query(FoodInteraction).all()

        items = []
        for interaction in interactions:
            content = (
                f"{interaction.drugName} food interaction: {interaction.interaction}"
            )

            metadata = {
                "drugName": interaction.drugName,
                "drugId": interaction.drugId,
                "interaction": interaction.interaction,
                "source": interaction.source,
            }

            items.append((content, "food_interactions", interaction.id, metadata))

        with tqdm(
            total=len(interactions), desc="🍎 Food Interactions", unit="interactions"
        ) as pbar:
            stats = await self.store_embeddings_batch_optimized(db, items, pbar)
            pbar.set_postfix(stats)

    async def index_drug_metadata_with_progress(self, db: Session) -> None:
        """Index drug metadata with optimized batched processing"""
        drugs = db.query(DrugMetadata).all()

        items = []
        for drug in drugs:
            content = f"Drug: {drug.name} (ID: {drug.drugId})"
            if drug.type:
                content += f", Type: {drug.type}"

            metadata = {
                "drugId": drug.drugId,
                "name": drug.name,
                "type": drug.type,
            }

            items.append((content, "drug_metadata", drug.id, metadata))

        with tqdm(total=len(drugs), desc="💊 Drug Metadata", unit="drugs") as pbar:
            stats = await self.store_embeddings_batch_optimized(db, items, pbar)
            pbar.set_postfix(stats)

    async def index_product_stages_with_progress(self, db: Session) -> None:
        """Index product stages with optimized batched processing"""
        stages = db.query(DrugProductStage).all()

        items = []
        for stage in stages:
            content = f"Drug {stage.drugName} is in product stage: {stage.productStage}"

            metadata = {
                "drugName": stage.drugName,
                "productStage": stage.productStage,
            }

            items.append((content, "product_stages", stage.id, metadata))

        with tqdm(total=len(stages), desc="🏭 Product Stages", unit="stages") as pbar:
            stats = await self.store_embeddings_batch_optimized(db, items, pbar)
            pbar.set_postfix(stats)

    async def index_stage_descriptions_with_progress(self, db: Session) -> None:
        """Index stage descriptions with optimized batched processing"""
        descriptions = db.query(ProductStageDescription).all()

        items = []
        for desc in descriptions:
            content = f"Product stage {desc.stageCode}: {desc.description}"

            metadata = {
                "stageCode": desc.stageCode,
                "description": desc.description,
            }

            items.append((content, "stage_descriptions", desc.id, metadata))

        with tqdm(
            total=len(descriptions), desc="📝 Stage Descriptions", unit="descriptions"
        ) as pbar:
            stats = await self.store_embeddings_batch_optimized(db, items, pbar)
            pbar.set_postfix(stats)

    async def index_drug_dosage_with_progress(self, db: Session) -> None:
        """Index drug dosage with optimized batched processing"""
        dosages = db.query(DrugDosage).all()

        items = []
        for dosage in dosages:
            # Create comprehensive content for dosage information
            content_parts = [f"Drug {dosage.drugId}"]

            if dosage.productName:
                content_parts.append(f"Product: {dosage.productName}")

            if dosage.dosageForm:
                content_parts.append(f"Dosage form: {dosage.dosageForm}")

            if dosage.route:
                content_parts.append(f"Route of administration: {dosage.route}")

            if dosage.strength:
                content_parts.append(f"Strength: {dosage.strength}")

            if dosage.manufacturer:
                content_parts.append(f"Manufacturer: {dosage.manufacturer}")

            content = ". ".join(content_parts) + "."

            metadata = {
                "drugId": dosage.drugId,
                "productName": dosage.productName,
                "dosageForm": dosage.dosageForm,
                "route": dosage.route,
                "strength": dosage.strength,
                "manufacturer": dosage.manufacturer,
            }

            items.append((content, "drug_dosage", dosage.id, metadata))

        with tqdm(total=len(dosages), desc="💊 Drug Dosages", unit="dosages") as pbar:
            stats = await self.store_embeddings_batch_optimized(db, items, pbar)
            pbar.set_postfix(stats)

    # Legacy methods for backward compatibility
    async def store_embeddings_batch(
        self,
        db: Session,
        items: List[Tuple[str, str, str, Dict[str, Any]]],
        pbar: Optional[tqdm] = None,
    ) -> None:
        """Legacy method - redirects to optimized version"""
        await self.store_embeddings_batch_optimized(db, items, pbar)

    async def index_drug_relations(self, db: Session, main_pbar=None) -> None:
        """Index drug relations for semantic search (legacy method)"""
        await self.index_drug_relations_with_progress(db)

    async def index_food_interactions(self, db: Session, main_pbar=None) -> None:
        """Index food interactions for semantic search (legacy method)"""
        await self.index_food_interactions_with_progress(db)

    async def index_drug_metadata(self, db: Session, main_pbar=None) -> None:
        """Index drug metadata for semantic search (legacy method)"""
        await self.index_drug_metadata_with_progress(db)

    async def index_product_stages(self, db: Session, main_pbar=None) -> None:
        """Index drug product stages for semantic search (legacy method)"""
        await self.index_product_stages_with_progress(db)

    async def index_stage_descriptions(self, db: Session, main_pbar=None) -> None:
        """Index product stage descriptions for semantic search (legacy method)"""
        await self.index_stage_descriptions_with_progress(db)

    async def index_drug_dosage(self, db: Session, main_pbar=None) -> None:
        """Index drug dosage information for semantic search (legacy method)"""
        await self.index_drug_dosage_with_progress(db)

    async def reindex_all(self, db: Session, force: bool = False) -> None:
        """Reindex all embeddings with optimized processing"""
        # print("🚀 Starting optimized vector embeddings creation...")
        # print(f"   • Using {self.model} with {self.dimensions}D vectors")
        # print(f"   • Batch size: {self.batch_size} items per API call")
        # print(f"   • Token budget: {self.token_budget:,} tokens per request")
        # print(f"   • Concurrency: {self.concurrency_limit} concurrent requests")

        if force:
            # Clear existing embeddings
            print("🗑️ Clearing existing embeddings...")
            db.query(VectorEmbedding).delete()
            db.commit()

        # Count total items to process
        total_relations = db.query(DrugRelation).count()
        total_food_interactions = db.query(FoodInteraction).count()
        total_drug_metadata = db.query(DrugMetadata).count()
        total_product_stages = db.query(DrugProductStage).count()
        total_stage_descriptions = db.query(ProductStageDescription).count()
        total_drug_dosages = db.query(DrugDosage).count()

        total_items = (
            total_relations
            + total_food_interactions
            + total_drug_metadata
            + total_product_stages
            + total_stage_descriptions
            + total_drug_dosages
        )

        print(f"📊 Total items to process: {total_items:,}")
        print(f"   • Drug Relations: {total_relations:,}")
        print(f"   • Food Interactions: {total_food_interactions:,}")
        print(f"   • Drug Metadata: {total_drug_metadata:,}")
        print(f"   • Product Stages: {total_product_stages:,}")
        print(f"   • Stage Descriptions: {total_stage_descriptions:,}")
        print(f"   • Drug Dosages: {total_drug_dosages:,}")
        print()

        # Check existing embeddings
        if not force:
            existing_count = db.query(VectorEmbedding).count()
            print(f"🔍 Existing embeddings: {existing_count:,}")
            print("   • Only new/changed content will be processed")
            print()

        # Reindex all data types with optimized batching
        await self.index_drug_relations_with_progress(db)
        await self.index_food_interactions_with_progress(db)
        await self.index_drug_metadata_with_progress(db)
        await self.index_product_stages_with_progress(db)
        await self.index_stage_descriptions_with_progress(db)
        await self.index_drug_dosage_with_progress(db)

        # final_count = db.query(VectorEmbedding).count()
        # print("✅ Optimized vector embeddings creation completed!")
        # print(f"   • Final embeddings count: {final_count:,}")
        # print(f"   • Batch size: {self.batch_size}")
        # print(f"   • Concurrency: {self.concurrency_limit}x faster")
        # print(f"   • Vector dimensions: {self.dimensions}D")
        # print("   • Storage: Binary format (3x smaller than JSON)")

    def get_embedding_stats(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive embedding statistics"""
        total_embeddings = db.query(VectorEmbedding).count()

        # Count by source
        source_stats = db.execute(
            select(VectorEmbedding.source, func.count(VectorEmbedding.id)).group_by(
                VectorEmbedding.source
            )
        ).all()

        # Token statistics
        token_stats = db.execute(
            select(
                func.sum(VectorEmbedding.token_count),
                func.avg(VectorEmbedding.token_count),
                func.max(VectorEmbedding.token_count),
                func.min(VectorEmbedding.token_count),
            ).where(VectorEmbedding.token_count.isnot(None))
        ).first()

        return {
            "total_embeddings": total_embeddings,
            "by_source": dict(source_stats),
            "token_stats": {
                "total_tokens": token_stats[0] or 0,
                "avg_tokens": round(token_stats[1] or 0, 2),
                "max_tokens": token_stats[2] or 0,
                "min_tokens": token_stats[3] or 0,
            },
            "model": self.model,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency_limit,
        }
