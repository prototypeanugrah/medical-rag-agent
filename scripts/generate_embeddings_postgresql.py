#!/usr/bin/env python3
"""
Embedding Generation Script for PostgreSQL Medical RAG Pipeline

This script generates embeddings for ALL data in PostgreSQL using:
- OpenAI text-embedding-3-small model
- 512 dimensions (optimized for speed)
- Efficient batching and progress tracking
- Cost monitoring and estimation

CRITICAL: Only run AFTER data ingestion is complete and verified!

Usage:
1. Ensure all data is ingested successfully
2. Set OPENAI_API_KEY in environment
3. Run: uv run scripts/generate_embeddings_postgresql.py
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any
import time

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from backend.lib.database import get_db, init_db_sync
from backend.lib.embeddings import EmbeddingService

# Load environment variables
load_dotenv()


class PostgreSQLEmbeddingGenerator:
    """Generate embeddings for all PostgreSQL data"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate environment
        if not self.database_url or not self.database_url.startswith("postgresql://"):
            raise ValueError(
                "DATABASE_URL must be PostgreSQL URL!\n"
                f"Current: {self.database_url}\n"
                "Expected: postgresql://user:password@localhost:5432/medical_rag_pg"
            )
        
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY not set!\n"
                "Please set your OpenAI API key in environment variables"
            )
        
        # Initialize embedding service with optimized settings
        self.embedding_service = EmbeddingService(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=512,  # Optimized for speed
            concurrency_limit=4,  # Moderate concurrency for cost control
            batch_size=512  # Token-aware batching
        )
        
        print(f"📊 PostgreSQL URL: {self.database_url}")
        print(f"🤖 Embedding Model: text-embedding-3-small (512 dimensions)")
        print(f"⚡ Concurrency: 4 parallel requests")
        print(f"📦 Batch size: 512 items per request")
    
    def verify_prerequisites(self):
        """Verify all prerequisites are met"""
        print("🔍 Verifying prerequisites...")
        
        try:
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Check database connection
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print(f"   ✅ PostgreSQL: Connected")
                
                # Check pgvector extension
                result = conn.execute(text("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname = 'vector';
                """))
                vector_ext = result.fetchone()
                if vector_ext:
                    print(f"   ✅ pgvector: version {vector_ext[1]}")
                else:
                    print("   ❌ pgvector extension not found!")
                    return False
                
                # Check data tables have data
                tables_to_check = [
                    ("drug_relations", "Knowledge graph relations"),
                    ("drug_metadata", "Drug metadata"),
                    ("food_interactions", "Food interactions"),
                    ("drug_dosage", "Drug dosage information"),
                    ("drug_product_stages", "Product stages"),
                    ("product_stage_descriptions", "Stage descriptions"),
                ]
                
                total_source_records = 0
                empty_tables = []
                
                for table_name, description in tables_to_check:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                    count = result.fetchone()[0]
                    total_source_records += count
                    
                    if count == 0:
                        empty_tables.append(table_name)
                        print(f"   ❌ {description}: EMPTY (no data to embed)")
                    else:
                        print(f"   ✅ {description}: {count:,} records")
                
                if empty_tables:
                    print(f"\n❌ Found {len(empty_tables)} empty tables!")
                    print("Run data ingestion first: uv run scripts/ingest_all_data_postgresql.py")
                    return False
                
                # Check if embeddings already exist
                result = conn.execute(text("SELECT COUNT(*) FROM vector_embeddings;"))
                existing_embeddings = result.fetchone()[0]
                
                if existing_embeddings > 0:
                    print(f"   ⚠️  Existing embeddings: {existing_embeddings:,}")
                    print("   This will add to existing embeddings (duplicates will be skipped)")
                else:
                    print("   📋 Vector embeddings table: empty (ready for generation)")
                
                print(f"\n📊 Total source records for embedding: {total_source_records:,}")
                
                return True
                
            engine.dispose()
            
        except Exception as e:
            print(f"❌ Prerequisites verification failed: {e}")
            return False
    
    def estimate_costs(self):
        """Estimate embedding generation costs"""
        print("\n💰 Estimating Embedding Generation Costs")
        print("-" * 40)
        
        try:
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Get record counts for cost estimation
                tables_for_embedding = [
                    "drug_relations",
                    "drug_metadata", 
                    "food_interactions",
                    "drug_dosage",
                    "drug_product_stages",
                    "product_stage_descriptions"
                ]
                
                total_records = 0
                for table in tables_for_embedding:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                    count = result.fetchone()[0]
                    total_records += count
                
                # Estimate token usage (average ~50 tokens per record)
                estimated_tokens = total_records * 50
                
                # text-embedding-3-small pricing: $0.02 per 1M tokens
                estimated_cost_usd = (estimated_tokens / 1_000_000) * 0.02
                
                # Estimate processing time (conservative estimate)
                # ~1000 records per minute with batching and concurrency
                estimated_minutes = total_records / 1000
                estimated_hours = estimated_minutes / 60
                
                print(f"📊 Records to embed: {total_records:,}")
                print(f"🔢 Estimated tokens: {estimated_tokens:,}")
                print(f"💵 Estimated cost: ${estimated_cost_usd:.2f} USD")
                print(f"⏱️  Estimated time: {estimated_hours:.1f} hours")
                print(f"🤖 Model: text-embedding-3-small (512 dimensions)")
                
                return True
                
            engine.dispose()
            
        except Exception as e:
            print(f"❌ Cost estimation failed: {e}")
            return False
    
    async def generate_all_embeddings(self):
        """Generate embeddings for all data sources"""
        print("\n🚀 Starting Embedding Generation")
        print("=" * 50)
        
        # Get database session
        db = next(get_db())
        
        try:
            start_time = time.time()
            
            # Get initial statistics
            initial_stats = self.embedding_service.get_embedding_stats(db)
            print(f"📊 Initial embeddings: {initial_stats['total_embeddings']:,}")
            
            print("\n📋 Embedding generation will proceed in this order:")
            print("1. Drug Relations (knowledge graph relationships)")
            print("2. Food Interactions (drug-food warnings)")
            print("3. Drug Metadata (basic drug information)")
            print("4. Product Stages (development phases)")
            print("5. Stage Descriptions (stage definitions)")
            print("6. Drug Dosage (product and dosage details)")
            print()
            
            # Generate embeddings for each data source
            embedding_functions = [
                ("Drug Relations", self.embedding_service.index_drug_relations_with_progress),
                ("Food Interactions", self.embedding_service.index_food_interactions_with_progress),
                ("Drug Metadata", self.embedding_service.index_drug_metadata_with_progress),
                ("Product Stages", self.embedding_service.index_product_stages_with_progress),
                ("Stage Descriptions", self.embedding_service.index_stage_descriptions_with_progress),
                ("Drug Dosage", self.embedding_service.index_drug_dosage_with_progress),
            ]
            
            completed_sources = 0
            
            for source_name, embedding_func in embedding_functions:
                print(f"\n📊 Processing: {source_name}")
                print("-" * 30)
                
                try:
                    await embedding_func(db)
                    completed_sources += 1
                    print(f"✅ {source_name} embedding generation completed!")
                    
                except Exception as e:
                    print(f"❌ {source_name} embedding generation failed: {e}")
                    print("Continuing with other sources...")
            
            # Final statistics
            end_time = time.time()
            processing_time = end_time - start_time
            
            final_stats = self.embedding_service.get_embedding_stats(db)
            new_embeddings = final_stats['total_embeddings'] - initial_stats['total_embeddings']
            
            print("\n" + "=" * 50)
            print("🎉 Embedding Generation Completed!")
            print(f"📊 New embeddings created: {new_embeddings:,}")
            print(f"📊 Total embeddings: {final_stats['total_embeddings']:,}")
            print(f"⏱️  Processing time: {processing_time/3600:.1f} hours")
            print(f"✅ Completed sources: {completed_sources}/{len(embedding_functions)}")
            
            # Show breakdown by source
            print(f"\nEmbedding breakdown by source:")
            for source, count in final_stats['by_source'].items():
                print(f"   • {source}: {count:,}")
            
            # Token statistics
            token_stats = final_stats['token_stats']
            print(f"\nToken usage:")
            print(f"   • Total tokens: {token_stats['total_tokens']:,}")
            print(f"   • Average per embedding: {token_stats['avg_tokens']:.1f}")
            print(f"   • Max tokens: {token_stats['max_tokens']:,}")
            
            return completed_sources == len(embedding_functions)
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            db.close()
    
    def create_vector_indexes(self):
        """Create optimized vector indexes for similarity search"""
        print("\n🔍 Creating Vector Similarity Indexes")
        print("-" * 40)
        
        try:
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Check if we have embeddings
                result = conn.execute(text("SELECT COUNT(*) FROM vector_embeddings;"))
                embedding_count = result.fetchone()[0]
                
                if embedding_count == 0:
                    print("❌ No embeddings found! Cannot create indexes.")
                    return False
                
                print(f"📊 Creating indexes for {embedding_count:,} embeddings...")
                
                # Create IVFFlat index for cosine similarity
                # Lists parameter: sqrt(rows) is a good starting point
                import math
                lists = max(100, min(1000, int(math.sqrt(embedding_count))))
                
                print(f"   • Using lists parameter: {lists}")
                print("   • Creating IVFFlat index (this may take several minutes)...")
                
                # Drop existing index if it exists
                conn.execute(text("DROP INDEX IF EXISTS ix_embedding_cosine;"))
                
                # Create new index
                index_sql = f"""
                    CREATE INDEX ix_embedding_cosine 
                    ON vector_embeddings 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = {lists});
                """
                
                start_time = time.time()
                conn.execute(text(index_sql))
                conn.commit()
                index_time = time.time() - start_time
                
                print(f"   ✅ Vector index created in {index_time:.1f} seconds")
                
                # Test index performance
                print("   • Testing index performance...")
                test_query = """
                    SELECT COUNT(*) 
                    FROM vector_embeddings 
                    WHERE embedding <=> '[0,0,0,0,0]'::vector < 1.0;
                """
                
                start_time = time.time()
                result = conn.execute(text(test_query))
                query_time = time.time() - start_time
                
                print(f"   ✅ Index test query: {query_time*1000:.1f}ms")
                
                # Update table statistics
                print("   • Updating table statistics...")
                conn.execute(text("ANALYZE vector_embeddings;"))
                conn.commit()
                
                print("✅ Vector indexes created successfully!")
                return True
                
            engine.dispose()
            
        except Exception as e:
            print(f"❌ Vector index creation failed: {e}")
            return False
    
    def run_embedding_generation(self):
        """Run the complete embedding generation process"""
        print("🚀 Starting PostgreSQL Embedding Generation")
        print("=" * 60)
        print("Model: OpenAI text-embedding-3-small (512 dimensions)")
        print("=" * 60)
        
        steps = [
            ("Prerequisites Verification", self.verify_prerequisites),
            ("Cost Estimation", self.estimate_costs),
        ]
        
        # Run verification steps
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}")
            print("-" * 40)
            
            success = step_func()
            if not success:
                print(f"\n❌ Failed at: {step_name}")
                return False
        
        # Confirmation prompt
        print("\n" + "⚠️ " * 20)
        print("IMPORTANT: Embedding generation will incur OpenAI API costs!")
        print("The estimated cost is shown above.")
        print("This process will take several hours to complete.")
        print("⚠️ " * 20)
        
        confirm = input("\nProceed with embedding generation? (yes/no): ").lower().strip()
        if confirm not in ["yes", "y"]:
            print("❌ Embedding generation cancelled by user")
            return False
        
        # Run embedding generation
        print(f"\n📋 Embedding Generation Process")
        print("-" * 40)
        
        success = asyncio.run(self.generate_all_embeddings())
        if not success:
            print(f"\n❌ Embedding generation failed")
            return False
        
        # Create vector indexes
        success = self.create_vector_indexes()
        if not success:
            print(f"\n❌ Vector index creation failed")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 Complete Embedding Generation Successful!")
        print("\nDatabase Status:")
        print("✅ PostgreSQL with pgvector ready")
        print("✅ All data ingested and embedded")
        print("✅ Vector similarity indexes created")
        print("🚀 Ready for RAG pipeline operations!")
        print("\nYou can now update your .env to use PostgreSQL:")
        print(f"DATABASE_URL={self.database_url}")
        
        return True


def main():
    """Main function"""
    try:
        generator = PostgreSQLEmbeddingGenerator()
        success = generator.run_embedding_generation()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Embedding generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()