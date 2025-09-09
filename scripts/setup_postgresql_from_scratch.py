#!/usr/bin/env python3
"""
PostgreSQL From-Scratch Setup Script for Medical RAG Pipeline

This script sets up a fresh PostgreSQL database with:
1. pgvector extension for vector operations
2. Optimized configuration for large vector workloads
3. All table creation using existing SQLAlchemy models
4. Performance indexes and constraints

Usage:
1. Ensure PostgreSQL is installed and running
2. Set DATABASE_URL environment variable
3. Run: uv run scripts/setup_postgresql_from_scratch.py
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from backend.lib.database import init_db_sync

# Load environment variables
load_dotenv()


class PostgreSQLSetupManager:
    """Manage PostgreSQL setup from scratch"""

    def __init__(self):
        self.pg_url = os.getenv("DATABASE_URL")

        if not self.pg_url:
            raise ValueError(
                "DATABASE_URL environment variable not set!\n"
                "Please set it to: postgresql://user:password@localhost:5432/medical_rag_pg"
            )

        if not self.pg_url.startswith("postgresql://"):
            raise ValueError(
                f"DATABASE_URL must be PostgreSQL URL, got: {self.pg_url}\n"
                "Expected format: postgresql://user:password@localhost:5432/medical_rag_pg"
            )

        print(f"📊 Using PostgreSQL URL: {self.pg_url}")

    def check_postgresql_connection(self):
        """Test PostgreSQL connection"""
        print("🔍 Testing PostgreSQL connection...")

        try:
            conn = psycopg2.connect(self.pg_url)
            cursor = conn.cursor()

            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version}")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            print("\n🛠️  Troubleshooting steps:")
            print("1. Ensure PostgreSQL is installed and running:")
            print("   brew services list | grep postgres  # macOS")
            print("   sudo systemctl status postgresql    # Linux")
            print("2. Create database user and database:")
            print("   createuser -s your_username")
            print("   createdb medical_rag_pg")
            print("3. Verify DATABASE_URL is correct")
            return False

    def setup_pgvector_extension(self):
        """Install and verify pgvector extension"""
        print("🧬 Setting up pgvector extension...")

        try:
            conn = psycopg2.connect(self.pg_url)
            conn.autocommit = True
            cursor = conn.cursor()

            # Install pgvector extension
            print("   • Installing pgvector extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Verify installation
            cursor.execute("""
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = 'vector';
            """)
            result = cursor.fetchone()

            if result:
                print(f"✅ pgvector extension installed: version {result[1]}")
            else:
                raise Exception("pgvector extension not found after installation")

            # Test vector functionality
            print("   • Testing vector functionality...")
            cursor.execute("SELECT '[1,2,3]'::vector(3);")
            test_vector = cursor.fetchone()[0]
            print(f"✅ Vector functionality working: {test_vector}")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            print(f"❌ pgvector setup failed: {e}")
            print("\n🛠️  pgvector installation steps:")
            print("1. Install pgvector:")
            print("   git clone https://github.com/pgvector/pgvector.git")
            print("   cd pgvector && make && sudo make install")
            print("2. Or install via package manager:")
            print("   brew install pgvector  # macOS")
            print("   sudo apt install postgresql-14-pgvector  # Ubuntu")
            return False

    def optimize_postgresql_configuration(self):
        """Apply performance optimizations for vector workloads"""
        print("⚡ Optimizing PostgreSQL configuration...")

        try:
            conn = psycopg2.connect(self.pg_url)
            conn.autocommit = True
            cursor = conn.cursor()

            # Get current configuration
            optimizations = [
                ("shared_buffers", "256MB", "Memory for shared buffers"),
                ("work_mem", "64MB", "Memory per operation"),
                ("maintenance_work_mem", "256MB", "Memory for maintenance operations"),
                ("effective_cache_size", "1GB", "Effective cache size"),
                ("random_page_cost", "1.1", "Random page cost for SSDs"),
            ]

            print("   • Current configuration:")
            for param, recommended_value, description in optimizations:
                cursor.execute(f"SHOW {param};")
                current_value = cursor.fetchone()[0]
                print(
                    f"     - {param}: {current_value} (recommended: {recommended_value}) - {description}"
                )

            print(
                "   • Note: For production, consider updating postgresql.conf with recommended values"
            )

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            print(f"⚠️  Configuration check failed: {e}")
            return False

    def create_database_schema(self):
        """Create all database tables using SQLAlchemy models"""
        print("📋 Creating database schema...")

        try:
            # Use existing database initialization
            print("   • Initializing database with SQLAlchemy models...")
            init_db_sync()

            # Verify table creation
            engine = create_engine(self.pg_url)
            inspector = inspect(engine)

            expected_tables = [
                "drug_relations",
                "drug_metadata",
                "drug_product_stages",
                "product_stage_descriptions",
                "food_interactions",
                "drug_dosage",
                "vector_embeddings",
                "chat_sessions",
                "chat_messages",
            ]

            created_tables = inspector.get_table_names()
            print(f"   • Created {len(created_tables)} tables")

            # Verify each expected table exists
            for table_name in expected_tables:
                if table_name in created_tables:
                    # Get column info
                    columns = inspector.get_columns(table_name)
                    print(f"     ✅ {table_name}: {len(columns)} columns")

                    # Special check for vector_embeddings table
                    if table_name == "vector_embeddings":
                        embedding_column = next(
                            (col for col in columns if col["name"] == "embedding"), None
                        )
                        if embedding_column:
                            print(
                                f"       • Embedding column type: {embedding_column['type']}"
                            )
                        else:
                            print("       ⚠️  Embedding column not found")
                else:
                    print(f"     ❌ {table_name}: NOT CREATED")

            engine.dispose()
            return True

        except Exception as e:
            print(f"❌ Schema creation failed: {e}")
            return False

    def create_performance_indexes(self):
        """Create performance indexes (excluding vector indexes)"""
        print("🔍 Creating performance indexes...")

        try:
            conn = psycopg2.connect(self.pg_url)
            conn.autocommit = True
            cursor = conn.cursor()

            # Create indexes for better query performance
            indexes = [
                # Drug relations indexes
                (
                    'CREATE INDEX IF NOT EXISTS ix_drug_relations_x_name ON drug_relations ("xName");',
                    "Drug relations x_name index",
                ),
                (
                    'CREATE INDEX IF NOT EXISTS ix_drug_relations_y_name ON drug_relations ("yName");',
                    "Drug relations y_name index",
                ),
                (
                    "CREATE INDEX IF NOT EXISTS ix_drug_relations_relation ON drug_relations (relation);",
                    "Drug relations relation index",
                ),
                # Drug metadata indexes
                (
                    'CREATE INDEX IF NOT EXISTS ix_drug_metadata_drug_id ON drug_metadata ("drugId");',
                    "Drug metadata drugId index",
                ),
                (
                    "CREATE INDEX IF NOT EXISTS ix_drug_metadata_name ON drug_metadata (name);",
                    "Drug metadata name index",
                ),
                # Food interactions indexes
                (
                    'CREATE INDEX IF NOT EXISTS ix_food_interactions_drug_name ON food_interactions ("drugName");',
                    "Food interactions drugName index",
                ),
                (
                    'CREATE INDEX IF NOT EXISTS ix_food_interactions_drug_id ON food_interactions ("drugId");',
                    "Food interactions drugId index",
                ),
                # Drug dosage indexes
                (
                    'CREATE INDEX IF NOT EXISTS ix_drug_dosage_drug_id ON drug_dosage ("drugId");',
                    "Drug dosage drugId index",
                ),
                (
                    'CREATE INDEX IF NOT EXISTS ix_drug_dosage_product_name ON drug_dosage ("productName");',
                    "Drug dosage productName index",
                ),
                # Chat system indexes
                (
                    "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id ON chat_messages (session_id);",
                    "Chat messages session_id index",
                ),
                (
                    "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions (user_id);",
                    "Chat sessions user_id index",
                ),
            ]

            for sql, description in indexes:
                print(f"   • Creating: {description}")
                cursor.execute(sql)

            print(f"✅ Created {len(indexes)} performance indexes")

            # Note about vector indexes
            print(
                "   📝 Note: Vector similarity indexes will be created after embedding population"
            )

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            print(f"❌ Index creation failed: {e}")
            return False

    def verify_setup(self):
        """Verify the complete setup"""
        print("🔍 Verifying PostgreSQL setup...")

        try:
            # Test database connection with SQLAlchemy
            engine = create_engine(self.pg_url)

            with engine.connect() as conn:
                # Test basic query
                result = conn.execute(text("SELECT 1 as test;"))
                assert result.fetchone()[0] == 1

                # Test vector functionality
                result = conn.execute(
                    text("SELECT '[1,2,3]'::vector(3) as test_vector;")
                )
                vector_result = result.fetchone()[0]
                print(f"   ✅ Vector test: {vector_result}")

                # Check table counts
                tables_to_check = [
                    "drug_relations",
                    "drug_metadata",
                    "drug_product_stages",
                    "product_stage_descriptions",
                    "food_interactions",
                    "drug_dosage",
                    "vector_embeddings",
                    "chat_sessions",
                    "chat_messages",
                ]

                for table in tables_to_check:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                    count = result.fetchone()[0]
                    print(f"   📊 {table}: {count} rows")

            engine.dispose()
            print("✅ PostgreSQL setup verification completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Setup verification failed: {e}")
            return False

    def run_complete_setup(self):
        """Run the complete PostgreSQL setup process"""
        print("🚀 Starting Complete PostgreSQL Setup for Medical RAG Pipeline")
        print("=" * 70)

        steps = [
            ("PostgreSQL Connection Test", self.check_postgresql_connection),
            ("pgvector Extension Setup", self.setup_pgvector_extension),
            ("Configuration Optimization", self.optimize_postgresql_configuration),
            ("Database Schema Creation", self.create_database_schema),
            ("Performance Indexes Creation", self.create_performance_indexes),
            ("Setup Verification", self.verify_setup),
        ]

        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            print("-" * 50)

            success = step_func()
            if not success:
                print(f"\n❌ Setup failed at step: {step_name}")
                print("Please resolve the issues above and try again.")
                return False

        print("\n" + "=" * 70)
        print("🎉 PostgreSQL setup completed successfully!")
        print("\nNext steps:")
        print("1. ✅ PostgreSQL with pgvector is ready")
        print("2. 📊 All tables created with proper schema")
        print("3. 🔍 Performance indexes created")
        print("4. 🚀 Ready for data ingestion")
        print("\nNext steps:")
        print("   • Ingest data: uv run scripts/ingest_all_data_postgresql.py")
        print("   • Manage data: uv run scripts/manage_postgresql_data.py stats")

        return True


def main():
    """Main function"""
    try:
        setup_manager = PostgreSQLSetupManager()
        success = setup_manager.run_complete_setup()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
