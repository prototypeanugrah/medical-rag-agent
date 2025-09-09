#!/usr/bin/env python3
"""
PostgreSQL Management CLI for Medical RAG Pipeline

The single comprehensive tool for all PostgreSQL operations:
- Setup and initialization 
- Data management (add, clear, stats, backup)
- Migration and embedding generation
- Validation and testing

Usage:
uv run scripts/pg.py --help

Core Commands:
uv run scripts/pg.py setup              # Setup PostgreSQL from scratch
uv run scripts/pg.py add --all          # Ingest all data from source files  
uv run scripts/pg.py embeddings         # Generate vector embeddings
uv run scripts/pg.py validate          # Validate complete setup

Data Management:
uv run scripts/pg.py stats             # Show table statistics
uv run scripts/pg.py clear <table>     # Clear specific table
uv run scripts/pg.py add <table>       # Add data to specific table
"""

import os
import sys
import argparse
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from backend.lib.database import get_db, init_db_sync
from backend.lib.data_ingestion import DataIngestionService

# Load environment variables
load_dotenv()


class PostgreSQLManager:
    """Unified PostgreSQL management system"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable not set!\n"
                "Please set it to: postgresql://user:password@localhost:5432/medical_rag_pg"
            )
        
        if not self.database_url.startswith("postgresql://"):
            raise ValueError(
                f"DATABASE_URL must be PostgreSQL URL, got: {self.database_url}\n"
                "Expected format: postgresql://user:password@localhost:5432/medical_rag_pg"
            )
        
        # Define table information
        self.tables = {
            "drug_relations": {
                "description": "Knowledge graph relations (1.7M+ records)",
                "source_file": "db_kg.csv",
                "ingestion_method": "ingest_knowledge_graph_from_csv",
                "primary": True,
                "large": True
            },
            "drug_metadata": {
                "description": "Drug metadata and basic information", 
                "source_file": "db_meta.txt",
                "ingestion_method": "ingest_drug_metadata",
                "primary": True,
                "large": False
            },
            "drug_product_stages": {
                "description": "Drug product development stages",
                "source_file": "db_product_stage.txt", 
                "ingestion_method": "ingest_drug_product_stages",
                "primary": True,
                "large": False
            },
            "product_stage_descriptions": {
                "description": "Product stage code descriptions",
                "source_file": "db_group_mapping.json",
                "ingestion_method": "ingest_product_stage_descriptions", 
                "primary": True,
                "large": False
            },
            "food_interactions": {
                "description": "Drug-food interaction warnings",
                "source_file": "db_food_interactions.txt",
                "ingestion_method": "ingest_drug_food_interactions",
                "primary": True,
                "large": False
            },
            "drug_dosage": {
                "description": "Drug dosage and product information",
                "source_file": "db_product.txt",
                "ingestion_method": "ingest_drug_dosage",
                "primary": True,
                "large": False
            },
            "vector_embeddings": {
                "description": "Vector embeddings for semantic search",
                "source_file": None,
                "ingestion_method": None,
                "primary": False,
                "large": True
            },
            "chat_sessions": {
                "description": "Chat session history",
                "source_file": None,
                "ingestion_method": None,
                "primary": False,
                "large": False
            },
            "chat_messages": {
                "description": "Chat message history", 
                "source_file": None,
                "ingestion_method": None,
                "primary": False,
                "large": False
            }
        }
        
        self.data_dir = self.find_data_directory()
        self.scripts_dir = Path(__file__).parent
        
        print(f"🐘 PostgreSQL Manager for Medical RAG Pipeline")
        print(f"🔗 Database: {self.database_url}")
        if self.data_dir:
            print(f"📁 Data directory: {self.data_dir}")
    
    def find_data_directory(self):
        """Find the directory containing data files"""
        possible_paths = ["./data", "../data", "./examples/data", "../examples/data", "./examples", "../examples"]
        
        for path in possible_paths:
            full_path = Path(path).resolve()
            if full_path.exists() and any((full_path / file).exists() for file in ["db_meta.txt", "db_kg.csv"]):
                return str(full_path)
        return None
    
    def run_script(self, script_name: str, args: list = None) -> bool:
        """Run a script with arguments"""
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            print(f"❌ Script not found: {script_name}")
            return False
        
        cmd = ["uv", "run", str(script_path)]
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(cmd, cwd=self.scripts_dir.parent)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Failed to run {script_name}: {e}")
            return False
    
    def get_table_stats(self, table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get statistics for tables"""
        if table_names is None:
            table_names = list(self.tables.keys())
        
        try:
            engine = create_engine(self.database_url)
            stats = {}
            
            with engine.connect() as conn:
                for table_name in table_names:
                    if table_name not in self.tables:
                        continue
                    
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                        count = result.fetchone()[0]
                        
                        result = conn.execute(text(f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'));"))
                        size = result.fetchone()[0]
                        
                        stats[table_name] = {
                            "count": count,
                            "size": size,
                            "description": self.tables[table_name]["description"],
                            "primary": self.tables[table_name]["primary"],
                            "large": self.tables[table_name]["large"]
                        }
                    except Exception as e:
                        stats[table_name] = {"count": 0, "size": "0 bytes", "error": str(e)}
            
            engine.dispose()
            return stats
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {}
    
    def display_stats(self, table_names: Optional[List[str]] = None):
        """Display formatted table statistics"""
        stats = self.get_table_stats(table_names)
        if not stats:
            return
        
        print("\n📊 PostgreSQL Table Statistics")
        print("=" * 90)
        
        total_records = 0
        for table_name, info in stats.items():
            if "error" in info:
                print(f"❌ {table_name:25} | ERROR: {info['error']}")
                continue
            
            count = info["count"]
            size = info["size"]
            desc = info["description"]
            
            icon = "📊" if info["primary"] else "🔧"
            warning = "🔥" if info["large"] and count > 100000 else ""
            
            print(f"{icon} {table_name:25} | {count:>10,} records | {size:>10} | {desc} {warning}")
            total_records += count
        
        print("-" * 90)
        print(f"📊 Total records: {total_records:,}")
    
    def clear_table(self, table_name: str, force: bool = False) -> bool:
        """Clear data from a specific table"""
        if table_name not in self.tables:
            print(f"❌ Unknown table: {table_name}")
            print(f"Available: {', '.join(self.tables.keys())}")
            return False
        
        stats = self.get_table_stats([table_name])
        if not stats or "error" in stats[table_name]:
            print(f"❌ Cannot access table: {table_name}")
            return False
        
        current_count = stats[table_name]["count"]
        if current_count == 0:
            print(f"✅ {table_name} is already empty")
            return True
        
        print(f"\n🗑️  Clear Table: {table_name}")
        print(f"📊 Current records: {current_count:,}")
        print(f"📝 {self.tables[table_name]['description']}")
        
        if self.tables[table_name]["large"] and current_count > 100000:
            print(f"⚠️  WARNING: Large table with {current_count:,} records!")
        
        if not force:
            response = input(f"Clear {current_count:,} records from {table_name}? (yes/no): ").lower().strip()
            if response not in ["yes", "y"]:
                print("❌ Cancelled")
                return False
        
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                if self.tables[table_name]["large"]:
                    # Batch delete for large tables
                    batch_size = 10000
                    while True:
                        result = conn.execute(text(f"DELETE FROM {table_name} WHERE id IN (SELECT id FROM {table_name} LIMIT {batch_size});"))
                        deleted = result.rowcount
                        conn.commit()
                        if deleted == 0:
                            break
                        print(f"   • Deleted {deleted:,} records...")
                else:
                    result = conn.execute(text(f"DELETE FROM {table_name};"))
                    conn.commit()
                
                print(f"✅ {table_name} cleared successfully!")
                return True
            
        except Exception as e:
            print(f"❌ Failed to clear {table_name}: {e}")
            return False
    
    def clear_all_tables(self, force: bool = False) -> bool:
        """Clear all tables in proper order"""
        stats = self.get_table_stats()
        total_records = sum(s["count"] for s in stats.values() if "count" in s)
        
        if total_records == 0:
            print("✅ All tables are already empty")
            return True
        
        print(f"\n🗑️  Clear All Tables ({total_records:,} total records)")
        
        if not force:
            print(f"⚠️  WARNING: This will permanently delete {total_records:,} records!")
            response = input("Type 'DELETE ALL' to confirm: ").strip()
            if response != "DELETE ALL":
                print("❌ Cancelled")
                return False
        
        # Clear in dependency order
        clear_order = ["chat_messages", "chat_sessions", "vector_embeddings", 
                      "drug_dosage", "food_interactions", "product_stage_descriptions", 
                      "drug_product_stages", "drug_relations", "drug_metadata"]
        
        for table in clear_order:
            if table in stats and stats[table].get("count", 0) > 0:
                self.clear_table(table, force=True)
        
        print("✅ All tables cleared")
        return True
    
    async def add_data_to_table(self, table_name: str) -> bool:
        """Add data to a specific table"""
        if table_name not in self.tables:
            print(f"❌ Unknown table: {table_name}")
            return False
        
        table_info = self.tables[table_name]
        print(f"\n➕ Add Data: {table_name}")
        print(f"📝 {table_info['description']}")
        
        if not table_info["source_file"] or not table_info["ingestion_method"]:
            if table_name == "vector_embeddings":
                print("⚠️  Use: uv run scripts/pg.py embeddings")
            else:
                print("⚠️  No data source available")
            return False
        
        if not self.data_dir:
            print("❌ Data directory not found")
            return False
        
        source_file = Path(self.data_dir) / table_info["source_file"]
        if not source_file.exists():
            print(f"❌ Source file not found: {source_file}")
            return False
        
        print(f"📁 Source: {source_file} ({source_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        try:
            db = next(get_db())
            try:
                ingestion_service = DataIngestionService(db, auto_create_tables=False)
                method = getattr(ingestion_service, table_info["ingestion_method"])
                
                if table_name == "drug_relations":
                    print("⚠️  Large file - this may take several minutes...")
                    result = await method(str(source_file))
                elif table_name == "drug_dosage":
                    result = method(str(source_file))
                else:
                    result = await method(str(source_file))
                
                if isinstance(result, dict):
                    count = result.get("count") or result.get("total_entries", 0)
                    print(f"✅ Added {count:,} records to {table_name}")
                    if result.get("errors"):
                        print(f"⚠️  {len(result['errors'])} errors occurred")
                
                return True
            finally:
                db.close()
        except Exception as e:
            print(f"❌ Failed to add data: {e}")
            return False
    
    async def add_all_data(self) -> bool:
        """Add data to all tables with sources"""
        tables_with_sources = [name for name, info in self.tables.items() 
                             if info["source_file"] and info["ingestion_method"]]
        
        print(f"\n➕ Add Data to All Tables ({len(tables_with_sources)} tables)")
        for table in tables_with_sources:
            print(f"   • {table}: {self.tables[table]['description']}")
        
        response = input(f"\nProceed? (yes/no): ").lower().strip()
        if response not in ["yes", "y"]:
            return False
        
        # Ingestion order
        order = ["drug_metadata", "drug_product_stages", "product_stage_descriptions", 
                "food_interactions", "drug_relations", "drug_dosage"]
        
        successful = 0
        for table in order:
            if table in tables_with_sources:
                if await self.add_data_to_table(table):
                    successful += 1
        
        print(f"\n✅ Successfully populated {successful}/{len(tables_with_sources)} tables")
        return successful == len(tables_with_sources)
    
    def setup_postgresql(self) -> bool:
        """Setup PostgreSQL from scratch"""
        return self.run_script("setup_postgresql_from_scratch.py")
    
    def generate_embeddings(self) -> bool:
        """Generate vector embeddings"""
        return self.run_script("generate_embeddings_postgresql.py")
    
    def validate_setup(self) -> bool:
        """Validate the complete setup"""
        return self.run_script("validate_postgresql_setup.py")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="PostgreSQL Management CLI for Medical RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Core Operations:
  setup                         Setup PostgreSQL from scratch
  add --all                     Ingest all data from source files
  embeddings                    Generate vector embeddings
  validate                      Validate complete setup

Data Management:
  stats                         Show all table statistics
  stats <table>                 Show specific table stats
  clear <table>                 Clear specific table
  clear --all                   Clear all tables
  add <table>                   Add data to specific table

Examples:
  uv run scripts/pg.py setup
  uv run scripts/pg.py add --all
  uv run scripts/pg.py stats
  uv run scripts/pg.py clear drug_relations
  uv run scripts/pg.py embeddings
        """
    )
    
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("target", nargs="?", help="Table name or target")
    parser.add_argument("--all", action="store_true", help="Apply to all tables")
    parser.add_argument("--force", action="store_true", help="Skip confirmations")
    
    args = parser.parse_args()
    
    try:
        manager = PostgreSQLManager()
        
        if args.command == "setup":
            return manager.setup_postgresql()
        
        elif args.command == "stats":
            if args.target:
                manager.display_stats([args.target])
            else:
                manager.display_stats()
            return True
        
        elif args.command == "clear":
            if args.all:
                return manager.clear_all_tables(force=args.force)
            elif args.target:
                return manager.clear_table(args.target, force=args.force)
            else:
                print("❌ Specify table name or use --all")
                return False
        
        elif args.command == "add":
            if args.all:
                return await manager.add_all_data()
            elif args.target:
                return await manager.add_data_to_table(args.target)
            else:
                print("❌ Specify table name or use --all")
                return False
        
        elif args.command == "embeddings":
            return manager.generate_embeddings()
        
        elif args.command == "validate":
            return manager.validate_setup()
        
        else:
            print(f"❌ Unknown command: {args.command}")
            print("Available: setup, stats, clear, add, embeddings, validate")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)