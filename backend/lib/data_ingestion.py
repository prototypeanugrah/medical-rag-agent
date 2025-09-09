"""
Data ingestion service for loading medical data into the system
"""

import csv
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from sqlalchemy.orm import Session
from tqdm import tqdm

from ..models.database import (
    DrugDosage,
    DrugMetadata,
    DrugProductStage,
    DrugRelation,
    FoodInteraction,
    ProductStageDescription,
    VectorEmbedding,
)
from ..models.schemas import (
    DrugRelationData,
    FoodInteractionData,
)
from .database import init_db_sync
from .embeddings import EmbeddingService
from .error_handling import DatabaseError

# Load environment variables
load_dotenv()


class DataIngestionService:
    """Service for ingesting various types of medical data"""

    def __init__(self, db: Session, auto_create_tables: bool = True):
        self.db = db

        # Automatically create tables if they don't exist
        if auto_create_tables:
            init_db_sync()

        # Initialize embedding service based on provider configuration
        try:
            provider = (os.getenv("EMBEDDING_PROVIDER", "openai")).lower()

            if provider == "openai":
                if (
                    os.getenv("OPENAI_API_KEY")
                    and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
                ):
                    self.embedding_service = EmbeddingService(provider="openai")
                else:
                    self.embedding_service = None
            elif provider == "ollama":
                # For local testing we don't require an OpenAI key
                self.embedding_service = EmbeddingService(provider="ollama")
            else:
                print(f"Unknown EMBEDDING_PROVIDER '{provider}', embeddings disabled.")
                self.embedding_service = None
        except Exception as e:
            print(f"Failed to initialize EmbeddingService: {e}")
            self.embedding_service = None

    async def ingest_drug_metadata(self, file_path: str) -> Dict[str, Any]:
        """Ingest drug metadata from tab-separated file"""
        errors = []
        success_count = 0
        drug_map = {}

        try:
            # First pass: count total lines for progress bar
            # total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

            with open(file_path, "r", encoding="utf-8") as file:
                # with tqdm(total=total_lines, desc="Drug Metadata", unit="rows") as pbar:
                for line_num, line in enumerate(file, 1):
                    try:
                        parts = line.strip().split("\t")
                        if len(parts) < 3:
                            # pbar.update(1)
                            continue

                        drug_id, field, value = parts[0], parts[1], parts[2]

                        if drug_id not in drug_map:
                            drug_map[drug_id] = {
                                "drugId": drug_id,
                                "name": None,
                                "type": None,
                                "products": [],
                            }

                        drug = drug_map[drug_id]

                        if field == "NAME":
                            drug["name"] = value
                        elif field == "TYPE":
                            drug["type"] = value
                        elif field == "PRODUCT":
                            drug["products"].append(value)

                    except Exception as e:
                        errors.append(f"Line {line_num}: Failed to parse - {str(e)}")

                        # pbar.update(1)

            # Insert into database
            with tqdm(
                total=len(drug_map), desc="Saving Drugs Metadata", unit="drugs"
            ) as pbar:
                for drug_id, drug_data in drug_map.items():
                    try:
                        # Upsert drug metadata
                        existing_drug = (
                            self.db.query(DrugMetadata)
                            .filter(DrugMetadata.drugId == drug_id)
                            .first()
                        )

                        if existing_drug:
                            existing_drug.name = drug_data["name"]
                            existing_drug.type = drug_data["type"]
                        else:
                            new_drug = DrugMetadata(
                                drugId=drug_data["drugId"],
                                name=drug_data["name"],
                                type=drug_data["type"],
                            )
                            self.db.add(new_drug)

                        success_count += 1

                    except Exception as e:
                        errors.append(
                            f"Failed to ingest drug metadata for {drug_id}: {str(e)}"
                        )

                    pbar.update(1)

            self.db.commit()
            return {"count": success_count, "errors": errors}

        except Exception as e:
            self.db.rollback()
            raise DatabaseError(f"Failed to read drug metadata file: {str(e)}")

    async def ingest_drug_product_stages(self, file_path: str) -> Dict[str, Any]:
        """Ingest drug product stages from tab-separated file"""
        errors = []
        success_count = 0
        drug_stages = {}  # Use dictionary to handle duplicates

        try:
            # First pass: count total lines for progress bar
            # total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

            with open(file_path, "r", encoding="utf-8") as file:
                # with tqdm(
                #     total=total_lines, desc="Product Stages", unit="rows"
                # ) as pbar:
                for line_num, line in enumerate(file, 1):
                    try:
                        parts = line.strip().split("\t")
                        if len(parts) < 3:
                            # pbar.update(1)
                            continue

                        drug_name, field, product_stage = (
                            parts[0],
                            parts[1],
                            parts[2],
                        )

                        if field == "PRODUCT_STAGE":
                            # Store the last product stage for each drug (handles duplicates)
                            drug_stages[drug_name] = product_stage

                    except Exception as e:
                        errors.append(f"Line {line_num}: Failed to parse - {str(e)}")

                        # pbar.update(1)

            # Now upsert all unique drug stages
            with tqdm(
                total=len(drug_stages), desc="Saving Product Stages", unit="drugs"
            ) as pbar:
                for drug_name, product_stage in drug_stages.items():
                    try:
                        # Upsert product stage
                        existing_stage = (
                            self.db.query(DrugProductStage)
                            .filter(DrugProductStage.drugName == drug_name)
                            .first()
                        )

                        if existing_stage:
                            existing_stage.productStage = product_stage
                        else:
                            new_stage = DrugProductStage(
                                drugName=drug_name, productStage=product_stage
                            )
                            self.db.add(new_stage)

                        success_count += 1

                    except Exception as e:
                        errors.append(
                            f"Failed to upsert stage for {drug_name}: {str(e)}"
                        )

                    pbar.update(1)

            self.db.commit()
            return {"count": success_count, "errors": errors}

        except Exception as e:
            self.db.rollback()
            raise DatabaseError(f"Failed to read drug product stages file: {str(e)}")

    async def ingest_product_stage_descriptions(self, file_path: str) -> Dict[str, Any]:
        """Ingest product stage descriptions from JSON file"""
        errors = []
        success_count = 0

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                stage_data = json.load(file)

            for stage_code, description in stage_data.items():
                try:
                    # Upsert product stage description
                    existing_desc = (
                        self.db.query(ProductStageDescription)
                        .filter(ProductStageDescription.stageCode == stage_code)
                        .first()
                    )

                    if existing_desc:
                        existing_desc.description = description
                    else:
                        new_desc = ProductStageDescription(
                            stageCode=stage_code, description=description
                        )
                        self.db.add(new_desc)

                    success_count += 1

                except Exception as e:
                    errors.append(
                        f"Failed to upsert description for {stage_code}: {str(e)}"
                    )

            self.db.commit()
            return {"count": success_count, "errors": errors}

        except Exception as e:
            self.db.rollback()
            raise DatabaseError(
                f"Failed to read product stage descriptions file: {str(e)}"
            )

    async def ingest_drug_food_interactions(self, file_path: str) -> Dict[str, Any]:
        """Ingest drug-food interactions from tab-separated file"""
        errors = []
        success_count = 0

        try:
            # First pass: count total lines for progress bar
            total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

            with open(file_path, "r", encoding="utf-8") as file:
                with tqdm(
                    total=total_lines,
                    desc="Saving Food Interactions",
                    unit="rows",
                ) as pbar:
                    for line_num, line in enumerate(file, 1):
                        try:
                            parts = line.strip().split("\t")
                            if len(parts) < 3:
                                pbar.update(1)
                                continue

                            drug_name, field, interaction = (
                                parts[0],
                                parts[1],
                                parts[2],
                            )

                            if field == "DRUG_FOOD_INTERACTION":
                                food_interaction = FoodInteraction(
                                    drugName=drug_name,
                                    interaction=interaction,
                                    source="DrugBank",
                                )
                                self.db.add(food_interaction)
                                success_count += 1

                        except Exception as e:
                            errors.append(
                                f"Line {line_num}: Failed to parse - {str(e)}"
                            )

                        pbar.update(1)

                        # Commit in batches for better performance
                        if line_num % 1000 == 0:
                            self.db.commit()

            self.db.commit()
            return {"count": success_count, "errors": errors}

        except Exception as e:
            self.db.rollback()
            raise DatabaseError(f"Failed to read drug food interactions file: {str(e)}")

    async def ingest_knowledge_graph_from_csv(
        self, file_path: str, max_rows: int = None
    ) -> Dict[str, Any]:
        """Ingest knowledge graph from CSV file"""
        errors = []
        success_count = 0
        batch_size = 1000
        batch = []

        try:
            # First pass: count total lines for progress bar
            with open(file_path, "r", encoding="utf-8") as file:
                total_lines = sum(1 for _ in csv.DictReader(file))
                if max_rows:
                    total_lines = min(total_lines, max_rows)

            with open(file_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                progress_desc = (
                    f"Saving Knowledge Graph (limit: {max_rows:,})"
                    if max_rows
                    else "Saving Knowledge Graph"
                )
                with tqdm(
                    total=total_lines, desc=progress_desc, unit="relations"
                ) as pbar:
                    for row_num, row in enumerate(reader, 1):
                        # Stop if we've reached the max_rows limit
                        if max_rows and row_num > max_rows:
                            pbar.set_description(f"{progress_desc} - Limit reached")
                            break

                        try:
                            relation = DrugRelation(
                                relation=row.get("relation", ""),
                                displayRelation=row.get("display_relation", ""),
                                xIndex=int(row.get("x_index", 0)),
                                xId=row.get("x_id", ""),
                                xType=row.get("x_type", ""),
                                xName=row.get("x_name", ""),
                                xSource=row.get("x_source", ""),
                                yIndex=int(row.get("y_index", 0)),
                                yId=row.get("y_id", ""),
                                yType=row.get("y_type", ""),
                                yName=row.get("y_name", ""),
                                ySource=row.get("y_source", ""),
                                relationType=row.get("relation_type"),
                            )

                            batch.append(relation)

                            if len(batch) >= batch_size:
                                self.db.add_all(batch)
                                self.db.commit()
                                success_count += len(batch)
                                batch = []

                        except Exception as e:
                            errors.append(f"Row {row_num}: Failed to parse - {str(e)}")

                        pbar.update(1)

                    # Process remaining batch
                    if batch:
                        self.db.add_all(batch)
                        self.db.commit()
                        success_count += len(batch)

            return {"count": success_count, "errors": errors}

        except Exception as e:
            self.db.rollback()
            raise DatabaseError(f"Failed to read knowledge graph CSV: {str(e)}")

    async def ingest_all_data_sources(
        self, base_dir: str, kg_max_rows: int = None
    ) -> Dict[str, Any]:
        """Ingest all data sources from a directory"""
        print("Starting comprehensive data ingestion...")
        if kg_max_rows:
            print(f"Knowledge graph limited to {kg_max_rows} rows for testing")

        results = {
            "drugMetadata": {"count": 0, "errors": []},
            "productStages": {"count": 0, "errors": []},
            "productStageDescriptions": {"count": 0, "errors": []},
            "foodInteractions": {"count": 0, "errors": []},
            "knowledgeGraph": {"count": 0, "errors": []},
            "dosage": {"count": 0, "errors": []},
        }

        try:
            # Ingest drug metadata
            # print("Ingesting drug metadata...")
            results["drugMetadata"] = await self.ingest_drug_metadata(
                os.path.join(base_dir, "db_meta.txt")
            )
            # print(f"Drug metadata: {results['drugMetadata']['count']} records ingested")

            # Ingest product stages
            # print("Ingesting product stages...")
            results["productStages"] = await self.ingest_drug_product_stages(
                os.path.join(base_dir, "db_product_stage.txt")
            )
            # print(
            #     f"Product stages: {results['productStages']['count']} records ingested"
            # )

            # Ingest product stage descriptions
            # print("Ingesting product stage descriptions...")
            results[
                "productStageDescriptions"
            ] = await self.ingest_product_stage_descriptions(
                os.path.join(base_dir, "db_group_mapping.json")
            )
            # print(
            #     f"Product stage descriptions: {results['productStageDescriptions']['count']} records ingested"
            # )

            # Ingest food interactions
            # print("Ingesting food interactions...")
            results["foodInteractions"] = await self.ingest_drug_food_interactions(
                os.path.join(base_dir, "db_food_interactions.txt")
            )
            # print(
            #     f"Food interactions: {results['foodInteractions']['count']} records ingested"
            # )

            # Ingest knowledge graph
            # print("Ingesting knowledge graph...")
            results["knowledgeGraph"] = await self.ingest_knowledge_graph_from_csv(
                os.path.join(base_dir, "db_kg.csv"), max_rows=kg_max_rows
            )
            # print(
            #     f"Knowledge graph: {results['knowledgeGraph']['count']} records ingested"
            # )

            # Ingest dosage information
            # print("Ingesting dosage information...")
            try:
                results["dosage"] = self.ingest_drug_dosage(
                    os.path.join(base_dir, "db_product.txt")
                )
                # print(
                #     f"Dosage info: {results['dosage']['total_entries']} records ingested"
                # )
            except Exception as e:
                print(f"Warning: Could not ingest dosage data: {e}")
                results["dosage"] = {"count": 0, "errors": [str(e)]}

            return results

        except Exception as e:
            raise DatabaseError(f"Failed during comprehensive data ingestion: {str(e)}")

    async def ingest_drug_relations(
        self, relations: List[DrugRelationData]
    ) -> Dict[str, Any]:
        """Ingest drug relations from list"""
        errors = []
        success_count = 0

        for relation_data in relations:
            try:
                relation = DrugRelation(
                    relation=relation_data.relation,
                    displayRelation=relation_data.display_relation,
                    xIndex=relation_data.x_index,
                    xId=relation_data.x_id,
                    xType=relation_data.x_type,
                    xName=relation_data.x_name,
                    xSource=relation_data.x_source,
                    yIndex=relation_data.y_index,
                    yId=relation_data.y_id,
                    yType=relation_data.y_type,
                    yName=relation_data.y_name,
                    ySource=relation_data.y_source,
                    relationType=relation_data.relation_type,
                )
                self.db.add(relation)
                success_count += 1

            except Exception as e:
                errors.append(
                    f"Failed to ingest relation {relation_data.x_name} - {relation_data.y_name}: {str(e)}"
                )

        self.db.commit()
        return {"count": success_count, "errors": errors}

    async def ingest_food_interactions(
        self, interactions: List[FoodInteractionData]
    ) -> Dict[str, Any]:
        """Ingest food interactions from list"""
        errors = []
        success_count = 0

        for interaction_data in interactions:
            try:
                interaction = FoodInteraction(
                    drugName=interaction_data.drugName,
                    drugId=interaction_data.drugId,
                    interaction=interaction_data.interaction,
                    source=interaction_data.source,
                )
                self.db.add(interaction)
                success_count += 1

            except Exception as e:
                errors.append(
                    f"Failed to ingest food interaction for {interaction_data.drugName}: {str(e)}"
                )

        self.db.commit()
        return {"count": success_count, "errors": errors}

    async def create_sample_data(self) -> None:
        """Create sample data for testing"""
        # Sample drug relations
        sample_drug_relations = [
            DrugRelationData(
                relation="contraindication",
                display_relation="contraindication",
                x_index=15193,
                x_id="DB05271",
                x_type="drug",
                x_name="Rotigotine",
                x_source="DrugBank",
                y_index=33577,
                y_id="5044",
                y_type="disease",
                y_name="hypertensive disorder",
                y_source="MONDO",
            ),
            DrugRelationData(
                relation="drug_drug",
                display_relation="synergistic interaction",
                x_index=16086,
                x_id="DB00001",
                x_type="drug",
                x_name="Lepirudin",
                x_source="DrugBank",
                y_index=15019,
                y_id="DB06605",
                y_type="drug",
                y_name="Apixaban",
                y_source="DrugBank",
            ),
        ]

        # Sample food interactions
        sample_food_interactions = [
            FoodInteractionData(
                drugName="Lepirudin",
                drugId="DB00001",
                interaction="Avoid herbs and supplements with anticoagulant/antiplatelet activity. Examples include chamomile, garlic, ginger, ginkgo and ginseng.",
                source="DrugBank",
            ),
            FoodInteractionData(
                drugName="Warfarin",
                drugId="DB00682",
                interaction="Avoid foods high in vitamin K such as leafy green vegetables. Maintain consistent intake of vitamin K-rich foods.",
                source="FDA",
            ),
        ]

        # Ingest sample data
        await self.ingest_drug_relations(sample_drug_relations)
        await self.ingest_food_interactions(sample_food_interactions)

    async def reindex_embeddings(self) -> None:
        """Reindex all embeddings"""
        if self.embedding_service is None:
            raise ValueError(
                "Embedding service not configured. Set EMBEDDING_PROVIDER and related env vars."
            )
        await self.embedding_service.reindex_all(self.db)

    def get_ingestion_stats(self) -> Dict[str, int]:
        """Get ingestion statistics"""
        return {
            "drugRelations": self.db.query(DrugRelation).count(),
            "foodInteractions": self.db.query(FoodInteraction).count(),
            "drugMetadata": self.db.query(DrugMetadata).count(),
            "drugProductStages": self.db.query(DrugProductStage).count(),
            "productStageDescriptions": self.db.query(ProductStageDescription).count(),
            "drugDosage": self.db.query(DrugDosage).count(),
            "vectorEmbeddings": self.db.query(VectorEmbedding).count(),
        }

    def clear_all_data(self) -> None:
        """Clear all data from database"""
        self.db.query(VectorEmbedding).delete()
        self.db.query(FoodInteraction).delete()
        self.db.query(DrugDosage).delete()
        self.db.query(DrugProductStage).delete()
        self.db.query(ProductStageDescription).delete()
        self.db.query(DrugMetadata).delete()
        self.db.query(DrugRelation).delete()
        self.db.commit()

    async def initialize_database(self) -> None:
        """Initialize database with sample data"""
        # Clear existing data
        self.clear_all_data()

        # Create sample data
        await self.create_sample_data()

        # Index embeddings
        await self.reindex_embeddings()

    def ingest_drug_dosage(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest drug dosage information from db_product.txt

        File format:
        DRUG_ID    FIELD_TYPE    VALUE
        DB00001    PRODUCT      Refludan
        DB00001    DOSAGE_FORM  Powder
        DB00001    ROUTE        Intravenous
        DB00001    STRENGTH     50 mg/1mL
        DB00001    MANUFACTURER Bayer healthcare pharmaceuticals inc
        """
        # print(f"🔬 Ingesting drug dosage data from {file_path}...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosage file not found: {file_path}")

        # Clear existing dosage data
        self.db.query(DrugDosage).delete()
        self.db.commit()

        # First pass: collect all manufacturers per drug
        drug_manufacturers = {}

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split("\t")
                    if len(parts) != 3:
                        continue

                    drug_id, field_type, value = parts

                    if field_type == "MANUFACTURER":
                        if drug_id not in drug_manufacturers:
                            drug_manufacturers[drug_id] = set()
                        # Handle multiple manufacturers separated by " + "
                        manufacturers = [m.strip() for m in value.split(" + ")]
                        drug_manufacturers[drug_id].update(manufacturers)

                except Exception:
                    continue

        # Convert sets to comma-separated strings
        for drug_id in drug_manufacturers:
            manufacturers_list = sorted(list(drug_manufacturers[drug_id]))
            drug_manufacturers[drug_id] = ", ".join(manufacturers_list)

        # Second pass: parse product entries and assign manufacturers
        current_product = {}
        completed_entries = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split("\t")
                    if len(parts) != 3:
                        continue

                    drug_id, field_type, value = parts

                    if field_type == "PRODUCT":
                        # If we have a completed product, save it
                        if current_product and current_product.get("productName"):
                            # Assign manufacturer from our collected data
                            current_product["manufacturer"] = drug_manufacturers.get(
                                current_product["drugId"]
                            )
                            completed_entries.append(current_product.copy())

                        # Start a new product entry
                        current_product = {
                            "drugId": drug_id,
                            "productName": value,
                            "dosageForm": None,
                            "route": None,
                            "strength": None,
                            "manufacturer": None,
                        }
                    elif field_type == "DOSAGE_FORM":
                        if current_product:
                            current_product["dosageForm"] = value
                    elif field_type == "ROUTE":
                        if current_product:
                            current_product["route"] = value
                    elif field_type == "STRENGTH":
                        if current_product:
                            current_product["strength"] = value
                    # Skip MANUFACTURER entries in second pass as we already collected them

                except Exception as e:
                    print(f"Error processing line {line_num}: {line} - {e}")
                    continue

        # Don't forget the last product
        if current_product and current_product.get("productName"):
            current_product["manufacturer"] = drug_manufacturers.get(
                current_product["drugId"]
            )
            completed_entries.append(current_product)

        # Remove duplicates and filter out incomplete entries
        unique_entries = []
        seen_combinations = set()

        for entry in completed_entries:
            # Create a unique key for each combination
            unique_key = (
                entry["drugId"],
                entry["productName"],
                entry.get("dosageForm"),
                entry.get("route"),
                entry.get("strength"),
            )

            # Only add if this combination hasn't been seen and has required fields
            if (
                unique_key not in seen_combinations
                and entry["drugId"]
                and entry["productName"]
            ):
                seen_combinations.add(unique_key)
                unique_entries.append(entry)

        # Save unique entries to database
        batch_size = 1000
        entries_list = unique_entries

        with tqdm(
            total=len(entries_list), desc="Saving dosage data", unit="records"
        ) as pbar:
            for i in range(0, len(entries_list), batch_size):
                batch = entries_list[i : i + batch_size]

                for entry in batch:
                    dosage = DrugDosage(**entry)
                    self.db.add(dosage)

                self.db.commit()
                pbar.update(len(batch))

        # Get statistics
        total_entries = self.db.query(DrugDosage).count()
        unique_drugs = self.db.query(DrugDosage.drugId).distinct().count()
        unique_products = self.db.query(DrugDosage.productName).distinct().count()
        unique_forms = (
            self.db.query(DrugDosage.dosageForm)
            .filter(DrugDosage.dosageForm.isnot(None))
            .distinct()
            .count()
        )
        unique_routes = (
            self.db.query(DrugDosage.route)
            .filter(DrugDosage.route.isnot(None))
            .distinct()
            .count()
        )

        return {
            "total_entries": total_entries,
            "unique_drugs": unique_drugs,
            "unique_products": unique_products,
            "unique_forms": unique_forms,
            "unique_routes": unique_routes,
        }
