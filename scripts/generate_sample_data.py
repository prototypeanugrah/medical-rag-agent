#!/usr/bin/env python3
"""
Generate Sample Medical Data for Development/Testing

This script creates minimal sample data files for development and testing
when the full medical datasets are not available.

Usage:
    python scripts/generate_sample_data.py --all --minimal
    python scripts/generate_sample_data.py --table drug_relations --count 100
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta


class SampleDataGenerator:
    def __init__(self):
        self.drug_ids = [f"DB{i:05d}" for i in range(1, 101)]  # DB00001 to DB00100
        self.drug_names = [
            "Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Metoprolol",
            "Omeprazole", "Simvastatin", "Losartan", "Hydrochlorothiazide", "Gabapentin",
            "Sertraline", "Montelukast", "Escitalopram", "Rosuvastatin", "Pantoprazole",
            "Azithromycin", "Amoxicillin", "Furosemide", "Trazodone", "Tramadol",
            "Prednisone", "Clonazepam", "Lorazepam", "Cyclobenzaprine", "Meloxicam",
            "Citalopram", "Duloxetine", "Venlafaxine", "Bupropion", "Mirtazapine",
            "Quetiapine", "Aripiprazole", "Risperidone", "Olanzapine", "Ziprasidone",
            "Warfarin", "Clopidogrel", "Aspirin", "Ibuprofen", "Naproxen",
            "Acetaminophen", "Codeine", "Morphine", "Oxycodone", "Fentanyl",
            "Insulin", "Glipizide", "Glyburide", "Pioglitazone", "Sitagliptin"
        ]
        self.categories = ["cardiovascular", "diabetes", "pain", "psychiatric", "antibiotic", "gastrointestinal"]
        self.stages = ["approved", "withdrawn", "investigational", "experimental"]
        self.interactions = [
            "may increase risk of hypoglycemia",
            "may enhance anticoagulant effect", 
            "may increase sedation",
            "may reduce absorption",
            "may increase blood pressure",
            "may cause drug interaction",
            "requires dose adjustment",
            "monitor for side effects"
        ]
        
    def generate_knowledge_graph(self, count: int = 100) -> str:
        """Generate sample drug relations CSV"""
        lines = ["subject_id,object_id,predicate,subject_name,object_name,subject_category,object_category,evidence"]
        
        predicates = ["interacts_with", "treats", "contraindicated_with", "enhances", "reduces"]
        evidence_types = ["clinical_study", "clinical_trial", "case_report", "in_vitro", "theoretical"]
        
        for i in range(count):
            subj_id = random.choice(self.drug_ids)
            obj_id = random.choice(self.drug_ids)
            while obj_id == subj_id:  # Avoid self-relations
                obj_id = random.choice(self.drug_ids)
                
            predicate = random.choice(predicates)
            subj_name = random.choice(self.drug_names)
            obj_name = random.choice(self.drug_names)
            subj_cat = random.choice(self.categories)
            obj_cat = random.choice(self.categories)
            evidence = random.choice(evidence_types)
            
            lines.append(f"{subj_id},{obj_id},{predicate},{subj_name},{obj_name},{subj_cat},{obj_cat},{evidence}")
            
        return "\n".join(lines)
    
    def generate_metadata(self, count: int = 50) -> str:
        """Generate sample drug metadata"""
        lines = []
        for i in range(count):
            drug_id = self.drug_ids[i] if i < len(self.drug_ids) else f"DB{i+1:05d}"
            name = self.drug_names[i % len(self.drug_names)]
            category = random.choice(self.categories)
            stage = random.choice(self.stages)
            description = f"{category} medication"
            
            lines.append(f"{drug_id}\t{name}\t{category}\t{stage}\t{description}")
            
        return "\n".join(lines)
    
    def generate_food_interactions(self, count: int = 50) -> str:
        """Generate sample food interactions"""
        lines = []
        foods = ["alcohol", "grapefruit", "vitamin K foods", "dairy products", "high sodium foods"]
        severities = ["high", "medium", "low"]
        mechanisms = ["liver_metabolism", "absorption", "coagulation_pathway", "electrolyte_balance"]
        
        for i in range(count):
            drug_id = self.drug_ids[i % len(self.drug_ids)]
            food = random.choice(foods)
            severity = random.choice(severities)
            mechanism = random.choice(mechanisms)
            interaction = f"Avoid {food}"
            
            lines.append(f"{drug_id}\t{interaction}\t{severity}\t{mechanism}")
            
        return "\n".join(lines)
    
    def generate_products(self, count: int = 100) -> str:
        """Generate sample drug products"""
        lines = []
        forms = ["tablet", "capsule", "injection", "liquid", "cream", "patch"]
        routes = ["oral", "intravenous", "topical", "subcutaneous", "intramuscular"]
        
        for i in range(count):
            drug_id = self.drug_ids[i % len(self.drug_ids)]
            form = random.choice(forms)
            strength = f"{random.randint(5, 500)}mg"
            route = random.choice(routes)
            name = f"{self.drug_names[i % len(self.drug_names)]} {strength} {form}"
            
            lines.append(f"{drug_id}\t{form}\t{strength}\t{route}\t{name}")
            
        return "\n".join(lines)
    
    def generate_product_stages(self, count: int = 50) -> str:
        """Generate sample product stages"""
        lines = []
        
        for i in range(count):
            drug_id = self.drug_ids[i % len(self.drug_ids)]
            stage = random.choice(self.stages)
            
            # Generate random date within last 5 years
            base_date = datetime.now() - timedelta(days=5*365)
            random_days = random.randint(0, 5*365)
            date = (base_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
            
            status = "active" if stage in ["approved", "investigational"] else "inactive"
            
            lines.append(f"{drug_id}\t{stage}\t{date}\t{status}")
            
        return "\n".join(lines)
    
    def generate_stage_mapping(self) -> str:
        """Generate stage descriptions JSON"""
        mapping = {
            "approved": "Currently marketed and available for prescription",
            "withdrawn": "Removed from market due to safety or efficacy concerns",
            "investigational": "Under clinical development and testing",
            "experimental": "Early research phase, not yet in clinical trials"
        }
        return json.dumps(mapping, indent=2)
    
    def generate_all(self, minimal: bool = True):
        """Generate all sample data files"""
        if minimal:
            counts = {
                "drug_relations": 100,
                "drug_metadata": 50,
                "food_interactions": 50,
                "products": 100,
                "product_stages": 50
            }
        else:
            counts = {
                "drug_relations": 1000,
                "drug_metadata": 100,
                "food_interactions": 100,
                "products": 500,
                "product_stages": 100
            }
        
        print("🎯 Generating sample medical data...")
        
        # Generate knowledge graph
        print(f"📊 Generating knowledge graph ({counts['drug_relations']} relations)...")
        with open("db_kg.csv", "w") as f:
            f.write(self.generate_knowledge_graph(counts["drug_relations"]))
        print(f"✅ Created db_kg.csv")
        
        # Generate metadata
        print(f"📝 Generating drug metadata ({counts['drug_metadata']} drugs)...")
        with open("db_meta.txt", "w") as f:
            f.write(self.generate_metadata(counts["drug_metadata"]))
        print(f"✅ Created db_meta.txt")
        
        # Generate food interactions
        print(f"🍽️ Generating food interactions ({counts['food_interactions']} interactions)...")
        with open("db_food_interactions.txt", "w") as f:
            f.write(self.generate_food_interactions(counts["food_interactions"]))
        print(f"✅ Created db_food_interactions.txt")
        
        # Generate products
        print(f"💊 Generating drug products ({counts['products']} products)...")
        with open("db_product.txt", "w") as f:
            f.write(self.generate_products(counts["products"]))
        print(f"✅ Created db_product.txt")
        
        # Generate product stages
        print(f"📈 Generating product stages ({counts['product_stages']} stages)...")
        with open("db_product_stage.txt", "w") as f:
            f.write(self.generate_product_stages(counts["product_stages"]))
        print(f"✅ Created db_product_stage.txt")
        
        # Generate stage mapping
        print("🗂️ Generating stage descriptions...")
        with open("db_group_mapping.json", "w") as f:
            f.write(self.generate_stage_mapping())
        print(f"✅ Created db_group_mapping.json")
        
        print("\n🎉 Sample data generation complete!")
        print("📁 Files created in current directory")
        print("🚀 Run 'npm run pg:add' to ingest the data")


def main():
    parser = argparse.ArgumentParser(description="Generate sample medical data for development")
    parser.add_argument("--all", action="store_true", help="Generate all data files")
    parser.add_argument("--minimal", action="store_true", help="Generate minimal dataset")
    parser.add_argument("--table", choices=["drug_relations", "drug_metadata", "food_interactions", "products", "product_stages"], help="Generate specific table")
    parser.add_argument("--count", type=int, default=100, help="Number of records to generate")
    
    args = parser.parse_args()
    
    generator = SampleDataGenerator()
    
    if args.all:
        generator.generate_all(minimal=args.minimal)
    elif args.table:
        print(f"Generating {args.table} with {args.count} records...")
        if args.table == "drug_relations":
            with open("db_kg.csv", "w") as f:
                f.write(generator.generate_knowledge_graph(args.count))
        elif args.table == "drug_metadata":
            with open("db_meta.txt", "w") as f:
                f.write(generator.generate_metadata(args.count))
        elif args.table == "food_interactions":
            with open("db_food_interactions.txt", "w") as f:
                f.write(generator.generate_food_interactions(args.count))
        elif args.table == "products":
            with open("db_product.txt", "w") as f:
                f.write(generator.generate_products(args.count))
        elif args.table == "product_stages":
            with open("db_product_stage.txt", "w") as f:
                f.write(generator.generate_product_stages(args.count))
        print(f"✅ Generated {args.table}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()