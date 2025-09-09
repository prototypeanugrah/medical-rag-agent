# Data Format Examples

## 1. Drug Relations (Knowledge Graph Format)

### JSON Format
```json
[
  {
    "relation": "treats",
    "display_relation": "treats",
    "x_index": 12345,
    "x_id": "DB00001",
    "x_type": "drug",
    "x_name": "Aspirin",
    "x_source": "DrugBank",
    "y_index": 67890,
    "y_id": "MONDO:0005036",
    "y_type": "disease",
    "y_name": "Cardiovascular disease",
    "y_source": "MONDO"
  }
]
```

### CSV Format
```csv
relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source
treats,treats,12345,DB00001,drug,Aspirin,DrugBank,67890,MONDO:0005036,disease,Cardiovascular disease,MONDO
contraindication,contraindication,12346,DB00002,drug,Warfarin,DrugBank,67891,MONDO:0002280,disease,Pregnancy,MONDO
```

## 2. Food Interactions

### JSON Format
```json
[
  {
    "drugName": "Warfarin",
    "drugId": "DB00682",
    "interaction": "Avoid foods high in vitamin K such as leafy green vegetables. Maintain consistent intake.",
    "source": "FDA"
  }
]
```

### CSV Format
```csv
drugName,drugId,interaction,source
Warfarin,DB00682,"Avoid foods high in vitamin K such as leafy green vegetables. Maintain consistent intake.",FDA
Grapefruit,DB01234,"May increase drug absorption and toxicity",Clinical Studies
```

## 3. Drug-Drug Interactions

### JSON Format
```json
[
  {
    "drug1Name": "Aspirin",
    "drug1Id": "DB00945",
    "drug2Name": "Warfarin",
    "drug2Id": "DB00682",
    "interaction": "Aspirin may increase the risk of bleeding when combined with Warfarin",
    "interactionType": "additive",
    "source": "FDA"
  }
]
```

### CSV Format
```csv
drug1Name,drug1Id,drug2Name,drug2Id,interaction,interactionType,source
Aspirin,DB00945,Warfarin,DB00682,"Aspirin may increase the risk of bleeding when combined with Warfarin",additive,FDA
Simvastatin,DB00641,Gemfibrozil,DB01241,"Increased risk of myopathy and rhabdomyolysis",synergistic,Clinical Studies
```

## 4. Custom Knowledge Graph Format

### Entities File (entities.json)
```json
[
  {
    "id": "drug_001",
    "name": "Metformin",
    "type": "drug",
    "properties": {
      "class": "biguanide",
      "indication": "diabetes",
      "mechanism": "glucose_reduction"
    }
  },
  {
    "id": "disease_001",
    "name": "Type 2 Diabetes",
    "type": "disease",
    "properties": {
      "category": "metabolic",
      "severity": "chronic"
    }
  }
]
```

### Relationships File (relationships.json)
```json
[
  {
    "source": "drug_001",
    "target": "disease_001",
    "type": "treats",
    "properties": {
      "efficacy": "high",
      "evidence_level": "strong"
    }
  }
]
```

## 5. Text Files

Any `.txt` or `.md` files can be ingested directly. The system will:
- Read the full content of each file
- Create vector embeddings for semantic search
- Store metadata including filename and source

### Example Text File Structure
```
# Medical Guidelines for Diabetes Management

## Overview
Type 2 diabetes is a chronic condition that affects how your body processes blood sugar...

## Treatment Options
- Metformin: First-line treatment for type 2 diabetes
- Insulin: Used when metformin is insufficient
- Lifestyle modifications: Diet and exercise

## Drug Interactions
- Metformin + Alcohol: May cause lactic acidosis
- Insulin + Beta-blockers: May mask hypoglycemia symptoms
```
