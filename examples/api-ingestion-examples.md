# Data Ingestion API Examples

## Using the REST API Endpoint

The ingestion API is available at `POST /api/ingest` and accepts the following data types:

### 1. Drug Relations (Knowledge Graph Data)

```bash
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "type": "drug_relations",
    "data": [
      {
        "relation": "treats",
        "display_relation": "treats",
        "x_index": 12345,
        "x_id": "DB00001",
        "x_type": "drug",
        "x_name": "Your Drug Name",
        "x_source": "YourDataSource",
        "y_index": 67890,
        "y_id": "MONDO:12345",
        "y_type": "disease",
        "y_name": "Your Disease Name",
        "y_source": "MONDO"
      }
    ],
    "reindex": true
  }'
```

### 2. Food Interactions

```bash
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "type": "food_interactions",
    "data": [
      {
        "drugName": "Your Drug",
        "drugId": "DB00001",
        "interaction": "Avoid grapefruit juice as it may increase drug levels",
        "source": "YourSource"
      }
    ],
    "reindex": true
  }'
```

### 3. Drug-Drug Interactions

```bash
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "type": "drug_drug_interactions",
    "data": [
      {
        "drug1Name": "Drug A",
        "drug1Id": "DB00001",
        "drug2Name": "Drug B", 
        "drug2Id": "DB00002",
        "interaction": "May increase risk of bleeding",
        "interactionType": "additive",
        "source": "YourSource"
      }
    ],
    "reindex": true
  }'
```

## JavaScript/TypeScript Example

```typescript
async function ingestData() {
  const response = await fetch('/api/ingest', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      type: 'drug_relations',
      data: yourKnowledgeGraphData,
      reindex: true
    })
  });
  
  const result = await response.json();
  console.log('Ingestion result:', result);
}
```
