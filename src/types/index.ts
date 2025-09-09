export interface DrugRelationData {
  relation: string;
  display_relation: string;
  x_index: number;
  x_id: string;
  x_type: string;
  x_name: string;
  x_source: string;
  y_index: number;
  y_id: string;
  y_type: string;
  y_name: string;
  y_source: string;
  relation_type?: string;
}

export interface FoodInteractionData {
  drugName: string;
  drugId?: string;
  interaction: string;
  source: string;
}

export interface DrugDrugInteractionData {
  drug1Name: string;
  drug1Id?: string;
  drug2Name: string;
  drug2Id?: string;
  interaction: string;
  interactionType?: string;
  source: string;
}

// New types for the actual data sources
export interface DrugMetadata {
  drugId: string;
  name?: string;
  type?: string;
  products: string[];
}

export interface DrugProductStage {
  drugName: string;
  productStage: string;
}

export interface DrugFoodInteraction {
  drugName: string;
  interaction: string;
}

export interface KnowledgeGraphRelation {
  relation: string;
  display_relation: string;
  x_index: number;
  x_id: string;
  x_type: string;
  x_name: string;
  x_source: string;
  y_index: number;
  y_id: string;
  y_type: string;
  y_name: string;
  y_source: string;
  relation_type: string;
}

export interface KnowledgeGraphQuery {
  drugName?: string;
  drugId?: string;
  disease?: string;
  symptom?: string;
  interactionType?: string;
}

export interface RetrievalResult {
  content: string;
  source: string;
  metadata: any;
  relevanceScore: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata?: any;
  createdAt: Date;
}

export interface DrugInfo {
  id: string;
  name: string;
  type: string;
  source: string;
}

export interface InteractionWarning {
  type: 'drug-drug' | 'food' | 'disease-contraindication';
  severity: 'low' | 'medium' | 'high';
  description: string;
  source: string;
  relatedDrugs?: string[];
  relatedFoods?: string[];
}