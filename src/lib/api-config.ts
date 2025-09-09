/**
 * API Configuration
 * Configure the backend URL for different environments
 */

// Use environment variable or default to Python backend
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// API endpoints
export const API_ENDPOINTS = {
  chat: '/api/chat',
  ingest: '/api/ingest', 
  setup: '/api/setup'
}

// Helper function to build full API URLs
export function getApiUrl(endpoint: string): string {
  return `${API_BASE_URL}${endpoint}`
}
