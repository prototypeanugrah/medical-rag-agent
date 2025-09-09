# Medical RAG Agent

A production-ready Retrieval-Augmented Generation (RAG) system designed for medical drug interactions, contraindications, and safety information. Built with PostgreSQL, pgvector, and OpenAI embeddings to serve **Informed Medication Decision Makers** who need comprehensive understanding of their medications.

## 🎯 Project Overview

This system transforms complex medical database information into clear, actionable insights for adults making important medication decisions. Unlike generic AI assistants that provide simple answers, this specialized system leverages 1.7M+ curated medical records to provide comprehensive, source-attributed responses with intelligent prioritization.

**📊 Data Note**: Medical data files are not included in this repository due to size and licensing constraints. See [`DATA_INGESTION_GUIDE.md`](./DATA_INGESTION_GUIDE.md) for instructions on obtaining the required medical databases or generating sample data for development.

### Target Users
- **Primary**: Adults prescribed new medications wanting full understanding
- **Secondary**: People experiencing side effects seeking comprehensive analysis  
- **Tertiary**: Long-term medication users wanting deeper insights
- **Use Case**: Anyone making important medication decisions who values detailed, trustworthy information

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js UI   │────│   FastAPI        │────│  RAG Pipeline   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌──────────────────────────────────┼─────────────────┐
                       │                                  │                 │
                ┌─────────────┐              ┌─────────────────┐    ┌──────────────┐
                │ PostgreSQL  │              │ Vector          │    │ AI Agent     │
                │ + pgvector  │              │ Embeddings      │    │ (OpenAI)     │
                │ (1.7M rows) │              │ (512D vectors)  │    │              │
                └─────────────┘              └─────────────────┘    └──────────────┘
```

## 🚀 Key Features

- **1.7M+ Medical Records**: Curated data from DrugBank, MONDO, and medical ontologies
- **PostgreSQL + pgvector**: Production-grade vector database with native similarity search
- **Intelligent Query Routing**: AI-powered classification determines optimal data sources
- **Content-Based Prioritization**: Critical warnings and interactions shown first
- **Product Availability Context**: Distinguishes between available and withdrawn medications
- **Cross-Table Relationships**: Connects dosage, availability, and safety information
- **Source Transparency**: Always cites specific database tables for information
- **Real-time Chat Interface**: Interactive Next.js web-based interface
- **Comprehensive Safety Analysis**: Prioritizes patient safety with evidence-based warnings

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **PostgreSQL**: Production database with ACID compliance
- **pgvector**: Vector similarity search extension
- **SQLAlchemy**: Python SQL toolkit and ORM
- **OpenAI API**: GPT-4O-mini for classification, text-embedding-3-small for vectors

### Frontend  
- **Next.js**: React framework with server-side rendering
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework

### AI/ML
- **RAG Pipeline**: Custom retrieval-augmented generation
- **Vector Embeddings**: 512-dimensional semantic search
- **Query Classification**: Intent detection and source routing
- **Content Analysis**: Severity detection from text patterns

### DevOps & Tools
- **UV Package Manager**: Fast Python dependency management
- **NPM**: Node.js package management
- **Git**: Version control
- **Environment Management**: Configurable deployment settings

## 📊 Database Schema

### Core Medical Data (1.7M+ records)
| Table | Records | Description |
|-------|---------|-------------|
| `drug_relations` | 1,443,729 | Knowledge graph relationships (DrugBank, MONDO) |
| `drug_dosage` | 265,376 | Product formulations and dosage information |
| `drug_metadata` | 17,430 | Basic drug information and classifications |
| `food_interactions` | 17,430 | Drug-food interaction warnings |
| `drug_product_stages` | 17,423 | Market availability and development stages |
| `product_stage_descriptions` | 7 | Stage definitions and descriptions |

### System Tables
| Table | Description |
|-------|-------------|
| `vector_embeddings` | 1,761,395 semantic search vectors (512D) |
| `chat_sessions` | User conversation sessions |
| `chat_messages` | Chat history and metadata |

---

## 🚀 Complete Setup Guide

Follow these steps chronologically for a successful installation:

### Prerequisites

1. **System Requirements**
   - Python 3.8+ (recommended: 3.11)
   - Node.js 18+ and NPM
   - PostgreSQL 14+ with pgvector extension
   - 8GB+ RAM (for embedding generation)
   - 10GB+ disk space

2. **Required Services**
   - OpenAI API account and API key
   - PostgreSQL server (local or remote)

### Step 1: Clone and Navigate

```bash
git clone git@github.com:prototypeanugrah/medical-rag-agent.git
cd medical-rag-agent
```

### Step 2: PostgreSQL Server Setup

#### Option A: macOS (Homebrew)
```bash
# Install PostgreSQL and pgvector
brew install postgresql pgvector

# Start PostgreSQL service
brew services start postgresql

# Create database and user
createdb medical_rag_pg
psql medical_rag_pg -c "CREATE EXTENSION vector;"
```

#### Option B: Linux (Ubuntu/Debian)
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector (requires compilation)
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres createdb medical_rag_pg
sudo -u postgres psql medical_rag_pg -c "CREATE EXTENSION vector;"
```

### Step 3: Environment Configuration

```bash
# Create .env file
cat > .env << EOF
# Database Configuration
DATABASE_URL="postgresql://$(whoami)@localhost:5432/medical_rag_pg"

# OpenAI Configuration
OPENAI_API_KEY="your_openai_api_key_here"
EMBEDDING_PROVIDER="openai"
OPENAI_EMBED_MODEL="text-embedding-3-small"  
EMBEDDING_DIMENSIONS="512"

# Application Configuration
PORT=8000
NODE_ENV=development
EOF

# Make sure to replace "your_openai_api_key_here" with your actual API key
```

### Step 4: Package Management with UV

This project uses **UV** for fast Python dependency management:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

# Create and activate virtual env
uv venv
source .venv/bin/activate

# Install Python dependencies with UV
uv sync

# Verify UV installation
uv --version
```

**Why UV?**
- **10-100x faster** than pip for dependency resolution
- **Lockfile management** for reproducible builds  
- **Virtual environment** handling built-in
- **Compatible** with standard requirements.txt

### Step 5: Node.js Dependencies

```bash
# Install Node.js dependencies
npm install

# Verify installation
npm list --depth=0
```

### Step 6: Database Initialization

```bash
# Test database connection
npm run pg:stats

# Initialize PostgreSQL with tables and indexes
npm run pg:setup

# Expected output:
# ✅ PostgreSQL: Connected
# ✅ pgvector: version 0.8.1  
# ✅ All tables created successfully
# ✅ Performance indexes created
```

### Step 7: Data Ingestion

⚠️ **Important**: Medical data files are **NOT included** in this repository due to size and licensing constraints.

**Option 1: Download from Google Drive (Recommended)**
```bash
# Show download instructions and open Google Drive folder
npm run data:info
npm run data:open

# [Google Drive Link](https://drive.google.com/drive/folders/17xzfvf0njVT7k9TDF6m2mCtDM4vMmV4F?usp=sharing)
# Download all files (~240MB total) to the project root directory
```

**Option 2: Development/Testing Sample Data**
```bash
# Generate sample data for testing (no download required)
npm run data:sample
```

**After obtaining data files (any option)**:
```bash
# Verify all files are in place
npm run data:check

# Add all medical data to PostgreSQL (takes 5-10 minutes)
npm run pg:add

# Monitor progress - you should see:
# ✅ Drug metadata: 17,430 records
# ✅ Drug relations: 1,443,729 records  
# ✅ Food interactions: 17,430 records
# ✅ Drug dosage: 265,376 records
# ✅ Product stages: 17,423 records
# ✅ Stage descriptions: 7 records
```

**📋 Quick Setup Reference**:
```bash
npm run data:guide    # Show complete setup guide
npm run data:info     # Show download instructions  
npm run data:open     # Open Google Drive in browser
npm run data:check    # Verify downloaded files
```

### Step 8: Vector Embeddings Generation

⚠️ **Important**: This step requires OpenAI API access and will incur costs (~$1.76)

```bash
# Generate embeddings (takes 4-6 hours, ~$1.76 cost)
npm run pg:embeddings

# The system will show:
# 💰 Estimated cost: $1.76 USD
# ⏱️ Estimated time: 29.4 hours (actual: ~2 hours)
# Proceed with embedding generation? (yes/no): yes
```

**Complete Setup Commands**
```bash
npm run pg:setup       # Database setup
npm run pg:add         # Add all data  
npm run pg:embeddings  # Generate embeddings
npm run pg:validate    # Validate setup
```

### Step 9: Validation

```bash
# Run comprehensive validation
npm run pg:validate

# Should show all tests passing:
# ✅ Database Connectivity: PASSED
# ✅ Table Structure: PASSED  
# ✅ Data Integrity: PASSED
# ✅ Embedding Functionality: PASSED
# ✅ RAG Pipeline: PASSED
# ✅ Performance Benchmarks: PASSED
# ✅ Validation Report: PASSED
```

### Step 10: Launch Application

```bash
# Terminal 1: Start FastAPI backend
python start_backend.py
# Backend running on: http://localhost:8000

# Terminal 2: Start Next.js frontend  
npm run dev
# Frontend running on: http://localhost:3000
```

---

## 🎮 Using the System

### Web Interface

1. **Visit**: `http://localhost:3000`
2. **Ask medical questions** like:
   - *"What are the interactions between warfarin and common foods?"*
   - *"Tell me about lisinopril dosage forms and any withdrawal history"*
   - *"I'm taking metformin and my doctor prescribed ibuprofen - what should I know?"*

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Chat API
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What foods should I avoid with warfarin?",
    "currentMedications": ["warfarin"],
    "drugs": []
  }'
```

#### Database Statistics  
```bash
curl http://localhost:8000/api/setup/
```

---

## 🛠️ Development Commands

### Database Management
```bash
# View database statistics
npm run pg:stats

# Clear specific table (use direct UV command)
uv run scripts/pg.py clear drug_relations

# Re-add specific data (use direct UV command)
uv run scripts/pg.py add drug_relations

# Complete reset (nuclear option)  
npm run pg:clear
npm run pg:add
npm run pg:embeddings
```

### Development Utilities
```bash
# Run backend in development mode
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Run frontend in development mode
npm run dev

# TypeScript type checking
npm run type-check

# Build for production
npm run build
npm start
```

### Testing & Validation
```bash
# Quick validation
npm run pg:validate

# Test specific functionality (use direct UV commands)
uv run scripts/pg.py stats drug_relations
uv run scripts/pg.py stats vector_embeddings
```

---

## ⚡ Performance Optimization

### Database Tuning
```bash
# Check index usage
psql $DATABASE_URL -c "
SELECT indexname, idx_scan, idx_tup_read 
FROM pg_stat_user_indexes 
WHERE relname = 'vector_embeddings';"

# Optimize for vector operations  
psql $DATABASE_URL -c "SET maintenance_work_mem = '512MB';"
```

### Memory Management
- **Minimum RAM**: 4GB for basic operation
- **Recommended RAM**: 8GB+ for optimal performance
- **Vector Index**: ~1GB memory usage
- **Embedding Generation**: Requires ~2GB during processing

### Expected Performance
- **Vector similarity search**: <2 seconds for 1.7M records
- **AI query classification**: ~500ms  
- **Database queries**: <100ms for most operations
- **Concurrent users**: 50+ simultaneous queries supported

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check if PostgreSQL is running
brew services list | grep postgres  # macOS
sudo systemctl status postgresql     # Linux

# Test connection
psql $DATABASE_URL -c "SELECT version();"
```

#### 2. pgvector Extension Missing
```bash
# Install pgvector
brew install pgvector  # macOS
# See PostgreSQL setup section for Linux

# Enable in database
psql $DATABASE_URL -c "CREATE EXTENSION vector;"
```

#### 3. OpenAI API Issues
```bash
# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Check rate limits in embedding generation output
```

#### 4. Memory Issues During Embedding Generation
```bash
# Monitor memory usage
htop  # Linux/macOS
# Reduce batch size if needed (edit backend/lib/embeddings.py)
```

#### 5. Frontend Build Issues
```bash
# Clear Next.js cache
rm -rf .next
npm run build

# Check Node.js version
node --version  # Should be 18+
```

### Performance Issues
```bash
# Check slow queries
psql $DATABASE_URL -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# Rebuild indexes if needed
npm run pg:setup  # Recreates indexes
```

---

## 📋 Data Sources & Attribution

### Medical Data Sources
- **DrugBank**: Comprehensive drug database
- **MONDO**: Medical ontology for diseases  
- **Medical Ontologies**: Structured medical terminology
- **FDA Drug Labels**: Official drug information
- **Clinical Guidelines**: Evidence-based medical protocols

### Data Processing
- **Total Records**: 1,761,395 processed entries
- **Vector Embeddings**: 512-dimensional using OpenAI text-embedding-3-small
- **Update Frequency**: Manual updates with new medical data releases
- **Quality Assurance**: Cross-validated against multiple medical sources

---

## 🏥 Medical Disclaimer

**⚠️ IMPORTANT MEDICAL DISCLAIMER**

This system is designed for **informational and educational purposes only** and should **not replace professional medical advice, diagnosis, or treatment**. 

**Always consult with qualified healthcare providers for:**
- Medical decisions and treatment plans
- Drug interaction assessments  
- Dosage modifications
- Side effect evaluation
- Emergency medical situations

**The system provides information to help users:**
- Understand their medications better
- Prepare informed questions for healthcare providers
- Make educated decisions about their healthcare
- Recognize potential concerns to discuss with doctors

**Users are responsible for:**
- Verifying information with healthcare professionals
- Not using this system for emergency medical decisions
- Understanding that AI-generated content may contain errors
- Seeking immediate medical attention for serious symptoms

---

## 🤝 Contributing

This is a portfolio project demonstrating advanced RAG system architecture, PostgreSQL optimization, and AI integration for medical applications. The system showcases production-ready code with comprehensive testing, error handling, and documentation.

### Key Technical Achievements
- **Scalable Architecture**: Handles 1.7M+ records efficiently
- **Intelligent Routing**: AI-powered query classification and source selection  
- **Production Database**: PostgreSQL with pgvector optimization
- **Safety-First Design**: Content-based prioritization and withdrawal detection
- **User-Focused UX**: Designed for informed medical decision makers

---

## 📄 License

This project is for educational and portfolio purposes. Medical data sources have their own licensing terms which must be respected when using this system.

---

**Built with ❤️ for Informed Medication Decision Makers**
