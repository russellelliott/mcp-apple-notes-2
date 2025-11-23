
# MCP Apple Notes - Advanced Analysis & Search

This repository provides advanced semantic analysis and clustering for Apple Notes data. It's optimized for high-performance Python processing and includes sophisticated clustering algorithms.

## Overview

This system provides:
- **Semantic clustering** of notes using two-pass HDBSCAN with dynamic quality scoring
- **Multi-strategy search** combining vector search, FTS, and database queries
- **Database management** tools for diagnostics and maintenance
- **2x faster processing** than TypeScript implementations (5s vs 11s for 200 notes)

## Prerequisites

- Python 3.8+
- Apple Notes data (use [mcp-apple-notes](https://github.com/russellelliott/mcp-apple-notes) for fetching)
- Virtual environment recommended ([setup guide](https://chatgpt.com/c/691d4f5e-777c-832d-8adf-5564a7896202))

## Core Scripts

### 🔍 Search Notes (`scripts/search_notes.py`)

Advanced multi-strategy semantic search with relevance scoring.

**Features:**
- **Vector semantic search** using embeddings
- **Full-text search (FTS)** on chunk content with re-scoring
- **Database-level exact phrase matching**
- **Combined relevance scoring** from multiple strategies
- **Chunk-aware results** with preview and context

**Usage:**
```bash
python scripts/search_notes.py "machine learning"
python scripts/search_notes.py "project ideas" --limit 10
```

**Search Strategies:**
1. **Vector Search**: Semantic similarity using embeddings
2. **FTS Search**: Full-text search with optional embedding re-scoring  
3. **Exact Phrase**: Database LIKE queries for precise matches

**Output:**
- Title, creation/modification dates
- Relevance score (0-1.0)
- Search strategy used
- Best matching chunk preview
- Chunk index and total chunks

### 🎯 Two-Pass Clustering (`scripts/two_pass_clustering.py`)

Sophisticated semantic clustering with dynamic quality assessment.

**Algorithm:**
1. **Pass 1**: Initial HDBSCAN clustering with configurable minimum cluster size
2. **Pass 2**: Semantic quality evaluation of outliers using cosine similarity
3. **Dynamic reassignment** based on average quality threshold
4. **Semantic preservation** - poor fits remain as outliers

**Features:**
- **Data-driven thresholds** (no hard-coded values)
- **Semantic quality scoring** (0-1 scale)
- **Variable cluster shapes** via HDBSCAN
- **Outlier reassignment** with quality gates
- **Full database persistence**

**Usage:**
```bash
# Default (balanced)
python scripts/two_pass_clustering.py

# Conservative (fewer, stronger clusters)
python scripts/two_pass_clustering.py --min-size=5

# High-precision (only very strong clusters)
python scripts/two_pass_clustering.py --min-size=10
```

**Configuration Guide:**
- `--min-size=2` (default): Balanced semantic quality
- `--min-size=5`: More robust initial clusters, less pollution
- `--min-size=10`: Only strongest clusters, more outliers

**Quality Metrics:**
- Dynamic threshold based on average semantic fit
- Cosine similarity scoring for reassignment decisions
- Preserves semantic integrity over spatial proximity

### 📊 Check Notes (`scripts/check_notes.py`)

Database diagnostics and overview tool.

**Information Provided:**
- **Cache status** with last sync timestamp
- **Database statistics** (total chunks, unique notes)
- **Chunks per note** distribution (top 20)
- **Schema information** and structure
- **Sample data** (5 newest chunks by insertion order)

**Usage:**
```bash
python scripts/check_notes.py
```

**Sample Output:**
```
📋 Cache Info:
   Last sync: 2024-11-22 10:30:15

📊 Database Stats:
   Total chunks: 1,247
   Unique notes: 203

📊 Chunks per Note (top 20):
     15 chunks  →  Machine Learning Research Notes
      8 chunks  →  Project Planning Documentation
      6 chunks  →  Meeting Notes - Q4 Review
   ...
```

## Database Management

### 🔧 Fix Database (`fix_db.py`)

Repairs database integrity by recreating tables from backup data.

**Operations:**
- Recreates `notes` table from `notes_new` backup
- Cleans up temporary test tables
- Verifies data integrity post-repair
- Provides detailed success/failure reporting

**Usage:**
```bash
python fix_db.py
```

**Use When:**
- Database corruption detected
- Table structure issues
- After failed migrations
- Inconsistent data states

### 🐛 Debug Database (`debug_db.py`)

Comprehensive database inspection and diagnostics.

**Diagnostics:**
- Lists all available tables
- Shows record counts per table
- Identifies table access issues
- Displays database file paths
- Error reporting for problematic tables

**Usage:**
```bash
python debug_db.py
```

**Output Example:**
```
Available tables:
  - notes
    Records: 1,247
  - notes_new
    Records: 1,247
  - test-cluster-20241122
    Error reading table: Schema mismatch

Database path: /Users/user/.mcp-apple-notes/data
```

## Data Pipeline

1. **Fetch Notes** → Use [mcp-apple-notes](https://github.com/russellelliott/mcp-apple-notes)
2. **Process & Embed** → `scripts/fetch_and_chunk_notes.py` (creates embeddings)
3. **Analyze** → `scripts/check_notes.py` (verify data)
4. **Cluster** → `scripts/two_pass_clustering.py` (semantic grouping)
5. **Search** → `scripts/search_notes.py` (query & discover)

## Performance

- **2x faster clustering** vs TypeScript (5s vs 11s for 200 notes)
- **Optimized embedding aggregation** (average of chunks per note)
- **Efficient vector operations** using NumPy
- **LanceDB integration** for high-performance vector search

## Database Schema

The system uses LanceDB with the following key fields:
- `title`: Note title
- `content`: Full note content
- `creation_date`: Note creation timestamp
- `modification_date`: Last modification timestamp
- `chunk_index`: Chunk position within note
- `total_chunks`: Total chunks for the note
- `chunk_content`: Content of this specific chunk
- `vector`: Embedding vector (384-dimensional)
- `cluster_id`: Assigned cluster (-1 for outliers)

## Troubleshooting

**No search results:**
```bash
python scripts/check_notes.py  # Verify data exists
```

**Database errors:**
```bash
python debug_db.py             # Diagnose issues
python fix_db.py               # Attempt repair
```

**Poor clustering:**
- Try `--min-size=5` for more conservative clustering
- Try `--min-size=1` for maximum coverage
- Check note content quality and diversity

**Dependencies:**
- Ensure all packages installed: `lancedb`, `scikit-learn`, `numpy`, `pandas`
- Verify embedding model initialization