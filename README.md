
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

**Hybrid semantic and text-based search** combining vector embeddings and full-text search for comprehensive note discovery.

#### Search Strategies

The system uses **three complementary search strategies** that work together:

1. **🎯 Vector Semantic Search**
   - Uses sentence embeddings to find conceptually related content
   - **Best for:** themes, concepts, related ideas, synonyms
   - **Example:** "machine learning" finds notes about AI, neural networks, data science
   - **Technology:** sentence-transformers/all-MiniLM-L6-v2 embeddings

2. **📝 Full-Text Search (FTS)**
   - Uses inverted indexes for exact text matching
   - **Best for:** specific terms, names, codes, IDs, acronyms
   - **Example:** "CRWN102" finds all notes containing that exact course code
   - **Technology:** LanceDB inverted indexes on `title`, `content`, and `chunk_content`

3. **🔄 Database-Level Filtering**
   - Fallback text scanning when other methods fail
   - **Best for:** comprehensive coverage and backup search
   - **Technology:** Pandas-based text filtering

#### Usage

```bash
# Basic search (uses all strategies)
python scripts/search_notes.py "restaurant data filtering"

# Limit results
python scripts/search_notes.py "machine learning" --limit 10

# Course codes and specific terms (primarily FTS)
python scripts/search_notes.py "CRWN102"

# Conceptual queries (benefits from semantic search)
python scripts/search_notes.py "project planning and organization"
```

#### Search Results

Each result shows:
- **Title** and creation/modification dates
- **Relevance score** (0-100, higher = more relevant)
- **Search source** (`vector_semantic`, `fts`, or `text_match`)
- **Chunk information** (which part of the note matched)
- **Preview** of matching content

#### Example Output

```
🔎 Searching for: 'restaurant data filtering'

1️⃣ Vector semantic search on chunks...
🎯 Found 10 relevant chunks
📋 Unique notes from vector search: 2

2️⃣ Full-text search on chunks...
📝 FTS results: 10 chunks

📊 Final results: 6 notes (from 6 total matches)
  1. "CRWN102 Data filtering" (score: 70.0, source: fts)
  2. "Restaurant Analytics Project" (score: 45.3, source: vector_semantic)
  3. "Data Processing Notes" (score: 42.1, source: vector_semantic)
```

#### When to Use Each Strategy

| Query Type | Best Strategy | Example |
|------------|---------------|---------|
| **Specific terms/codes** | FTS | "CRWN102", "iPhone 14", "Project Alpha" |
| **Concepts/themes** | Vector Semantic | "machine learning approaches", "project planning" |
| **Mixed queries** | Hybrid (both) | "CRWN102 restaurant data" |
| **Exploratory** | Vector Semantic | "innovative solutions", "creative ideas" |

#### Setup Requirements

**⚠️ Important:** FTS requires inverted indexes to be created first:

```bash
# Create indexes (run once)
python scripts/create_inverted_index.py

# Then search works fully
python scripts/search_notes.py "your query"
```

**Without indexes:** Only semantic search works; FTS returns 0 results.
**With indexes:** Full hybrid search with both semantic and exact matching.

#### **Index Maintenance**

**When you add new notes**, the inverted indexes persist but new data won't be searchable via FTS until indexed:

```bash
# After adding new notes, update indexes:
python -c "
from main import NotesDatabase
db = NotesDatabase()
table = db.get_or_create_table()
table.optimize()  # Updates existing indexes with new data
print('✅ Indexes updated with new data')
"

# Or recreate indexes entirely:
python scripts/create_inverted_index.py
```

**Best practice:** Run `table.optimize()` after adding new notes to ensure full search coverage.

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

### Search Issues

**No search results / FTS returning 0 results:**
```bash
# 1. Check if inverted indexes exist
python scripts/create_inverted_index.py

# 2. Verify data exists
python scripts/check_notes.py

# 3. Test with different query types
python scripts/search_notes.py "machine learning"  # conceptual
python scripts/search_notes.py "specific term"     # exact match
```

**Only getting FTS results (no semantic results):**
- This is normal for specific terms like course codes ("CRWN102")
- Try conceptual queries like "data analysis" or "project planning"
- Check similarity threshold (lowered to 0.01 for better coverage)

**Vector search showing 0 unique notes:**
- Indicates similarity threshold too strict or calculation error
- Fixed in recent updates with proper L2 distance conversion
- Should now show both `vector_semantic` and `fts` sources

### Database Issues

**Database errors:**
```bash
python debug_db.py             # Diagnose issues
python fix_db.py               # Attempt repair
```

**Wrong database path:**
- Scripts use `~/.mcp-apple-notes/data` (not `~/.mcp-apple-notes-2/data`)
- Ensure consistency across all scripts

### Clustering Issues

**Poor clustering:**
- Try `--min-size=5` for more conservative clustering
- Try `--min-size=1` for maximum coverage
- Check note content quality and diversity

### Dependencies

**Missing packages:**
- Ensure all packages installed: `lancedb`, `scikit-learn`, `numpy`, `pandas`, `sentence-transformers`
- Verify embedding model initialization
- Check GPU/MPS availability for faster processing