# MCP Apple Notes System

This system integrates Apple Notes with the Model Context Protocol (MCP), providing advanced semantic analysis, clustering, and a visual frontend.

## 🚀 Quick Start

We provide convenience scripts in the root directory to manage the application lifecycle.

### 1. Standard Pipeline
To process **all** new or modified notes and update the analysis:

```bash
./run_pipeline.sh
```
This script runs the full incremental update process:
- Fetches all new/modified notes from Apple Notes.
- Updates the vector database.
- Runs the BERTopic analysis to update clusters.

### 2. Limited Pipeline (Batched)
To process a specific number of notes:

```bash
./run_pipeline_limit.sh <number_of_notes>
# Example: ./run_pipeline_limit.sh 50
```
This script iterates through notes and stops when the limit is reached. It is particularly useful for fetching older notes that may not have been covered in previous runs.

### 3. Start Application
To start the full application (Backend API + Frontend UI):

```bash
./start.sh
```
This script will:
- Start the Python FastAPI backend server (`backend/scripts/server.py`) in the background.
- Start the Electron/React frontend (`frontend/`).
- Automatically shut down the backend when you exit the frontend.

---

## 📂 System Components

This repository is divided into three main components. Please refer to their respective READMEs for detailed documentation.

### 1. [Server (MCP)](./server/README.md)
 Located in `/server`.
 
 This is the core Model Context Protocol server. It handles:
 - Fetching notes from Apple Notes.
 - Creating embeddings and vector storage.
 - `cli.ts` for command-line management.

### 2. [Backend Analysis](./backend/README.md)
 Located in `/backend`.
 
 A Python-based analysis engine that provides:
 - **Advanced Search**: Hybrid semantic/text search with ordered proximity boosting for multi-word queries.
 - **Clustering**: BERTopic modeling to find themes in your notes.
 - **API**: A FastAPI server (`server.py`) that serves data to the frontend.

### 3. [Frontend](./frontend/README.md)
 Located in `/frontend`.
 
 An Electron + React application that visualizes your note data.
 - **Cluster Viz**: Interactive 3D/2D visualization of note clusters (UMAP).
 - **Search UI**: User-friendly interface to query your notes using the backend API.

---

## 🔍 Search Behavior

### Hybrid Retrieval
- **Vector search** captures semantic similarity.
- **FTS (Full-Text Search)** captures exact keyword matches.
- Results from both strategies are merged and ranked by relevance score.

### Multi-Word Keyword Proximity Boosting

For queries with multiple meaningful terms (e.g., `docker backup`), the search system applies a **proximity bonus** on top of the base hybrid score. This improves ranking for multi-word queries without breaking single-word, empty, or semantic-only results.

#### How it works

1. **Broad keyword retrieval is preserved.** A query like `docker backup` still returns notes matching *either* term, not just exact phrase matches.

2. **All-terms coverage bonus.** If all distinct query terms appear somewhere in the chunk/parent note (regardless of order), a small bonus is added:
   - **+0.08** to the hybrid score

3. **Ordered proximity bonus.** If the terms appear in the same order as the user typed them, an additional bonus is applied. This bonus decays with the number of extra words between matching terms:
   ```
   ordered_proximity_bonus = ORDERED_MAX / (1 + extra_tokens)
   where: extra_tokens = best_span_length - distinct_term_count
   ```
   - Adjacent terms (`docker backup` → "docker backup"): **+0.12** (maximum)
   - One word between (`docker postgres backup`): **+0.06**
   - Many words between: smaller bonus, approaching zero

4. **Exact phrase bonus.** If all terms appear consecutively in order (zero extra tokens), a flat bonus is added:
   - **+0.15** for exact adjacent match

5. **Total bonus is capped.** The sum of all bonuses is bounded by `TOTAL_BONUS_CAP = 0.30` to prevent proximity from overwhelming semantic signals.

#### Quoted queries

Wrapping terms in double quotes (e.g., `"docker backup"`) signals explicit phrase intent. Quoted queries always receive the maximum exact phrase bonus (+0.15) regardless of whether the terms are actually adjacent in the text, in addition to coverage and ordered proximity bonuses if applicable.

#### Scoring constants

| Component | Max Value | Description |
|---|---|---|
| `ALL_TERMS_COVERAGE_MAX` | 0.08 | All distinct query terms present |
| `ORDERED_MAX` | 0.12 | Decaying bonus for ordered proximity (adjacent = max) |
| `EXACT_PHRASE_BONUS` | 0.15 | Flat bonus for consecutive ordered terms |
| `TOTAL_BONUS_CAP` | 0.30 | Absolute maximum proximity bonus |

#### Expected ranking order

For query `docker backup`:

| Rank | Chunk | Why |
|---|---|---|
| 1 | "Docker backup procedure" | Highest lexical/proximity boost (adjacent) |
| 2 | "Docker PostgreSQL backup procedure" | Strong ordered proximity boost |
| 3 | "Docker database migration and backup procedure" | Positive ordered proximity, lower |
| 4 | "Backup Docker volumes before deployment" | Term coverage only; no ordered bonus |
| 5 | "Docker setup notes ... many words ... backup checklist" | Small ordered proximity boost |
| 6 | Notes matching only `docker` or only `backup` | Partial keyword match |

#### Edge cases

- **Single-word queries**: No proximity bonus applied (semantic + FTS only).
- **Empty/whitespace queries**: Returns all notes with no proximity boost.
- **Stop words filtered**: Queries like "the and or" produce zero meaningful terms → score = 0.
- **Reverse order** (`backup docker` vs query `docker backup`): Coverage bonus only, NO ordered proximity bonus.
- **Repeated query terms** (`backup docker backup`): Deduplicated to unique terms before scoring.

---

## 🔧 Configuration

All configuration is managed via environment variables and scripts in the repository root.