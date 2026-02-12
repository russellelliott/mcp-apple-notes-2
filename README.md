# MCP Apple Notes System

This system integrates Apple Notes with the Model Context Protocol (MCP), providing advanced semantic analysis, clustering, and a visual frontend.

## 🚀 Quick Start

We provide convenience scripts in the root directory to manage the application lifecycle.

### 1. Data Pipeline
To update the incremental note data and run the BERTopic analysis:

```bash
./run_pipeline.sh
```
This script will:
- Run the MCP server CLI (`server/cli.ts`) to fetch incremental updates.
- Run the Python analysis (`backend/analysis/run_bertopic.py`) to cluster and verify data.

### 2. Start Application
To start the full application (Backend API + Frontend UI):

```bash
./start_frontend.sh
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
 - **Advanced Search**: Hybrid semantic/text search.
 - **Clustering**: BERTopic modeling to find themes in your notes.
 - **API**: A FastAPI server (`server.py`) that serves data to the frontend.

### 3. [Frontend](./frontend/README.md)
 Located in `/frontend`.
 
 An Electron + React application that visualizes your note data.
 - **Cluster Viz**: Interactive 3D/2D visualization of note clusters (UMAP).
 - **Search UI**: User-friendly interface to query your notes using the backend API.
