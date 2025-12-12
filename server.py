import uvicorn
from fastapi import FastAPI, HTTPException
import lancedb
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

from scripts.two_pass_clustering import (
    list_clusters,
    get_notes_in_cluster,
    DB_PATH,
    TABLE_NAME
)

app = FastAPI(title="Apple Notes Clusters API")

def get_db_connection():
    try:
        db = lancedb.connect(str(DB_PATH))
        return db.open_table(TABLE_NAME)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.get("/clusters")
async def get_clusters_and_notes():
    """
    Get all clusters and their associated notes.
    """
    try:
        table = get_db_connection()
        
        # Get all clusters
        clusters_summary = list_clusters(table)
        
        results = []
        
        # Sort clusters by ID numerically for consistent output
        # Handle potential non-integer IDs gracefully, though they should be ints or '-1'
        def sort_key(c):
            try:
                return int(c['cluster_id'])
            except ValueError:
                return float('inf')

        clusters_summary.sort(key=sort_key)
        
        for cluster in clusters_summary:
            cluster_id = cluster['cluster_id']
            
            # Get notes for this cluster
            notes = get_notes_in_cluster(table, cluster_id)
            
            cluster_data = {
                "cluster_id": cluster_id,
                "label": cluster['cluster_label'],
                "note_count": cluster['count'],
                "notes": notes
            }
            results.append(cluster_data)
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
