from pathlib import Path
from typing import List, Dict, Any
import lancedb

# Configuration
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def list_clusters(notes_table) -> List[Dict[str, Any]]:
    """List all clusters in the database."""
    all_chunks = notes_table.to_pandas().to_dict('records')
    
    # Get unique clusters
    clusters = {}
    for chunk in all_chunks:
        cluster_id = chunk.get('cluster_id', '-1')
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                'cluster_id': cluster_id,
                'cluster_label': chunk.get('cluster_label', 'Unknown'),
                'notes': set()
            }
        clusters[cluster_id]['notes'].add(f"{chunk['title']}|||{chunk['creation_date']}")
    
    # Convert to list
    result = []
    for cluster_id, data in clusters.items():
        result.append({
            'cluster_id': cluster_id,
            'cluster_label': data['cluster_label'],
            'count': len(data['notes'])
        })
    
    result.sort(key=lambda x: x['count'], reverse=True)
    return result


def get_notes_in_cluster(notes_table, cluster_id: str) -> List[Dict[str, str]]:
    """Get all unique notes in a cluster."""
    all_chunks = notes_table.to_pandas().to_dict('records')
    
    notes = {}
    for chunk in all_chunks:
        if chunk.get('cluster_id') == cluster_id:
            key = f"{chunk['title']}|||{chunk['creation_date']}"
            if key not in notes:
                notes[key] = {
                    'title': chunk['title'],
                    'creation_date': chunk['creation_date'],
                    'cluster_confidence': chunk.get('cluster_confidence')
                }
    
    return list(notes.values())
