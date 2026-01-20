#!/usr/bin/env python3
"""
Script to display existing clusters and their notes.
"""

import lancedb
from scripts.cluster_utils import (
    list_clusters,
    get_notes_in_cluster,
    DB_PATH,
    TABLE_NAME
)

def display_clusters():
    """
    Display all clusters and their notes.
    """
    print(f"Connecting to database at {DB_PATH}...")
    try:
        db = lancedb.connect(str(DB_PATH))
        notes_table = db.open_table(TABLE_NAME)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    print("\nFetching clusters...")
    clusters = list_clusters(notes_table)
    
    # Filter out the outlier cluster for the main list, or keep it?
    # The user asked for "list off all the cluster titles".
    # Usually outliers are cluster_id '-1'.
    
    real_clusters = [c for c in clusters if c['cluster_id'] != '-1']
    # Sort by cluster ID numerically
    real_clusters.sort(key=lambda x: int(x['cluster_id']))
    
    outliers = [c for c in clusters if c['cluster_id'] == '-1']

    print(f"\nFound {len(real_clusters)} clusters (plus {len(outliers)} outlier group).")

    # 1. List off all the cluster titles, and how many notes are in said clusters
    print("\n" + "="*50)
    print("CLUSTER SUMMARY")
    print("="*50)
    for cluster in real_clusters:
        print(f"Cluster {cluster['cluster_id']}: {cluster['cluster_label']} ({cluster['count']} notes)")
    
    if outliers:
        print(f"Outliers: {outliers[0]['count']} notes")

    # 2. List off all the titles of the notes in a given cluster
    print("\n" + "="*50)
    print("DETAILED CLUSTER CONTENTS")
    print("="*50)
    
    # Sort clusters by ID for consistent output, or by count? 
    # The list_clusters function sorts by count descending.
    
    for cluster in real_clusters:
        cluster_id = cluster['cluster_id']
        label = cluster['cluster_label']
        count = cluster['count']
        
        print(f"\n📌 Cluster {cluster_id}: \"{label}\" ({count} notes)")
        
        notes = get_notes_in_cluster(notes_table, cluster_id)
        for i, note in enumerate(notes, 1):
            print(f"   {i}. {note['title']}")

    if outliers:
        print(f"\n📌 Outliers ({outliers[0]['count']} notes)")
        notes = get_notes_in_cluster(notes_table, '-1')
        for i, note in enumerate(notes, 1):
            print(f"   {i}. {note['title']}")

if __name__ == "__main__":
    display_clusters()
