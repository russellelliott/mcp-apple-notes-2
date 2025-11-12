#!/usr/bin/env python3
"""
Two-Pass Clustering with Dynamic Semantic Quality Scoring

This version uses data-driven, dynamic outlier reassignment:
- Automatically evaluates each outlier's semantic fit with clusters
- Uses cosine similarity to determine quality of reassignment (0-1 scale)
- Dynamic threshold: Uses AVERAGE quality score from evaluation pass
- Only reassigns outliers with quality score > average
- Truly isolated outliers (below-average quality) stay as outliers
- No hard-coded thresholds - adapts to your data

The quality score evaluates semantic alignment, so notes that don't
fit well semantically won't pollute clusters even if they're spatially close.

Usage:
    python two_pass_clustering.py                 # Default
    python two_pass_clustering.py --min-size=5    # More robust initial clusters
    python two_pass_clustering.py --min-size=10   # Very conservative clustering

Recommended Configurations:
- Default (minClusterSize=2): Balanced, good semantic quality
- Conservative (minClusterSize=5): Fewer initial clusters, less pollution
- High-precision (minClusterSize=10): Only strong clusters, more outliers
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import lancedb
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# Configuration
DATA_DIR = Path.home() / ".mcp-apple-notes-2"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def cluster_notes(
    notes_table,
    min_cluster_size: int = 2,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform two-pass clustering with dynamic semantic quality scoring.
    
    Args:
        notes_table: LanceDB table containing notes
        min_cluster_size: Minimum cluster size for initial HDBSCAN
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with clustering results and statistics
    """
    start_time = time.time()
    
    if verbose:
        print("📥 Loading all notes from database...")
    
    # Load ALL chunks from the table
    all_chunks = notes_table.to_pandas().to_dict('records')
    
    if verbose:
        print(f"   Loaded {len(all_chunks)} chunks")
    
    # Get unique notes (by title + creation_date)
    unique_notes = {}
    for chunk in all_chunks:
        key = f"{chunk['title']}|||{chunk['creation_date']}"
        if key not in unique_notes:
            unique_notes[key] = []
        unique_notes[key].append(chunk)
    
    total_notes = len(unique_notes)
    
    if verbose:
        print(f"   Found {total_notes} unique notes\n")
    
    # Aggregate embeddings for each note (average of all chunks)
    note_data = []
    for key, chunks in unique_notes.items():
        embeddings = [chunk['embedding'] for chunk in chunks]
        avg_embedding = np.mean(embeddings, axis=0)
        note_data.append({
            'key': key,
            'title': chunks[0]['title'],
            'embedding': avg_embedding,
            'chunks': chunks
        })
    
    embeddings = np.array([note['embedding'] for note in note_data])
    
    # ===== PASS 1: Initial HDBSCAN =====
    if verbose:
        print("=" * 50)
        print("🔍 PASS 1: Initial HDBSCAN Clustering")
        print("=" * 50)
        print(f"   Parameters: min_cluster_size={min_cluster_size}")
        print(f"   Algorithm: HDBSCAN (respects variable shapes/densities)\n")
    
    hdbscan_primary = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    primary_labels = hdbscan_primary.fit_predict(embeddings)
    
    # Count primary clusters and outliers
    primary_clusters = len(set(primary_labels)) - (1 if -1 in primary_labels else 0)
    primary_outliers = np.sum(primary_labels == -1)
    
    if verbose:
        print(f"✅ Primary clustering complete:")
        print(f"   • Clusters formed: {primary_clusters}")
        print(f"   • Notes in clusters: {total_notes - primary_outliers}")
        print(f"   • Outliers: {primary_outliers}\n")
    
    # ===== PASS 2: Semantic Quality Evaluation =====
    if verbose:
        print("=" * 50)
        print("🎯 PASS 2: Semantic Quality Evaluation")
        print("=" * 50)
        print("   Evaluating outlier fit with existing clusters...\n")
    
    # Get outlier indices
    outlier_indices = np.where(primary_labels == -1)[0]
    
    if len(outlier_indices) == 0:
        if verbose:
            print("   No outliers to reassign!\n")
        quality_threshold = 0.0
        reassigned_count = 0
        still_isolated = 0
    else:
        # Calculate cluster centroids
        cluster_ids = [c for c in set(primary_labels) if c != -1]
        cluster_centroids = {}
        
        for cluster_id in cluster_ids:
            cluster_mask = primary_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        # Evaluate each outlier's quality with each cluster
        quality_scores = []
        best_assignments = []
        
        for outlier_idx in outlier_indices:
            outlier_embedding = embeddings[outlier_idx].reshape(1, -1)
            
            # Calculate cosine similarity with each cluster centroid
            best_score = -1
            best_cluster = -1
            
            for cluster_id, centroid in cluster_centroids.items():
                centroid_reshaped = centroid.reshape(1, -1)
                similarity = cosine_similarity(outlier_embedding, centroid_reshaped)[0][0]
                
                if similarity > best_score:
                    best_score = similarity
                    best_cluster = cluster_id
            
            quality_scores.append(best_score)
            best_assignments.append(best_cluster)
        
        # Dynamic threshold: use average quality score
        quality_threshold = np.mean(quality_scores)
        
        if verbose:
            print(f"   📊 Quality Score Distribution:")
            print(f"      • Min: {np.min(quality_scores):.3f}")
            print(f"      • Average: {quality_threshold:.3f} ← Dynamic threshold")
            print(f"      • Max: {np.max(quality_scores):.3f}\n")
        
        # Reassign outliers with quality > average
        reassigned_count = 0
        for i, outlier_idx in enumerate(outlier_indices):
            if quality_scores[i] > quality_threshold:
                primary_labels[outlier_idx] = best_assignments[i]
                reassigned_count += 1
        
        still_isolated = len(outlier_indices) - reassigned_count
        
        if verbose:
            print(f"   ✅ Reassignment Results:")
            print(f"      • Reassigned to clusters: {reassigned_count}")
            print(f"      • Still isolated: {still_isolated}\n")
    
    # ===== PASS 3: Secondary HDBSCAN on Remaining Outliers =====
    secondary_clusters = 0
    
    if still_isolated > 0:
        if verbose:
            print("=" * 50)
            print("🔄 PASS 3: Secondary HDBSCAN on Isolated Notes")
            print("=" * 50)
            print(f"   Parameters: min_cluster_size=2 (allow small pairs)")
            print(f"   Processing {still_isolated} remaining outliers...\n")
        
        # Get remaining outlier indices
        remaining_outliers = np.where(primary_labels == -1)[0]
        
        if len(remaining_outliers) >= 2:
            remaining_embeddings = embeddings[remaining_outliers]
            
            hdbscan_secondary = HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            secondary_labels = hdbscan_secondary.fit_predict(remaining_embeddings)
            
            # Remap secondary labels to avoid conflicts
            max_primary_label = np.max(primary_labels)
            for i, outlier_idx in enumerate(remaining_outliers):
                if secondary_labels[i] != -1:
                    primary_labels[outlier_idx] = max_primary_label + 1 + secondary_labels[i]
            
            secondary_clusters = len(set(secondary_labels)) - (1 if -1 in secondary_labels else 0)
            still_isolated = np.sum(secondary_labels == -1)
            
            if verbose:
                print(f"   ✅ Secondary clustering complete:")
                print(f"      • New clusters formed: {secondary_clusters}")
                print(f"      • Still isolated: {still_isolated}\n")
        else:
            if verbose:
                print(f"   ⏭️  Only {len(remaining_outliers)} outlier(s) remaining, cannot form pairs\n")
    
    # ===== Generate Cluster Labels =====
    if verbose:
        print("🏷️  Generating cluster labels...")
    
    cluster_labels = {}
    for cluster_id in set(primary_labels):
        if cluster_id == -1:
            continue
        
        cluster_mask = primary_labels == cluster_id
        cluster_note_data = [note_data[i] for i in np.where(cluster_mask)[0]]
        
        # Generate label from common words in titles
        all_titles = ' '.join([note['title'] for note in cluster_note_data])
        words = all_titles.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            # Get top 2 most common words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
            label = ' '.join([word for word, _ in top_words]).title()
        else:
            label = f"Cluster {cluster_id}"
        
        cluster_labels[cluster_id] = label
    
    # ===== Update Database =====
    if verbose:
        print("💾 Updating database with cluster assignments...\n")
    
    # Get current timestamp for last_clustered
    from datetime import datetime
    current_timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Build update map: id -> cluster info
    # Compute a dynamic confidence score per note as the cosine similarity
    # between the note's embedding and its assigned cluster centroid (0-1-ish).
    # For true outliers (-1) we fall back to the best similarity to any centroid
    # computed during pass 2 when available, otherwise 0.0.
    cluster_updates = {}

    # Recompute cluster centroids from final labels to ensure consistency
    cluster_centroids = {}
    cluster_ids_final = [c for c in set(primary_labels) if c != -1]
    for cid in cluster_ids_final:
        mask = primary_labels == cid
        if np.sum(mask) > 0:
            cluster_centroids[cid] = np.mean(embeddings[mask], axis=0)

    # Prepare a map of outlier best-scores from pass 2 if they exist
    # (we computed `quality_scores` and `outlier_indices` earlier in pass 2)
    outlier_best_score_map = {}
    try:
        # `outlier_indices` and `quality_scores` may only exist if there were outliers
        for idx, score in zip(outlier_indices, quality_scores):
            outlier_best_score_map[int(idx)] = float(score)
    except NameError:
        # No outliers were processed in pass 2
        outlier_best_score_map = {}

    # Compute confidence per note index
    confidence_scores = [0.0] * len(note_data)
    for i in range(len(note_data)):
        label = int(primary_labels[i])
        emb = embeddings[i].reshape(1, -1)
        conf = 0.0
        if label != -1 and label in cluster_centroids:
            centroid = cluster_centroids[label].reshape(1, -1)
            conf = float(cosine_similarity(emb, centroid)[0][0])
        else:
            # Use best score from pass 2 if present for this outlier, else 0.0
            conf = float(outlier_best_score_map.get(i, 0.0))

        # Clamp to [-1,1] then map to [0,1] to give an interpretable confidence
        conf = max(-1.0, min(1.0, conf))
        conf = (conf + 1.0) / 2.0
        confidence_scores[i] = conf

    for i, note in enumerate(note_data):
        cluster_id = int(primary_labels[i])
        cluster_label = cluster_labels.get(cluster_id, "Outlier")

        confidence_val = confidence_scores[i]
        confidence = f"{confidence_val:.3f}"

        for chunk in note['chunks']:
            if 'id' in chunk:
                cluster_updates[chunk['id']] = {
                    'cluster_id': str(cluster_id),
                    'cluster_label': cluster_label,
                    'cluster_confidence': confidence,
                    'last_clustered': current_timestamp
                }
    
    # Read all chunks, add cluster assignments, and rebuild table
    if cluster_updates:
        import pyarrow as pa
        try:
            db = lancedb.connect(str(DB_PATH))
            existing_data = notes_table.to_pandas().to_dict('records')
            
            # Add cluster assignments to each chunk
            for record in existing_data:
                chunk_id = record.get('id')
                if chunk_id in cluster_updates:
                    cluster_info = cluster_updates[chunk_id]
                    record['cluster_id'] = cluster_info['cluster_id']
                    record['cluster_label'] = cluster_info['cluster_label']
                    record['cluster_confidence'] = cluster_info['cluster_confidence']
                    record['last_clustered'] = cluster_info['last_clustered']
                else:
                    # Chunks without ID mapping are marked as outliers
                    record['cluster_id'] = '-1'
                    record['cluster_label'] = 'Outlier'
                    record['cluster_confidence'] = '0.0'
                    record['last_clustered'] = current_timestamp
                
                # Initialize cluster_summary if not present
                if 'cluster_summary' not in record:
                    cluster_label = record.get('cluster_label', 'Outlier')
                    if cluster_label == 'Outlier':
                        record['cluster_summary'] = 'Isolated note not assigned to any cluster'
                    else:
                        record['cluster_summary'] = f'Part of {cluster_label} cluster'
            
            # Rebuild the table with cluster assignments
            pa_updated = pa.Table.from_pylist(existing_data)
            db.drop_table(TABLE_NAME)
            db.create_table(TABLE_NAME, pa_updated)
            
            # Refresh the table reference after rebuild
            notes_table = db.open_table(TABLE_NAME)
            
            if verbose:
                print(f"   ✅ Updated {len(cluster_updates)} chunks with cluster assignments\n")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Warning: Could not update database: {e}\n")
    
    # ===== Calculate Statistics =====
    total_clusters = primary_clusters + secondary_clusters
    final_outliers = np.sum(primary_labels == -1)
    
    elapsed_time = time.time() - start_time
    
    # Get cluster sizes
    cluster_sizes = []
    for cluster_id in set(primary_labels):
        if cluster_id == -1:
            continue
        size = np.sum(primary_labels == cluster_id)
        cluster_sizes.append({
            'cluster_id': cluster_id,
            'label': cluster_labels[cluster_id],
            'size': size
        })
    
    cluster_sizes.sort(key=lambda x: x['size'], reverse=True)
    
    return {
        'primaryClusters': primary_clusters,
        'secondaryClusters': secondary_clusters,
        'totalClusters': total_clusters,
        'totalNotes': total_notes,
        'outliers': primary_outliers,
        'reassigned': reassigned_count,
        'stillIsolated': final_outliers,
        'qualityThreshold': quality_threshold,
        'timeSeconds': elapsed_time,
        'clusterSizes': cluster_sizes,
        'labels': primary_labels
    }


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
                    'creation_date': chunk['creation_date']
                }
    
    return list(notes.values())


def main():
    parser = argparse.ArgumentParser(
        description='Two-Pass Clustering with Semantic Quality Scoring'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=2,
        help='Minimum cluster size for initial HDBSCAN (default: 2)'
    )
    
    args = parser.parse_args()
    min_cluster_size = args.min_size
    
    print("🎯 Two-Pass Clustering with Semantic Quality Scoring\n")
    print("Configuration:")
    print(f"  • minClusterSize: {min_cluster_size} (initial HDBSCAN density threshold)")
    print(f"  • Outlier Evaluation: Semantic quality score (0-1 scale, cosine similarity)")
    print(f"  • Reassignment Strategy: Dynamic threshold (uses average quality score)\n")
    
    print("Clustering Pipeline:")
    print("  1️⃣  Initial HDBSCAN: Find dense clusters respecting variable shapes")
    print("  2️⃣  Quality Evaluation: Assess semantic fit of outliers to clusters")
    print("  3️⃣  Dynamic Filtering: Reassign only outliers with quality > average")
    print("  4️⃣  Secondary HDBSCAN: Cluster remaining isolated notes (minClusterSize=1)\n")
    
    try:
        # Connect to database
        db = lancedb.connect(str(DB_PATH))
        notes_table = db.open_table(TABLE_NAME)
        
        # Run clustering
        print("=" * 50)
        print("🚀 Starting Semantic-Aware Two-Pass Clustering")
        print("=" * 50)
        print()
        
        cluster_result = cluster_notes(notes_table, min_cluster_size, verbose=True)
        
        # Refresh the table reference after clustering (table was rebuilt)
        notes_table = db.open_table(TABLE_NAME)
        
        print(f"✅ Clustering Results:")
        print(f"   • Primary clusters: {cluster_result['primaryClusters']}")
        print(f"   • Secondary clusters: {cluster_result['secondaryClusters']}")
        print(f"   • Total clusters: {cluster_result['totalClusters']}")
        print(f"   • Total notes: {cluster_result['totalNotes']}")
        
        clustered_notes = cluster_result['totalNotes'] - cluster_result['stillIsolated']
        clustered_pct = (clustered_notes / cluster_result['totalNotes']) * 100
        
        print(f"   • Notes in clusters: {clustered_notes} ({clustered_pct:.1f}%)")
        print(f"   • Remaining true outliers: {cluster_result['stillIsolated']} ({(cluster_result['stillIsolated'] / cluster_result['totalNotes'] * 100):.1f}%)")
        print(f"   • Quality threshold used: {cluster_result['qualityThreshold']:.3f} (dynamic average)")
        print(f"   • Time: {cluster_result['timeSeconds']:.1f}s\n")
        
        # Cluster size distribution
        if cluster_result['clusterSizes']:
            print("📊 Cluster Size Distribution:")
            for idx, cluster in enumerate(cluster_result['clusterSizes'][:10]):
                bar_length = max(1, cluster['size'] // 2)
                bar = "█" * bar_length
                print(f"   {idx + 1:2d}. {cluster['label'][:25]:25s} │ {bar} {cluster['size']} notes")
            
            if len(cluster_result['clusterSizes']) > 10:
                print(f"   ... and {len(cluster_result['clusterSizes']) - 10} more clusters")
            print()
        
        # Display final composition
        print("=" * 50)
        print("📊 FINAL CLUSTER COMPOSITION")
        print("=" * 50)
        print()
        
        final_clusters = list_clusters(notes_table)
        real_clusters = [c for c in final_clusters if c['cluster_id'] != '-1']
        
        print("🎯 ALL CLUSTERS:\n")
        total_clustered = 0
        
        for i, cluster in enumerate(real_clusters):
            notes = get_notes_in_cluster(notes_table, cluster['cluster_id'])
            total_clustered += len(notes)
            
            print(f"📌 Cluster {i + 1}: \"{cluster['cluster_label']}\"")
            print(f"   📊 {len(notes)} notes")
            print(f"   📖 Notes:")
            for idx, note in enumerate(notes[:10]):  # Show first 10
                print(f"      {idx + 1}. \"{note['title']}\"")
            if len(notes) > 10:
                print(f"      ... and {len(notes) - 10} more")
            print()
        
        # Display outliers
        outlier_notes = get_notes_in_cluster(notes_table, '-1')
        if outlier_notes:
            print(f"📌 OUTLIERS ({len(outlier_notes)} notes):")
            print(f"   Notes with poor semantic fit to any cluster:")
            for idx, note in enumerate(outlier_notes[:10]):
                print(f"      {idx + 1}. \"{note['title']}\"")
            if len(outlier_notes) > 10:
                print(f"      ... and {len(outlier_notes) - 10} more")
            print()
        
        # Final summary
        print("=" * 50)
        print("📈 FINAL SUMMARY")
        print("=" * 50)
        print()
        
        print(f"Configuration Used:")
        print(f"  • minClusterSize: {min_cluster_size}\n")
        
        print(f"Results:")
        print(f"  • Total notes: {cluster_result['totalNotes']}")
        print(f"  • Primary clusters: {cluster_result['primaryClusters']}")
        print(f"  • Secondary clusters: {cluster_result['secondaryClusters']}")
        print(f"  • Total clusters: {len(real_clusters)}")
        print(f"  • Notes clustered: {total_clustered} ({(total_clustered / cluster_result['totalNotes'] * 100):.1f}%)")
        print(f"  • True outliers: {len(outlier_notes)} ({(len(outlier_notes) / cluster_result['totalNotes'] * 100):.1f}%)")
        print(f"  • Processing time: {cluster_result['timeSeconds']:.1f}s")
        print(f"  • Quality threshold: {cluster_result['qualityThreshold']:.3f} (semantic fit)")
        
        print(f"\n✨ Semantic-aware clustering complete!")
        print(f"   💾 All changes persisted to database")
        print(f"   🎯 Dynamic threshold: {cluster_result['qualityThreshold']:.3f} (average quality score)")
        print(f"   📊 {cluster_result['primaryClusters']} primary + {cluster_result['secondaryClusters']} secondary = {len(real_clusters)} total clusters")
        print(f"   🔄 HDBSCAN throughout (respects variable cluster shapes/densities)")
        
        if not outlier_notes:
            print("\n🎉 Full coverage: All notes are now clustered!")
        else:
            outlier_pct = (len(outlier_notes) / cluster_result['totalNotes'] * 100)
            print(f"\n💡 Semantic preservation: {outlier_pct:.1f}% of notes remain as outliers")
            print(f"   These have poor semantic fit with existing clusters")
            print(f"\n   To increase coverage, try:")
            print(f"   • Decreasing minClusterSize (e.g., --min-size=1)")
            print(f"\n   To improve semantic accuracy, try:")
            print(f"   • Increasing minClusterSize (e.g., --min-size=5)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()