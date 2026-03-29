#!/usr/bin/env python3
"""Generate per-cluster chunk statistics and write them to JSON."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lancedb

from backend.analysis.cluster_utils import DB_PATH, TABLE_NAME


def _cluster_sort_key(cluster_id: Any) -> Tuple[int, int, str]:
    """Sort numerically by cluster ID when possible, placing -1 (outliers) last."""
    cluster_id_str = str(cluster_id)
    try:
        cluster_id_int = int(cluster_id_str)
        is_outlier = 1 if cluster_id_int == -1 else 0
        return (0, is_outlier, f"{cluster_id_int:020d}")
    except ValueError:
        return (1, 1, cluster_id_str)


def _safe_percentage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (count / total) * 100.0


def _display_topic_sort_key(display_topic_id: str) -> Tuple[int, ...]:
    """Sort display topic IDs like 12.0, 12.1, 13 numerically by parts."""
    parts = str(display_topic_id).split(".")
    key: List[int] = []
    for part in parts:
        try:
            key.append(int(part))
        except ValueError:
            # Keep deterministic ordering for non-numeric fragments.
            key.append(10**9)
    return tuple(key)


def build_cluster_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_chunks = len(records)

    # Aggregate at base-cluster level (main BERTopic retrieval cluster).
    clusters: Dict[str, Dict[str, Any]] = {}
    # Track child display clusters for mega-cluster splits.
    subclusters_by_base: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for record in records:
        cluster_id = str(record.get("cluster_id", "-1"))
        cluster_label = record.get("cluster_label") or "Unknown"
        base_topic_id = str(record.get("base_topic_id", cluster_id))
        display_topic_id = str(record.get("display_topic_id", cluster_id))

        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_label,
                "chunk_count": 0,
                "subclusters": [],
            }

        clusters[cluster_id]["chunk_count"] += 1

        # A subcluster is recognized as parent.child (e.g. 12.0) that maps to a base topic.
        if "." in display_topic_id and base_topic_id == cluster_id:
            if cluster_id not in subclusters_by_base:
                subclusters_by_base[cluster_id] = {}
            if display_topic_id not in subclusters_by_base[cluster_id]:
                subclusters_by_base[cluster_id][display_topic_id] = {
                    "display_topic_id": display_topic_id,
                    "chunk_count": 0,
                }
            subclusters_by_base[cluster_id][display_topic_id]["chunk_count"] += 1

    cluster_list = sorted(clusters.values(), key=lambda c: _cluster_sort_key(c["cluster_id"]))

    for cluster in cluster_list:
        percent = _safe_percentage(cluster["chunk_count"], total_chunks)
        # Keep 2 decimal places in the output for readability and stable JSON.
        cluster["percent_of_total_chunks"] = round(percent, 2)

        base_cluster_id = cluster["cluster_id"]
        subcluster_map = subclusters_by_base.get(base_cluster_id, {})
        if subcluster_map:
            sorted_subclusters = sorted(
                subcluster_map.values(),
                key=lambda s: _display_topic_sort_key(s["display_topic_id"]),
            )
            for subcluster in sorted_subclusters:
                subcluster["percent_of_total_chunks"] = round(
                    _safe_percentage(subcluster["chunk_count"], total_chunks), 2
                )
            cluster["subclusters"] = sorted_subclusters

    return {
        "total_chunks": total_chunks,
        "cluster_count": len(cluster_list),
        "clusters": cluster_list,
    }


def print_cluster_stats(report: Dict[str, Any]) -> None:
    total_chunks = report["total_chunks"]
    cluster_count = report["cluster_count"]

    print(f"Total chunks: {total_chunks}")
    print(f"Total clusters: {cluster_count}")
    print("-" * 80)
    print(f"{'Cluster ID':<12} {'Cluster Name':<40} {'Chunks':>10} {'% of Total':>12}")
    print("-" * 80)

    for cluster in report["clusters"]:
        cluster_id = cluster["cluster_id"]
        cluster_name = str(cluster["cluster_name"]).replace("\n", " ").strip()
        if len(cluster_name) > 40:
            cluster_name = f"{cluster_name[:37]}..."

        chunk_count = cluster["chunk_count"]
        percent = cluster["percent_of_total_chunks"]
        print(f"{cluster_id:<12} {cluster_name:<40} {chunk_count:>10} {percent:>11.2f}%")

        subclusters = cluster.get("subclusters", [])
        for subcluster in subclusters:
            subcluster_id = subcluster["display_topic_id"]
            subcluster_count = subcluster["chunk_count"]
            subcluster_percent = subcluster["percent_of_total_chunks"]
            print(
                f"{'':<12} {'-> ' + subcluster_id:<40} {subcluster_count:>10} {subcluster_percent:>11.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List each cluster by ID with cluster name, chunk count, and percentage "
            "of total chunks; print to stdout and write JSON output."
        )
    )
    parser.add_argument(
        "--output",
        default="cluster_stats.json",
        help="Path to JSON output file (default: cluster_stats.json)",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    db = lancedb.connect(str(DB_PATH))
    notes_table = db.open_table(TABLE_NAME)
    records = notes_table.to_pandas().to_dict("records")

    report = build_cluster_stats(records)

    print_cluster_stats(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print("-" * 80)
    print(f"Wrote JSON report to: {output_path}")


if __name__ == "__main__":
    main()
