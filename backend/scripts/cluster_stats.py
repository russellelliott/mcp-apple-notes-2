#!/usr/bin/env python3
"""Generate per-cluster chunk statistics and write them to JSON."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
    # Track nested display clusters (supports multiple levels: 0.0, 0.0.0, 0.0.0.0, etc.)
    nested_clusters: Dict[str, Dict[str, Any]] = {}

    for record in records:
        cluster_id = str(record.get("cluster_id", "-1"))
        cluster_label = record.get("cluster_label") or "Unknown"
        base_cluster_label = record.get("base_cluster_label") or cluster_label or "Unknown"
        base_topic_id = str(record.get("base_topic_id", cluster_id))
        display_topic_id = str(record.get("display_topic_id", cluster_id))

        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "cluster_name": str(base_cluster_label),
                "chunk_count": 0,
                "nested_clusters": {},
            }

        clusters[cluster_id]["chunk_count"] += 1

        # Track any display_topic_id that differs from base (indicates a split cluster)
        if display_topic_id != cluster_id:
            if display_topic_id not in nested_clusters:
                nested_clusters[display_topic_id] = {
                    "display_topic_id": display_topic_id,
                    "cluster_name": str(cluster_label),
                    "chunk_count": 0,
                    "base_topic_id": cluster_id,
                }
            nested_clusters[display_topic_id]["chunk_count"] += 1

    # Infer intermediate cluster levels from their children
    # E.g., if we have "0.0.0" and "0.0.1", create "0.0" if it doesn't exist
    inferred_clusters: Dict[str, Dict[str, Any]] = {}
    for display_id in list(nested_clusters.keys()):
        parts = display_id.split('.')
        base_id = parts[0]

        # For each level between base and this display_id, create intermediate entries
        for level in range(1, len(parts)):
            intermediate_id = '.'.join(parts[:level + 1])
            if intermediate_id not in nested_clusters and intermediate_id not in inferred_clusters:
                inferred_clusters[intermediate_id] = {
                    "display_topic_id": intermediate_id,
                    "cluster_name": f"Intermediate {intermediate_id}",  # Placeholder name
                    "chunk_count": 0,
                    "base_topic_id": base_id,
                    "is_inferred": True,
                }

    # Merge inferred clusters with actual ones
    nested_clusters.update(inferred_clusters)

    # Calculate chunk counts for inferred clusters (bottom-up: deepest first)
    def direct_child_ids(cluster_id: str, all_clusters: Dict[str, Dict[str, Any]]) -> List[str]:
        cluster_parts = cluster_id.split('.')
        out: List[str] = []
        prefix = cluster_id + '.'
        for other_id in all_clusters.keys():
            if other_id == cluster_id:
                continue
            if not other_id.startswith(prefix):
                continue
            other_parts = other_id.split('.')
            if len(other_parts) == len(cluster_parts) + 1:
                out.append(other_id)
        return out

    # Process deepest inferred nodes first so children's counts are finalized.
    for inferred_id in sorted(inferred_clusters.keys(), key=lambda x: len(x.split('.')), reverse=True):
        child_ids = direct_child_ids(inferred_id, nested_clusters)
        nested_clusters[inferred_id]["chunk_count"] = sum(
            nested_clusters[child_id]["chunk_count"] for child_id in child_ids
        )

    cluster_list = sorted(clusters.values(), key=lambda c: _cluster_sort_key(c["cluster_id"]))

    for cluster in cluster_list:
        percent = _safe_percentage(cluster["chunk_count"], total_chunks)
        # Keep 2 decimal places in the output for readability and stable JSON.
        cluster["percent_of_total_chunks"] = round(percent, 2)

        base_cluster_id = cluster["cluster_id"]

        # Collect all nested clusters that belong to this base cluster
        belonging_nested = {
            display_id: info
            for display_id, info in nested_clusters.items()
            if info["base_topic_id"] == base_cluster_id
        }

        if belonging_nested:
            # Build a tree structure for nested clusters
            cluster_tree = _build_nested_tree(belonging_nested, base_cluster_id, total_chunks, cluster["chunk_count"])
            cluster["nested_clusters"] = cluster_tree

    return {
        "total_chunks": total_chunks,
        "cluster_count": len(cluster_list),
        "clusters": cluster_list,
    }


def _build_nested_tree(
    nested_clusters: Dict[str, Dict[str, Any]],
    base_cluster_id: str,
    total_chunks: int,
    base_cluster_count: int
) -> Dict[str, Dict[str, Any]]:
    """Build a proper tree — only direct children of base go in the root."""
    tree: Dict[str, Dict[str, Any]] = {}

    def _make_node(display_id: str, parent_count: int) -> Dict[str, Any]:
        info = nested_clusters.get(display_id, {})
        count = info.get("chunk_count", 0)
        return {
            "display_topic_id": display_id,
            "cluster_name": info.get("cluster_name", f"Intermediate {display_id}"),
            "chunk_count": count,
            "percent_of_total_chunks": round(_safe_percentage(count, total_chunks), 2),
            "percent_of_parent_cluster": round(_safe_percentage(count, parent_count), 2),
            "nested_clusters": {},
        }

    def _get_or_create(tree_node: Dict[str, Dict[str, Any]], display_id: str, parent_count: int) -> Dict[str, Any]:
        if display_id not in tree_node:
            tree_node[display_id] = _make_node(display_id, parent_count)
        return tree_node[display_id]

    def _insert(display_id: str) -> None:
        parts = display_id.split('.')
        # Direct children of base cluster go at the tree root (e.g., '0.0')
        if len(parts) == 2:
            _get_or_create(tree, display_id, base_cluster_count)
            return

        # For deeper nodes, walk down from the root creating/getting ancestors
        current_node = tree
        # Walk from depth=2 ('0.0') down to the parent of the target node
        for depth in range(2, len(parts)):
            ancestor_id = '.'.join(parts[:depth])
            # Use the ancestor's own chunk_count when available; fallback to base_cluster_count
            ancestor_count = nested_clusters.get(ancestor_id, {}).get("chunk_count", base_cluster_count)
            node = _get_or_create(current_node, ancestor_id, base_cluster_count if depth == 2 else ancestor_count)
            current_node = node["nested_clusters"]

        # current_node now refers to the parent's nested_clusters dict
        parent_id = '.'.join(parts[:-1])
        parent_count_actual = nested_clusters.get(parent_id, {}).get("chunk_count", base_cluster_count) or base_cluster_count
        _get_or_create(current_node, display_id, parent_count_actual)

    sorted_displays = sorted(nested_clusters.keys(), key=_display_topic_sort_key)
    for display_id in sorted_displays:
        _insert(display_id)

    return tree


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

        # Recursively print nested clusters, seeding parent count for correct percentages
        nested = cluster.get("nested_clusters", {})
        if nested:
            _print_nested_clusters(nested, indent=1, parent_chunk_count=cluster["chunk_count"])


def _print_nested_clusters(nested_dict: Dict[str, Dict[str, Any]], indent: int = 1, parent_chunk_count: Optional[int] = None) -> None:
    """Recursively print nested clusters with proper indentation and parent percentages."""
    for display_id in sorted(nested_dict.keys(), key=_display_topic_sort_key):
        cluster = nested_dict[display_id]
        indent_str = " " * (indent * 4)

        cluster_name = str(cluster.get("cluster_name", "Unknown")).replace("\n", " ").strip()

        # Compute a name width that shrinks with indentation so columns stay aligned
        max_name_width = max(10, 52 - len(indent_str))
        cluster_display = f"-> {display_id} | {cluster_name}"
        if len(cluster_display) > max_name_width:
            cluster_display = cluster_display[: max_name_width - 3] + "..."

        chunk_count = cluster.get("chunk_count", 0)
        percent_total = cluster.get("percent_of_total_chunks", 0.0)

        # Compute percent relative to parent using the passed-in parent_chunk_count
        if parent_chunk_count and parent_chunk_count > 0:
            percent_parent = round(_safe_percentage(chunk_count, parent_chunk_count), 2)
        else:
            percent_parent = 0.0

        # Print the cluster line
        print(f"{indent_str}{cluster_display:<{max_name_width}} {chunk_count:>10} {percent_total:>11.2f}%")

        # Print the parent percentage line (aligned under the name column)
        parent_line = f"   rate within parent: {percent_parent:.2f}%"
        print(f"{indent_str}{parent_line:<{max_name_width}} {'':>10} {'':>12}")

        # Recursively print children, passing this cluster's count as the parent count
        children = cluster.get("nested_clusters", {})
        if children:
            _print_nested_clusters(children, indent + 1, parent_chunk_count=chunk_count)


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
