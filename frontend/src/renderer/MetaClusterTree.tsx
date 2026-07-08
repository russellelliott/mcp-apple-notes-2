import React, { useCallback, useEffect, useMemo, useState } from 'react';

// ── API types ────────────────────────────────────────────────────────────────
interface MetaChildCluster {
  cluster_id: string;
  label: string;
  chunk_count: number;
  color?: string | null;
  centroid?: number[] | null;
}

interface MetaClusterInfo {
  meta_cluster_id: string;
  label: string;
  child_clusters: MetaChildCluster[];
}

type ClusterSortMetric = 'recency' | 'momentum' | 'az' | 'size' | 'search' | 'history' | 'similarity';

interface Props {
  /**Called when user clicks a cluster (child or meta). Passes the child cluster_id.*/
  onClusterSelect: (clusterId: string) => void;
  /**Currently selected cluster ID — highlights it in the tree */
  selectedClusterId?: string | null;
  /**Sort order at both meta and child levels */
  sortMetric: ClusterSortMetric;
  /**Optional search filter to narrow visible children */
  filterText?: string;
  /**Map of cluster_id → hex color (from /cluster_colors API) */
  clusterColors?: Record<string, string>;
}

// ── Sort helpers ─────────────────────────────────────────────────────────────
const compareTopicIds = (a: string, b: string) => {
  const aParts = String(a).split('.').map((p) => (p.match(/^-?\d+$/) ? Number(p) : p));
  const bParts = String(b).split('.').map((p) => (p.match(/^-?\d+$/) ? Number(p) : p));
  const maxLen = Math.max(aParts.length, bParts.length);
  for (let i = 0; i < maxLen; i += 1) {
    const aVal = aParts[i];
    const bVal = bParts[i];
    if (aVal === undefined) return -1;
    if (bVal === undefined) return 1;
    if (typeof aVal === 'number' && typeof bVal === 'number' && aVal !== bVal) return aVal - bVal;
    if (String(aVal) !== String(bVal)) return String(aVal).localeCompare(String(bVal), undefined, { numeric: true });
  }
  return 0;
};

const sortChildren = (children: MetaChildCluster[], metric: ClusterSortMetric): MetaChildCluster[] => {
  const sorted = [...children];
  switch (metric) {
    case 'size':
      return sorted.sort((a, b) => b.chunk_count - a.chunk_count);
    case 'search':
    case 'similarity':
      return sorted.sort((a, b) => compareTopicIds(a.cluster_id, b.cluster_id));
    default:
      return sorted.sort((a, b) => compareTopicIds(a.cluster_id, b.cluster_id));
  }
};

// ── Helper: hex to rgba with alpha ───────────────────────────────────────────
const hexToRgba = (hex: string, alpha: number): string => {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

// ── TreeNode component ───────────────────────────────────────────────────────
interface TreeNodeProps {
  meta: MetaClusterInfo;
  depth: number;
  expandedMetaIds: Set<string>;
  toggleMeta: (id: string) => void;
  onClusterSelect: (clusterId: string) => void;
  selectedClusterId?: string | null;
  sortMetric: ClusterSortMetric;
  filterText?: string;
  clusterColors?: Record<string, string>;
}

const TreeNode: React.FC<TreeNodeProps> = ({
  meta,
  depth,
  expandedMetaIds,
  toggleMeta,
  onClusterSelect,
  selectedClusterId,
  sortMetric,
  filterText,
  clusterColors,
}) => {
  const isExpanded = expandedMetaIds.has(meta.meta_cluster_id);

  const filteredChildren = useMemo(() => {
    let children = meta.child_clusters;
    if (filterText) {
      const lower = filterText.toLowerCase();
      children = children.filter(
        (c) =>
          c.cluster_id.toLowerCase().includes(lower) ||
          c.label.toLowerCase().includes(lower),
      );
    }
    return sortChildren(children, sortMetric);
  }, [meta.child_clusters, sortMetric, filterText]);

  // If filter is active, show all expanded
  const effectiveExpanded = filterText ? new Set([meta.meta_cluster_id]) : expandedMetaIds;
  const currentlyExpanded = effectiveExpanded.has(meta.meta_cluster_id);

  return (
    <div style={{ marginLeft: depth * 12 }}>
      {/* Meta-cluster folder row */}
      <div
        onClick={() => toggleMeta(meta.meta_cluster_id)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          cursor: 'pointer',
          padding: '4px 6px',
          borderRadius: 4,
          fontSize: 12,
          fontWeight: 700,
          color: '#1d4ed8',
          userSelect: 'none',
          background: currentlyExpanded ? '#eff6ff' : 'transparent',
        }}
      >
        <span style={{ fontSize: 10, transition: 'transform 0.15s', display: 'inline-block', transform: currentlyExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
          ▶
        </span>
        <span style={{ fontSize: 14 }}>📁</span>
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
          {meta.label}
        </span>
        <span style={{ fontSize: 10, color: '#9ca3af' }}>
          {filteredChildren.length}
        </span>
      </div>

      {/* Child clusters */}
      {currentlyExpanded && filteredChildren.map((child) => {
        const isSelected = selectedClusterId === child.cluster_id;
        const dotColor = child.color || clusterColors?.[child.cluster_id] || '#6b7280';
        const rowBg = isSelected
          ? '#3b82f6'
          : hexToRgba(dotColor, 0.12);

        return (
          <div
            key={child.cluster_id}
            onClick={(e) => { e.stopPropagation(); onClusterSelect(child.cluster_id); }}
            style={{
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
              padding: '3px 6px',
              borderRadius: 4,
              fontSize: 11,
              color: isSelected ? '#fff' : '#374151',
              background: rowBg,
              marginLeft: 18,
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: dotColor,
                flex: '0 0 8px',
              }}
            />
            {/* cluster_id */}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginRight: 4, flexShrink: 0, minWidth: 0, fontFamily: 'monospace', fontSize: 10 }}>
              {child.cluster_id}
            </span>
            {/* cluster name / label */}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginRight: 4, flexShrink: 1, minWidth: 0 }}>
              {child.label}
            </span>
            {/* chunk count */}
            <span style={{ fontSize: 10, color: isSelected ? '#bfdbfe' : '#9ca3af', flexShrink: 0, marginLeft: 2 }}>
              {child.chunk_count}
            </span>
          </div>
        );
      })}
    </div>
  );
};

// ── Main component ───────────────────────────────────────────────────────────
export const MetaClusterTree: React.FC<Props> = ({
  onClusterSelect,
  selectedClusterId,
  sortMetric,
  filterText,
  clusterColors,
}) => {
  const [metaData, setMetaData] = useState<MetaClusterInfo[]>([]);
  const [expandedMetaIds, setExpandedMetaIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch meta-cluster data
  useEffect(() => {
    const fetchMetaClusters = async () => {
      try {
        setLoading(true);
        const res = await fetch('http://localhost:8000/meta_clusters');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: MetaClusterInfo[] = await res.json();
        setMetaData(data);

        // Auto-expand the meta-cluster containing the selected cluster (if any)
        if (selectedClusterId && data.length > 0) {
          const targetMeta = data.find((m) =>
            m.child_clusters.some((c) => c.cluster_id === selectedClusterId),
          );
          if (targetMeta) {
            setExpandedMetaIds(new Set([targetMeta.meta_cluster_id]));
          } else if (data.length <= 5) {
            // If data is small, expand all
            setExpandedMetaIds(new Set(data.map((m) => m.meta_cluster_id)));
          }
        } else if (data.length <= 5) {
          setExpandedMetaIds(new Set(data.map((m) => m.meta_cluster_id)));
        }
      } catch (e: any) {
        setError(e.message || 'Failed to load meta-clusters');
      } finally {
        setLoading(false);
      }
    };
    fetchMetaClusters();
  }, []);

  const toggleMeta = useCallback((id: string) => {
    setExpandedMetaIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  // Expand the meta-cluster containing a selected child
  useEffect(() => {
    if (!selectedClusterId || metaData.length === 0) return;
    const target = metaData.find((m) =>
      m.child_clusters.some((c) => c.cluster_id === selectedClusterId),
    );
    if (target && !expandedMetaIds.has(target.meta_cluster_id)) {
      setExpandedMetaIds(new Set([target.meta_cluster_id]));
    }
  }, [selectedClusterId, metaData, expandedMetaIds]);

  const handleExpandAll = useCallback(() => {
    setExpandedMetaIds(new Set(metaData.map((m) => m.meta_cluster_id)));
  }, [metaData]);

  const handleCollapseAll = useCallback(() => {
    setExpandedMetaIds(new Set());
  }, []);

  if (loading) return <div style={{ padding: 12, color: '#9ca3af', fontSize: 12 }}>Loading meta-clusters…</div>;
  if (error) return <div style={{ padding: 12, color: '#dc2626', fontSize: 12 }}>Error: {error}</div>;
  if (metaData.length === 0) return <div style={{ padding: 12, color: '#9ca3af', fontSize: 12 }}>No meta-clusters yet. Run the clustering pipeline.</div>;

  const totalClusters = metaData.reduce((sum, m) => sum + m.child_clusters.length, 0);
  const totalChunks = metaData.reduce(
    (sum, m) => sum + m.child_clusters.reduce((s, c) => s + c.chunk_count, 0),
    0,
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{ padding: '8px 10px', borderBottom: '1px solid #e5e7eb' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#1f2937' }}>Meta-Clusters</span>
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              type="button"
              onClick={handleExpandAll}
              style={{
                fontSize: 10, padding: '2px 6px', cursor: 'pointer',
                background: 'none', border: '1px solid #d1d5db', borderRadius: 3, color: '#4b5563',
              }}
            >Expand All</button>
            <button
              type="button"
              onClick={handleCollapseAll}
              style={{
                fontSize: 10, padding: '2px 6px', cursor: 'pointer',
                background: 'none', border: '1px solid #d1d5db', borderRadius: 3, color: '#4b5563',
              }}
            >Collapse All</button>
          </div>
        </div>
        <div style={{ fontSize: 10, color: '#9ca3af' }}>
          {metaData.length} meta-clusters · {totalClusters} clusters · {totalChunks.toLocaleString()} chunks
        </div>
      </div>

      {/* Optional filter input */}
      {filterText !== undefined && (
        <div style={{ padding: '6px 10px', borderBottom: '1px solid #e5e7eb' }}>
          <input
            type="text"
            placeholder="Filter clusters…"
            value={filterText}
            onChange={(e) => {}}
            style={{
              width: '100%', padding: '4px 8px', fontSize: 11,
              border: '1px solid #d1d5db', borderRadius: 3, boxSizing: 'border-box',
            }}
          />
        </div>
      )}

      {/* Tree */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
        {metaData.map((meta) => (
          <TreeNode
            key={meta.meta_cluster_id}
            meta={meta}
            depth={0}
            expandedMetaIds={expandedMetaIds}
            toggleMeta={toggleMeta}
            onClusterSelect={onClusterSelect}
            selectedClusterId={selectedClusterId}
            sortMetric={sortMetric}
            filterText={filterText}
            clusterColors={clusterColors}
          />
        ))}
      </div>
    </div>
  );
};

export default MetaClusterTree;
