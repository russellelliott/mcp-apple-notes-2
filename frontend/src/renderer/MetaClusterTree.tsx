import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

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
  last_modified?: string | null;
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
    /**Optional date range to filter meta-clusters */
  dateFrom?: string;
  dateTo?: string;
    /**When search mode is active, only show clusters represented in these IDs */
  searchClusterIds?: Set<string>;
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

const sortMetaClusters = (metaList: MetaClusterInfo[]): MetaClusterInfo[] => {
   // Sort by last_modified DESC (nulls last), then by label for stable ordering
  return [...metaList].sort((a, b) => {
    const aMod = a.last_modified ?? '';
    const bMod = b.last_modified ?? '';
    if (!aMod && !bMod) return a.label.localeCompare(b.label, undefined, { numeric: true });
    if (!aMod) return 1;    // nulls last
    if (!bMod) return -1;
    if (bMod > aMod) return 1;
    if (bMod < aMod) return -1;
    return a.label.localeCompare(b.label, undefined, { numeric: true });
   });
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
  searchClusterIds?: Set<string>;
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
  searchClusterIds,
}) => {
  const isExpanded = expandedMetaIds.has(meta.meta_cluster_id);

  const filteredChildren = useMemo(() => {
    let children = meta.child_clusters;
     // Filter to only clusters represented in search results (if search mode active)
    if (searchClusterIds && searchClusterIds.size > 0) {
      children = children.filter((c) => searchClusterIds.has(c.cluster_id));
     }
    if (filterText) {
      const lower = filterText.toLowerCase();
      children = children.filter(
         (c) =>
          c.cluster_id.toLowerCase().includes(lower) ||
          c.label.toLowerCase().includes(lower),
       );
      }
    return sortChildren(children, sortMetric);
   }, [meta.child_clusters, sortMetric, filterText, searchClusterIds]);

   // Skip rendering this meta-cluster entirely if no children pass the filter
  const hasVisibleChildren = filteredChildren.length > 0;
  if (!hasVisibleChildren) return null;

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
           <span style={{ flex: 1, wordBreak: 'break-word', whiteSpace: 'normal', lineHeight: 1.3 }}>
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

         return (
            <div
             key={child.cluster_id}
             onClick={(e) => { e.stopPropagation(); onClusterSelect(child.cluster_id); }}
             style={{
             display: 'flex',
             alignItems: 'flex-start',
             cursor: 'pointer',
               padding: '3px 6px',
               borderRadius: 4,
               fontSize: 11,
               color: isSelected ? '#fff' : '#374151',
               background: isSelected ? '#3b82f6' : 'transparent',
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
               <span style={{ fontFamily: 'monospace', fontSize: 10, flexShrink: 0, marginRight: 4 }}>
                 {child.cluster_id}
               </span>
               {/* cluster name — wraps freely, fills remaining space */}
               <span style={{ flex: 1, minWidth: 0, marginRight: 4, wordBreak: 'break-word', whiteSpace: 'normal', lineHeight: 1.3 }}>
                 {child.label}
               </span>
               {/* chunk count — always right-aligned */}
               <span style={{ fontSize: 10, color: isSelected ? '#bfdbfe' : '#9ca3af', flexShrink: 0, marginLeft: 'auto', whiteSpace: 'nowrap' }}>
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
  dateFrom,
  dateTo,
  searchClusterIds,
}) => {
  const [metaData, setMetaData] = useState<MetaClusterInfo[]>([]);
  const [expandedMetaIds, setExpandedMetaIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

     // Determine which endpoint to use — memoized but only changes when date range actually changes
    const useFiltered = useMemo(() => !!(dateFrom || dateTo), [dateFrom, dateTo]);

      // Fetch meta-cluster data — only when date range changes (not on cluster select)
   // Uses the memoized `useFiltered` to decide endpoint, but never re-fetches just because selection changed
  useEffect(() => {
    let active = true;
     (async () => {
      try {
        setLoading(true);
        let url = 'http://localhost:8000/meta_clusters';
        if (useFiltered) {
            // Use the filtered endpoint which properly restricts to notes in date range
          url = 'http://localhost:8000/meta_clusters_filtered';
          const params = new URLSearchParams();
          if (dateFrom) params.set('date_from', dateFrom);
          if (dateTo) params.set('date_to', dateTo);
          url += `?${params.toString()}`;
           }
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: MetaClusterInfo[] = await res.json();
        if (!active) return;

           // Sort meta-clusters by last_modified DESC
        const sorted = sortMetaClusters(data);

        setMetaData(sorted);

           // Initialize expanded state on first load (if still empty)
        setExpandedMetaIds((prev) => {
          if (prev.size > 0) return prev; // already initialized
          if (selectedClusterId && data.length > 0) {
            const targetMeta = data.find((m) =>
              m.child_clusters.some((c) => c.cluster_id === selectedClusterId),
               );
            if (targetMeta) return new Set([targetMeta.meta_cluster_id]);
           }
          return data.length <= 5 ? new Set(data.map((m) => m.meta_cluster_id)) : prev;
         });
         } catch (e: any) {
        if (!active) return;
        setError(e.message || 'Failed to load meta-clusters');
        setLoading(false);
         } finally {
        if (!active) return;
        setLoading(false);
         }
       })();
     return () => { active = true; };
      }, [useFiltered, dateFrom, dateTo]); // ← selection only updates highlight, never re-fetches

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
     if (!target) return;
     setExpandedMetaIds((prev) => {
       if (prev.has(target.meta_cluster_id)) return prev; // no change → no re-render
       const next = new Set(prev);
       next.add(target.meta_cluster_id);           // add, don't replace
       return next;
       });
     }, [selectedClusterId, metaData]); // ← expandedMetaIds removed

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
           {searchClusterIds && searchClusterIds.size > 0 && (
             <span style={{ fontSize: 10, color: '#6b7280', fontStyle: 'italic' }}>Search mode</span>
           )}
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
            searchClusterIds={searchClusterIds}
           />
         ))}
       </div>
     </div>
   );
};

export default MetaClusterTree;
