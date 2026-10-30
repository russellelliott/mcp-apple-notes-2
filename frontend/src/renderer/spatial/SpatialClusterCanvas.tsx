/**
 * SpatialClusterCanvas — a pan/zoomable 2D spatial canvas of metaclusters.
 *
 * Modeled after the reference CanvasWorkspace:
 *  - metaclusters are large soft circles; clusters are discs nested inside
 *  - base:   all metaclusters at their deterministic UMAP-derived positions
 *  - focus:  selected cluster's metacluster centered, others orbiting by
 *            semantic distance, with similarity rays + connective web links
 *  - search: matches highlighted, non-matches dimmed
 *
 * Pure presentation: all data is computed by the parent (NoteClusters) and
 * passed via props.
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import {
  SearchMatchCounts,
  SpatialCluster,
  SpatialMetaCluster,
  SpatialViewMode,
  XY,
} from './types';
import {
  WORLD_CENTER,
  WORLD_HEIGHT,
  WORLD_WIDTH,
  calculateDynamicCircleWebLayout,
  clusterDiscSize,
  cosineSimilarity,
  getMetaClusterCentroid,
  mixWithBlack,
  mixWithWhite,
} from './layout';

interface SpatialClusterCanvasProps {
  metaClusters: SpatialMetaCluster[];
  selectedClusterId: string | null;
  /** Called with null when the background is clicked or focus is cleared */
  onSelectCluster: (clusterId: string | null) => void;
  /** Optional search highlighting: clusterId -> match count */
  searchMatches?: SearchMatchCounts;
  isSearchActive?: boolean;
}

const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

/** Similarity above which an orbiting metacluster stays fully visible */
const DIM_SIM_THRESHOLD = 0.5;

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 3;

const MODE_META: Record<SpatialViewMode, { title: string; color: string }> = {
  base: { title: '2D Semantic Map', color: '#5b64a8' },
  focus: { title: 'Proximity Orbit', color: '#2f7d6d' },
  search: { title: 'Search Results', color: '#b3691e' },
};

export default function SpatialClusterCanvas({
  metaClusters,
  selectedClusterId,
  onSelectCluster,
  searchMatches,
  isSearchActive,
}: SpatialClusterCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // ── view state ──────────────────────────────────────────────────────────
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<XY>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{ startX: number; startY: number; basePan: XY } | null>(null);

  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  zoomRef.current = zoom;
  panRef.current = pan;

  const [hoveredClusterId, setHoveredClusterId] = useState<string | null>(null);
  const [hoveredMetaId, setHoveredMetaId] = useState<string | null>(null);

  // ── derived data ────────────────────────────────────────────────────────
  const allClusters = useMemo(() => metaClusters.flatMap((m) => m.clusters), [metaClusters]);

  const selectedCluster = useMemo<SpatialCluster | null>(
    () => (selectedClusterId ? allClusters.find((c) => c.id === selectedClusterId) ?? null : null),
    [allClusters, selectedClusterId],
  );

  const activeMeta = useMemo<SpatialMetaCluster | null>(
    () =>
      selectedCluster
        ? metaClusters.find((m) => m.id === selectedCluster.metaClusterId) ?? null
        : null,
    [metaClusters, selectedCluster],
  );

  const viewMode: SpatialViewMode = selectedCluster ? 'focus' : isSearchActive ? 'search' : 'base';

  // ── fit / zoom ──────────────────────────────────────────────────────────
  const fitView = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    const z = Math.min(rect.width / WORLD_WIDTH, rect.height / WORLD_HEIGHT) * 0.94;
    setZoom(z);
    setPan({
      x: (rect.width - WORLD_WIDTH * z) / 2,
      y: (rect.height - WORLD_HEIGHT * z) / 2,
    });
  }, []);

  // Fit once when data first arrives
  const hasFittedRef = useRef(false);
  useLayoutEffect(() => {
    if (metaClusters.length > 0 && !hasFittedRef.current) {
      fitView();
      hasFittedRef.current = true;
    }
  }, [metaClusters.length, fitView]);

  // Non-passive wheel listener so preventDefault works (cursor-anchored zoom)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const prevZoom = zoomRef.current;
      const prevPan = panRef.current;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prevZoom * factor));
      const k = nextZoom / prevZoom;
      setZoom(nextZoom);
      setPan({ x: mx - (mx - prevPan.x) * k, y: my - (my - prevPan.y) * k });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const handleMouseDown = (e: ReactMouseEvent) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    // Let node clicks through — only start panning on empty space / circles
    if (target.closest('.cluster-node-target')) return;
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      basePan: { ...panRef.current },
    };
    setIsDragging(true);
  };

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      setPan({
        x: d.basePan.x + (e.clientX - d.startX),
        y: d.basePan.y + (e.clientY - d.startY),
      });
    };
    const onUp = (e: MouseEvent) => {
      const d = dragRef.current;
      const moved = d ? Math.hypot(e.clientX - d.startX, e.clientY - d.startY) : 99;
      dragRef.current = null;
      setIsDragging(false);
      // Clicking empty background (not a drag) clears the current focus
      if (moved < 4) onSelectCluster(null);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDragging, onSelectCluster]);

  // Center-anchored zoom for the control buttons
  const zoomBy = (factor: number) => {
    const el = containerRef.current;
    const rect = el ? el.getBoundingClientRect() : { width: 0, height: 0 };
    const mx = rect.width / 2;
    const my = rect.height / 2;
    const prevZoom = zoomRef.current;
    const prevPan = panRef.current;
    const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prevZoom * factor));
    const k = nz / prevZoom;
    setZoom(nz);
    setPan({ x: mx - (mx - prevPan.x) * k, y: my - (my - prevPan.y) * k });
  };

  // ── layout + proximity ──────────────────────────────────────────────────
  const layout = useMemo(
    () => calculateDynamicCircleWebLayout(selectedCluster, metaClusters, viewMode),
    [selectedCluster, metaClusters, viewMode],
  );

  const metaCentroidMap = useMemo(() => {
    const map: Record<string, number[] | null> = {};
    for (const m of metaClusters) map[m.id] = getMetaClusterCentroid(m);
    return map;
  }, [metaClusters]);

  // Best similarity of each cluster to every *other* metacluster (centroid cosine)
  const proximities = useMemo(() => {
    const map: Record<string, { bestSim: number; bestMetaId: string | null }> = {};
    for (const c of allClusters) {
      let bestSim = 0;
      let bestMetaId: string | null = null;
      for (const m of metaClusters) {
        if (m.id === c.metaClusterId) continue;
        const sim = cosineSimilarity(c.centroid, metaCentroidMap[m.id]);
        if (sim > bestSim) {
          bestSim = sim;
          bestMetaId = m.id;
        }
      }
      map[c.id] = { bestSim, bestMetaId };
    }
    return map;
  }, [allClusters, metaClusters, metaCentroidMap]);

  const metaMatchCounts = useMemo(() => {
    const map: Record<string, number> = {};
    if (!isSearchActive || !searchMatches) return map;
    for (const m of metaClusters) {
      let n = 0;
      for (const c of m.clusters) n += searchMatches[c.id] || 0;
      if (n > 0) map[m.id] = n;
    }
    return map;
  }, [metaClusters, searchMatches, isSearchActive]);

  const totalMatches = useMemo(
    () => Object.values(metaMatchCounts).reduce((a, b) => a + b, 0),
    [metaMatchCounts],
  );

  const hoveredCluster = hoveredClusterId
    ? allClusters.find((c) => c.id === hoveredClusterId) ?? null
    : null;

  const selectedSimilarity =
    selectedCluster && hoveredCluster && hoveredCluster.id !== selectedCluster.id
      ? cosineSimilarity(selectedCluster.centroid, hoveredCluster.centroid)
      : null;

  const activeMetaId = activeMeta?.id ?? null;

  const getMetaOpacity = (m: SpatialMetaCluster, isCenter: boolean): number => {
    if (hoveredMetaId === m.id) return 1;
    if (viewMode === 'focus' && !isCenter && selectedCluster) {
      const sim = cosineSimilarity(selectedCluster.centroid, metaCentroidMap[m.id]);
      return sim >= DIM_SIM_THRESHOLD ? 0.85 : 0.45;
    }
    if (viewMode === 'search' && !metaMatchCounts[m.id]) return 0.35;
    return 1;
  };

  // ── empty state ─────────────────────────────────────────────────────────
  if (metaClusters.length === 0) {
    return (
      <div style={{ ...styles.container, cursor: 'default', backgroundColor: '#f6f7fb' }}>
        <div style={styles.emptyState}>Loading metaclusters…</div>
      </div>
    );
  }

  const modeSub =
    viewMode === 'focus' && selectedCluster
      ? `centered on “${selectedCluster.title}”`
      : viewMode === 'search'
        ? `${totalMatches} matching cluster${totalMatches === 1 ? '' : 's'}`
        : `${metaClusters.length} metaclusters · ${allClusters.length} clusters`;

  return (
    <div
      ref={containerRef}
      style={{
        ...styles.container,
        cursor: isDragging ? 'grabbing' : 'grab',
        backgroundColor: '#f6f7fb',
        backgroundImage:
          'radial-gradient(circle, rgba(120, 130, 160, 0.16) 1px, transparent 1px)',
        backgroundSize: `${24 * zoom}px ${24 * zoom}px`,
        backgroundPosition: `${pan.x}px ${pan.y}px`,
      }}
      onMouseDown={handleMouseDown}
    >
      <div
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: WORLD_WIDTH,
          height: WORLD_HEIGHT,
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: '0 0',
        }}
      >
        <svg
          width={WORLD_WIDTH}
          height={WORLD_HEIGHT}
          style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible' }}
        >

          {/* Orbit guide rings in focus mode */}
          {viewMode === 'focus' && (
            <g>
              <circle
                cx={WORLD_CENTER.x}
                cy={WORLD_CENTER.y}
                r={520}
                fill="none"
                stroke="#9aa3c0"
                strokeOpacity={0.25}
                strokeDasharray="4 10"
              />
              <circle
                cx={WORLD_CENTER.x}
                cy={WORLD_CENTER.y}
                r={1480}
                fill="none"
                stroke="#9aa3c0"
                strokeOpacity={0.18}
                strokeDasharray="4 10"
              />
              <text
                x={WORLD_CENTER.x}
                y={WORLD_CENTER.y - 526}
                textAnchor="middle"
                fontSize={10}
                fill="#7c86a8"
              >
                core
              </text>
              <text
                x={WORLD_CENTER.x}
                y={WORLD_CENTER.y - 1486}
                textAnchor="middle"
                fontSize={10}
                fill="#7c86a8"
              >
                similar orbit
              </text>
            </g>
          )}
        </svg>

        {/* Metacluster circles */}
        {metaClusters.map((m) => {
          const center = layout.metaCircleCenters[m.id];
          if (!center) return null;
          const isCenter = viewMode === 'focus' && m.id === activeMetaId;
          const isHovered = hoveredMetaId === m.id;
          const r = center.radius;
          return (
            <div
              key={m.id}
              onMouseEnter={() => setHoveredMetaId(m.id)}
              onMouseLeave={() => setHoveredMetaId((prev) => (prev === m.id ? null : prev))}
              style={{
                position: 'absolute',
                left: center.x - r,
                top: center.y - r,
                width: r * 2,
                height: r * 2,
                borderRadius: '50%',
                background: `radial-gradient(circle at 50% 40%, ${m.color.bg}, ${mixWithWhite(
                  m.color.accent,
                  0.97,
                )})`,
                border: `1.5px solid ${
                  isCenter ? mixWithBlack(m.color.accent, 0.1) : m.color.border
                }`,
                boxShadow: isCenter
                  ? `0 0 0 3px ${mixWithWhite(m.color.accent, 0.6)}, 0 8px 28px rgba(20, 24, 40, 0.14)`
                  : isHovered
                    ? `0 0 0 2px ${mixWithWhite(m.color.accent, 0.65)}, 0 6px 18px rgba(20, 24, 40, 0.1)`
                    : '0 2px 10px rgba(20, 24, 40, 0.05)',
                opacity: getMetaOpacity(m, isCenter),
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: 8,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  textAlign: 'center',
                  maxWidth: r * 1.7,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  pointerEvents: 'none',
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 700, color: m.color.text }}>
                  {m.title}
                </div>
                <div
                  style={{ fontSize: 9, color: m.color.text, opacity: 0.65, fontFamily: MONO }}
                >
                  {m.clusters.length} clusters · {m.totalChunks} chunks
                  {viewMode === 'search' && metaMatchCounts[m.id]
                    ? ` · ${metaMatchCounts[m.id]} matches`
                    : ''}
                </div>
              </div>
            </div>
          );
        })}

        {/* Cluster discs */}
        {metaClusters.flatMap((m) =>
          m.clusters.map((c) => {
            const pos = layout.clusterCoordinates[c.id];
            if (!pos) return null;
            const isSelected = c.id === selectedClusterId;
            const isHovered = hoveredClusterId === c.id;
            const inActiveMeta = c.metaClusterId === activeMetaId;
            const matchCount = searchMatches?.[c.id] || 0;
            const hasMatch = viewMode === 'search' && matchCount > 0;
            const dimmedBySearch = viewMode === 'search' && matchCount === 0;

            let opacity = 1;
            if (viewMode === 'focus') {
              opacity = isSelected ? 1 : inActiveMeta ? 0.9 : 0.3;
            } else if (dimmedBySearch) {
              opacity = 0.18;
            }
            if (isHovered || isSelected) opacity = 1;

            const size = clusterDiscSize(m.clusters.length) * (isSelected ? 1.35 : 1);
            const showSimBadge =
              viewMode === 'focus' &&
              isHovered &&
              !!selectedCluster &&
              c.id !== selectedClusterId;

            return (
              <div
                key={c.id}
                className="cluster-node-target"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectCluster(isSelected ? null : c.id);
                }}
                onMouseEnter={() => setHoveredClusterId(c.id)}
                onMouseLeave={() =>
                  setHoveredClusterId((prev) => (prev === c.id ? null : prev))
                }
                style={{
                  position: 'absolute',
                  left: pos.x - size / 2,
                  top: pos.y - size / 2,
                  width: size,
                  height: size,
                  zIndex: isSelected ? 30 : isHovered ? 25 : 10,
                  opacity,
                  cursor: 'pointer',
                }}
              >
                {showSimBadge && (
                  <div
                    style={{
                      position: 'absolute',
                      top: -18,
                      left: '50%',
                      transform: 'translateX(-50%)',
                      background: '#1f2436',
                      color: '#fff',
                      fontSize: 9,
                      fontFamily: MONO,
                      padding: '2px 6px',
                      borderRadius: 6,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {(cosineSimilarity(selectedCluster!.centroid, c.centroid) * 100).toFixed(
                      0,
                    )}
                    % similar
                  </div>
                )}
                <div
                  style={{
                    width: '100%',
                    height: '100%',
                    borderRadius: '50%',
                    background: `radial-gradient(circle at 35% 30%, ${mixWithWhite(
                      c.color,
                      0.35,
                    )}, ${mixWithBlack(c.color, 0.15)})`,
                    border: isSelected || hasMatch
                      ? `2.5px solid ${mixWithBlack(c.color, 0.3)}`
                      : `1.5px solid ${mixWithWhite(c.color, 0.3)}`,
                    boxShadow: isSelected || isHovered || hasMatch
                      ? `0 0 0 3px ${mixWithWhite(c.color, 0.55)}, 0 4px 12px rgba(20, 24, 40, 0.18)`
                      : '0 1px 4px rgba(20, 24, 40, 0.12)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {size >= 34 && (
                    <span
                      style={{
                        fontSize: 9,
                        fontWeight: 700,
                        color: 'rgba(255, 255, 255, 0.95)',
                        fontFamily: MONO,
                      }}
                    >
                      {c.totalChunks}
                    </span>
                  )}
                </div>
                {hasMatch && (
                  <div
                    style={{
                      position: 'absolute',
                      top: -6,
                      right: -6,
                      minWidth: 16,
                      height: 16,
                      padding: '0 4px',
                      borderRadius: 8,
                      background: mixWithBlack(c.color, 0.1),
                      color: '#fff',
                      fontSize: 9,
                      fontWeight: 700,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontFamily: MONO,
                    }}
                  >
                    {matchCount}
                  </div>
                )}
                <div
                  title={c.title}
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    marginTop: 3,
                    maxWidth: 120,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    fontSize: isSelected || isHovered ? 10 : 8.5,
                    fontWeight: isSelected ? 700 : 500,
                    color: '#3a4058',
                    opacity: isSelected || isHovered || size >= 44 ? 1 : 0.55,
                  }}
                >
                  {c.title}
                </div>
              </div>
            );
          }),
        )}
      </div>

      {/* ── overlay chrome ── */}
      <div style={styles.banner}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ ...styles.modeDot, background: MODE_META[viewMode].color }} />
          <span style={{ fontSize: 13, fontWeight: 700, color: '#23283c' }}>
            {MODE_META[viewMode].title}
          </span>
          {selectedCluster && (
            <button style={styles.clearBtn} onClick={() => onSelectCluster(null)}>
              ✕ clear focus
            </button>
          )}
        </div>
        <div style={{ fontSize: 11, color: '#6b7288', marginTop: 4 }}>{modeSub}</div>
      </div>

      <div style={styles.controls}>
        <button style={styles.controlBtn} title="Zoom out" onClick={() => zoomBy(1 / 1.3)}>
          −
        </button>
        <button style={styles.controlBtn} title="Zoom in" onClick={() => zoomBy(1.3)}>
          +
        </button>
        <button
          style={{ ...styles.controlBtn, width: 'auto', padding: '0 10px', fontSize: 11 }}
          title="Fit to view"
          onClick={fitView}
        >
          fit
        </button>
      </div>

      <div style={styles.legend}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              width: 14,
              height: 14,
              borderRadius: '50%',
              background: '#dfe3f4',
              border: '1.5px solid #8b93b8',
              display: 'inline-block',
            }}
          />
          Metacluster
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#5b64a8',
              display: 'inline-block',
            }}
          />
          Cluster
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#5b64a8',
              boxShadow: '0 0 0 3px rgba(91, 100, 168, 0.45)',
              display: 'inline-block',
            }}
          />
          Selected / match
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, borderTop: '2px dashed #8b93b8', display: 'inline-block' }} />
          Similarity web
        </div>
        <div style={{ opacity: 0.6 }}>scroll to zoom · drag to pan · click cluster to focus</div>
      </div>

      {/* hover card */}
      {hoveredCluster && (
        <div style={styles.hoverCard}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 3,
                background: hoveredCluster.color,
                flexShrink: 0,
              }}
            />
            <span
              style={{
                fontWeight: 700,
                color: '#23283c',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {hoveredCluster.title}
            </span>
          </div>
          <div style={{ color: '#6b7288' }}>
            {metaClusters.find((m) => m.id === hoveredCluster.metaClusterId)?.title ?? '—'}
          </div>
          <div style={{ color: '#6b7288', fontFamily: MONO }}>
            {hoveredCluster.totalChunks} chunks
          </div>
          {viewMode === 'focus' &&
            selectedCluster &&
            hoveredCluster.id !== selectedCluster.id &&
            selectedSimilarity !== null && (
              <div style={{ color: '#2f7d6d', fontFamily: MONO }}>
                {(selectedSimilarity * 100).toFixed(1)}% similar to selection
              </div>
            )}
          {viewMode === 'search' && (
            <div style={{ color: '#b3691e', fontFamily: MONO }}>
              {searchMatches?.[hoveredCluster.id] || 0} matches
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── styles ──────────────────────────────────────────────────────────────────

const styles: Record<string, CSSProperties> = {
  container: { position: 'relative', width: '100%', height: '100%', overflow: 'hidden', userSelect: 'none' },
  banner: {
    position: 'absolute',
    top: 12,
    left: 12,
    zIndex: 50,
    background: 'rgba(255, 255, 255, 0.92)',
    backdropFilter: 'blur(8px)',
    border: '1px solid rgba(60, 70, 110, 0.15)',
    borderRadius: 10,
    padding: '10px 14px',
    boxShadow: '0 4px 16px rgba(20, 24, 40, 0.08)',
    maxWidth: 340,
  },
  modeDot: { width: 8, height: 8, borderRadius: '50%', display: 'inline-block', flexShrink: 0 },
  clearBtn: {
    marginLeft: 4,
    fontSize: 10.5,
    border: '1px solid rgba(60, 70, 110, 0.25)',
    background: 'transparent',
    color: '#4a5270',
    borderRadius: 6,
    padding: '2px 8px',
    cursor: 'pointer',
  },
  controls: { position: 'absolute', top: 12, right: 12, zIndex: 50, display: 'flex', gap: 6 },
  controlBtn: {
    width: 30,
    height: 30,
    borderRadius: 8,
    border: '1px solid rgba(60, 70, 110, 0.15)',
    background: 'rgba(255, 255, 255, 0.92)',
    backdropFilter: 'blur(8px)',
    cursor: 'pointer',
    fontSize: 15,
    color: '#23283c',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 2px 8px rgba(20, 24, 40, 0.06)',
  },
  legend: {
    position: 'absolute',
    bottom: 12,
    left: 12,
    zIndex: 50,
    background: 'rgba(255, 255, 255, 0.92)',
    backdropFilter: 'blur(8px)',
    border: '1px solid rgba(60, 70, 110, 0.15)',
    borderRadius: 10,
    padding: '10px 12px',
    fontSize: 10.5,
    color: '#3a4058',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  hoverCard: {
    position: 'absolute',
    top: 86,
    left: 12,
    zIndex: 60,
    background: 'rgba(255, 255, 255, 0.96)',
    border: '1px solid rgba(60, 70, 110, 0.18)',
    borderRadius: 10,
    padding: '10px 12px',
    fontSize: 11,
    maxWidth: 280,
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
    boxShadow: '0 6px 20px rgba(20, 24, 40, 0.12)',
    pointerEvents: 'none',
  },
  emptyState: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 13,
    color: '#8b93ad',
  },
};