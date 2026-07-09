import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

// ── API types ────────────────────────────────────────────────────────────────
interface SimilarClusterInfo {
  cluster_id: string;
  label: string;
  similarity: number;
  chunk_count: number;
  color?: string | null;
}

interface Props {
    /**The currently selected cluster ID. When set, computes its neighborhood. */
  selectedClusterId: string | null;
    /**Callback when user clicks an orbiting node (promotes it to center) */
  onNodeClick: (clusterId: string) => void;
    /**Current note chunk index for tooltip */
  activeNoteKey?: string | null;
    /**Map of cluster_id → color for coloring the center node */
  clusterColors?: Record<string, string>;
}

// ── Constants ────────────────────────────────────────────────────────────────
const SIMILARITY_THRESHOLD = 0.35;
const MIN_SIMILARITY_FOR_NODE = 0.15;
const MAX_NODES = 30;
const NODE_MIN_RADIUS = 14;
const NODE_MAX_RADIUS = 36;
const CENTER_NODE_RADIUS = 56;
const SIMILARITY_TO_RADIUS_FACTOR = 0.6; // how much similarity affects visual radius

const DEFAULT_CENTER_COLOR = '#1e40af';

// ── Shared SVG style for the hub container ───────────────────────────────────
const svgStyle: React.CSSProperties = {
  display: 'block',
  width: '100%',
  height: '100%',
  background: '#fafbfc',
  borderRadius: 8,
  flex: 1,
};

// ── Word-wrap a cluster name to fit inside the center circle ──────────────────
function wrapText(text: string, maxLineWidthPx: number, fontSizePx: number): string[] {
  const avgCharWidth = fontSizePx * 0.55;
  const maxCharsPerLine = Math.floor(maxLineWidthPx / avgCharWidth);
  if (maxCharsPerLine <= 0) return [text];
  if (text.length <= maxCharsPerLine) return [text];

  const words = text.split(/\s+/);
  const lines: string[] = [];
  let currentLine = '';

  for (const word of words) {
    const candidate = currentLine ? `${currentLine} ${word}` : word;
    if (candidate.length > maxCharsPerLine && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = candidate;
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines;
}

// ── Tooltip with text-wrapping for orbiting node hover ───────────────────────
interface TooltipBoxProps {
  cx: number;
  cy: number;
  node: { x: number; y: number; radius: number; label: string; chunk_count: number; similarity: number };
}
function TooltipBox({ cx, cy, node }: TooltipBoxProps) {
  const fontSize = 10;
  const avgCharW = fontSize * 0.55;
  const maxTooltipW = 200;
  const textAreaW = maxTooltipW - 20;
  const maxChars = Math.floor(textAreaW / avgCharW);
  const words = node.label.split(/\s+/);
  const lines: string[] = [];
  let cur = '';
  for (const w of words) {
    const cand = cur ? `${cur} ${w}` : w;
    if (cand.length > maxChars && cur) { lines.push(cur); cur = w; } else { cur = cand; }
  }
  if (cur) lines.push(cur);
  const lineH = fontSize + 3;
  const metaLine = `${node.chunk_count} chunks · ${Math.round(node.similarity * 100)}% similar`;
  const totalHeight = lines.length * lineH + lineH + 14;
  const boxX = (cx + node.x) - maxTooltipW / 2;
  const boxY = (cy + node.y) - node.radius - totalHeight - 6;
  return (
    <g>
      <rect x={boxX} y={boxY} width={maxTooltipW} height={totalHeight} rx={6} fill="#1f2937" opacity={0.95} />
      {lines.map((line: string, li: number) => (
        <text key={li} x={cx + node.x} y={boxY + 8 + (li + 1) * lineH} textAnchor="middle" fill="#fff" fontSize={fontSize} fontWeight={600}>
          {line}
        </text>
      ))}
      <text x={cx + node.x} y={boxY + 8 + (lines.length + 1) * lineH} textAnchor="middle" fill="#d1d5db" fontSize={9}>
        {metaLine}
      </text>
    </g>
  );
}

// ── RadialHub component ──────────────────────────────────────────────────────
export const RadialSimilarityHub: React.FC<Props> = ({
  selectedClusterId,
  onNodeClick,
  activeNoteKey,
  clusterColors: propClusterColors,
}) => {
  const [similarData, setSimilarData] = useState<SimilarClusterInfo[]>([]);
  const [targetLabel, setTargetLabel] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Fetch similar clusters whenever selection changes
  useEffect(() => {
    if (!selectedClusterId) {
      setSimilarData([]);
      setTargetLabel('');
      return;
    }

    const fetchSimilar = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(
          `http://localhost:8000/similar_clusters?cluster_id=${encodeURIComponent(selectedClusterId)}&min_similarity=${SIMILARITY_THRESHOLD}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setTargetLabel(data.target_label || selectedClusterId);

         // Filter and sort
        const filtered: SimilarClusterInfo[] = (data.similar_clusters || [])
           .filter((c: SimilarClusterInfo) => c.similarity >= MIN_SIMILARITY_FOR_NODE)
           .sort((a: SimilarClusterInfo, b: SimilarClusterInfo) => b.similarity - a.similarity)
           .slice(0, MAX_NODES);

        setSimilarData(filtered);
      } catch (e: any) {
        setError(e.message || 'Failed to load similar clusters');
      } finally {
        setLoading(false);
      }
    };

    fetchSimilar();
  }, [selectedClusterId]);

  // Resolve center node color from the clusterColors map or API data
  const centerNodeColor = useMemo(() => {
    if (!selectedClusterId) return DEFAULT_CENTER_COLOR;
    return propClusterColors?.[selectedClusterId] || similarData.find((c) => c.cluster_id === selectedClusterId)?.color || DEFAULT_CENTER_COLOR;
  }, [selectedClusterId, propClusterColors, similarData]);

  // Compute the effective color for any orbiting node — prefer its own API color, fall back to assigned palette
  const getNodeColor = useCallback(
    (node: SimilarClusterInfo): string => {
      return node.color || propClusterColors?.[node.cluster_id] || `hsl(${210 + node.cluster_id.charCodeAt(0) * 15}, 75%, 45%)`;
    },
    [propClusterColors],
  );

  // Compute positions using polar → cartesian with full-circle multi-ring layout
  const nodePositions = useMemo(() => {
    if (similarData.length === 0) return [];

    // Sort by similarity descending, then arrange in order around the circle
    const sorted = [...similarData].sort((a, b) => b.similarity - a.similarity);

    const maxSim = sorted[0]?.similarity ?? 1;
    const minSim = sorted[sorted.length - 1]?.similarity ?? 0;
    const simRange = maxSim - minSim || 1;

    // SVG dimensions for layout
    const svgWidth = 500;
    const centerX = svgWidth / 2;

    // Compute minimum angular gap to prevent dot overlap at the outermost radius
    const rMin = CENTER_NODE_RADIUS + NODE_MAX_RADIUS * 1.2 + 8;
    const maxOrbitRadius = Math.min(200, (svgWidth / 2) - 50);

    // Minimum angle needed between dots at the outermost orbit to prevent overlap
    const minAngularGap = (Math.PI * 2) / (Math.floor((2 * Math.PI * maxOrbitRadius) / (NODE_MAX_RADIUS * 2.5)));

    // Determine nodes per ring based on available angular space
    const nodesPerRing = Math.max(3, Math.floor(minAngularGap > 0 ? (Math.PI * 2) / minAngularGap : sorted.length));

    return sorted.map((node, i) => {
      const n = sorted.length;
      // Determine which ring this node goes into
      const ringIndex = Math.floor(i / nodesPerRing);
      const positionInRing = i % nodesPerRing;
      const totalRings = Math.ceil(n / nodesPerRing);

      // Radius scales down by 18% per inner ring
      const radiusScale = Math.pow(0.82, ringIndex);
      const baseRadius = rMin + (maxOrbitRadius - rMin) * radiusScale;

      // Distribute evenly across full 360° within this ring
      const anglePerNode = (Math.PI * 2) / nodesPerRing;
      // Start from top (-π/2) for intuitive placement
      const startAngle = -Math.PI / 2;
      const baseAngle = startAngle + positionInRing * anglePerNode;

      // Add slight jitter based on similarity (more similar = tighter to top)
      const similarityNorm = (node.similarity - minSim) / simRange;
      const angleJitter = (1 - similarityNorm) * 0.2;
      const finalAngle = baseAngle + (Math.random() - 0.5) * angleJitter;

      // Radius: more similar = closer to center (smaller radius within the ring)
      const normalizedSim = (node.similarity - minSim) / simRange;
      const ringRMin = rMin;
      const ringRMax = baseRadius;
      const radius = ringRMax - normalizedSim * (ringRMax - ringRMin);

      const x = radius * Math.cos(finalAngle);
      const y = radius * Math.sin(finalAngle);

      // Node size based on chunk count and similarity
      const maxChunks = sorted.reduce((max, nd) => Math.max(max, nd.chunk_count), 1);
      const minChunks = sorted.reduce((min, nd) => Math.min(min, nd.chunk_count), 1);
      const chunkRange = maxChunks - minChunks || 1;
      const chunkNorm = (node.chunk_count - minChunks) / chunkRange;
      const nodeRadius = NODE_MIN_RADIUS + chunkNorm * (NODE_MAX_RADIUS - NODE_MIN_RADIUS) * SIMILARITY_TO_RADIUS_FACTOR;

      return {
        ...node,
        x,
        y,
        radius: nodeRadius,
        angle: finalAngle,
        similarityNorm,
        ringIndex,
      };
    });
  }, [similarData]);

  const handleNodeClick = useCallback((clusterId: string) => {
    onNodeClick(clusterId);
  }, [onNodeClick]);

  // SVG dimensions — responsive to container (use % for width/height, viewBox for aspect ratio)
  const svgWidth = '100%';
  const viewBoxWidth = 500;
  const viewBoxHeight = 460;
  const cx = viewBoxWidth / 2;
  const cy = viewBoxHeight / 2;

  // Compute maxOrbitRadius for grid circles (matches outermost ring)
  const maxOrbitRadius = Math.min(200, (viewBoxWidth / 2) - 50);

  if (loading) {
    return (
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
        preserveAspectRatio="xMidYMid meet"
        style={svgStyle}
      >
        <text x={cx} y={cy} textAnchor="middle" fill="#9ca3af" fontSize="14">
          Loading similar clusters…
        </text>
      </svg>
    );
  }

  if (error) {
    return (
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
        preserveAspectRatio="xMidYMid meet"
        style={svgStyle}
      >
        <text x={cx} y={cy} textAnchor="middle" fill="#dc2626" fontSize="14">
          Error: {error}
        </text>
      </svg>
    );
  }

  if (!selectedClusterId) {
    return (
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
        preserveAspectRatio="xMidYMid meet"
        style={svgStyle}
      >
        <text x={cx} y={cy - 10} textAnchor="middle" fill="#6b7280" fontSize="14">
          Select a cluster to view its semantic neighborhood
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill="#9ca3af" fontSize="11">
          Click a cluster in the meta-cluster tree to begin
        </text>
      </svg>
    );
  }

  // Draw connections from center to orbiting nodes
  const connections = nodePositions.map((node) => ({
    x1: cx,
    y1: cy,
    x2: cx + node.x,
    y2: cy + node.y,
    similarity: node.similarityNorm,
  }));

  // Gradient definitions for connection lines
  const gradients = nodePositions.map((node, i) => (
    <linearGradient key={`grad-${i}`} id={`hub-grad-${i}`} x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stopColor={centerNodeColor} stopOpacity="0.6" />
      <stop offset="100%" stopColor={node.color || '#6366f1'} stopOpacity="0.3" />
    </linearGradient>
  ));

   // Compute center label sizing and wrapping
   const centerLabelFontSize = Math.min(10, CENTER_NODE_RADIUS * 0.22);
  const availableWidthForText = (CENTER_NODE_RADIUS * 2) - 8; // minus small padding
  const wrappedCenterLines = wrapText(targetLabel, availableWidthForText, centerLabelFontSize);
  const lineCount = wrappedCenterLines.length;
  const totalTextHeight = lineCount * (centerLabelFontSize + 2); // line height + small gap
  const centerYOffset = -totalTextHeight / 2 + (centerLabelFontSize / 2);
  const centerTextStartY = cy + centerYOffset;

  return (
    <svg
      ref={svgRef}
      width="100%"
      height="100%"
      viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
      preserveAspectRatio="xMidYMid meet"
      style={svgStyle}
    >
      <defs>{gradients}</defs>

      {/* Grid circles */}
      {[0.33, 0.66, 1.0].map((factor, i) => (
        <circle
          key={`grid-${i}`}
          cx={cx}
          cy={cy}
          r={maxOrbitRadius * factor}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={1}
          strokeDasharray={i === 2 ? 'none' : '4 4'}
        />
      ))}

      {/* Connection lines */}
      {connections.map((conn, i) => (
        <line
          key={`conn-${i}`}
          x1={conn.x1}
          y1={conn.y1}
          x2={conn.x2}
          y2={conn.y2}
          stroke={`url(#hub-grad-${i})`}
          strokeWidth={1 + conn.similarity * 2}
          opacity={0.4 + conn.similarity * 0.3}
        />
      ))}

      {/* Center node — selected cluster */}
      <g style={{ cursor: 'default' }}>
        <circle cx={cx} cy={cy} r={CENTER_NODE_RADIUS} fill={centerNodeColor} />
        {wrappedCenterLines.map((line, li) => (
          <text
            key={li}
            x={cx}
            y={centerTextStartY + li * (centerLabelFontSize + 2)}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#fff"
            fontSize={centerLabelFontSize}
            fontWeight={700}
          >
            {line}
          </text>
        ))}
        <text
          x={cx}
          y={cy + CENTER_NODE_RADIUS - 14}
          textAnchor="middle"
          fill="#bfdbfe"
          fontSize="9"
        >
          {selectedClusterId}
        </text>
      </g>

      {/* Orbiting nodes */}
      {nodePositions.map((node) => {
        const isHovered = hoveredNodeId === node.cluster_id;
        const nodeColor = getNodeColor(node);

        return (
          <g
            key={node.cluster_id}
            onClick={() => handleNodeClick(node.cluster_id)}
            onMouseEnter={() => setHoveredNodeId(node.cluster_id)}
            onMouseLeave={() => setHoveredNodeId(null)}
            style={{ cursor: 'pointer' }}
          >
            {/* Glow on hover */}
            {isHovered && (
              <circle
                cx={cx + node.x}
                cy={cy + node.y}
                r={(node.radius + 6) * 1.5}
                fill={nodeColor}
                opacity={0.2}
              />
            )}

            {/* Main circle */}
            <circle
              cx={cx + node.x}
              cy={cy + node.y}
              r={isHovered ? node.radius * 1.2 : node.radius}
              fill={nodeColor}
              stroke={isHovered ? '#fff' : 'none'}
              strokeWidth={isHovered ? 2 : 0}
              opacity={0.85 + node.similarityNorm * 0.15}
            />

            {/* Label */}
            <text
              x={cx + node.x}
              y={cy + node.y + node.radius + 14}
              textAnchor="middle"
              fill="#374151"
              fontSize={isHovered ? 11 : 10}
              fontWeight={isHovered ? 600 : 400}
            >
              {node.cluster_id.length > 8 ? node.cluster_id.slice(0, 6) + '…' : node.cluster_id}
            </text>

              {/* Similarity percentage on hover — full title, text-wrapped */}
              {isHovered && (
                <TooltipBox cx={cx} cy={cy} node={node} />
              )}
          </g>
        );
      })}
    </svg>
  );
};

export default RadialSimilarityHub;
