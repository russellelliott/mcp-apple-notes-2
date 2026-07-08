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
}

// ── Constants ────────────────────────────────────────────────────────────────
const SIMILARITY_THRESHOLD = 0.35;
const MIN_SIMILARITY_FOR_NODE = 0.15;
const MAX_NODES = 30;
const NODE_MIN_RADIUS = 8;
const NODE_MAX_RADIUS = 24;
const CENTER_NODE_RADIUS = 32;
const SIMILARITY_TO_RADIUS_FACTOR = 0.6; // how much similarity affects visual radius

// ── RadialHub component ──────────────────────────────────────────────────────
export const RadialSimilarityHub: React.FC<Props> = ({
  selectedClusterId,
  onNodeClick,
  activeNoteKey,
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
          .sort((a, b) => b.similarity - a.similarity)
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

  // Compute positions using polar → cartesian
  const nodePositions = useMemo(() => {
    if (similarData.length === 0) return [];

    // Sort by similarity descending, then arrange in order around the circle
    const sorted = [...similarData].sort((a, b) => b.similarity - a.similarity);

    // The radius for orbiting nodes: closer = more similar → closer to center
    const maxSim = sorted[0]?.similarity ?? 1;
    const minSim = sorted[sorted.length - 1]?.similarity ?? 0;
    const simRange = maxSim - minSim || 1;

    return sorted.map((node, i) => {
      const n = sorted.length;
      // Distribute around the circle, but cluster by angle similarity
      const baseAngle = (2 * Math.PI * i) / n;

      // Jitter angle based on similarity (more similar = tighter to top)
      const similarityNorm = (node.similarity - minSim) / simRange;
      const angleJitter = (1 - similarityNorm) * 0.5;
      const finalAngle = baseAngle + (Math.random() - 0.5) * angleJitter;

      // Radius: more similar = closer to center (smaller radius)
      // Map similarity [minSim, maxSim] → radius [maxR, minR]
      const rMin = CENTER_NODE_RADIUS + NODE_MIN_RADIUS + 8;
      const rMax = Math.min(200, 300 - n * 4); // Dynamic outer radius based on node count
      const normalizedSim = (node.similarity - minSim) / simRange;
      const radius = rMax - normalizedSim * (rMax - rMin);

      const x = radius * Math.cos(finalAngle);
      const y = radius * Math.sin(finalAngle);

      // Node size based on chunk count and similarity
      const maxChunks = sorted.reduce((max, n) => Math.max(max, n.chunk_count), 1);
      const minChunks = sorted.reduce((min, n) => Math.min(min, n.chunk_count), 1);
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
      };
    });
   }, [similarData]);

  const handleNodeClick = useCallback((clusterId: string) => {
    onNodeClick(clusterId);
   }, [onNodeClick]);

  // SVG dimensions (responsive to container)
  const svgWidth = 500;
  const svgHeight = 500;
  const centerX = svgWidth / 2;
  const centerY = svgHeight / 2;

  if (loading) {
    return (
       <svg ref={svgRef} width={svgWidth} height={svgHeight} style={{ background: '#fafbfc', borderRadius: 8 }}>
        <text x={centerX} y={centerY} textAnchor="middle" fill="#9ca3af" fontSize="14">
          Loading similar clusters…
        </text>
       </svg>
     );
   }

  if (error) {
    return (
       <svg width={svgWidth} height={svgHeight} style={{ background: '#fafbfc', borderRadius: 8 }}>
        <text x={centerX} y={centerY} textAnchor="middle" fill="#dc2626" fontSize="14">
          Error: {error}
        </text>
       </svg>
     );
   }

  if (!selectedClusterId) {
    return (
       <svg width={svgWidth} height={svgHeight} style={{ background: '#fafbfc', borderRadius: 8 }}>
        <text x={centerX} y={centerY - 10} textAnchor="middle" fill="#6b7280" fontSize="14">
          Select a cluster to view its semantic neighborhood
        </text>
        <text x={centerX} y={centerY + 14} textAnchor="middle" fill="#9ca3af" fontSize="11">
          Click a cluster in the meta-cluster tree to begin
        </text>
       </svg>
     );
   }

   // Draw connections from center to orbiting nodes
  const connections = nodePositions.map((node) => ({
    x1: centerX,
    y1: centerY,
    x2: centerX + node.x,
    y2: centerY + node.y,
    similarity: node.similarityNorm,
   }));

   // Gradient definitions for connection lines
  const gradients = nodePositions.map((node, i) => (
     <linearGradient key={`grad-${i}`} id={`hub-grad-${i}`} x1="0%" y1="0%" x2="100%" y2="0%">
       <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.6" />
       <stop offset="100%" stopColor={node.color || '#6366f1'} stopOpacity="0.3" />
     </linearGradient>
   ));

  return (
     <svg
      ref={svgRef}
      width={svgWidth}
      height={svgHeight}
      style={{ background: '#fafbfc', borderRadius: 8 }}
     >
       <defs>{gradients}</defs>

       {/* Grid circles */}
       {[0.33, 0.66, 1.0].map((factor, i) => (
         <circle
          key={`grid-${i}`}
          cx={centerX}
          cy={centerY}
          r={200 * factor}
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
       <g
        onClick={() => {}} // center doesn't promote, it's the anchor
        style={{ cursor: 'default' }}
       >
         <circle
          cx={centerX}
          cy={centerY}
          r={CENTER_NODE_RADIUS}
          fill="#1e40af"
          stroke="#3b82f6"
          strokeWidth={3}
         />
         <text
          x={centerX}
          y={centerY - 4}
          textAnchor="middle"
          fill="#fff"
          fontSize="10"
          fontWeight={700}
         >
           {targetLabel.length > 20 ? targetLabel.slice(0, 18) + '…' : targetLabel}
         </text>
         <text
          x={centerX}
          y={centerY + 10}
          textAnchor="middle"
          fill="#bfdbfe"
          fontSize="9"
         >
           {selectedClusterId}
         </text>
       </g>

       {/* Orbiting nodes */}
       {nodePositions.map((node, i) => {
        const isHovered = hoveredNodeId === node.cluster_id;
        const nodeColor = node.color || `hsl(${210 + i * 15}, 75%, 45%)`;
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
                cx={centerX + node.x}
                cy={centerY + node.y}
                r={(node.radius + 6) * 1.5}
                fill={nodeColor}
                opacity={0.2}
               />
             )}

             {/* Main circle */}
             <circle
              cx={centerX + node.x}
              cy={centerY + node.y}
              r={isHovered ? node.radius * 1.2 : node.radius}
              fill={nodeColor}
              stroke={isHovered ? '#fff' : 'none'}
              strokeWidth={isHovered ? 2 : 0}
              opacity={0.85 + node.similarityNorm * 0.15}
             />

             {/* Label */}
             <text
              x={centerX + node.x}
              y={centerY + node.y + node.radius + 12}
              textAnchor="middle"
              fill="#374151"
              fontSize={isHovered ? 10 : 9}
              fontWeight={isHovered ? 600 : 400}
             >
               {node.cluster_id.length > 8 ? node.cluster_id.slice(0, 6) + '…' : node.cluster_id}
             </text>

             {/* Similarity percentage on hover */}
             {isHovered && (
               <text
                x={centerX + node.x}
                y={centerY + node.y + 4}
                textAnchor="middle"
                fill="#fff"
                fontSize="9"
                fontWeight={600}
               >
                 {Math.round(node.similarity * 100)}%
               </text>
             )}

             {/* Tooltip */}
             {isHovered && (
               <g>
                 <rect
                  x={(centerX + node.x) - 50}
                  y={(centerY + node.y) - node.radius - 30}
                  width={100}
                  height={28}
                  rx={4}
                  fill="#1f2937"
                  opacity={0.95}
                 />
                 <text
                  x={centerX + node.x}
                  y={(centerY + node.y) - node.radius - 18}
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="10"
                  fontWeight={600}
                 >
                   {node.label.length > 25 ? node.label.slice(0, 23) + '…' : node.label}
                 </text>
                 <text
                  x={centerX + node.x}
                  y={(centerY + node.y) - node.radius - 6}
                  textAnchor="middle"
                  fill="#d1d5db"
                  fontSize="9"
                 >
                   {node.chunk_count} chunks · {Math.round(node.similarity * 100)}% similar
                 </text>
               </g>
             )}
           </g>
         );
       })}

       {/* Similarity legend */}
       <g transform={`translate(${svgWidth - 90}, ${svgHeight - 60})`}>
         <rect width={80} height={50} rx={4} fill="#fff" stroke="#e5e7eb" strokeWidth={1} />
         <text x={40} y={16} textAnchor="middle" fill="#6b7280" fontSize="9" fontWeight={600}>
           Similarity
         </text>
         <line x1={15} y1={26} x2={65} y2={26} stroke="#1e40af" strokeWidth={3} />
         <text x={15} y={38} fill="#9ca3af" fontSize="8">High</text>
         <text x={65} y={38} fill="#9ca3af" fontSize="8" textAnchor="end">Low</text>
       </g>
     </svg>
   );
};

export default RadialSimilarityHub;
