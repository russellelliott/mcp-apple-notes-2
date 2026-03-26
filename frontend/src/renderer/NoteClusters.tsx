import React, { useEffect, useState, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

interface NotePoint {
  unique_key: string;
  title: string;
  chunk_index: number;
  total_chunks?: number;
  cluster_id?: string;
  cluster_label: string;
  umap_x: number;
  umap_y: number;
  umap_z: number;
}

interface SearchResult {
  unique_key: string;
  title: string;
  chunk_index: number;
  total_chunks?: number;
  distance: number;
  cluster_id?: string;
  cluster_label: string;
  preview: string;
}

interface NoteContent {
  title: string;
  chunk_index: number;
  content: string;
  total_chunks: number;
  unique_key: string;
  cluster_id?: string;
  cluster_label?: string;
}

const ClusterLabel = ({
  label,
  hasHits,
  isSelected,
  isDimmed,
}: {
  label: string;
  hasHits: boolean;
  isSelected: boolean;
  isDimmed: boolean;
}) => {
    const [isOverflowing, setIsOverflowing] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLSpanElement>(null);

    const checkOverflow = () => {
        if (containerRef.current && contentRef.current) {
            setIsOverflowing(contentRef.current.scrollWidth > containerRef.current.clientWidth);
        }
    };

    useEffect(() => {
        checkOverflow();
        window.addEventListener('resize', checkOverflow);
        return () => window.removeEventListener('resize', checkOverflow);
    }, [label]);

    return (
        <div
            className={`cluster-label-container ${isOverflowing ? 'overflowing' : ''}`}
            title={label}
            ref={containerRef}
            onMouseEnter={checkOverflow}
        >
             <span
                className="cluster-label-content"
              style={{
                fontWeight: isSelected ? 700 : (hasHits ? 600 : 400),
                color: isDimmed ? '#8c8c8c' : '#222'
              }}
                ref={contentRef}
             >
                 {label}
             </span>
        </div>
    );
};

export default function NoteClusters() {
  const [data, setData] = useState<NotePoint[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NoteContent | null>(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [selectedClusters, setSelectedClusters] = useState<Set<string>>(new Set());
  const [searchSelectedClusters, setSearchSelectedClusters] = useState<Set<string>>(new Set());

  const isSearchMode = searchQuery.trim() !== '';

  const fetchNoteContent = async (title: string, chunk_index: number, initial_cluster_id?: string, initial_cluster_label?: string) => {
      setIsLoadingContent(true);
      try {
          const response = await axios.get(`http://127.0.0.1:8000/note_content?title=${encodeURIComponent(title)}&chunk_index=${chunk_index}`);

          // Try to find the node in our dataset to get the correct cluster info for this specific chunk
          // This is important because different chunks of the same note could theoretically be in different clusters,
          // or we just want to be consistent.
          const newUniqueKey = `${title}_${chunk_index}`;
          const existingNode = data.find(d => d.unique_key === newUniqueKey);

          const cluster_id = existingNode ? existingNode.cluster_id : initial_cluster_id;
          const cluster_label = existingNode ? existingNode.cluster_label : initial_cluster_label;

          setSelectedNode({
              ...response.data,
              cluster_id,
              cluster_label,
              unique_key: newUniqueKey
          });
      } catch (err) {
          console.error(err);
      } finally {
          setIsLoadingContent(false);
      }
  };

  const handlePlotClick = (event: any) => {
    const point = event.points[0];
    const { title, chunk_index, cluster_id, cluster_label } = point.customdata;

    // Set hoveredId to null to avoid confusing interactions
    setHoveredId(null);

    fetchNoteContent(title, chunk_index, cluster_id, cluster_label);
  };

  const handleNextChunk = () => {
      if (!selectedNode) return;
      let nextIndex = selectedNode.chunk_index + 1;
      if (nextIndex >= selectedNode.total_chunks) nextIndex = 0;
      fetchNoteContent(selectedNode.title, nextIndex, selectedNode.cluster_id, selectedNode.cluster_label);
  };

  const handlePrevChunk = () => {
      if (!selectedNode) return;
      let prevIndex = selectedNode.chunk_index - 1;
      if (prevIndex < 0) prevIndex = selectedNode.total_chunks - 1;
      fetchNoteContent(selectedNode.title, prevIndex, selectedNode.cluster_id, selectedNode.cluster_label);
  };

  const closePopup = () => setSelectedNode(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/points');
        // Ensure response.data is an array
        setData(Array.isArray(response.data) ? response.data : []);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearchQuery(val);
    if (val === '') {
      setDebouncedQuery('');
      setSearchResults([]);
    }
  };

  useEffect(() => {
    // If the search query is empty, we don't need a debounce timer
    // because we handled it immediately in handleSearch for instant feedback.
    if (searchQuery === '') return;

    const handler = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 2000);

    return () => {
      clearTimeout(handler);
    };
  }, [searchQuery]);

  useEffect(() => {
    let active = true;

    const runSearch = async () => {
      // If empty query, clear results immediately (though handleSearch does this too)
      if (!debouncedQuery.trim()) {
        if (active) setSearchResults([]);
        return;
      }

      try {
        const response = await axios.get(`http://127.0.0.1:8000/search?q=${encodeURIComponent(debouncedQuery)}&limit=1000&max_distance=0.8`);
        if (active) {
          setSearchResults(response.data.results || []);
        }
      } catch (error) {
        console.error('Error searching:', error);
      }
    };

    runSearch();

    return () => {
      active = false;
    };
  }, [debouncedQuery]);

  useEffect(() => {
    // Search-mode cluster filtering is temporary and should not persist
    // once the query is cleared.
    if (!isSearchMode) {
      setSearchSelectedClusters(new Set());
    }
  }, [isSearchMode]);

  const searchScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    searchResults.forEach(r => map.set(r.unique_key, r.distance));
    return map;
  }, [searchResults]);

  const toggleClusterSelection = (label: string) => {
    const setter = isSearchMode ? setSearchSelectedClusters : setSelectedClusters;
    setter(prev => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  const clearActiveClusterFilter = () => {
    if (isSearchMode) {
      setSearchSelectedClusters(new Set());
    } else {
      setSelectedClusters(new Set());
    }
  };

  const { clusterGroups, clusterColors, clusterTints, clusterHoverTints, clusterOpaqueTints } = useMemo(() => {
    // Logic calculation - always run it, but safely handle empty data
    const processingGroups: { [key: string]: { x: number[]; y: number[]; z: number[]; customdata: any[]; text: string[]; clusterId?: string } } = {};
    let globalSumX = 0;
    let globalSumY = 0;
    let count = 0;

    if (data.length > 0) {
      data.forEach(point => {
          const label = point.cluster_label || 'Unclustered';
          if (!processingGroups[label]) {
              processingGroups[label] = { x: [], y: [], z: [], customdata: [], text: [] };
          }
          processingGroups[label].x.push(point.umap_x);
          processingGroups[label].y.push(point.umap_y);
          processingGroups[label].z.push(point.umap_z);

          const total = point.total_chunks || '?';
          // Use cluster_id if available for shorter display, but fallback to label for unclustered
          const cid = (point.cluster_id && point.cluster_id !== '-1') ? point.cluster_id : label;
          if (!processingGroups[label].clusterId) {
             processingGroups[label].clusterId = cid;
          }
          processingGroups[label].text.push(`<b>${point.title}</b><br>Chunk ${point.chunk_index + 1} of ${total}<br>Cluster: ${cid}`);

          // Store full context in customdata for click handling
          processingGroups[label].customdata.push({
              unique_key: point.unique_key,
              title: point.title,
              chunk_index: point.chunk_index,
              cluster_id: cid,
              cluster_label: label
          });

          globalSumX += point.umap_x;
          globalSumY += point.umap_y;
          count++;
      });
    }

    const centerX = count > 0 ? globalSumX / count : 0;
    const centerY = count > 0 ? globalSumY / count : 0;

    const processingColors: { [key: string]: string } = {};
    const processingTints: { [key: string]: string } = {};
    const processingHoverTints: { [key: string]: string } = {};
    const processingOpaqueTints: { [key: string]: string } = {};

    Object.keys(processingGroups).forEach(label => {
        const points = processingGroups[label];
        const cx = points.x.reduce((a, b) => a + b, 0) / points.x.length;
        const cy = points.y.reduce((a, b) => a + b, 0) / points.y.length;

        const dx = cx - centerX;
        const dy = cy - centerY;
        let angle = (Math.atan2(dy, dx) * 180 / Math.PI);
        if (angle < 0) angle += 360;

        processingColors[label] = `hsl(${Math.round(angle)}, 75%, 45%)`;
        processingTints[label] = `hsla(${Math.round(angle)}, 75%, 45%, 0.25)`;
        processingHoverTints[label] = `hsla(${Math.round(angle)}, 75%, 45%, 0.45)`;
        // Opaque version of the tint (approximate mixing 25% color with 75% white)
        // L=45% * 0.25 + 100% * 0.75 = ~86%
        processingOpaqueTints[label] = `hsl(${Math.round(angle)}, 75%, 92%)`;
    });

    return { clusterGroups: processingGroups, clusterColors: processingColors, clusterTints: processingTints, clusterHoverTints: processingHoverTints, clusterOpaqueTints: processingOpaqueTints };
  }, [data]);

  const clusterCentroids = useMemo(() => {
    const centroids = new Map<string, { x: number; y: number; z: number }>();
    Object.keys(clusterGroups).forEach(label => {
      const group = clusterGroups[label];
      if (!group || group.x.length === 0) return;
      const x = group.x.reduce((sum, value) => sum + value, 0) / group.x.length;
      const y = group.y.reduce((sum, value) => sum + value, 0) / group.y.length;
      const z = group.z.reduce((sum, value) => sum + value, 0) / group.z.length;
      centroids.set(label, { x, y, z });
    });
    return centroids;
  }, [clusterGroups]);

  const sortedLabels = useMemo(() => {
    let sorted = Object.keys(clusterGroups);

    if (isSearchMode && searchResults.length > 0) {
        // Sort by relevance (closest distance first)
        sorted.sort((a, b) => {
            const groupA = clusterGroups[a];
            const groupB = clusterGroups[b];

            // Find best (minimum) distance in each group
            let minDistA = Infinity;
            groupA.customdata.forEach(data => {
                const s = searchScoreMap.get(data.unique_key);
                if (s !== undefined && s < minDistA) minDistA = s;
            });

            let minDistB = Infinity;
            groupB.customdata.forEach(data => {
                const s = searchScoreMap.get(data.unique_key);
                if (s !== undefined && s < minDistB) minDistB = s;
            });

            if (minDistA === Infinity && minDistB === Infinity) return 0;
            if (minDistA === Infinity) return 1;
            if (minDistB === Infinity) return -1;

            return minDistA - minDistB;
        });
    } else if (!isSearchMode && selectedClusters.size > 0) {
        // Selected clusters first, then remaining clusters by nearest centroid
        // distance to any selected cluster.
        const selectedLabels = Array.from(selectedClusters).filter(label => clusterCentroids.has(label));

        const minDistanceToSelection = (label: string) => {
          if (selectedClusters.has(label)) return -1;
          const current = clusterCentroids.get(label);
          if (!current || selectedLabels.length === 0) return Number.POSITIVE_INFINITY;

          let minDistance = Number.POSITIVE_INFINITY;
          selectedLabels.forEach(selectedLabel => {
            const selectedCentroid = clusterCentroids.get(selectedLabel);
            if (!selectedCentroid) return;
            const dx = current.x - selectedCentroid.x;
            const dy = current.y - selectedCentroid.y;
            const dz = current.z - selectedCentroid.z;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (distance < minDistance) minDistance = distance;
          });
          return minDistance;
        };

        sorted.sort((a, b) => {
          const aSelected = selectedClusters.has(a);
          const bSelected = selectedClusters.has(b);

          if (aSelected && !bSelected) return -1;
          if (!aSelected && bSelected) return 1;

          if (aSelected && bSelected) {
            const idA = clusterGroups[a].clusterId || a;
            const idB = clusterGroups[b].clusterId || b;
            return String(idA).localeCompare(String(idB), undefined, { numeric: true });
          }

          const distA = minDistanceToSelection(a);
          const distB = minDistanceToSelection(b);
          if (distA !== distB) return distA - distB;

          const idA = clusterGroups[a].clusterId || a;
          const idB = clusterGroups[b].clusterId || b;
          return String(idA).localeCompare(String(idB), undefined, { numeric: true });
        });
    } else {
         sorted.sort((a, b) => {
             const idA = clusterGroups[a].clusterId || a;
             const idB = clusterGroups[b].clusterId || b;
             const numA = parseInt(idA);
             const numB = parseInt(idB);
             if (!isNaN(numA) && !isNaN(numB)) {
                 return numA - numB;
             }
             return String(idA).localeCompare(String(idB));
        });
    }
    return sorted;
  }, [clusterCentroids, clusterGroups, isSearchMode, searchScoreMap, searchResults.length, selectedClusters]);

  const activeSelectedClusters = isSearchMode ? searchSelectedClusters : selectedClusters;
  const hasActiveClusterFilter = activeSelectedClusters.size > 0;
  const visibleLabels = hasActiveClusterFilter
    ? sortedLabels.filter(label => activeSelectedClusters.has(label))
    : sortedLabels;

  const plotData: any[] = useMemo(() => {
    return visibleLabels.map(label => {
      const group = clusterGroups[label];

      const markerSizes: number[] = [];
      const markerOpacities: number[] = [];
      const lineWidths: number[] = [];
      const lineColors: string[] = [];

      let hasHits = false;

      group.customdata.forEach(data => {
          const id = data.unique_key;
          const score = searchScoreMap.get(id);
          if (score !== undefined) hasHits = true;
          const isHovered = id === hoveredId;

          let size = 8;
          let opacity = 0.6;
          let lw = 0;
          // Default line color to transparent so no outline is visible
          let lc = 'rgba(0,0,0,0)';

          if (score !== undefined) {
             size = 15 + (score * 20);
             opacity = 0.9;
             // Highlight search results slightly if desired, or keep no outline
             // lw = 1; lc = 'white';
          } else if (searchResults.length > 0) {
             opacity = 0.1;
             size = 5;
          }

          if (isHovered) {
              size = size * 1.5;
              if (size < 12) size = 12;
              opacity = 1.0;
              lw = 2;
              // Add a dark outline on hover for contrast
              lc = '#333';
          }

          markerSizes.push(size);
          markerOpacities.push(opacity);
          lineWidths.push(lw);
          lineColors.push(lc);
      });

      const cid = group.clusterId || label;
      let displayName = label;
      if (cid && cid !== '-1' && cid !== label) {
          displayName = `Cluster ${cid}: ${label}`;
      } else if (cid && cid !== '-1' && cid === label) {
           // If label is "1" and cid is "1"
           displayName = `Cluster ${cid}`;
      }

      if (hasHits && searchResults.length > 0) {
          displayName = `<b>${displayName}</b>`;
      }

      return {
        x: group.x,
        y: group.y,
        z: group.z,
        text: group.text,
        customdata: group.customdata,
        mode: 'markers',
        type: 'scatter3d',
        name: displayName,
        marker: {
            size: markerSizes,
            opacity: markerOpacities,
            color: clusterColors[label],
            line: {
                color: lineColors,
                width: lineWidths
            }
        },
        hoverinfo: 'text'
      };
    });
  }, [clusterGroups, clusterColors, hoveredId, searchResults.length, searchScoreMap, visibleLabels]);

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: '#333' }}>
            Loading visualization...
        </div>
      ) : (
        <div style={{ display: 'flex', width: '100%', height: '100%' }}>
            {/* Sidebar */}
            <div style={{
                width: '400px',
                flexShrink: 0,
                display: 'flex',
                flexDirection: 'column',
                borderRight: '1px solid #e0e0e0',
                backgroundColor: '#f9f9f9',
                padding: '10px',
                boxSizing: 'border-box',
                zIndex: 10
            }}>
                <div style={{ marginBottom: '15px' }}>
                    <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>Search Notes</h3>
                    <input
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={handleSearch}
                        style={{
                            width: '100%',
                            padding: '10px',
                            fontSize: '14px',
                            borderRadius: '4px',
                            border: '1px solid #ccc',
                            boxSizing: 'border-box'
                        }}
                    />
                </div>

                <div style={{ flex: 1, overflowY: 'auto' }}>
                    {searchResults.length === 0 && searchQuery && (
                        <div style={{ color: '#666', fontStyle: 'italic', padding: '10px' }}>No results found.</div>
                    )}
                    {searchResults.map((result) => (
                        <div
                            key={result.unique_key}
                            onMouseEnter={() => setHoveredId(result.unique_key)}
                            onMouseLeave={() => setHoveredId(null)}
                            onClick={() => fetchNoteContent(result.title, result.chunk_index, result.cluster_id, result.cluster_label)}
                            style={{
                                padding: '12px',
                                marginBottom: '8px',
                                borderRadius: '6px',
                                backgroundColor: (hoveredId === result.unique_key) ? (clusterHoverTints[result.cluster_label] || '#e6f7ff') : (clusterTints[result.cluster_label] || 'white'),
                                border: '1px solid #eee',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                overflowWrap: 'anywhere',
                                wordBreak: 'break-word'
                            }}
                        >
                             <div style={{ fontWeight: 'bold', marginBottom: '4px', color: 'black' }}>
                                {result.title}
                                <span style={{ fontWeight: 'normal', color: '#555', fontSize: '0.85em', marginLeft: '6px' }}>
                                    (Chunk {result.chunk_index + 1} of {result.total_chunks || '?'})
                                </span>
                             </div>
                             <div style={{ fontSize: '0.8em', color: '#444', marginBottom: '6px', fontStyle: 'italic' }}>
                                Cluster {(result.cluster_id && result.cluster_id !== '-1') ? result.cluster_id : '?'}: {result.cluster_label}
                             </div>
                             <div style={{ fontSize: '0.9em', color: '#555', lineHeight: '1.4' }}>
                                {result.preview}
                             </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Plot Area */}
            <div style={{ flex: 1, position: 'relative', height: '100%', minWidth: 0 }}>
                {selectedNode && (
                  <div style={{
                      position: 'absolute',
                      top: '20px',
                      right: '20px',
                      width: '400px',
                      maxHeight: '80vh',
                      backgroundColor: clusterOpaqueTints[selectedNode.cluster_label || ''] || 'white',
                      border: '1px solid #ccc',
                      borderRadius: '8px',
                      padding: '20px',
                      boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                      zIndex: 100,
                      display: 'flex',
                      flexDirection: 'column',
                      overflow: 'hidden'
                  }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                          <h3 style={{ margin: 0, fontSize: '1.1em', wordBreak: 'break-word', color: '#333' }}>{selectedNode.title}</h3>
                          <button onClick={closePopup} style={{ background: 'none', border: 'none', fontSize: '1.5em', cursor: 'pointer', padding: '0 5px', lineHeight: '0.8', color: '#666' }}>&times;</button>
                      </div>

                      <div style={{ fontSize: '0.9em', color: '#555', marginBottom: '10px' }}>
                         Cluster {(selectedNode.cluster_id && selectedNode.cluster_id !== '-1') ? selectedNode.cluster_id : '?'}: {selectedNode.cluster_label}
                      </div>

                      <div style={{ flex: 1, overflowY: 'auto', marginBottom: '15px', whiteSpace: 'pre-wrap', fontSize: '0.95em', lineHeight: '1.5', padding: '10px', backgroundColor: 'rgba(255,255,255,0.5)', borderRadius: '4px', border: '1px solid rgba(0,0,0,0.05)' }}>
                          {isLoadingContent ? 'Loading content...' : selectedNode.content}
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto' }}>
                          <button onClick={handlePrevChunk} disabled={isLoadingContent} style={{ padding: '6px 12px', cursor: 'pointer', borderRadius: '4px', border: '1px solid #ccc', backgroundColor: '#fff' }}>&lt; Prev</button>
                          <span style={{ fontSize: '0.9em', color: '#444' }}>Chunk {selectedNode.chunk_index + 1} of {selectedNode.total_chunks}</span>
                          <button onClick={handleNextChunk} disabled={isLoadingContent} style={{ padding: '6px 12px', cursor: 'pointer', borderRadius: '4px', border: '1px solid #ccc', backgroundColor: '#fff' }}>Next &gt;</button>
                      </div>
                  </div>
                )}

                <Plot
                    data={plotData}
                    onClick={handlePlotClick}
                    layout={{
                        title: 'Notes Landscape (3D)',
                        autosize: true,
                        hovermode: 'closest',
                        showlegend: false,
                        paper_bgcolor: 'white',
                        scene: {
                            xaxis: { title: 'X', showgrid: false, zeroline: false, showticklabels: false },
                            yaxis: { title: 'Y', showgrid: false, zeroline: false, showticklabels: false },
                            zaxis: { title: 'Z', showgrid: false, zeroline: false, showticklabels: false },
                            aspectmode: 'cube',
                            camera: {
                                projection: { type: 'orthographic' }
                            }
                        },
                        margin: { t: 40, r: 20, b: 20, l: 20 },
                        // legend: { ... } - hidden
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                />
            </div>
            {/* Custom Legend Column */}
            <div style={{
                width: '450px',
                flexShrink: 0,
                display: 'flex',
                flexDirection: 'column',
                borderLeft: '1px solid #e0e0e0',
                backgroundColor: '#f9f9f9',
                padding: '10px',
                boxSizing: 'border-box',
                zIndex: 10,
                fontSize: '11px',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
            }}>
                <style>{`
                    @keyframes marquee-scroll {
                        0% { transform: translateX(0); }
                        100% { transform: translateX(-100%); }
                    }
                    .cluster-label-container {
                        overflow: hidden;
                        white-space: nowrap;
                        text-overflow: ellipsis;
                        flex: 1;
                        padding-left: 5px;
                    }
                    .cluster-label-content {
                        display: inline-block;
                    }
                    .cluster-label-container.overflowing:hover {
                         text-overflow: clip;
                    }
                    .cluster-label-container.overflowing:hover .cluster-label-content {
                        animation: marquee-scroll 5s linear infinite;
                        padding-right: 20px;
                    }
                    .cluster-row {
                        display: flex;
                        align-items: center;
                        margin-bottom: 4px;
                        cursor: default;
                        padding: 2px 0;
                    }
                    .cluster-id {
                        width: 30px;
                        text-align: right;
                        margin-right: 5px;
                        font-weight: bold;
                        color: #555;
                        flex-shrink: 0;
                    }
                    .cluster-dot {
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        margin-right: 5px;
                        flex-shrink: 0;
                    }
                `}</style>
                <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>Clusters</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                  <button
                    onClick={clearActiveClusterFilter}
                    disabled={!hasActiveClusterFilter}
                    style={{
                      padding: '6px 10px',
                      border: '1px solid #ccc',
                      borderRadius: '4px',
                      fontSize: '11px',
                      cursor: hasActiveClusterFilter ? 'pointer' : 'not-allowed',
                      backgroundColor: hasActiveClusterFilter ? '#fff' : '#f2f2f2',
                      color: hasActiveClusterFilter ? '#333' : '#999',
                      boxShadow: 'none',
                      opacity: 1,
                      transform: 'none'
                    }}
                  >
                    Reset Full View
                  </button>
                  {hasActiveClusterFilter && (
                    <span style={{ color: '#666', fontSize: '11px' }}>
                      Showing {activeSelectedClusters.size} cluster{activeSelectedClusters.size === 1 ? '' : 's'}
                    </span>
                  )}
                </div>
                <div style={{ flex: 1, overflowY: 'auto' }}>
                    {sortedLabels.map(label => {
                        const group = clusterGroups[label];
                        const cid = group.clusterId || label;
                        const hasHits = group.customdata.some(d => searchScoreMap.has(d.unique_key));
                        const color = clusterColors[label];
                    const isSelected = activeSelectedClusters.has(label);
                    const isDimmed = hasActiveClusterFilter && !isSelected;

                        return (
                      <div
                        key={label}
                        className="cluster-row"
                        onClick={() => toggleClusterSelection(label)}
                        style={{
                          opacity: isDimmed ? 0.45 : 1,
                          cursor: 'pointer',
                          backgroundColor: isSelected ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                          borderRadius: '4px',
                          transition: 'opacity 0.15s ease, background-color 0.15s ease'
                        }}
                      >
                        <div className="cluster-dot" style={{ backgroundColor: color, opacity: isDimmed ? 0.5 : 1 }}></div>
                        <div className="cluster-id" style={{ color: isDimmed ? '#9a9a9a' : '#555', fontWeight: isSelected ? 700 : 600 }}>#{cid}</div>
                        <ClusterLabel label={label} hasHits={hasHits} isSelected={isSelected} isDimmed={isDimmed} />
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
      )}
    </div>
  );
}
