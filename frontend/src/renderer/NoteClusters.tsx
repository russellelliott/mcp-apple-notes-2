import React, { useEffect, useState, useMemo } from 'react';
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

export default function NoteClusters() {
  const [data, setData] = useState<NotePoint[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NoteContent | null>(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);

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

  const searchScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    searchResults.forEach(r => map.set(r.unique_key, r.distance));
    return map;
  }, [searchResults]);

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

  const plotData: any[] = useMemo(() => {
    let sortedLabels = Object.keys(clusterGroups);

    if (searchResults.length > 0) {
        // Sort by relevance (closest distance first)
        sortedLabels.sort((a, b) => {
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
    } else {
         sortedLabels.sort((a, b) => {
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

    return sortedLabels.map(label => {
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
  }, [clusterGroups, clusterColors, searchScoreMap, hoveredId, searchResults.length]);

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
                width: '350px',
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
            <div style={{ flex: 1, position: 'relative', height: '100%' }}>
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
                        showlegend: true,
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
                        legend: {
                            orientation: 'v',
                            y: 1,
                            x: 1,
                            xanchor: 'left',
                            yanchor: 'top',
                            bgcolor: 'rgba(255,255,255,0.8)'
                        }
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                />
            </div>
        </div>
      )}
    </div>
  );
}
