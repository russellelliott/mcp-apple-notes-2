import React, { useEffect, useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

interface NotePoint {
  unique_key: string;
  title: string;
  chunk_index: number;
  cluster_label: string;
  umap_x: number;
  umap_y: number;
}

interface SearchResult {
  unique_key: string;
  title: string;
  chunk_index: number;
  distance: number;
  cluster_label: string;
  preview: string;
}

export default function NoteClusters() {
  const [data, setData] = useState<NotePoint[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

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

  const handleSearch = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      const response = await axios.get(`http://127.0.0.1:8000/search?q=${encodeURIComponent(query)}&limit=20`);
      setSearchResults(response.data.results || []);
    } catch (error) {
      console.error('Error searching:', error);
    }
  };

  const searchScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    searchResults.forEach(r => map.set(r.unique_key, r.distance));
    return map;
  }, [searchResults]);

  // Logic calculation - always run it, but safely handle empty data
  const clusterGroups: { [key: string]: { x: number[]; y: number[]; ids: string[]; text: string[] } } = {};
  let globalSumX = 0;
  let globalSumY = 0;
  let count = 0;

  if (data.length > 0) {
    data.forEach(point => {
        const label = point.cluster_label || 'Unclustered';
        if (!clusterGroups[label]) {
            clusterGroups[label] = { x: [], y: [], ids: [], text: [] };
        }
        clusterGroups[label].x.push(point.umap_x);
        clusterGroups[label].y.push(point.umap_y);
        clusterGroups[label].ids.push(point.unique_key);
        clusterGroups[label].text.push(`<b>${point.title}</b><br>Chunk ${point.chunk_index}<br>Cluster: ${label}`);
        
        globalSumX += point.umap_x;
        globalSumY += point.umap_y;
        count++;
    });
  }
  
  const centerX = count > 0 ? globalSumX / count : 0;
  const centerY = count > 0 ? globalSumY / count : 0;

  const clusterColors: { [key: string]: string } = {};
  
  Object.keys(clusterGroups).forEach(label => {
      const points = clusterGroups[label];
      const cx = points.x.reduce((a, b) => a + b, 0) / points.x.length;
      const cy = points.y.reduce((a, b) => a + b, 0) / points.y.length;
      
      const dx = cx - centerX;
      const dy = cy - centerY;
      let angle = (Math.atan2(dy, dx) * 180 / Math.PI);
      if (angle < 0) angle += 360;
      
      clusterColors[label] = `hsl(${Math.round(angle)}, 75%, 45%)`;
  });

  const plotData: any[] = Object.keys(clusterGroups).map(label => {
      const group = clusterGroups[label];
      
      const markerSizes: number[] = [];
      const markerOpacities: number[] = [];
      const lineWidths: number[] = [];
      const lineColors: string[] = [];

      group.ids.forEach(id => {
          const score = searchScoreMap.get(id);
          const isHovered = id === hoveredId;
          
          let size = 8;
          let opacity = 0.6;
          let lw = 0;
          let lc = 'white';

          if (score !== undefined) {
             size = 15 + (score * 20); 
             opacity = 0.9;
          } else if (searchResults.length > 0) {
             opacity = 0.1;
             size = 5;
          }

          if (isHovered) {
              size = size * 1.5;
              if (size < 12) size = 12;
              opacity = 1.0;
              lw = 2;
              lc = '#333';
          }
          
          markerSizes.push(size);
          markerOpacities.push(opacity);
          lineWidths.push(lw);
          lineColors.push(lc);
      });

      return {
        x: group.x,
        y: group.y,
        text: group.text,
        customdata: group.ids,
        mode: 'markers',
        type: 'scatter',
        name: label,
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
                            style={{
                                padding: '12px',
                                marginBottom: '8px',
                                borderRadius: '6px',
                                backgroundColor: (hoveredId === result.unique_key) ? '#e6f7ff' : 'white',
                                border: '1px solid #eee',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                            }}
                        >
                             <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#007bff' }}>
                                {result.title}
                                <span style={{ fontWeight: 'normal', color: '#999', fontSize: '0.85em', marginLeft: '6px' }}>
                                    (Chunk {result.chunk_index})
                                </span>
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
                <Plot
                    data={plotData}
                    layout={{
                        title: 'Notes Landscape',
                        autosize: true,
                        hovermode: 'closest',
                        showlegend: true,
                        paper_bgcolor: 'white',
                        plot_bgcolor: 'white',
                        margin: { t: 40, r: 20, b: 20, l: 20 },
                        xaxis: {
                            showgrid: false,
                            zeroline: false,
                            showticklabels: false
                        },
                        yaxis: {
                            showgrid: false,
                            zeroline: false,
                            showticklabels: false
                        },
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
