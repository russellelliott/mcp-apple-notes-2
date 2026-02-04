import React, { useEffect, useState } from 'react';
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

const NoteClusters: React.FC = () => {
  const [data, setData] = useState<NotePoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/points');
        setData(response.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', color: '#333' }}>
        Loading visualization data...
      </div>
    );
  }

  // 1. Calculate Cluster Centroids & Global Center for Continuous Coloring
  const clusterGroups: { [key: string]: { x: number[]; y: number[]; text: string[] } } = {};
  let globalSumX = 0;
  let globalSumY = 0;
  let count = 0;

  data.forEach(point => {
      const label = point.cluster_label || 'Unclustered';
      if (!clusterGroups[label]) {
          clusterGroups[label] = { x: [], y: [], text: [] };
      }
      clusterGroups[label].x.push(point.umap_x);
      clusterGroups[label].y.push(point.umap_y);
      clusterGroups[label].text.push(`<b>${point.title}</b><br>Chunk ${point.chunk_index}<br>Cluster: ${label}`);
      
      globalSumX += point.umap_x;
      globalSumY += point.umap_y;
      count++;
  });
  
  const centerX = count > 0 ? globalSumX / count : 0;
  const centerY = count > 0 ? globalSumY / count : 0;

  // 2. Assign Colors based on Centroid Angle relative to Global Center
  const clusterColors: { [key: string]: string } = {};
  
  Object.keys(clusterGroups).forEach(label => {
      const points = clusterGroups[label];
      const cx = points.x.reduce((a, b) => a + b, 0) / points.x.length;
      const cy = points.y.reduce((a, b) => a + b, 0) / points.y.length;
      
      // Calculate angle in degrees (0-360)
      const dx = cx - centerX;
      const dy = cy - centerY;
      let angle = (Math.atan2(dy, dx) * 180 / Math.PI);
      if (angle < 0) angle += 360;
      
      // Map angle to Hue in HSL (0-360)
      clusterColors[label] = `hsl(${Math.round(angle)}, 75%, 45%)`;
  });

  const plotData: any[] = Object.keys(clusterGroups).map(label => ({
      x: clusterGroups[label].x,
      y: clusterGroups[label].y,
      text: clusterGroups[label].text,
      mode: 'markers',
      type: 'scatter',
      name: label,
      marker: { 
        size: 10,
        opacity: 0.7,
        color: clusterColors[label]
      },
      hoverinfo: 'text'
  }));

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
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
  );
};

export default NoteClusters;
