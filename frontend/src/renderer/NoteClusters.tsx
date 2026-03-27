import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Canvas, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

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

interface ClusterPointMeta {
  unique_key: string;
  title: string;
  chunk_index: number;
  cluster_id: string;
  cluster_label: string;
}

interface ClusterGroup {
  x: number[];
  y: number[];
  z: number[];
  customdata: ClusterPointMeta[];
  text: string[];
  clusterId?: string;
}

interface VisualPoint extends ClusterPointMeta {
  x: number;
  y: number;
  z: number;
  dotColor: string;
}

interface PointBucket {
  key: string;
  sizeMetric: number;
  color: string;
  points: VisualPoint[];
}

const colorToRgba = (color: THREE.Color, alpha: number) => {
  const r = Math.round(color.r * 255);
  const g = Math.round(color.g * 255);
  const b = Math.round(color.b * 255);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const mixColorWithWhite = (baseColor: string, whiteMix: number) => {
  const mixed = new THREE.Color(baseColor).lerp(new THREE.Color('#ffffff'), whiteMix);
  return mixed.getStyle();
};

const getDotSurfaceTint = (dotColor: string) => mixColorWithWhite(dotColor, 0.35);

const DotInstances = ({
  bucket,
  sphereRadius,
  onPointerOver,
  onPointerMove,
  onPointerOut,
  onClick,
}: {
  bucket: PointBucket;
  sphereRadius: number;
  onPointerOver: (event: ThreeEvent<PointerEvent>) => void;
  onPointerMove: (event: ThreeEvent<PointerEvent>) => void;
  onPointerOut: (event: ThreeEvent<PointerEvent>) => void;
  onClick: (event: ThreeEvent<MouseEvent>) => void;
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const temp = new THREE.Object3D();
    bucket.points.forEach((point, index) => {
      temp.position.set(point.x, point.y, point.z);
      temp.scale.setScalar(sphereRadius);
      temp.updateMatrix();
      mesh.setMatrixAt(index, temp.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
  }, [bucket.points, sphereRadius]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, bucket.points.length]}
      userData={{ points: bucket.points }}
      onPointerOver={onPointerOver}
      onPointerMove={onPointerMove}
      onPointerOut={onPointerOut}
      onClick={onClick}
    >
      <sphereGeometry args={[1, 12, 12]} />
      <meshBasicMaterial color={bucket.color} toneMapped={false} />
    </instancedMesh>
  );
};

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
          fontWeight: isSelected ? 700 : hasHits ? 600 : 400,
          color: isDimmed ? '#8c8c8c' : '#222',
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
  const [hoverSource, setHoverSource] = useState<'canvas' | 'list' | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<NoteContent | null>(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [selectedClusters, setSelectedClusters] = useState<Set<string>>(new Set());
  const [searchSelectedClusters, setSearchSelectedClusters] = useState<Set<string>>(new Set());
  const plotAreaRef = useRef<HTMLDivElement>(null);

  const isSearchMode = searchQuery.trim() !== '';

  const fetchNoteContent = async (
    title: string,
    chunk_index: number,
    initial_cluster_id?: string,
    initial_cluster_label?: string,
  ) => {
    setIsLoadingContent(true);
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/note_content?title=${encodeURIComponent(title)}&chunk_index=${chunk_index}`,
      );

      const newUniqueKey = `${title}_${chunk_index}`;
      const existingNode = data.find((d) => d.unique_key === newUniqueKey);

      const cluster_id = existingNode ? existingNode.cluster_id : initial_cluster_id;
      const cluster_label = existingNode ? existingNode.cluster_label : initial_cluster_label;

      setSelectedNode({
        ...response.data,
        cluster_id,
        cluster_label,
        unique_key: newUniqueKey,
      });
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoadingContent(false);
    }
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
      if (!debouncedQuery.trim()) {
        if (active) setSearchResults([]);
        return;
      }

      try {
        const response = await axios.get(
          `http://127.0.0.1:8000/search?q=${encodeURIComponent(debouncedQuery)}&limit=1000&max_distance=0.8`,
        );
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
    if (!isSearchMode) {
      setSearchSelectedClusters(new Set());
    }
  }, [isSearchMode]);

  const searchScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    searchResults.forEach((r) => map.set(r.unique_key, r.distance));
    return map;
  }, [searchResults]);

  const toggleClusterSelection = (label: string) => {
    const setter = isSearchMode ? setSearchSelectedClusters : setSelectedClusters;
    setter((prev) => {
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
    const processingGroups: Record<string, ClusterGroup> = {};
    let globalSumX = 0;
    let globalSumY = 0;
    let count = 0;

    if (data.length > 0) {
      data.forEach((point) => {
        const label = point.cluster_label || 'Unclustered';
        if (!processingGroups[label]) {
          processingGroups[label] = { x: [], y: [], z: [], customdata: [], text: [] };
        }
        processingGroups[label].x.push(point.umap_x);
        processingGroups[label].y.push(point.umap_y);
        processingGroups[label].z.push(point.umap_z);

        const total = point.total_chunks || '?';
        const cid = point.cluster_id && point.cluster_id !== '-1' ? point.cluster_id : label;
        if (!processingGroups[label].clusterId) {
          processingGroups[label].clusterId = cid;
        }

        processingGroups[label].text.push(
          `<b>${point.title}</b><br>Chunk ${point.chunk_index + 1} of ${total}<br>Cluster: ${cid}`,
        );

        processingGroups[label].customdata.push({
          unique_key: point.unique_key,
          title: point.title,
          chunk_index: point.chunk_index,
          cluster_id: cid,
          cluster_label: label,
        });

        globalSumX += point.umap_x;
        globalSumY += point.umap_y;
        count += 1;
      });
    }

    const centerX = count > 0 ? globalSumX / count : 0;
    const centerY = count > 0 ? globalSumY / count : 0;

    const processingColors: Record<string, string> = {};
    const processingTints: Record<string, string> = {};
    const processingHoverTints: Record<string, string> = {};
    const processingOpaqueTints: Record<string, string> = {};

    Object.keys(processingGroups).forEach((label) => {
      const points = processingGroups[label];
      const cx = points.x.reduce((a, b) => a + b, 0) / points.x.length;
      const cy = points.y.reduce((a, b) => a + b, 0) / points.y.length;

      const dx = cx - centerX;
      const dy = cy - centerY;
      let angle = (Math.atan2(dy, dx) * 180) / Math.PI;
      if (angle < 0) angle += 360;

      processingColors[label] = `hsl(${Math.round(angle)}, 75%, 45%)`;
    });

    Object.entries(processingColors).forEach(([label, baseColor]) => {
      const color = new THREE.Color(baseColor);
      processingTints[label] = colorToRgba(color, 0.25);
      processingHoverTints[label] = colorToRgba(color, 0.45);
      // Use dot color as the single source of truth, then lift toward white for surfaces.
      processingOpaqueTints[label] = mixColorWithWhite(baseColor, 0.35);
    });

    return {
      clusterGroups: processingGroups,
      clusterColors: processingColors,
      clusterTints: processingTints,
      clusterHoverTints: processingHoverTints,
      clusterOpaqueTints: processingOpaqueTints,
    };
  }, [data]);

  const clusterCentroids = useMemo(() => {
    const centroids = new Map<string, { x: number; y: number; z: number }>();
    Object.keys(clusterGroups).forEach((label) => {
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
    const sorted = Object.keys(clusterGroups);

    if (isSearchMode && searchResults.length > 0) {
      sorted.sort((a, b) => {
        const groupA = clusterGroups[a];
        const groupB = clusterGroups[b];

        let minDistA = Infinity;
        groupA.customdata.forEach((pointData) => {
          const score = searchScoreMap.get(pointData.unique_key);
          if (score !== undefined && score < minDistA) minDistA = score;
        });

        let minDistB = Infinity;
        groupB.customdata.forEach((pointData) => {
          const score = searchScoreMap.get(pointData.unique_key);
          if (score !== undefined && score < minDistB) minDistB = score;
        });

        if (minDistA === Infinity && minDistB === Infinity) return 0;
        if (minDistA === Infinity) return 1;
        if (minDistB === Infinity) return -1;

        return minDistA - minDistB;
      });
    } else if (!isSearchMode && selectedClusters.size > 0) {
      const selectedLabels = Array.from(selectedClusters).filter((label) => clusterCentroids.has(label));

      const minDistanceToSelection = (label: string) => {
        if (selectedClusters.has(label)) return -1;
        const current = clusterCentroids.get(label);
        if (!current || selectedLabels.length === 0) return Number.POSITIVE_INFINITY;

        let minDistance = Number.POSITIVE_INFINITY;
        selectedLabels.forEach((selectedLabel) => {
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
        const numA = parseInt(idA, 10);
        const numB = parseInt(idB, 10);
        if (!Number.isNaN(numA) && !Number.isNaN(numB)) {
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
    ? sortedLabels.filter((label) => activeSelectedClusters.has(label))
    : sortedLabels;

  const sceneBounds = useMemo(() => {
    if (data.length === 0) {
      return {
        center: new THREE.Vector3(0, 0, 0),
        radius: 10,
      };
    }

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    data.forEach((point) => {
      minX = Math.min(minX, point.umap_x);
      minY = Math.min(minY, point.umap_y);
      minZ = Math.min(minZ, point.umap_z);
      maxX = Math.max(maxX, point.umap_x);
      maxY = Math.max(maxY, point.umap_y);
      maxZ = Math.max(maxZ, point.umap_z);
    });

    const center = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;
    const radius = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5, 1);

    return { center, radius };
  }, [data]);

  const pointRenderColorMap = useMemo(() => {
    const map = new Map<string, string>();
    const hasSearchHits = searchResults.length > 0;

    Object.keys(clusterGroups).forEach((label) => {
      const group = clusterGroups[label];
      const baseColor = new THREE.Color(clusterColors[label] || '#4b5563');

      group.customdata.forEach((meta) => {
        const score = searchScoreMap.get(meta.unique_key);
        const isHit = score !== undefined;

        const color = baseColor.clone();
        if (hasSearchHits && !isHit) {
          color.lerp(new THREE.Color('#c4c4c4'), 0.65);
        }
        map.set(meta.unique_key, color.getStyle());
      });
    });

    return map;
  }, [clusterColors, clusterGroups, searchResults.length, searchScoreMap]);

  const { buckets, pointLookup } = useMemo(() => {
    const bucketMap = new Map<string, PointBucket>();
    const lookup = new Map<string, VisualPoint>();
    const hasSearchHits = searchResults.length > 0;

    visibleLabels.forEach((label) => {
      const group = clusterGroups[label];

      group.customdata.forEach((meta, index) => {
        const score = searchScoreMap.get(meta.unique_key);
        const isHit = score !== undefined;
        const isHovered = hoveredId === meta.unique_key;

        let size = 0.02;
        if (hasSearchHits && isHit) {
          const clamped = Math.max(0, Math.min(1, score));
          size = 0.018 + (1 - clamped) * 0.018;
        } else if (hasSearchHits && !isHit) {
          size = 0.012;
        }

        if (isHovered) {
          size = Math.max(size * 1.35, 0.024);
        }

        const dotColor = pointRenderColorMap.get(meta.unique_key) || (clusterColors[label] || '#4b5563');
        const quantizedSize = Math.max(0.008, Math.round(size * 1000) / 1000);
        const bucketKey = `${quantizedSize}|${dotColor}`;
        if (!bucketMap.has(bucketKey)) {
          bucketMap.set(bucketKey, {
            key: bucketKey,
            sizeMetric: quantizedSize,
            color: dotColor,
            points: [],
          });
        }

        const visualPoint: VisualPoint = {
          ...meta,
          x: group.x[index],
          y: group.y[index],
          z: group.z[index],
          dotColor,
        };
        lookup.set(meta.unique_key, visualPoint);

        const bucket = bucketMap.get(bucketKey)!;
        bucket.points.push(visualPoint);
      });
    });

    return {
      buckets: Array.from(bucketMap.values()).sort((a, b) => a.sizeMetric - b.sizeMetric),
      pointLookup: lookup,
    };
  }, [clusterColors, clusterGroups, hoveredId, pointRenderColorMap, searchResults.length, searchScoreMap, visibleLabels]);

  const hoveredPoint = hoveredId ? pointLookup.get(hoveredId) || null : null;
  const hoveredClusterColor = hoveredPoint ? getDotSurfaceTint(hoveredPoint.dotColor) : '#ffffff';

  const selectedNodeColor = useMemo(() => {
    if (!selectedNode) return '#ffffff';
    const dotColor = pointRenderColorMap.get(selectedNode.unique_key)
      || clusterColors[selectedNode.cluster_label || '']
      || '#ffffff';
    return getDotSurfaceTint(dotColor);
  }, [clusterColors, pointRenderColorMap, selectedNode]);

  const updateTooltipPosition = useCallback((nativeEvent: PointerEvent | MouseEvent) => {
    if (!plotAreaRef.current) return;
    const rect = plotAreaRef.current.getBoundingClientRect();
    setTooltipPosition({
      x: nativeEvent.clientX - rect.left,
      y: nativeEvent.clientY - rect.top,
    });
  }, []);

  const handleCanvasPointMove = useCallback(
    (point: VisualPoint, event: ThreeEvent<PointerEvent>) => {
      event.stopPropagation();
      setHoveredId(point.unique_key);
      setHoverSource('canvas');
      updateTooltipPosition(event.nativeEvent);
    },
    [updateTooltipPosition],
  );

  const resolvePointFromIntersections = useCallback(
    (intersections: ThreeEvent<PointerEvent | MouseEvent>['intersections']) => {
      if (!intersections || intersections.length === 0) {
        return null;
      }

      let best: VisualPoint | null = null;
      let bestDistance = Number.POSITIVE_INFINITY;

      intersections.forEach((intersection) => {
        const sourcePoints = intersection.object.userData?.points as VisualPoint[] | undefined;
        if (!sourcePoints) return;
        const hitIndex = intersection.instanceId ?? intersection.index;
        if (hitIndex === undefined || hitIndex === null) return;

        const point = sourcePoints[hitIndex];
        if (!point) return;

        if (intersection.distance < bestDistance) {
          best = point;
          bestDistance = intersection.distance;
        }
      });

      return best;
    },
    [],
  );

  const resolvePointFromEvent = useCallback(
    (event: ThreeEvent<PointerEvent | MouseEvent>) => {
      const points = (event.eventObject as THREE.Object3D).userData?.points as VisualPoint[] | undefined;
      if (points && event.instanceId !== undefined && event.instanceId !== null) {
        return points[event.instanceId] || null;
      }
      return resolvePointFromIntersections(event.intersections);
    },
    [resolvePointFromIntersections],
  );

  const handleCanvasPointClick = useCallback(
    (point: VisualPoint, event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation();
      setHoveredId(null);
      setHoverSource(null);
      fetchNoteContent(point.title, point.chunk_index, point.cluster_id, point.cluster_label);
    },
    [data],
  );

  const cameraPosition = useMemo(() => {
    const { center, radius } = sceneBounds;
    return [
      center.x + radius * 1.7,
      center.y + radius * 1.3,
      center.z + radius * 1.7,
    ] as [number, number, number];
  }, [sceneBounds]);

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {loading ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100%',
            color: '#333',
          }}
        >
          Loading visualization...
        </div>
      ) : (
        <div style={{ display: 'flex', width: '100%', height: '100%' }}>
          <div
            style={{
              width: '400px',
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              borderRight: '1px solid #e0e0e0',
              backgroundColor: '#f9f9f9',
              padding: '10px',
              boxSizing: 'border-box',
              zIndex: 10,
            }}
          >
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
                  boxSizing: 'border-box',
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
                  onMouseEnter={() => {
                    setHoveredId(result.unique_key);
                    setHoverSource('list');
                  }}
                  onMouseLeave={() => {
                    setHoveredId(null);
                    setHoverSource(null);
                  }}
                  onClick={() =>
                    fetchNoteContent(result.title, result.chunk_index, result.cluster_id, result.cluster_label)
                  }
                  style={{
                    padding: '12px',
                    marginBottom: '8px',
                    borderRadius: '6px',
                    backgroundColor:
                      hoveredId === result.unique_key
                        ? clusterHoverTints[result.cluster_label] || '#e6f7ff'
                        : clusterTints[result.cluster_label] || 'white',
                    border: '1px solid #eee',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                    overflowWrap: 'anywhere',
                    wordBreak: 'break-word',
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '4px', color: 'black' }}>
                    {result.title}
                    <span
                      style={{
                        fontWeight: 'normal',
                        color: '#555',
                        fontSize: '0.85em',
                        marginLeft: '6px',
                      }}
                    >
                      (Chunk {result.chunk_index + 1} of {result.total_chunks || '?'})
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: '0.8em',
                      color: '#444',
                      marginBottom: '6px',
                      fontStyle: 'italic',
                    }}
                  >
                    Cluster{' '}
                    {result.cluster_id && result.cluster_id !== '-1' ? result.cluster_id : '?'}: {result.cluster_label}
                  </div>
                  <div style={{ fontSize: '0.9em', color: '#555', lineHeight: '1.4' }}>{result.preview}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ flex: 1, position: 'relative', height: '100%', minWidth: 0 }} ref={plotAreaRef}>
            {selectedNode && (
              <div
                style={{
                  position: 'absolute',
                  top: '20px',
                  right: '20px',
                  width: '400px',
                  maxHeight: '80vh',
                  backgroundColor: selectedNodeColor,
                  border: '1px solid #ccc',
                  borderRadius: '8px',
                  padding: '20px',
                  boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                  zIndex: 100,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: '10px',
                  }}
                >
                  <h3 style={{ margin: 0, fontSize: '1.1em', wordBreak: 'break-word', color: '#333' }}>
                    {selectedNode.title}
                  </h3>
                  <button
                    onClick={closePopup}
                    style={{
                      background: 'none',
                      border: 'none',
                      fontSize: '1.5em',
                      cursor: 'pointer',
                      padding: '0 5px',
                      lineHeight: '0.8',
                      color: '#666',
                    }}
                  >
                    &times;
                  </button>
                </div>

                <div style={{ fontSize: '0.9em', color: '#555', marginBottom: '10px' }}>
                  Cluster {selectedNode.cluster_id && selectedNode.cluster_id !== '-1' ? selectedNode.cluster_id : '?'}:{' '}
                  {selectedNode.cluster_label}
                </div>

                <div
                  style={{
                    flex: 1,
                    overflowY: 'auto',
                    marginBottom: '15px',
                    whiteSpace: 'pre-wrap',
                    fontSize: '0.95em',
                    lineHeight: '1.5',
                    padding: '10px',
                    backgroundColor: 'rgba(255,255,255,0.5)',
                    borderRadius: '4px',
                    border: '1px solid rgba(0,0,0,0.05)',
                  }}
                >
                  {isLoadingContent ? 'Loading content...' : selectedNode.content}
                </div>

                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginTop: 'auto',
                  }}
                >
                  <button
                    onClick={handlePrevChunk}
                    disabled={isLoadingContent}
                    style={{
                      padding: '6px 12px',
                      cursor: 'pointer',
                      borderRadius: '4px',
                      border: '1px solid #ccc',
                      backgroundColor: '#fff',
                    }}
                  >
                    &lt; Prev
                  </button>
                  <span style={{ fontSize: '0.9em', color: '#444' }}>
                    Chunk {selectedNode.chunk_index + 1} of {selectedNode.total_chunks}
                  </span>
                  <button
                    onClick={handleNextChunk}
                    disabled={isLoadingContent}
                    style={{
                      padding: '6px 12px',
                      cursor: 'pointer',
                      borderRadius: '4px',
                      border: '1px solid #ccc',
                      backgroundColor: '#fff',
                    }}
                  >
                    Next &gt;
                  </button>
                </div>
              </div>
            )}

            {hoverSource === 'canvas' && hoveredPoint && (
              <div
                style={{
                  position: 'absolute',
                  left: tooltipPosition.x + 12,
                  top: tooltipPosition.y + 12,
                  pointerEvents: 'none',
                  zIndex: 120,
                  backgroundColor: hoveredClusterColor,
                  border: '1px solid rgba(0, 0, 0, 0.2)',
                  borderRadius: '6px',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.14)',
                  padding: '8px 10px',
                  maxWidth: '300px',
                  color: '#222',
                  fontSize: '12px',
                  lineHeight: 1.4,
                  overflowWrap: 'anywhere',
                }}
              >
                <div style={{ fontWeight: 700 }}>{hoveredPoint.title}</div>
                <div style={{ color: '#333' }}>Chunk {hoveredPoint.chunk_index + 1}</div>
                <div style={{ color: '#333' }}>
                  Cluster {hoveredPoint.cluster_id && hoveredPoint.cluster_id !== '-1' ? hoveredPoint.cluster_id : '?'}:{' '}
                  {hoveredPoint.cluster_label}
                </div>
              </div>
            )}

            <Canvas
              camera={{
                fov: 45,
                near: 0.1,
                far: Math.max(sceneBounds.radius * 15, 100),
                position: cameraPosition,
              }}
              onPointerMissed={() => {
                setHoveredId(null);
                setHoverSource(null);
              }}
              onPointerLeave={() => {
                setHoveredId(null);
                setHoverSource(null);
              }}
              style={{ width: '100%', height: '100%', background: 'white' }}
            >
              <ambientLight intensity={0.9} />
              <OrbitControls
                makeDefault
                enablePan
                enableZoom
                enableRotate
                target={[sceneBounds.center.x, sceneBounds.center.y, sceneBounds.center.z]}
              />

              {buckets.map((bucket) => {
                const sphereRadius = (bucket.sizeMetric / 0.02) * Math.max(sceneBounds.radius * 0.006, 0.04);
                return (
                  <DotInstances
                    key={`bucket-${bucket.key}-${bucket.points.length}`}
                    bucket={bucket}
                    sphereRadius={sphereRadius}
                    onPointerOver={(event: ThreeEvent<PointerEvent>) => {
                      const point = resolvePointFromEvent(event);
                      if (!point) return;
                      handleCanvasPointMove(point, event);
                    }}
                    onPointerMove={(event: ThreeEvent<PointerEvent>) => {
                      const point = resolvePointFromEvent(event);
                      if (!point) {
                        setHoveredId(null);
                        setHoverSource(null);
                        return;
                      }

                      handleCanvasPointMove(point, event);
                    }}
                    onPointerOut={(event: ThreeEvent<PointerEvent>) => {
                      event.stopPropagation();
                      if (event.intersections.length === 0) {
                        setHoveredId(null);
                        setHoverSource(null);
                      }
                    }}
                    onClick={(event: ThreeEvent<MouseEvent>) => {
                      const point = resolvePointFromEvent(event);
                      if (!point) return;

                      handleCanvasPointClick(point, event);
                    }}
                  />
                );
              })}
            </Canvas>
          </div>

          <div
            style={{
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
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            }}
          >
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
                cursor: pointer;
                padding: 2px 0;
                border-radius: 4px;
                transition: opacity 0.15s ease, background-color 0.15s ease;
              }
              .cluster-row:hover {
                background-color: rgba(0, 0, 0, 0.08);
              }
              .cluster-row.cluster-row-selected {
                background-color: rgba(0, 0, 0, 0.04);
              }
              .cluster-row.cluster-row-selected:hover {
                background-color: rgba(0, 0, 0, 0.1);
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
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '10px',
                minHeight: '30px',
              }}
            >
              {hasActiveClusterFilter ? (
                <button
                  onClick={clearActiveClusterFilter}
                  style={{
                    padding: '6px 10px',
                    height: '30px',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    fontSize: '11px',
                    cursor: 'pointer',
                    backgroundColor: '#fff',
                    color: '#333',
                    boxShadow: 'none',
                    opacity: 1,
                    transform: 'none',
                  }}
                >
                  Reset Full View
                </button>
              ) : (
                <div style={{ width: '112px', height: '30px', flexShrink: 0 }} />
              )}
              <span style={{ color: '#666', fontSize: '11px' }}>
                Showing {visibleLabels.length} cluster{visibleLabels.length === 1 ? '' : 's'}
              </span>
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {sortedLabels.map((label) => {
                const group = clusterGroups[label];
                const cid = group.clusterId || label;
                const hasHits = group.customdata.some((pointData) =>
                  searchScoreMap.has(pointData.unique_key),
                );
                const color = clusterColors[label];
                const isSelected = activeSelectedClusters.has(label);
                const isDimmed = hasActiveClusterFilter && !isSelected;

                return (
                  <div
                    key={label}
                    className={`cluster-row ${isSelected ? 'cluster-row-selected' : ''}`}
                    onClick={() => toggleClusterSelection(label)}
                    style={{
                      opacity: isDimmed ? 0.45 : 1,
                      cursor: 'pointer',
                    }}
                  >
                    <div
                      className="cluster-dot"
                      style={{ backgroundColor: color, opacity: isDimmed ? 0.5 : 1 }}
                    ></div>
                    <div
                      className="cluster-id"
                      style={{ color: isDimmed ? '#9a9a9a' : '#555', fontWeight: isSelected ? 700 : 600 }}
                    >
                      #{cid}
                    </div>
                    <ClusterLabel
                      label={label}
                      hasHits={hasHits}
                      isSelected={isSelected}
                      isDimmed={isDimmed}
                    />
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
