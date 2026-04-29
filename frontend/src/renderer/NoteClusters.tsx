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
  base_topic_id?: string;
  display_topic_id?: string;
  cluster_label: string;
  umap_x: number;
  umap_y: number;
  umap_z: number;
  creation_date?: string;
  modification_date?: string;
}

interface SearchResult {
  unique_key: string;
  title: string;
  chunk_index: number;
  total_chunks?: number;
  distance: number;
  cluster_id?: string;
  base_topic_id?: string;
  display_topic_id?: string;
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
  base_topic_id?: string;
  display_topic_id?: string;
  cluster_label?: string;
  creation_date?: string;
  modification_date?: string;
}

interface SidebarChunkData {
  chunk_index: number;
  cluster_id: string;
  cluster_name: string;
  in_cluster: boolean;
  text?: string | null;
}

interface SidebarNoteData {
  note_key: string;
  title: string;
  creation_date?: string;
  modification_date?: string;
  chunks: SidebarChunkData[];
}

type ClusterOrderMode = 'spike' | 'momentum';
type SearchLegendOrderMode = 'results' | 'similarity';
type NotesSortMetric = 'modified' | 'size';
type SortDirection = 'desc' | 'asc';
type VisualizationMode = 'linear' | 'condensed';

interface ClusterPointMeta {
  unique_key: string;
  title: string;
  chunk_index: number;
  total_chunks?: number;
  cluster_id: string;
  base_topic_id?: string;
  display_topic_id?: string;
  cluster_label: string;
  creation_date?: string;
  modification_date?: string;
}

interface ClusterGroup {
  x: number[];
  y: number[];
  z: number[];
  customdata: ClusterPointMeta[];
  text: string[];
  clusterId?: string;
  clusterLabel?: string;
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
  glowOpacity: number;
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

const DOT_RADIUS_BASE = 0.016;

const compareTopicIds = (a: string, b: string) => {
  const aParts = String(a).split('.').map((part) => (part.match(/^-?\d+$/) ? Number(part) : part));
  const bParts = String(b).split('.').map((part) => (part.match(/^-?\d+$/) ? Number(part) : part));
  const maxLen = Math.max(aParts.length, bParts.length);

  for (let i = 0; i < maxLen; i += 1) {
    const aVal = aParts[i];
    const bVal = bParts[i];
    if (aVal === undefined) return -1;
    if (bVal === undefined) return 1;
    if (typeof aVal === 'number' && typeof bVal === 'number' && aVal !== bVal) {
      return aVal - bVal;
    }
    if (String(aVal) !== String(bVal)) {
      return String(aVal).localeCompare(String(bVal), undefined, { numeric: true });
    }
  }
  return 0;
};

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
  const glowMeshRef = useRef<THREE.InstancedMesh>(null);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    const glowMesh = glowMeshRef.current;
    if (!mesh) return;

    const temp = new THREE.Object3D();
    bucket.points.forEach((point, index) => {
      temp.position.set(point.x, point.y, point.z);
      temp.scale.setScalar(sphereRadius);
      temp.updateMatrix();
      mesh.setMatrixAt(index, temp.matrix);

      if (glowMesh) {
        temp.scale.setScalar(sphereRadius * 2.4);
        temp.updateMatrix();
        glowMesh.setMatrixAt(index, temp.matrix);
      }
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (glowMesh) {
      glowMesh.instanceMatrix.needsUpdate = true;
    }
  }, [bucket.points, sphereRadius]);

  return (
    <>
      {bucket.glowOpacity > 0 && (
        <instancedMesh ref={glowMeshRef} args={[undefined, undefined, bucket.points.length]}>
          <sphereGeometry args={[1, 10, 10]} />
          <meshBasicMaterial
            color={bucket.color}
            transparent
            opacity={bucket.glowOpacity}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </instancedMesh>
      )}

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
    </>
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

const SegmentedRail = ({
  chunks,
  activeClusterId,
  getClusterColor,
  onActiveDotClick,
  onInactiveDashClick,
}: {
  chunks: SidebarChunkData[];
  activeClusterId: string;
  getClusterColor: (clusterId: string) => string;
  onActiveDotClick: (chunk: SidebarChunkData) => void;
  onInactiveDashClick: (chunk: SidebarChunkData) => void;
}) => {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '3px',
        flexWrap: 'wrap',
        margin: '6px 0 8px 0',
        backgroundColor: '#000000',
        borderRadius: '6px',
        padding: '4px 6px',
      }}
    >
      {chunks.map((chunk) => {
        const isActive = chunk.cluster_id === activeClusterId;
        const symbol = isActive ? '●' : '−';
        return (
          <button
            type="button"
            key={`rail-${chunk.chunk_index}-${chunk.cluster_id}`}
            onClick={() => (isActive ? onActiveDotClick(chunk) : onInactiveDashClick(chunk))}
            title={`Chunk ${chunk.chunk_index + 1} | Cluster: ${chunk.cluster_name}`}
            style={{
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              color: isActive ? '#ffffff' : getClusterColor(chunk.cluster_id),
              fontSize: isActive ? '15px' : '14px',
              lineHeight: 1,
              padding: 0,
              margin: 0,
            }}
            aria-label={`Chunk ${chunk.chunk_index + 1} in cluster ${chunk.cluster_name}`}
          >
            {symbol}
          </button>
        );
      })}
    </div>
  );
};

export default function NoteClusters() {
  const [data, setData] = useState<NotePoint[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>('linear');
  const [hideOtherClusters, setHideOtherClusters] = useState(false);
  const [clusterOrderMode, setClusterOrderMode] = useState<ClusterOrderMode>('spike');
  const [notesSortMetric, setNotesSortMetric] = useState<NotesSortMetric>('modified');
  const [notesSortDirection, setNotesSortDirection] = useState<SortDirection>('desc');
  const [searchLegendOrderMode, setSearchLegendOrderMode] = useState<SearchLegendOrderMode>('results');
  const [sidebarNotes, setSidebarNotes] = useState<SidebarNoteData[]>([]);
  const [isLoadingSidebar, setIsLoadingSidebar] = useState(false);
  const [loadedSidebarCluster, setLoadedSidebarCluster] = useState<string | null>(null);
  const [pendingScrollNoteKey, setPendingScrollNoteKey] = useState<string | null>(null);
  const [pendingScrollNoteTitle, setPendingScrollNoteTitle] = useState<string | null>(null);
  const [pendingScrollTargetCluster, setPendingScrollTargetCluster] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [hoverSource, setHoverSource] = useState<'canvas' | 'list' | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0, containerWidth: 0 });
  const [selectedNode, setSelectedNode] = useState<NoteContent | null>(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [selectedClusters, setSelectedClusters] = useState<Set<string>>(new Set());
  const plotAreaRef = useRef<HTMLDivElement>(null);
  const sidebarCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const legendClusterRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const toggleClusterSelection = (label: string) => {
    setSelectedClusters((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  const fetchNoteContent = async (
    title: string,
    chunkIndex: number,
    initialClusterId?: string,
    initialClusterLabel?: string,
    initialDisplayTopicId?: string,
    initialBaseTopicId?: string,
    creationDate?: string,
    modificationDate?: string,
  ) => {
    setIsLoadingContent(true);
    try {
      const params = new URLSearchParams({
        title,
        chunk_index: String(chunkIndex),
      });
      if (creationDate) params.set('creation_date', creationDate);
      if (modificationDate) params.set('modification_date', modificationDate);

      const response = await axios.get(
        `http://127.0.0.1:8000/note_content?${params.toString()}`,
      );

      const newUniqueKey = `${title}_${chunkIndex}`;
      const existingNode = data.find((d) => d.unique_key === newUniqueKey);

      const cluster_id = existingNode ? existingNode.cluster_id : initialClusterId;
      const cluster_label = existingNode ? existingNode.cluster_label : initialClusterLabel;
      const display_topic_id = existingNode ? existingNode.display_topic_id : initialDisplayTopicId;
      const base_topic_id = existingNode ? existingNode.base_topic_id : initialBaseTopicId;

      setSelectedNode({
        ...response.data,
        cluster_id,
        display_topic_id,
        base_topic_id,
        cluster_label,
        creation_date: creationDate,
        modification_date: modificationDate,
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
    fetchNoteContent(
      selectedNode.title,
      nextIndex,
      selectedNode.cluster_id,
      selectedNode.cluster_label,
      selectedNode.display_topic_id,
      selectedNode.base_topic_id,
      selectedNode.creation_date,
      selectedNode.modification_date,
    );
  };

  const handlePrevChunk = () => {
    if (!selectedNode) return;
    let prevIndex = selectedNode.chunk_index - 1;
    if (prevIndex < 0) prevIndex = selectedNode.total_chunks - 1;
    fetchNoteContent(
      selectedNode.title,
      prevIndex,
      selectedNode.cluster_id,
      selectedNode.cluster_label,
      selectedNode.display_topic_id,
      selectedNode.base_topic_id,
      selectedNode.creation_date,
      selectedNode.modification_date,
    );
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
        if (active) {
          setSearchResults([]);
        }
      }
    };

    runSearch();

    return () => {
      active = false;
    };
  }, [debouncedQuery]);

  const searchScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    searchResults.forEach((r) => map.set(r.unique_key, r.distance));
    return map;
  }, [searchResults]);

  const { clusterGroups, clusterColors, clusterTints, clusterHoverTints, clusterOpaqueTints } = useMemo(() => {
    const processingGroups: Record<string, ClusterGroup> = {};
    let globalSumX = 0;
    let globalSumY = 0;
    let count = 0;

    if (data.length > 0) {
      data.forEach((point) => {
        const clusterKey = point.display_topic_id || point.cluster_id || '-1';
        const clusterLabel = point.cluster_label || `Cluster ${clusterKey}`;
        if (!processingGroups[clusterKey]) {
          processingGroups[clusterKey] = {
            x: [],
            y: [],
            z: [],
            customdata: [],
            text: [],
            clusterId: clusterKey,
            clusterLabel,
          };
        }
        processingGroups[clusterKey].x.push(point.umap_x);
        processingGroups[clusterKey].y.push(point.umap_y);
        processingGroups[clusterKey].z.push(point.umap_z);

        const total = point.total_chunks || '?';
        const cid = clusterKey;
        processingGroups[clusterKey].clusterLabel = clusterLabel;

        processingGroups[clusterKey].text.push(
          `<b>${point.title}</b><br>Chunk ${point.chunk_index + 1} of ${total}<br>Cluster: ${cid} (${clusterLabel})`,
        );

        processingGroups[clusterKey].customdata.push({
          unique_key: point.unique_key,
          title: point.title,
          chunk_index: point.chunk_index,
          total_chunks: point.total_chunks,
          cluster_id: cid,
          base_topic_id: point.base_topic_id,
          display_topic_id: point.display_topic_id,
          cluster_label: clusterLabel,
          creation_date: point.creation_date,
          modification_date: point.modification_date,
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

  const clusterOrderScores = useMemo(() => {
    const scores = new Map<string, { spike: number; momentum: number }>();
    const sums = new Map<string, { sum: number; count: number; max: number }>();

    data.forEach((point) => {
      const label = point.display_topic_id || point.cluster_id || '-1';
      const raw = point.modification_date || '';
      const ts = Date.parse(String(raw));
      if (!Number.isFinite(ts)) return;

      const current = sums.get(label) || { sum: 0, count: 0, max: Number.NEGATIVE_INFINITY };
      current.sum += ts;
      current.count += 1;
      current.max = Math.max(current.max, ts);
      sums.set(label, current);
    });

    Object.keys(clusterGroups).forEach((label) => {
      const stat = sums.get(label);
      if (!stat || stat.count === 0) {
        scores.set(label, { spike: Number.NEGATIVE_INFINITY, momentum: Number.NEGATIVE_INFINITY });
      } else {
        scores.set(label, { spike: stat.max, momentum: stat.sum / stat.count });
      }
    });

    return scores;
  }, [clusterGroups, data]);

  const sortedLabels = useMemo(() => {
    const sorted = Object.keys(clusterGroups);

    const distanceToNearest = (label: string, anchors: string[]) => {
      const current = clusterCentroids.get(label);
      if (!current || anchors.length === 0) return Number.POSITIVE_INFINITY;

      let minDistance = Number.POSITIVE_INFINITY;
      anchors.forEach((anchorLabel) => {
        const centroid = clusterCentroids.get(anchorLabel);
        if (!centroid) return;
        const dx = current.x - centroid.x;
        const dy = current.y - centroid.y;
        const dz = current.z - centroid.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (distance < minDistance) minDistance = distance;
      });
      return minDistance;
    };

    // When there are search results and similarity order is selected
    if (searchLegendOrderMode === 'similarity' && selectedClusters.size > 0 && searchResults.length > 0) {
      const selectedLabels = Array.from(selectedClusters).filter((label) => clusterCentroids.has(label));

      const minDistanceToSelection = (label: string) => {
        if (selectedClusters.has(label)) return -1;
        return distanceToNearest(label, selectedLabels);
      };

      sorted.sort((a, b) => {
        const aSelected = selectedClusters.has(a);
        const bSelected = selectedClusters.has(b);

        if (aSelected && !bSelected) return -1;
        if (!aSelected && bSelected) return 1;

        if (aSelected && bSelected) {
          const idA = clusterGroups[a].clusterId || a;
          const idB = clusterGroups[b].clusterId || b;
          return compareTopicIds(String(idA), String(idB));
        }

        const distA = minDistanceToSelection(a);
        const distB = minDistanceToSelection(b);
        if (distA !== distB) return distA - distB;

        const idA = clusterGroups[a].clusterId || a;
        const idB = clusterGroups[b].clusterId || b;
        return compareTopicIds(String(idA), String(idB));
      });
    } else if (searchLegendOrderMode === 'similarity' && searchResults.length > 0) {
      const hitLabels = sorted.filter((label) => {
        const group = clusterGroups[label];
        return group.customdata.some((pointData) => searchScoreMap.has(pointData.unique_key));
      });

      const distanceToNearestHitCluster = (label: string) => {
        return distanceToNearest(label, hitLabels);
      };

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

        if (minDistA === Infinity && minDistB === Infinity) {
          const clusterDistA = distanceToNearestHitCluster(a);
          const clusterDistB = distanceToNearestHitCluster(b);
          if (clusterDistA !== clusterDistB) return clusterDistA - clusterDistB;

          const idA = clusterGroups[a].clusterId || a;
          const idB = clusterGroups[b].clusterId || b;
          return compareTopicIds(String(idA), String(idB));
        }
        if (minDistA === Infinity) return 1;
        if (minDistB === Infinity) return -1;

        if (minDistA !== minDistB) return minDistA - minDistB;

        const idA = clusterGroups[a].clusterId || a;
        const idB = clusterGroups[b].clusterId || b;
        return compareTopicIds(String(idA), String(idB));
      });
    } else if (searchLegendOrderMode === 'results' && searchResults.length > 0) {
      const firstSeenRank = new Map<string, number>();
      searchResults.forEach((result, idx) => {
        const label = result.display_topic_id || result.cluster_id || '-1';
        if (clusterGroups[label] && !firstSeenRank.has(label)) {
          firstSeenRank.set(label, idx);
        }
      });

      const hitLabels = Array.from(firstSeenRank.keys());

      sorted.sort((a, b) => {
        const rankA = firstSeenRank.get(a);
        const rankB = firstSeenRank.get(b);

        if (rankA !== undefined && rankB !== undefined) return rankA - rankB;
        if (rankA !== undefined) return -1;
        if (rankB !== undefined) return 1;

        const distA = distanceToNearest(a, hitLabels);
        const distB = distanceToNearest(b, hitLabels);
        if (distA !== distB) return distA - distB;

        const idA = clusterGroups[a].clusterId || a;
        const idB = clusterGroups[b].clusterId || b;
        return compareTopicIds(String(idA), String(idB));
      });
    } else {
      sorted.sort((a, b) => {
        const scoreA = clusterOrderScores.get(a);
        const scoreB = clusterOrderScores.get(b);
        const metricA = clusterOrderMode === 'spike' ? scoreA?.spike ?? Number.NEGATIVE_INFINITY : scoreA?.momentum ?? Number.NEGATIVE_INFINITY;
        const metricB = clusterOrderMode === 'spike' ? scoreB?.spike ?? Number.NEGATIVE_INFINITY : scoreB?.momentum ?? Number.NEGATIVE_INFINITY;

        if (metricA !== metricB) return metricB - metricA;

        const idA = clusterGroups[a].clusterId || a;
        const idB = clusterGroups[b].clusterId || b;
        return compareTopicIds(String(idA), String(idB));
      });
    }
    return sorted;
  }, [
    clusterCentroids,
    clusterGroups,
    clusterOrderMode,
    clusterOrderScores,
    searchLegendOrderMode,
    selectedClusters,
    searchScoreMap,
    searchResults.length,
  ]);

  const hasActiveClusterFilter = selectedClusters.size > 0;
  const visibleLabels = hideOtherClusters && hasActiveClusterFilter
    ? sortedLabels.filter((label) => selectedClusters.has(label))
    : sortedLabels;

  useEffect(() => {
    let active = true;

    const fetchSidebar = async () => {
      // Fetch notes for the first selected cluster if any are selected
      if (selectedClusters.size === 0) {
        if (active) {
          setSidebarNotes([]);
          setLoadedSidebarCluster(null);
        }
        return;
      }

      const clusterToFetch = Array.from(selectedClusters)[0];
      setIsLoadingSidebar(true);
      try {
        const response = await axios.get(
          `http://127.0.0.1:8000/cluster_sidebar?active_cluster_id=${encodeURIComponent(clusterToFetch)}`,
        );
        if (active) {
          setSidebarNotes(Array.isArray(response.data?.notes) ? response.data.notes : []);
          setLoadedSidebarCluster(clusterToFetch);
        }
      } catch (error) {
        console.error('Error loading sidebar notes:', error);
        if (active) {
          setSidebarNotes([]);
          setLoadedSidebarCluster(null);
        }
      } finally {
        if (active) setIsLoadingSidebar(false);
      }
    };

    fetchSidebar();

    return () => {
      active = false;
    };
  }, [selectedClusters]);

  useEffect(() => {
    if (!pendingScrollNoteKey) return;
    if (!pendingScrollTargetCluster) return;
    if (isLoadingSidebar) return;
    if (loadedSidebarCluster !== pendingScrollTargetCluster) return;

    const byKey = sidebarNotes.find((note) => note.note_key === pendingScrollNoteKey);
    const byTitle = pendingScrollNoteTitle
      ? sidebarNotes.find((note) => note.title === pendingScrollNoteTitle)
      : undefined;
    const matched = byKey || byTitle;

    if (!matched) {
      setPendingScrollNoteKey(null);
      setPendingScrollNoteTitle(null);
      setPendingScrollTargetCluster(null);
      return;
    }

    const element = sidebarCardRefs.current[matched.note_key];
    if (!element) return;
    element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setPendingScrollNoteKey(null);
    setPendingScrollNoteTitle(null);
    setPendingScrollTargetCluster(null);
  }, [
    isLoadingSidebar,
    loadedSidebarCluster,
    pendingScrollNoteKey,
    pendingScrollNoteTitle,
    pendingScrollTargetCluster,
    sidebarNotes,
  ]);

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

  const pointPositionMap = useMemo(() => {
    const map = new Map<string, { linear: THREE.Vector3; log: THREE.Vector3; condensed: THREE.Vector3 }>();
    const CLUSTER_RADIUS = 2;
    const LOG_FACTOR = 0.5;
    const WORLD_SIZE = 30;

    Object.keys(clusterGroups).forEach((label) => {
      const group = clusterGroups[label];
      const centroid = clusterCentroids.get(label);
      if (!centroid) return;

      // Calculate cluster center
      const clusterCenter = new THREE.Vector3(
        group.x.reduce((a, b) => a + b, 0) / group.x.length,
        group.y.reduce((a, b) => a + b, 0) / group.y.length,
        group.z.reduce((a, b) => a + b, 0) / group.z.length,
      );

      group.customdata.forEach((meta, index) => {
        // 1. Linear: Raw position from UMAP
        const linearPos = new THREE.Vector3(group.x[index], group.y[index], group.z[index]);

        // 2. Local Log Scale: Compress distances relative to cluster center
        const relativePos = new THREE.Vector3(
          group.x[index] - clusterCenter.x,
          group.y[index] - clusterCenter.y,
          group.z[index] - clusterCenter.z,
        );
        const dist = relativePos.length();
        const scaledDist =
          Math.log(1 + dist * LOG_FACTOR) *
          (CLUSTER_RADIUS / Math.log(1 + CLUSTER_RADIUS * LOG_FACTOR));
        const logPos = new THREE.Vector3(
          clusterCenter.x + (relativePos.x / (dist || 1)) * scaledDist,
          clusterCenter.y + (relativePos.y / (dist || 1)) * scaledDist,
          clusterCenter.z + (relativePos.z / (dist || 1)) * scaledDist,
        );

        // 3. Condensed Log: Map to cluster centroid, then log-scale relative to world origin
        const originDist = Math.sqrt(
          centroid.x * centroid.x + centroid.y * centroid.y + centroid.z * centroid.z,
        );
        const scaledOriginDist =
          Math.log(1 + originDist * (LOG_FACTOR * 0.1)) *
          (WORLD_SIZE / Math.log(1 + WORLD_SIZE * (LOG_FACTOR * 0.1)));
        const condensedPos = new THREE.Vector3(
          (centroid.x / (originDist || 1)) * scaledOriginDist,
          (centroid.y / (originDist || 1)) * scaledOriginDist,
          (centroid.z / (originDist || 1)) * scaledOriginDist,
        );

        map.set(meta.unique_key, {
          linear: linearPos,
          log: logPos,
          condensed: condensedPos,
        });
      });
    });

    return map;
  }, [clusterGroups, clusterCentroids]);

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
          color.lerp(new THREE.Color('#4b5563'), 0.85);
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

    if (visualizationMode === 'condensed') {
      // In condensed mode, show only one representative point per cluster
      visibleLabels.forEach((label) => {
        const group = clusterGroups[label];
        const positionData = pointPositionMap.get(group.customdata[0]?.unique_key);
        const clusterColor = clusterColors[label] || '#4b5563';

        // Create one representative point per cluster
        let size = 0.04;
        if (hoveredId && group.customdata.some((meta) => meta.unique_key === hoveredId)) {
          size = Math.max(size * 1.35, 0.054);
        }
        if (highlightedNodeId && group.customdata.some((meta) => meta.unique_key === highlightedNodeId)) {
          size = Math.max(size * 1.45, 0.058);
        }

        const targetPos = positionData?.condensed || new THREE.Vector3(0, 0, 0);
        const glowOpacity = 0.2;
        const quantizedSize = Math.max(0.008, Math.round(size * 1000) / 1000);
        const bucketKey = `${quantizedSize}|${clusterColor}|${glowOpacity}`;

        if (!bucketMap.has(bucketKey)) {
          bucketMap.set(bucketKey, {
            key: bucketKey,
            sizeMetric: quantizedSize,
            color: clusterColor,
            glowOpacity,
            points: [],
          });
        }

        // Use first point's data as representative for the cluster
        const firstMeta = group.customdata[0];
        const visualPoint: VisualPoint = {
          ...firstMeta,
          x: targetPos.x,
          y: targetPos.y,
          z: targetPos.z,
          dotColor: clusterColor,
        };

        // Store under cluster label for hover/selection
        lookup.set(`cluster-${label}`, visualPoint);
        // Also store under first point's key for backward compatibility
        lookup.set(firstMeta.unique_key, visualPoint);

        const bucket = bucketMap.get(bucketKey)!;
        bucket.points.push(visualPoint);
      });
    } else {
      // In linear and log modes, show all individual points
      visibleLabels.forEach((label) => {
        const group = clusterGroups[label];

        group.customdata.forEach((meta, index) => {
          const score = searchScoreMap.get(meta.unique_key);
          const isHit = score !== undefined;
          const isHovered = hoveredId === meta.unique_key;
          const isHighlighted = highlightedNodeId === meta.unique_key;

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

          if (isHighlighted) {
            size = Math.max(size * 1.45, 0.026);
          }

          const dotColor = pointRenderColorMap.get(meta.unique_key) || (clusterColors[label] || '#4b5563');
          const glowOpacity = hasSearchHits ? (isHit ? 0.24 : 0) : 0.2;
          const quantizedSize = Math.max(0.008, Math.round(size * 1000) / 1000);
          const bucketKey = `${quantizedSize}|${dotColor}|${glowOpacity}`;
          if (!bucketMap.has(bucketKey)) {
            bucketMap.set(bucketKey, {
              key: bucketKey,
              sizeMetric: quantizedSize,
              color: dotColor,
              glowOpacity,
              points: [],
            });
          }

          // Get position based on visualization mode
          const positionData = pointPositionMap.get(meta.unique_key);
          const targetPos = positionData
            ? positionData.linear
            : new THREE.Vector3(group.x[index], group.y[index], group.z[index]);

          const visualPoint: VisualPoint = {
            ...meta,
            x: targetPos.x,
            y: targetPos.y,
            z: targetPos.z,
            dotColor,
          };
          lookup.set(meta.unique_key, visualPoint);

          const bucket = bucketMap.get(bucketKey)!;
          bucket.points.push(visualPoint);
        });
      });
    }

    return {
      buckets: Array.from(bucketMap.values()).sort((a, b) => a.sizeMetric - b.sizeMetric),
      pointLookup: lookup,
    };
  }, [
    clusterColors,
    clusterGroups,
    highlightedNodeId,
    hoveredId,
    pointRenderColorMap,
    pointPositionMap,
    searchResults.length,
    searchScoreMap,
    visibleLabels,
    visualizationMode,
  ]);

  const hoveredPoint = hoveredId ? pointLookup.get(hoveredId) || null : null;
  const hoveredClusterColor = hoveredPoint ? getDotSurfaceTint(hoveredPoint.dotColor) : '#ffffff';

  const selectedNodeColor = useMemo(() => {
    if (!selectedNode) return '#ffffff';
    const selectedClusterKey = selectedNode.display_topic_id || selectedNode.cluster_id || '';
    const dotColor = pointRenderColorMap.get(selectedNode.unique_key)
      || clusterColors[selectedClusterKey]
      || '#ffffff';
    return getDotSurfaceTint(dotColor);
  }, [clusterColors, pointRenderColorMap, selectedNode]);

  const updateTooltipPosition = useCallback((nativeEvent: PointerEvent | MouseEvent) => {
    if (!plotAreaRef.current) return;
    const rect = plotAreaRef.current.getBoundingClientRect();
    setTooltipPosition({
      x: nativeEvent.clientX - rect.left,
      y: nativeEvent.clientY - rect.top,
      containerWidth: rect.width,
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

      if (visualizationMode === 'condensed') {
        // In condensed mode, select/deselect the cluster
        const clusterId = point.display_topic_id || point.cluster_id || '';
        setSelectedClusters((prev) => {
          const next = new Set(prev);
          if (next.has(clusterId)) {
            next.delete(clusterId);
          } else {
            next.add(clusterId);
          }
          return next;
        });
      } else {
        // In linear/log mode, open the note
        setHighlightedNodeId(point.unique_key);
        fetchNoteContent(
          point.title,
          point.chunk_index,
          point.cluster_id,
          point.cluster_label,
          point.display_topic_id,
          point.base_topic_id,
          point.creation_date,
          point.modification_date,
        );
      }
    },
    [visualizationMode],
  );

  const tooltipStyle = useMemo(() => {
    const OFFSET = 12;
    const containerWidth = tooltipPosition.containerWidth || 0;
    const shouldPositionLeft = tooltipPosition.x > containerWidth * 0.5;

    if (shouldPositionLeft) {
      return {
        positionLeft: true,
        left: 'auto',
        right: containerWidth - tooltipPosition.x + OFFSET,
        top: tooltipPosition.y + OFFSET,
      };
    }

    return {
      positionLeft: false,
      left: tooltipPosition.x + OFFSET,
      right: 'auto',
      top: tooltipPosition.y + OFFSET,
    };
  }, [tooltipPosition]);

  const modalStyle = useMemo(() => {
    const OFFSET = 20;
    const plotWidth = plotAreaRef.current?.getBoundingClientRect().width || window.innerWidth;
    const shouldPositionLeft = tooltipPosition.x > plotWidth * 0.5;

    const baseStyle: React.CSSProperties = {
      position: 'absolute',
      top: '20px',
      width: '400px',
      maxHeight: '80vh',
      border: '1px solid #ccc',
      borderRadius: '8px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingLeft: shouldPositionLeft ? '10px' : '10px',
      paddingRight: shouldPositionLeft ? '20px' : '20px',
      boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
      zIndex: 100,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      textAlign: 'left',
      ...(shouldPositionLeft
        ? { right: Math.max(0, plotWidth - tooltipPosition.x + OFFSET) }
        : { right: '20px' }),
    };

    const headerStyle: React.CSSProperties = {
      display: 'flex',
      justifyContent: 'flex-start',
      alignItems: 'flex-start',
      marginBottom: '10px',
      flexDirection: 'row',
    };

    return {
      positionLeft: shouldPositionLeft,
      base: baseStyle,
      header: headerStyle,
    };
  }, [tooltipPosition]);

  const cameraPosition = useMemo(() => {
    const { center, radius } = sceneBounds;
    return [
      center.x + radius * 1.7,
      center.y + radius * 1.3,
      center.z + radius * 1.7,
    ] as [number, number, number];
  }, [sceneBounds]);

  const openSidebarNote = useCallback(
    (note: SidebarNoteData, preferredChunkIndex?: number) => {
      const inClusterChunk = note.chunks.find((chunk) => chunk.in_cluster);
      const fallbackChunk = note.chunks[0];
      const chunkIndex = preferredChunkIndex ?? inClusterChunk?.chunk_index ?? fallbackChunk?.chunk_index ?? 0;

      // Use the first selected cluster or the note's cluster
      const targetCluster = Array.from(selectedClusters)[0] || inClusterChunk?.cluster_id || fallbackChunk?.cluster_id;
      const clusterGroup = targetCluster ? clusterGroups[targetCluster] : null;

      fetchNoteContent(
        note.title,
        chunkIndex,
        fallbackChunk?.cluster_id || undefined,
        clusterGroup?.clusterLabel,
        targetCluster || undefined,
        targetCluster || undefined,
        note.creation_date,
        note.modification_date,
      );
    },
    [selectedClusters, clusterGroups],
  );

  const handleActiveRailClick = useCallback((note: SidebarNoteData, chunk: SidebarChunkData) => {
    const uniqueKey = `${note.title}_${chunk.chunk_index}`;
    setHighlightedNodeId(uniqueKey);
    setHoveredId(uniqueKey);
    setHoverSource('list');
    openSidebarNote(note, chunk.chunk_index);
  }, [openSidebarNote]);

  const handleInactiveRailClick = useCallback((note: SidebarNoteData, chunk: SidebarChunkData) => {
    // In the merged view, select the cluster and open the note
    setSelectedClusters((prev) => {
      const next = new Set(prev);
      next.add(chunk.cluster_id);
      return next;
    });
    setPendingScrollNoteKey(note.note_key);
    setPendingScrollNoteTitle(note.title);
    setPendingScrollTargetCluster(chunk.cluster_id);
  }, []);

  const getClusterColor = useCallback(
    (clusterId: string) => {
      const base = clusterColors[clusterId] || '#6b7280';
      return mixColorWithWhite(base, 0.45);
    },
    [clusterColors],
  );

  const displayedSidebarNotes = useMemo(() => {
    const deduped = sidebarNotes.map((note) => {
      const seen = new Set<number>();
      const chunks = note.chunks
        .slice()
        .sort((a, b) => a.chunk_index - b.chunk_index)
        .filter((chunk) => {
          if (seen.has(chunk.chunk_index)) return false;
          seen.add(chunk.chunk_index);
          return true;
        });
      return { ...note, chunks };
    });

    const sorted = deduped.slice();
    sorted.sort((a, b) => {
      let cmp = 0;
      if (notesSortMetric === 'size') {
        cmp = b.chunks.length - a.chunks.length;
      } else {
        const aTs = Date.parse(String(a.modification_date || ''));
        const bTs = Date.parse(String(b.modification_date || ''));
        const safeA = Number.isFinite(aTs) ? aTs : Number.NEGATIVE_INFINITY;
        const safeB = Number.isFinite(bTs) ? bTs : Number.NEGATIVE_INFINITY;
        cmp = safeB - safeA;
      }

      if (cmp === 0) {
        cmp = a.title.localeCompare(b.title, undefined, { numeric: true });
      }

      return notesSortDirection === 'asc' ? -cmp : cmp;
    });

    return sorted;
  }, [notesSortDirection, notesSortMetric, sidebarNotes]);

  const modalRailChunks = useMemo(() => {
    if (!selectedNode) return [] as SidebarChunkData[];

    const noteRows = data
      .filter((row) => {
        if (row.title !== selectedNode.title) return false;
        if (selectedNode.creation_date && row.creation_date !== selectedNode.creation_date) return false;
        if (selectedNode.modification_date && row.modification_date !== selectedNode.modification_date) return false;
        return true;
      })
      .sort((a, b) => a.chunk_index - b.chunk_index);

    const seen = new Set<number>();
    const selectedClusterId = selectedNode.display_topic_id || selectedNode.cluster_id || '';
    return noteRows
      .filter((row) => {
        if (seen.has(row.chunk_index)) return false;
        seen.add(row.chunk_index);
        return true;
      })
      .map((row) => ({
        chunk_index: row.chunk_index,
        cluster_id: row.display_topic_id || row.cluster_id || '-1',
        cluster_name: row.cluster_label || 'Unclustered',
        in_cluster: (row.display_topic_id || row.cluster_id || '') === selectedClusterId,
        text: null,
      }));
  }, [data, selectedNode]);

  const handleModalChunkJump = useCallback(
    (chunkIndex: number) => {
      if (!selectedNode) return;
      fetchNoteContent(
        selectedNode.title,
        chunkIndex,
        selectedNode.cluster_id,
        selectedNode.cluster_label,
        selectedNode.display_topic_id,
        selectedNode.base_topic_id,
        selectedNode.creation_date,
        selectedNode.modification_date,
      );
    },
    [selectedNode],
  );

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
            <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
              <button
                type="button"
                onClick={() => setVisualizationMode('linear')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  fontSize: '11px',
                  borderRadius: '5px',
                  border: visualizationMode === 'linear' ? '1px solid #1f2937' : '1px solid #ccc',
                  background: visualizationMode === 'linear' ? '#e5e7eb' : '#fff',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                Linear
              </button>
              <button
                type="button"
                onClick={() => setVisualizationMode('condensed')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  fontSize: '11px',
                  borderRadius: '5px',
                  border: visualizationMode === 'condensed' ? '1px solid #1f2937' : '1px solid #ccc',
                  background: visualizationMode === 'condensed' ? '#e5e7eb' : '#fff',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                Condensed
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
              <button
                type="button"
                onClick={() => setHideOtherClusters(!hideOtherClusters)}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  fontSize: '11px',
                  borderRadius: '5px',
                  border: hideOtherClusters ? '1px solid #1f2937' : '1px solid #ccc',
                  background: hideOtherClusters ? '#fee2e2' : '#fff',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                {hideOtherClusters ? 'Hide Others' : 'Show All'}
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
              <button
                type="button"
                onClick={() => setClusterOrderMode('spike')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  fontSize: '11px',
                  borderRadius: '5px',
                  border: clusterOrderMode === 'spike' ? '1px solid #1f2937' : '1px solid #ccc',
                  background: clusterOrderMode === 'spike' ? '#dbeafe' : '#fff',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
                title="Peak Recency (The Spike)"
              >
                Peak Recency
              </button>
              <button
                type="button"
                onClick={() => setClusterOrderMode('momentum')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  fontSize: '11px',
                  borderRadius: '5px',
                  border: clusterOrderMode === 'momentum' ? '1px solid #1f2937' : '1px solid #ccc',
                  background: clusterOrderMode === 'momentum' ? '#dcfce7' : '#fff',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
                title="Thematic Momentum (The Average)"
              >
                Momentum
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', fontSize: '12px', color: '#4b5563', marginBottom: '4px' }}>
                  Sort Notes By
                </label>
                <select
                  value={notesSortMetric}
                  onChange={(e) => setNotesSortMetric(e.target.value as NotesSortMetric)}
                  style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #ccc' }}
                >
                  <option value="modified">Modification Date</option>
                  <option value="size">Note Size (Chunks)</option>
                </select>
              </div>
              <div style={{ width: '130px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: '#4b5563', marginBottom: '4px' }}>
                  Direction
                </label>
                <select
                  value={notesSortDirection}
                  onChange={(e) => setNotesSortDirection(e.target.value as SortDirection)}
                  style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #ccc' }}
                >
                  <option value="desc">Decreasing</option>
                  <option value="asc">Increasing</option>
                </select>
              </div>
            </div>

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
              {searchResults.length === 0 && !searchQuery && selectedClusters.size === 0 && (
                <div style={{ color: '#666', fontStyle: 'italic', padding: '10px' }}>
                  Select clusters to view their notes.
                </div>
              )}
              {searchResults.map((result) => {
                const resultClusterKey = result.display_topic_id || result.cluster_id || '-1';
                const preview = (result.preview || '').trim();
                const isHovered = hoveredId === result.unique_key;
                const pointMatch = data.find((d) => d.unique_key === result.unique_key);
                return (
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
                      fetchNoteContent(
                        result.title,
                        result.chunk_index,
                        result.cluster_id,
                        result.cluster_label,
                        result.display_topic_id,
                        result.base_topic_id,
                        pointMatch?.creation_date,
                        pointMatch?.modification_date,
                      )
                    }
                    style={{
                      padding: '12px',
                      marginBottom: '8px',
                      borderRadius: '6px',
                      backgroundColor: clusterTints[resultClusterKey] || 'white',
                      border: '1px solid #eee',
                      cursor: 'pointer',
                      transition: 'transform 0.15s ease, box-shadow 0.15s ease',
                      transform: isHovered ? 'scale(1.015)' : 'scale(1)',
                      transformOrigin: 'left center',
                      boxShadow: isHovered ? '0 6px 12px rgba(0,0,0,0.12)' : '0 1px 2px rgba(0,0,0,0.05)',
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
                      {result.cluster_id && result.cluster_id !== '-1' ? result.cluster_id : '?'}:{' '}
                      {result.cluster_label}
                    </div>
                    <div style={{ fontSize: '12px', color: '#555', lineHeight: 1.35 }}>
                      {preview.slice(0, 180)}
                      {preview.length > 180 ? '...' : ''}
                    </div>
                  </div>
                );
              })}
              {displayedSidebarNotes.map((note) => (
                <div
                  key={note.note_key}
                  ref={(el) => {
                    sidebarCardRefs.current[note.note_key] = el;
                  }}
                  style={{
                    padding: '10px',
                    marginBottom: '8px',
                    borderRadius: '8px',
                    border: '1px solid #e5e7eb',
                    backgroundColor: '#ffffff',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                  }}
                >
                  <div
                    style={{
                      padding: 0,
                      margin: 0,
                      fontWeight: 700,
                      color: '#111827',
                      textAlign: 'left',
                      width: '100%',
                      overflowWrap: 'anywhere',
                      userSelect: 'none',
                    }}
                    title={note.title}
                  >
                    {note.title}
                  </div>

                  {note.chunks.length > 1 && (
                    <SegmentedRail
                      chunks={note.chunks}
                      activeClusterId=""
                      getClusterColor={getClusterColor}
                      onActiveDotClick={(chunk) => handleActiveRailClick(note, chunk)}
                      onInactiveDashClick={(chunk) => handleInactiveRailClick(note, chunk)}
                    />
                  )}

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {(() => {
                      const rows: React.ReactNode[] = [];
                      let i = 0;
                      while (i < note.chunks.length) {
                        const chunk = note.chunks[i];
                        if (chunk.in_cluster) {
                          const preview = (chunk.text || '').trim() || '(Empty chunk)';
                          rows.push(
                            <button
                              type="button"
                              key={`snippet-${note.note_key}-${chunk.chunk_index}`}
                              onClick={() => openSidebarNote(note, chunk.chunk_index)}
                              style={{
                                border: '1px solid #e5e7eb',
                                background: 'rgba(255,255,255,0.8)',
                                borderRadius: '6px',
                                padding: '6px 8px',
                                cursor: 'pointer',
                                textAlign: 'left',
                                color: '#374151',
                                fontSize: '12px',
                                lineHeight: 1.35,
                                whiteSpace: 'normal',
                                overflowWrap: 'anywhere',
                                wordBreak: 'break-word',
                              }}
                              title={`Open chunk ${chunk.chunk_index + 1}`}
                            >
                              <strong>Chunk {chunk.chunk_index + 1}:</strong> {preview.slice(0, 180)}
                              {preview.length > 180 ? '...' : ''}
                            </button>,
                          );
                          i += 1;
                          continue;
                        }

                        let gapCount = 0;
                        while (i < note.chunks.length && !note.chunks[i].in_cluster) {
                          gapCount += 1;
                          i += 1;
                        }

                        rows.push(
                          <div
                            key={`gap-${note.note_key}-${i}-${gapCount}`}
                            style={{
                              fontSize: '11px',
                              color: '#6b7280',
                              fontStyle: 'italic',
                              padding: '2px 4px',
                            }}
                          >
                            --- {gapCount} Chunks ---
                          </div>,
                        );
                      }
                      return rows;
                    })()}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ flex: 1, position: 'relative', height: '100%', minWidth: 0 }} ref={plotAreaRef}>
            {selectedNode && (
              <div
                style={{
                  ...modalStyle.base,
                  backgroundColor: selectedNodeColor,
                }}
              >
                <div style={modalStyle.header}>
                  <h3 style={{ margin: 0, fontSize: '1.1em', wordBreak: 'break-word', color: '#333', textAlign: 'left', flex: 1 }}>
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

                <div style={{ fontSize: '0.9em', color: '#555', marginBottom: '10px', textAlign: 'left' }}>
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
                    textAlign: 'left',
                  }}
                >
                  {isLoadingContent ? 'Loading content...' : selectedNode.content}
                </div>

                <div
                  style={{
                    marginTop: 'auto',
                    width: '100%',
                  }}
                >
                  {selectedNode.total_chunks > 1 ? (
                    <div style={{ width: '100%' }}>
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '3px',
                          flexWrap: 'wrap',
                          margin: '0 0 8px 0',
                          backgroundColor: '#000000',
                          borderRadius: '6px',
                          padding: '4px 6px',
                        }}
                      >
                        {modalRailChunks.map((chunk) => {
                          const selectedClusterId = selectedNode.display_topic_id || selectedNode.cluster_id || '';
                          const isDot = chunk.cluster_id === selectedClusterId;
                          const symbol = isDot ? '●' : '−';
                          const isCurrent = chunk.chunk_index === selectedNode.chunk_index;
                          return (
                            <button
                              type="button"
                              key={`modal-rail-${chunk.chunk_index}-${chunk.cluster_id}`}
                              onClick={() => handleModalChunkJump(chunk.chunk_index)}
                              title={`Chunk ${chunk.chunk_index + 1} | Cluster: ${chunk.cluster_name}`}
                              style={{
                                border: 'none',
                                background: 'transparent',
                                cursor: 'pointer',
                                color: isDot ? '#ffffff' : getClusterColor(chunk.cluster_id),
                                fontSize: isDot ? '15px' : '14px',
                                lineHeight: 1,
                                padding: 0,
                                margin: 0,
                                opacity: isCurrent ? 1 : 0.82,
                                transform: isCurrent ? 'scale(1.15)' : 'scale(1)',
                              }}
                              aria-label={`Jump to chunk ${chunk.chunk_index + 1}`}
                            >
                              {symbol}
                            </button>
                          );
                        })}
                      </div>

                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: '1fr auto 1fr',
                          alignItems: 'center',
                          width: '100%',
                        }}
                      >
                        <button
                          type="button"
                          onClick={handlePrevChunk}
                          disabled={isLoadingContent}
                          style={{
                            justifySelf: 'start',
                            width: '34px',
                            height: '30px',
                            cursor: 'pointer',
                            borderRadius: '4px',
                            border: '1px solid #ccc',
                            backgroundColor: '#fff',
                            fontSize: '18px',
                            lineHeight: 1,
                          }}
                          aria-label="Previous chunk"
                        >
                          &#8592;
                        </button>
                        <span style={{ fontSize: '0.9em', color: '#444', whiteSpace: 'nowrap', justifySelf: 'center' }}>
                          Chunk {selectedNode.chunk_index + 1} of {selectedNode.total_chunks}
                        </span>
                        <button
                          type="button"
                          onClick={handleNextChunk}
                          disabled={isLoadingContent}
                          style={{
                            justifySelf: 'end',
                            width: '34px',
                            height: '30px',
                            cursor: 'pointer',
                            borderRadius: '4px',
                            border: '1px solid #ccc',
                            backgroundColor: '#fff',
                            fontSize: '18px',
                            lineHeight: 1,
                          }}
                          aria-label="Next chunk"
                        >
                          &#8594;
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                      <span style={{ fontSize: '0.9em', color: '#444', whiteSpace: 'nowrap' }}>
                        Chunk 1 of 1
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {hoverSource === 'canvas' && hoveredPoint && (
              <div
                style={{
                  position: 'absolute',
                  left: tooltipStyle.left,
                  right: tooltipStyle.right,
                  top: tooltipStyle.top,
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
                  textAlign: tooltipStyle.positionLeft ? 'right' : 'left',
                  overflowWrap: 'anywhere',
                }}
              >
                <div style={{ fontWeight: 700 }}>{hoveredPoint.title}</div>
                <div style={{ color: '#333' }}>
                  Cluster {hoveredPoint.cluster_id && hoveredPoint.cluster_id !== '-1' ? hoveredPoint.cluster_id : '?'}:{' '}
                  {hoveredPoint.cluster_label}
                </div>
                <div style={{ color: '#333' }}>
                  Chunk {hoveredPoint.chunk_index + 1} of {hoveredPoint.total_chunks || '?'}
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
              style={{ width: '100%', height: '100%', background: '#000000' }}
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
                const sphereRadius = (bucket.sizeMetric / 0.02) * DOT_RADIUS_BASE;
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
              width: '360px',
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
            <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>
              Showing {visibleLabels.length} cluster{visibleLabels.length === 1 ? '' : 's'}
            </h3>
            <div style={{ minHeight: '30px', marginBottom: '8px' }}>
              {searchResults.length > 0 && (
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    type="button"
                    onClick={() => setSearchLegendOrderMode('results')}
                    style={{
                      flex: 1,
                      padding: '5px 8px',
                      fontSize: '11px',
                      borderRadius: '5px',
                      border: searchLegendOrderMode === 'results' ? '1px solid #1f2937' : '1px solid #c9c9c9',
                      backgroundColor: searchLegendOrderMode === 'results' ? '#e5e7eb' : '#fff',
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                  >
                    Results Order
                  </button>
                  <button
                    type="button"
                    onClick={() => setSearchLegendOrderMode('similarity')}
                    style={{
                      flex: 1,
                      padding: '5px 8px',
                      fontSize: '11px',
                      borderRadius: '5px',
                      border: searchLegendOrderMode === 'similarity' ? '1px solid #1f2937' : '1px solid #c9c9c9',
                      backgroundColor: searchLegendOrderMode === 'similarity' ? '#e5e7eb' : '#fff',
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                  >
                    Similarity Order
                  </button>
                </div>
              )}
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {sortedLabels.map((label) => {
                const group = clusterGroups[label];
                const cid = group.clusterId || label;
                const displayLabel = group.clusterLabel || label;
                const hasHits = group.customdata.some((pointData) =>
                  searchScoreMap.has(pointData.unique_key),
                );
                const color = clusterColors[label];
                const isSelected = selectedClusters.has(label);
                const isSearchDimmed = searchResults.length > 0 && !hasHits;
                const isFilterDimmed = hasActiveClusterFilter && !isSelected;
                const isHardDimmed = isFilterDimmed || (isSearchDimmed && !isSelected);
                const isSoftDimmed = isSearchDimmed && isSelected;
                const isDimmed = isHardDimmed || isSoftDimmed;
                const rowOpacity = isHardDimmed ? 0.5 : isSoftDimmed ? 0.76 : 1;
                const dotOpacity = isHardDimmed ? 0.55 : isSoftDimmed ? 0.78 : 1;
                const idColor = isHardDimmed ? '#8a8a8a' : isSoftDimmed ? '#666' : '#555';

                return (
                  <div
                    key={label}
                    ref={(el) => {
                      legendClusterRefs.current[label] = el;
                    }}
                    className={`cluster-row ${isSelected ? 'cluster-row-selected' : ''}`}
                    onClick={() => {
                      toggleClusterSelection(label);
                    }}
                    style={{
                      opacity: rowOpacity,
                      cursor: 'pointer',
                    }}
                  >
                    <div
                      className="cluster-dot"
                      style={{ backgroundColor: color, opacity: dotOpacity }}
                    ></div>
                    <div
                      className="cluster-id"
                      style={{ color: idColor, fontWeight: isSelected ? 700 : 600 }}
                    >
                      #{cid}
                    </div>
                    <ClusterLabel
                      label={displayLabel}
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
