import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Canvas, type ThreeEvent, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Billboard, Text } from '@react-three/drei';
import * as THREE from 'three';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import ZoomOutMapIcon from '@mui/icons-material/ZoomOutMap';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';

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
type NotesSortMetric = 'modified' | 'size' | 'search';
type SortDirection = 'desc' | 'asc';
type ClusterSortMetric = 'recency' | 'momentum' | 'az' | 'size' | 'search' | 'similarity';

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
const GLOBAL_LAYOUT_SPREAD = 2.45;

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
  hoveredId,
  highlightedId,
  onPointerOver,
  onPointerMove,
  onPointerOut,
  onClick,
}: {
  bucket: PointBucket;
  sphereRadius: number;
  hoveredId: string | null;
  highlightedId: string | null;
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
      // scale up hovered / highlighted nodes for emphasis
      const isHovered = hoveredId === point.unique_key;
      const isHighlighted = highlightedId === point.unique_key;
      const scale = isHovered ? sphereRadius * 1.8 : isHighlighted ? sphereRadius * 1.35 : sphereRadius;
      temp.scale.setScalar(scale);
      temp.updateMatrix();
      mesh.setMatrixAt(index, temp.matrix);

      if (glowMesh) {
        // glow is larger and stronger for hovered points
        const glowScale = isHovered ? scale * 3.0 : scale * 2.4;
        temp.scale.setScalar(glowScale);
        temp.updateMatrix();
        glowMesh.setMatrixAt(index, temp.matrix);
      }
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (glowMesh) {
      glowMesh.instanceMatrix.needsUpdate = true;
    }
  }, [bucket.points, sphereRadius, hoveredId, highlightedId]);

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
  const [marqueeDuration, setMarqueeDuration] = useState(5);
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLSpanElement>(null);
  const marqueeSpeedPxPerSecond = 40;

  const checkOverflow = () => {
    if (containerRef.current && contentRef.current) {
      const contentWidth = contentRef.current.scrollWidth;
      const containerWidth = containerRef.current.clientWidth;
      setIsOverflowing(contentWidth > containerWidth);
      setMarqueeDuration(Math.max((contentWidth + 20) / marqueeSpeedPxPerSecond, 1));
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
          animationDuration: `${marqueeDuration}s`,
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
  activeClusterIds,
  currentChunkIndex,
  getClusterColor,
  onActiveDotClick,
  onInactiveDashClick,
}: {
  chunks: SidebarChunkData[];
  activeClusterIds: Set<string>;
  currentChunkIndex: number | null;
  getClusterColor: (clusterId: string) => string;
  onActiveDotClick: (chunk: SidebarChunkData, e: React.MouseEvent) => void;
  onInactiveDashClick: (chunk: SidebarChunkData, e: React.MouseEvent) => void;
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
        const isDot = activeClusterIds.has(chunk.cluster_id);
        const symbol = isDot ? '●' : '−';
        const isCurrent = currentChunkIndex !== null && chunk.chunk_index === currentChunkIndex;
        return (
          <button
            type="button"
            key={`rail-${chunk.chunk_index}-${chunk.cluster_id}`}
            onClick={(e) => (isDot ? onActiveDotClick(chunk, e) : onInactiveDashClick(chunk, e))}
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
  const [hideOtherClusters, setHideOtherClusters] = useState(false);
  const [clusterOrderMode, setClusterOrderMode] = useState<ClusterOrderMode>('spike');
  const [clusterSortMetric, setClusterSortMetric] = useState<ClusterSortMetric>('recency');
  const [clusterSortDirection, setClusterSortDirection] = useState<SortDirection>('desc');
  const [notesSortMetric, setNotesSortMetric] = useState<NotesSortMetric>('modified');
  const [notesSortDirection, setNotesSortDirection] = useState<SortDirection>('desc');
  const [selectedSearchNotes, setSelectedSearchNotes] = useState<Set<string>>(new Set());
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
  const [savedClustersBeforeSearch, setSavedClustersBeforeSearch] = useState<Set<string>>(new Set());
  const plotAreaRef = useRef<HTMLDivElement>(null);
  const sidebarCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const legendClusterRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const legendContainerRef = useRef<HTMLDivElement | null>(null);
  const notesListRef = useRef<HTMLDivElement | null>(null);
  const recentClusterInfoCache = useRef<Map<string, { clusterId: string; clusterLabel: string }>>(new Map());

  const makePointCacheKey = useCallback(
    (uniqueKey: string, creationDate?: string, modificationDate?: string) =>
      `${uniqueKey}__${creationDate || ''}__${modificationDate || ''}`,
    [],
  );

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

      // Prefer authoritative values from the backend row for the opened chunk.
      const cluster_id = response.data?.cluster_id || initialClusterId;
      const cluster_label = response.data?.cluster_label || initialClusterLabel;
      const display_topic_id = response.data?.display_topic_id || initialDisplayTopicId;
      const base_topic_id = response.data?.base_topic_id || initialBaseTopicId;

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
      // Cache the authoritative cluster info from backend for this unique_key
      const cacheClusId = display_topic_id || cluster_id || '-1';
      const cacheKey = makePointCacheKey(newUniqueKey, creationDate, modificationDate);
      recentClusterInfoCache.current.set(cacheKey, {
        clusterId: cacheClusId,
        clusterLabel: cluster_label || '',
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

  const formatDateMMDDYYYY = (iso?: string | number | null) => {
    if (!iso) return '';
    const t = typeof iso === 'number' ? new Date(iso) : new Date(String(iso));
    if (Number.isNaN(t.getTime())) return '';
    const mm = String(t.getMonth() + 1).padStart(2, '0');
    const dd = String(t.getDate()).padStart(2, '0');
    const yyyy = String(t.getFullYear());
    return `${mm}/${dd}/${yyyy}`;
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

  // Manage cluster selection in search mode
  useEffect(() => {
    if (searchResults.length > 0) {
      // Search is active - save current selection and clear it
      if (selectedClusters.size > 0 && savedClustersBeforeSearch.size === 0) {
        setSavedClustersBeforeSearch(new Set(selectedClusters));
        setSelectedClusters(new Set());
      }
    } else if (searchResults.length === 0 && savedClustersBeforeSearch.size > 0) {
      // Search ended - restore previously saved clusters
      setSelectedClusters(new Set(savedClustersBeforeSearch));
      setSavedClustersBeforeSearch(new Set());
    }
  }, [searchResults.length, selectedClusters.size, savedClustersBeforeSearch.size]);

  const searchScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    searchResults.forEach((r) => map.set(r.unique_key, r.distance));
    return map;
  }, [searchResults]);

  const clusterNameById = useMemo(() => {
    const map = new Map<string, string>();
    data.forEach((point) => {
      const clusterId = point.display_topic_id || point.cluster_id || '-1';
      if (!map.has(clusterId) && point.cluster_label) {
        map.set(clusterId, point.cluster_label);
      }
    });
    return map;
  }, [data]);

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

  const clusterAverageRelevance = useMemo(() => {
    // compute average relevance per cluster from searchResults (lower distance = better)
    if (searchResults.length === 0) return new Map<string, number>();
    const byCluster = new Map<string, number[]>();
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    searchResults.forEach((r) => {
      const label = r.display_topic_id || r.cluster_id || '-1';
      const arr = byCluster.get(label) || [];
      arr.push(r.distance);
      byCluster.set(label, arr);
      if (r.distance < min) min = r.distance;
      if (r.distance > max) max = r.distance;
    });
    const map = new Map<string, number>();
    const range = Math.max(1e-6, max - min);
    byCluster.forEach((arr, label) => {
      const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
      const normalized = 1 - (avg - min) / range; // higher => more relevant
      map.set(label, Math.max(0, Math.min(1, normalized)));
    });
    return map;
  }, [searchResults]);

  // When similarity ordering is requested and notes are selected, compute a centroid
  const selectedNotesCentroid = useMemo(() => {
    if (selectedSearchNotes.size === 0) return null;
    const pts = data.filter((d) => selectedSearchNotes.has(d.unique_key));
    if (pts.length === 0) return null;
    const acc = pts.reduce((acc2, p) => ({ x: acc2.x + p.umap_x, y: acc2.y + p.umap_y, z: acc2.z + p.umap_z }), { x: 0, y: 0, z: 0 });
    acc.x /= pts.length; acc.y /= pts.length; acc.z /= pts.length;
    const norm = Math.sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z) || 1;
    return { x: acc.x, y: acc.y, z: acc.z, norm };
  }, [selectedSearchNotes, data]);

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

  const selectedClustersCentroid = useMemo(() => {
    if (selectedClusters.size === 0) return null;
    const centroids = Array.from(selectedClusters)
      .map((label) => clusterCentroids.get(label))
      .filter((centroid): centroid is { x: number; y: number; z: number } => !!centroid);
    if (centroids.length === 0) return null;
    const sum = centroids.reduce((acc, centroid) => ({
      x: acc.x + centroid.x,
      y: acc.y + centroid.y,
      z: acc.z + centroid.z,
    }), { x: 0, y: 0, z: 0 });
    sum.x /= centroids.length;
    sum.y /= centroids.length;
    sum.z /= centroids.length;
    const norm = Math.sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z) || 1;
    return { x: sum.x, y: sum.y, z: sum.z, norm };
  }, [clusterCentroids, selectedClusters]);

  const distanceToNearestCluster = useCallback(
    (label: string, anchors: string[]) => {
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
    },
    [clusterCentroids],
  );

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

  // On first load, default-select the peak-recency cluster (highest `spike` score)
  useEffect(() => {
    if (selectedClusters.size > 0) return;
    if (!clusterOrderScores || clusterOrderScores.size === 0) return;
    let bestLabel: string | null = null;
    let bestVal = Number.NEGATIVE_INFINITY;
    clusterOrderScores.forEach((scores, label) => {
      const val = scores?.spike ?? Number.NEGATIVE_INFINITY;
      if (val > bestVal) {
        bestVal = val;
        bestLabel = label;
      }
    });
    if (bestLabel) setSelectedClusters(new Set([bestLabel]));
  }, [clusterOrderScores]);

  const sortedLabels = useMemo(() => {
    const sorted = Object.keys(clusterGroups);

    // When similarity ordering is requested via selected clusters or selected notes
    if (searchLegendOrderMode === 'similarity' && selectedSearchNotes.size > 0 && selectedNotesCentroid) {
      const collective = selectedNotesCentroid;

      const cosineSimilarity = (a: { x: number; y: number; z: number }) => {
        const dot = a.x * collective.x + a.y * collective.y + a.z * collective.z;
        const normA = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z) || 1;
        return dot / (normA * collective.norm);
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

        const centroidA = clusterCentroids.get(a);
        const centroidB = clusterCentroids.get(b);

        const simA = centroidA ? cosineSimilarity(centroidA) : -Infinity;
        const simB = centroidB ? cosineSimilarity(centroidB) : -Infinity;

        if (simA !== simB) return simB - simA; // higher similarity first

        const idA = clusterGroups[a].clusterId || a;
        const idB = clusterGroups[b].clusterId || b;
        return compareTopicIds(String(idA), String(idB));
      });
      return sorted;
    } else if (searchLegendOrderMode === 'similarity' && selectedClusters.size > 0) {
      // Compute the collective centroid (average) of currently selected clusters
      const selectedCentroids: { x: number; y: number; z: number }[] = Array.from(selectedClusters)
        .map((label) => clusterCentroids.get(label))
        .filter((c): c is { x: number; y: number; z: number } => !!c);

      if (selectedCentroids.length === 0) {
        // fallback to previous behavior when there are no available centroids for selection
        // (falls through to other branches below)
      } else {
        const collective = selectedCentroids.reduce(
          (acc, cur) => ({ x: acc.x + cur.x, y: acc.y + cur.y, z: acc.z + cur.z }),
          { x: 0, y: 0, z: 0 },
        );
        collective.x /= selectedCentroids.length;
        collective.y /= selectedCentroids.length;
        collective.z /= selectedCentroids.length;

        const collectiveNorm = Math.sqrt(collective.x * collective.x + collective.y * collective.y + collective.z * collective.z) || 1;

        const cosineSimilarity = (a: { x: number; y: number; z: number }) => {
          const dot = a.x * collective.x + a.y * collective.y + a.z * collective.z;
          const normA = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z) || 1;
          return dot / (normA * collectiveNorm);
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

          const centroidA = clusterCentroids.get(a);
          const centroidB = clusterCentroids.get(b);

          const simA = centroidA ? cosineSimilarity(centroidA) : -Infinity;
          const simB = centroidB ? cosineSimilarity(centroidB) : -Infinity;

          if (simA !== simB) return simB - simA; // higher similarity first

          const idA = clusterGroups[a].clusterId || a;
          const idB = clusterGroups[b].clusterId || b;
          return compareTopicIds(String(idA), String(idB));
        });
        return sorted;
      }
    } else if (searchLegendOrderMode === 'similarity' && searchResults.length > 0) {
      const hitLabels = sorted.filter((label) => {
        const group = clusterGroups[label];
        return group.customdata.some((pointData) => searchScoreMap.has(pointData.unique_key));
      });

      const distanceToNearestHitCluster = (label: string) => {
        return distanceToNearestCluster(label, hitLabels);
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

        const distA = distanceToNearestCluster(a, hitLabels);
        const distB = distanceToNearestCluster(b, hitLabels);
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
    selectedSearchNotes,
    selectedNotesCentroid,
  ]);

  const displayedClusterLabels = useMemo(() => {
    let list = sortedLabels.slice();
    if (clusterSortMetric === 'az') {
      list = list.slice().sort((a, b) => {
        const aLabel = (clusterGroups[a]?.clusterLabel || a).toLowerCase();
        const bLabel = (clusterGroups[b]?.clusterLabel || b).toLowerCase();
        return aLabel.localeCompare(bLabel, undefined, { numeric: true });
      });
    } else if (clusterSortMetric === 'size') {
      list = list.slice().sort((a, b) => {
        const sa = clusterGroups[a]?.customdata.length || 0;
        const sb = clusterGroups[b]?.customdata.length || 0;
        return clusterSortDirection === 'asc' ? sa - sb : sb - sa;
      });
    } else if (clusterSortMetric === 'search') {
      const firstSeenRank = new Map<string, number>();
      searchResults.forEach((result, index) => {
        const label = result.display_topic_id || result.cluster_id || '-1';
        if (clusterGroups[label] && !firstSeenRank.has(label)) {
          firstSeenRank.set(label, index);
        }
      });

      const hitLabels = Array.from(firstSeenRank.keys());

      list = list.slice().sort((a, b) => {
        const rankA = firstSeenRank.get(a);
        const rankB = firstSeenRank.get(b);

        if (rankA !== undefined && rankB !== undefined) {
          return clusterSortDirection === 'asc' ? rankA - rankB : rankB - rankA;
        }
        if (rankA !== undefined) return -1;
        if (rankB !== undefined) return 1;

        const distA = distanceToNearestCluster(a, hitLabels);
        const distB = distanceToNearestCluster(b, hitLabels);
        if (distA !== distB) {
          return clusterSortDirection === 'asc' ? distA - distB : distB - distA;
        }

        const idA = clusterGroups[a].clusterId || a;
        const idB = clusterGroups[b].clusterId || b;
        return compareTopicIds(String(idA), String(idB));
      });
    } else if (clusterSortMetric === 'similarity') {
      list = list.slice().sort((a, b) => {
        const centroid = selectedClustersCentroid;
        if (!centroid) return compareTopicIds(String(clusterGroups[a]?.clusterId || a), String(clusterGroups[b]?.clusterId || b));

        const similarity = (label: string) => {
          const point = clusterCentroids.get(label);
          if (!point) return Number.NEGATIVE_INFINITY;
          const dot = point.x * centroid.x + point.y * centroid.y + point.z * centroid.z;
          const norm = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z) || 1;
          return dot / (norm * centroid.norm);
        };

        const sa = similarity(a);
        const sb = similarity(b);
        return clusterSortDirection === 'asc' ? sa - sb : sb - sa;
      });
    } else if (clusterSortMetric === 'recency') {
      list = list.slice().sort((a, b) => {
        const as = clusterGroups[a]?.customdata.map((d) => Date.parse(String(d.modification_date || ''))).filter(Number.isFinite) || [];
        const bs = clusterGroups[b]?.customdata.map((d) => Date.parse(String(d.modification_date || ''))).filter(Number.isFinite) || [];
        const aMax = as.length ? Math.max(...as) : Number.NEGATIVE_INFINITY;
        const bMax = bs.length ? Math.max(...bs) : Number.NEGATIVE_INFINITY;
        return clusterSortDirection === 'asc' ? aMax - bMax : bMax - aMax;
      });
    } else if (clusterSortMetric === 'momentum') {
      // use previously computed clusterOrderScores (momentum = average)
      list = list.slice().sort((a, b) => {
        const sa = clusterOrderScores.get(a)?.momentum ?? Number.NEGATIVE_INFINITY;
        const sb = clusterOrderScores.get(b)?.momentum ?? Number.NEGATIVE_INFINITY;
        return clusterSortDirection === 'asc' ? sa - sb : sb - sa;
      });
    }

    return list;
  }, [sortedLabels, clusterGroups, clusterSortMetric, clusterSortDirection, clusterOrderScores, clusterAverageRelevance, clusterCentroids, selectedClustersCentroid]);

  const displayedSearchResults = useMemo(() => {
    const sorted = searchResults.slice();

    const getModifiedTimestamp = (result: SearchResult) => {
      const match = data.find((point) => point.unique_key === result.unique_key);
      const parsed = Date.parse(String(match?.modification_date || ''));
      return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
    };

    const getSize = (result: SearchResult) => {
      const match = data.find((point) => point.unique_key === result.unique_key);
      return match?.total_chunks ?? result.total_chunks ?? 0;
    };

    if (notesSortMetric === 'search') {
      const order = new Map(searchResults.map((result, index) => [result.unique_key, index]));
      sorted.sort((a, b) => {
        const ra = order.get(a.unique_key) ?? Number.POSITIVE_INFINITY;
        const rb = order.get(b.unique_key) ?? Number.POSITIVE_INFINITY;
        if (ra !== rb) {
          return notesSortDirection === 'asc' ? ra - rb : rb - ra;
        }
        return notesSortDirection === 'asc'
          ? a.title.localeCompare(b.title, undefined, { numeric: true })
          : b.title.localeCompare(a.title, undefined, { numeric: true });
      });
      return sorted;
    }

    sorted.sort((a, b) => {
      let cmp = 0;
      if (notesSortMetric === 'size') {
        cmp = getSize(a) - getSize(b);
      } else {
        cmp = getModifiedTimestamp(a) - getModifiedTimestamp(b);
      }

      if (cmp === 0) {
        cmp = a.title.localeCompare(b.title, undefined, { numeric: true });
      }

      return notesSortDirection === 'asc' ? cmp : -cmp;
    });

    return sorted;
  }, [data, notesSortDirection, notesSortMetric, searchResults]);

  const getClusterSectionTitle = useCallback(
    (label: string) => {
      if (clusterSortMetric === 'recency' || clusterSortMetric === 'momentum') {
        const dates = (clusterGroups[label]?.customdata || [])
          .map((point) => Date.parse(String(point.modification_date || '')))
          .filter(Number.isFinite) as number[];

        if (dates.length === 0) return 'Unknown date';

        const representativeDate =
          clusterSortMetric === 'momentum'
            ? Math.round(dates.reduce((sum, value) => sum + value, 0) / dates.length)
            : Math.max(...dates);

        const oneDay = 24 * 60 * 60 * 1000;
        const diffDays = Math.floor((Date.now() - representativeDate) / oneDay);
        if (diffDays <= 1) return 'Today & Yesterday';
        if (diffDays <= 7) return 'Last 7 Days';
        if (diffDays <= 30) return 'Last 30 Days';

        return new Intl.DateTimeFormat(undefined, { month: 'long', year: 'numeric' }).format(
          new Date(representativeDate),
        );
      }

      return '';
    },
    [clusterGroups, clusterSortMetric],
  );

  useEffect(() => {
    if (clusterSortMetric === 'search' && searchResults.length === 0) {
      setClusterSortMetric('momentum');
    }
    if (clusterSortMetric === 'similarity' && selectedClusters.size === 0) {
      setClusterSortMetric('momentum');
    }
  }, [clusterSortMetric, searchResults.length, selectedClusters.size]);


  const hasActiveClusterFilter = selectedClusters.size > 0;
  const visibleLabels = hideOtherClusters && hasActiveClusterFilter
    ? displayedClusterLabels.filter((label) => selectedClusters.has(label))
    : displayedClusterLabels;

  useEffect(() => {
    let active = true;

    const fetchSidebar = async () => {
      // Fetch notes for all selected clusters
      if (selectedClusters.size === 0) {
        if (active) {
          setSidebarNotes([]);
          setLoadedSidebarCluster(null);
        }
        return;
      }

      const clusterArray = Array.from(selectedClusters);
      setIsLoadingSidebar(true);
      try {
        // Fetch notes for each selected cluster
        const allNotes: SidebarNoteData[] = [];
        const noteTitles = new Set<string>();

        for (const clusterId of clusterArray) {
          const response = await axios.get(
            `http://127.0.0.1:8000/cluster_sidebar?active_cluster_id=${encodeURIComponent(clusterId)}`,
          );
          const notes = Array.isArray(response.data?.notes) ? response.data.notes : [];
          notes.forEach((note: SidebarNoteData) => {
            if (!noteTitles.has(note.title)) {
              allNotes.push(note);
              noteTitles.add(note.title);
            }
          });
        }

        if (active) {
          setSidebarNotes(allNotes);
          setLoadedSidebarCluster(clusterArray[0]);
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
    const INTRA_CLUSTER_RADIUS = 1.75;
    const INTRA_LOG_STRENGTH = 4.2;
    const CONDENSED_RADIUS_SCALE = 1.7;
    const CONDENSED_CURVE = 0.72;

      const clusterCondensed = new Map<string, THREE.Vector3>();
      const clusterCenterMap = new Map<string, THREE.Vector3>();
      const uniqueToLabel = new Map<string, string>();

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
      clusterCenterMap.set(label, clusterCenter);

      const localOffsets = group.customdata.map((_, idx) => (
        new THREE.Vector3(
          group.x[idx] - clusterCenter.x,
          group.y[idx] - clusterCenter.y,
          group.z[idx] - clusterCenter.z,
        )
      ));
      const maxLocalDist = Math.max(
        ...localOffsets.map((offset) => offset.length()),
        1e-6,
      );

      group.customdata.forEach((meta, index) => {
        // 1. Linear: Raw position from UMAP
        const linearPos = new THREE.Vector3(group.x[index], group.y[index], group.z[index]);

        // 2. Local Log Scale (bounded): keep cluster points compact while preserving orientation.
        const relativePos = localOffsets[index];
        const dist = relativePos.length();
        const normLocal = Math.min(dist / maxLocalDist, 1);
        const logNorm = Math.log1p(normLocal * INTRA_LOG_STRENGTH) / Math.log1p(INTRA_LOG_STRENGTH);
        const scaledDist = logNorm * INTRA_CLUSTER_RADIUS;
        const logPos = dist > 0
          ? clusterCenter.clone().add(relativePos.clone().normalize().multiplyScalar(scaledDist))
          : clusterCenter.clone();

        // 3. Condensed: remap cluster centroids around the scene center using a bounded radial curve.
        // This increases separation for nearby clusters while keeping the overall cloud compact.
            const sceneOffset = new THREE.Vector3(centroid.x, centroid.y, centroid.z).sub(sceneBounds.center);
        const sceneDistance = sceneOffset.length();
        const normalizedDistance = Math.min(sceneDistance / sceneBounds.radius, 1);
        const curvedDistance = Math.pow(normalizedDistance, CONDENSED_CURVE) * (sceneBounds.radius * CONDENSED_RADIUS_SCALE);
        const condensedPos = sceneDistance > 0
          ? new THREE.Vector3(sceneBounds.center.x, sceneBounds.center.y, sceneBounds.center.z)
            .add(sceneOffset.normalize().multiplyScalar(curvedDistance))
          : new THREE.Vector3(sceneBounds.center.x, sceneBounds.center.y, sceneBounds.center.z);

            clusterCondensed.set(label, condensedPos);
            map.set(meta.unique_key, {
              linear: linearPos,
              log: logPos,
              condensed: clusterCondensed.get(label) as THREE.Vector3,
            });
            uniqueToLabel.set(meta.unique_key, label);
      });
    });

      // Post-process condensed positions:
      // 1) Apply a mild log-based compression for long distances so far-away clusters
      //    are pulled closer while preserving small-distance expansion from the curve.
      // 2) Spread exact duplicates on a small deterministic circle.
      // 3) Run an iterative repulsion pass to resolve remaining near-overlaps.
      if (clusterCondensed.size > 0) {
        const entries = Array.from(clusterCondensed.entries());

        // Compression parameters
        const LOG_COMPRESSOR = 3.0;
        const COMPRESS_WEIGHT = 0.45;
        const maxScale = sceneBounds.radius * CONDENSED_RADIUS_SCALE;

        // 1) compression: remap radii using a log-style normalizer blended with curved distance
        entries.forEach(([label, vec]) => {
          const dir = new THREE.Vector3(vec.x - sceneBounds.center.x, vec.y - sceneBounds.center.y, vec.z - sceneBounds.center.z);
          const r = dir.length();
          if (r === 0) return;
          const norm = Math.min(r / maxScale, 1);
          const curvedNorm = Math.pow(norm, CONDENSED_CURVE);
          const compressedNorm = Math.log1p(norm * LOG_COMPRESSOR) / Math.log1p(LOG_COMPRESSOR);
          const blended = curvedNorm * (1 - COMPRESS_WEIGHT) + compressedNorm * COMPRESS_WEIGHT;
          const finalR = blended * maxScale;
          dir.normalize().multiplyScalar(finalR);
          vec.x = sceneBounds.center.x + dir.x;
          vec.y = sceneBounds.center.y + dir.y;
          vec.z = sceneBounds.center.z + dir.z;
        });

        // 2) exact duplicates: spread evenly on a small circle
        const bins = new Map<string, string[]>();
        entries.forEach(([label, vec]) => {
          const key = `${vec.x.toFixed(6)}|${vec.y.toFixed(6)}|${vec.z.toFixed(6)}`;
          const arr = bins.get(key) || [];
          arr.push(label);
          bins.set(key, arr);
        });

        bins.forEach((labels) => {
          if (labels.length <= 1) return;
          const n = labels.length;
          const smallRadius = Math.max(0.02 * sceneBounds.radius, 0.9) * (1 + n * 0.08);
          for (let i = 0; i < n; i += 1) {
            const angle = (2 * Math.PI * i) / n;
            const base = clusterCondensed.get(labels[i])!;
            base.x += Math.cos(angle) * smallRadius;
            base.y += Math.sin(angle) * smallRadius;
          }
        });

        // 3) iterative repulsion for near overlaps
        const minSep = Math.max(0.12 * sceneBounds.radius, 3.2);
        const repulseIters = 12;
        for (let iter = 0; iter < repulseIters; iter += 1) {
          const all = Array.from(clusterCondensed.entries());
          for (let i = 0; i < all.length; i += 1) {
            for (let j = i + 1; j < all.length; j += 1) {
              const a = clusterCondensed.get(all[i][0])!;
              const b = clusterCondensed.get(all[j][0])!;
              const dx = b.x - a.x;
              const dy = b.y - a.y;
              const dz = b.z - a.z;
              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
              if (dist < minSep) {
                const overlap = (minSep - dist) * 0.55; // fraction to move per iter
                const nx = dx / dist;
                const ny = dy / dist;
                const nz = dz / dist;
                a.x -= nx * overlap;
                a.y -= ny * overlap;
                a.z -= nz * overlap;
                b.x += nx * overlap;
                b.y += ny * overlap;
                b.z += nz * overlap;
              }
            }
          }
        }

        // Pull cluster centers inward to reduce the large middle void.
        const VOID_PULL = 1.08;
        entries.forEach(([, vec]) => {
          const dx = vec.x - sceneBounds.center.x;
          const dy = vec.y - sceneBounds.center.y;
          const dz = vec.z - sceneBounds.center.z;
          vec.x = sceneBounds.center.x + dx * VOID_PULL;
          vec.y = sceneBounds.center.y + dy * VOID_PULL;
          vec.z = sceneBounds.center.z + dz * VOID_PULL;
        });

        // Re-apply a lighter repulsion so inward pull doesn't re-introduce overlaps.
        const postPullMinSep = Math.max(0.11 * sceneBounds.radius, 3.0);
        for (let iter = 0; iter < 12; iter += 1) {
          const all = Array.from(clusterCondensed.entries());
          for (let i = 0; i < all.length; i += 1) {
            for (let j = i + 1; j < all.length; j += 1) {
              const a = clusterCondensed.get(all[i][0])!;
              const b = clusterCondensed.get(all[j][0])!;
              const dx = b.x - a.x;
              const dy = b.y - a.y;
              const dz = b.z - a.z;
              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
              if (dist < postPullMinSep) {
                const overlap = (postPullMinSep - dist) * 0.5;
                const nx = dx / dist;
                const ny = dy / dist;
                const nz = dz / dist;
                a.x -= nx * overlap;
                a.y -= ny * overlap;
                a.z -= nz * overlap;
                b.x += nx * overlap;
                b.y += ny * overlap;
                b.z += nz * overlap;
              }
            }
          }
        }

        // Anchor intra-cluster log positions to the computed cluster centers (Tier 1 + Tier 2 composition)
        uniqueToLabel.forEach((label, uniqueKey) => {
          const entry = map.get(uniqueKey);
          const clusterCenter = clusterCenterMap.get(label);
          const condensedCenter = clusterCondensed.get(label);
          if (!entry || !clusterCenter || !condensedCenter) return;

          const dir = new THREE.Vector3(entry.log.x - clusterCenter.x, entry.log.y - clusterCenter.y, entry.log.z - clusterCenter.z);
          const dist = dir.length();
          if (dist === 0) {
            entry.log = new THREE.Vector3(
              condensedCenter.x + (Math.random() - 0.5) * 1e-3,
              condensedCenter.y + (Math.random() - 0.5) * 1e-3,
              condensedCenter.z + (Math.random() - 0.5) * 1e-3,
            );
          } else {
            dir.normalize().multiplyScalar(dist);
            entry.log = new THREE.Vector3(condensedCenter.x + dir.x, condensedCenter.y + dir.y, condensedCenter.z + dir.z);
          }
          entry.condensed = condensedCenter.clone();
        });

        // Per-cluster anti-overlap for chunk dots
        const labelToKeys = new Map<string, string[]>();
        uniqueToLabel.forEach((label, key) => {
          const arr = labelToKeys.get(label) || [];
          arr.push(key);
          labelToKeys.set(label, arr);
        });

        const minNodeSep = Math.max(0.0065 * sceneBounds.radius, 0.42);
        const nodeRepulseIters = 6;
        labelToKeys.forEach((keys) => {
          for (let iter = 0; iter < nodeRepulseIters; iter += 1) {
            for (let i = 0; i < keys.length; i += 1) {
              for (let j = i + 1; j < keys.length; j += 1) {
                const a = map.get(keys[i]);
                const b = map.get(keys[j]);
                if (!a || !b) continue;

                const dx = b.log.x - a.log.x;
                const dy = b.log.y - a.log.y;
                const dz = b.log.z - a.log.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
                if (dist < minNodeSep) {
                  const overlap = (minNodeSep - dist) * 0.5;
                  const nx = dx / dist;
                  const ny = dy / dist;
                  const nz = dz / dist;
                  a.log.x -= nx * overlap;
                  a.log.y -= ny * overlap;
                  a.log.z -= nz * overlap;
                  b.log.x += nx * overlap;
                  b.log.y += ny * overlap;
                  b.log.z += nz * overlap;
                }
              }
            }
          }
        });

        // Outlier pull: use standard deviation to identify and pull in far-away dots.
        // Use a tighter threshold and stronger iterative pull toward the cluster centroid
        // so extreme outliers are rapidly moved into the main cloud.
        const outlierPullStrength = 0.85; // much stronger pull per iteration
        const stdDevMultiplier = 0.55; // tighter threshold: mean + 0.55*stdDev
        const outlierPullIters = 40; // more iterations for extreme convergence

        labelToKeys.forEach((keys) => {
          const condensedCenter = clusterCondensed.get(uniqueToLabel.get(keys[0])!)!;

          // Compute mean and standard deviation of distances
          const distances = keys.map((key) => {
            const entry = map.get(key)!;
            const dx = entry.log.x - condensedCenter.x;
            const dy = entry.log.y - condensedCenter.y;
            const dz = entry.log.z - condensedCenter.z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
          });

          const meanDist = distances.reduce((a, b) => a + b, 0) / Math.max(distances.length, 1);
          const variance = distances.reduce((a, d) => a + (d - meanDist) ** 2, 0) / Math.max(distances.length, 1);
          const stdDev = Math.sqrt(variance);
          const outlierThreshold = meanDist + stdDev * stdDevMultiplier;

          // Pull outliers inward iteratively
          for (let pullIter = 0; pullIter < outlierPullIters; pullIter += 1) {
            keys.forEach((key) => {
              const entry = map.get(key)!;
              const dx = entry.log.x - condensedCenter.x;
              const dy = entry.log.y - condensedCenter.y;
              const dz = entry.log.z - condensedCenter.z;
              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;

              if (dist > outlierThreshold) {
                // Determine a target distance: bring the point down to just inside the
                // mean + scaled stdDev boundary, then move toward that target by
                // a strong fraction each iteration so extreme points converge quickly.
                const targetDist = Math.max(meanDist + stdDev * 0.2, outlierThreshold);
                const excess = dist - targetDist;
                // scale the per-iteration movement by outlierPullStrength and how far beyond target
                const move = Math.min(excess, excess * outlierPullStrength);
                const newDist = Math.max(dist - move, targetDist);
                const scale = newDist / dist;
                const newDx = (dx * scale);
                const newDy = (dy * scale);
                const newDz = (dz * scale);
                entry.log.x = condensedCenter.x + newDx;
                entry.log.y = condensedCenter.y + newDy;
                entry.log.z = condensedCenter.z + newDz;
              }
            });
          }
        });

        // Final light repulsion pass after outlier pull to avoid any new overlaps
        const outlierMinNodeSep = Math.max(0.005 * sceneBounds.radius, 0.3);
        labelToKeys.forEach((keys) => {
          for (let iter = 0; iter < 4; iter += 1) {
            for (let i = 0; i < keys.length; i += 1) {
              for (let j = i + 1; j < keys.length; j += 1) {
                const a = map.get(keys[i]);
                const b = map.get(keys[j]);
                if (!a || !b) continue;

                const dx = b.log.x - a.log.x;
                const dy = b.log.y - a.log.y;
                const dz = b.log.z - a.log.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
                if (dist < outlierMinNodeSep) {
                  const overlap = (outlierMinNodeSep - dist) * 0.35;
                  const nx = dx / dist;
                  const ny = dy / dist;
                  const nz = dz / dist;
                  a.log.x -= nx * overlap;
                  a.log.y -= ny * overlap;
                  a.log.z -= nz * overlap;
                  b.log.x += nx * overlap;
                  b.log.y += ny * overlap;
                  b.log.z += nz * overlap;
                }
              }
            }
          }
        });

        // Hard clamp for extreme outliers: snap any remaining very-distant points
        // back toward the cluster condensed center with smoothing. This handles
        // stubborn patches that survive the iterative pull above.
        const clampStdMultiplier = 2.5; // consider anything beyond mean + 2.5*std as extreme
        const clampTargetStd = 0.9; // bring them to mean + 0.9*std
        const absoluteMaxFraction = 0.65; // absolute max as fraction of scene radius
        labelToKeys.forEach((keys) => {
          const condensedCenter = clusterCondensed.get(uniqueToLabel.get(keys[0])!)!;
          // recompute distances for this cluster
          const dists = keys.map((key) => {
            const e = map.get(key)!;
            const dx = e.log.x - condensedCenter.x;
            const dy = e.log.y - condensedCenter.y;
            const dz = e.log.z - condensedCenter.z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
          });
          const meanD = dists.reduce((a, b) => a + b, 0) / Math.max(dists.length, 1);
          const varD = dists.reduce((a, v, i) => a + (v - meanD) ** 2, 0) / Math.max(dists.length, 1);
          const stdD = Math.sqrt(varD) || 1e-6;
          const extremeThreshold = meanD + stdD * clampStdMultiplier;
          const absoluteMax = Math.max(meanD + stdD * 3.0, sceneBounds.radius * absoluteMaxFraction);

          keys.forEach((key) => {
            const entry = map.get(key)!;
            const dx = entry.log.x - condensedCenter.x;
            const dy = entry.log.y - condensedCenter.y;
            const dz = entry.log.z - condensedCenter.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;

            if (dist > extremeThreshold || dist > absoluteMax) {
              // Target distance just inside the cluster spread
              const targetDist = Math.max(meanD + stdD * clampTargetStd, meanD * 0.6);
              const scale = targetDist / dist;
              const targetX = condensedCenter.x + dx * scale;
              const targetY = condensedCenter.y + dy * scale;
              const targetZ = condensedCenter.z + dz * scale;

              // Hard snap: immediately set to the target to remove stubborn outliers
              entry.log.x = targetX;
              entry.log.y = targetY;
              entry.log.z = targetZ;
            }
          });
        });

        // Final absolute-cap pass: enforce a strict maximum distance from the cluster
        // center so any stubborn points are brought well inside the scene. This is
        // deliberately aggressive and uses strong smoothing to avoid popping.
        const absoluteCapFraction = 0.55; // fraction of scene radius considered too far
        const absoluteMoveSmooth = 0.95; // move 95% of the way to the target
        labelToKeys.forEach((keys) => {
          const condensedCenter = clusterCondensed.get(uniqueToLabel.get(keys[0])!)!;
          const dists = keys.map((k) => {
            const e = map.get(k)!;
            const dx = e.log.x - condensedCenter.x;
            const dy = e.log.y - condensedCenter.y;
            const dz = e.log.z - condensedCenter.z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
          });
          const meanD = dists.reduce((a, b) => a + b, 0) / Math.max(dists.length, 1);
          const varD = dists.reduce((a, v) => a + (v - meanD) ** 2, 0) / Math.max(dists.length, 1);
          const stdD = Math.sqrt(varD) || 1e-6;
          const absoluteCap = Math.max(sceneBounds.radius * absoluteCapFraction, meanD + stdD * 1.0);
          const fallbackTarget = Math.min(meanD + stdD * 0.9, sceneBounds.radius * 0.45);

          keys.forEach((k) => {
            const entry = map.get(k)!;
            const dx = entry.log.x - condensedCenter.x;
            const dy = entry.log.y - condensedCenter.y;
            const dz = entry.log.z - condensedCenter.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
              if (dist > absoluteCap) {
                const scale = fallbackTarget / dist;
                const tx = condensedCenter.x + dx * scale;
                const ty = condensedCenter.y + dy * scale;
                const tz = condensedCenter.z + dz * scale;
                // Hard snap: set directly to the computed target
                entry.log.x = tx;
                entry.log.y = ty;
                entry.log.z = tz;
              }
          });
        });

        // Global spread so islands and internal points breathe more in world space.
        // Extra robust outlier pass using median + MAD (no cluster-specific hardcoding):
        // - Compute median distance per-cluster and MAD-scaled metric
        // - Any point beyond median + 3*MAD_scaled is an outlier and will be snapped
        //   to a safer distance (median + 1.5*MAD_scaled)
        const MAD_SCALE = 1.4826; // approximate conversion to std
        labelToKeys.forEach((keys) => {
          const condensedCenter = clusterCondensed.get(uniqueToLabel.get(keys[0])!)!;
          const dists = keys.map((k) => {
            const e = map.get(k)!;
            const dx = e.log.x - condensedCenter.x;
            const dy = e.log.y - condensedCenter.y;
            const dz = e.log.z - condensedCenter.z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
          }).sort((a, b) => a - b);

          if (dists.length === 0) return;
          const mid = Math.floor(dists.length / 2);
          const median = dists.length % 2 === 1 ? dists[mid] : (dists[mid - 1] + dists[mid]) / 2;

          const absDevs = keys.map((k) => {
            const e = map.get(k)!;
            const dx = e.log.x - condensedCenter.x;
            const dy = e.log.y - condensedCenter.y;
            const dz = e.log.z - condensedCenter.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            return Math.abs(dist - median);
          }).sort((a, b) => a - b);
          const mad = absDevs.length % 2 === 1 ? absDevs[Math.floor(absDevs.length / 2)] : (absDevs[Math.floor(absDevs.length / 2) - 1] + absDevs[Math.floor(absDevs.length / 2)]) / 2;
          const madScaled = Math.max(mad * MAD_SCALE, 1e-6);

          const outlierThreshold = median + 3 * madScaled;
          const targetDist = Math.max(median + 1.5 * madScaled, Math.min(sceneBounds.radius * 0.45, median + 2 * madScaled));

          keys.forEach((k) => {
            const entry = map.get(k)!;
            const dx = entry.log.x - condensedCenter.x;
            const dy = entry.log.y - condensedCenter.y;
            const dz = entry.log.z - condensedCenter.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
            if (dist > outlierThreshold) {
              const scale = targetDist / dist;
              entry.log.x = condensedCenter.x + dx * scale;
              entry.log.y = condensedCenter.y + dy * scale;
              entry.log.z = condensedCenter.z + dz * scale;
            }
          });
        });

        // KNN-based neighbor snapping pass for any remaining stubborn outliers.
        // For each cluster, detect outliers (median + 2*MAD) then move the outlier
        // to the mean of its nearest same-cluster neighbors (k-nearest), preserving
        // local shape without hardcoding cluster ids.
        labelToKeys.forEach((keys) => {
          if (keys.length <= 4) return; // too small to compute neighbors
          const points = keys.map((k) => {
            const e = map.get(k)!;
            return { key: k, x: e.log.x, y: e.log.y, z: e.log.z };
          });

          const dists = points.map((p) => {
            const dx = p.x - (clusterCondensed.get(uniqueToLabel.get(p.key)!)!.x);
            const dy = p.y - (clusterCondensed.get(uniqueToLabel.get(p.key)!)!.y);
            const dz = p.z - (clusterCondensed.get(uniqueToLabel.get(p.key)!)!.z);
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
          }).sort((a, b) => a - b);

          const mid = Math.floor(dists.length / 2);
          const median = dists.length % 2 === 1 ? dists[mid] : (dists[mid - 1] + dists[mid]) / 2;
          const absDevs = dists.map((d) => Math.abs(d - median));
          const mad = absDevs.length % 2 === 1 ? absDevs[Math.floor(absDevs.length / 2)] : (absDevs[Math.floor(absDevs.length / 2) - 1] + absDevs[Math.floor(absDevs.length / 2)]) / 2;
          const madScaled = Math.max(mad * 1.4826, 1e-6);
          const knnOutlierThreshold = median + 2 * madScaled;

          // For each point flagged as outlier, find k nearest neighbors (excluding itself)
          const K = Math.min(6, Math.max(2, Math.floor(points.length * 0.12)));
          points.forEach((p) => {
            const dx = p.x - (clusterCondensed.get(uniqueToLabel.get(p.key)!)!.x);
            const dy = p.y - (clusterCondensed.get(uniqueToLabel.get(p.key)!)!.y);
            const dz = p.z - (clusterCondensed.get(uniqueToLabel.get(p.key)!)!.z);
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
            if (dist <= knnOutlierThreshold) return;

            // compute distances to other points in same cluster
            const neighbors = points
              .map((q) => {
                if (q.key === p.key) return null;
                const ddx = q.x - p.x;
                const ddy = q.y - p.y;
                const ddz = q.z - p.z;
                return { key: q.key, d: Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz), x: q.x, y: q.y, z: q.z };
              })
              .filter(Boolean) as { key: string; d: number; x: number; y: number; z: number }[];

            neighbors.sort((a, b) => a.d - b.d);
            const chosen = neighbors.slice(0, K);
            if (chosen.length === 0) return;
            const avgX = chosen.reduce((s, n) => s + n.x, 0) / chosen.length;
            const avgY = chosen.reduce((s, n) => s + n.y, 0) / chosen.length;
            const avgZ = chosen.reduce((s, n) => s + n.z, 0) / chosen.length;

            const entry = map.get(p.key)!;
            // place the outlier at the neighbor mean + small jitter
            const jitter = (Math.random() - 0.5) * (0.002 * sceneBounds.radius);
            entry.log.x = avgX + jitter;
            entry.log.y = avgY + jitter;
            entry.log.z = avgZ + jitter;
          });
        });

        map.forEach((entry) => {
          const expand = (vec: THREE.Vector3) =>
            sceneBounds.center.clone().add(vec.clone().sub(sceneBounds.center).multiplyScalar(GLOBAL_LAYOUT_SPREAD));
          entry.linear = expand(entry.linear);
          entry.log = expand(entry.log);
          entry.condensed = expand(entry.condensed);
        });
      }

    return map;
  }, [clusterGroups, clusterCentroids, sceneBounds]);

  const authoritativeClusterByKey = useMemo(() => {
    const map = new Map<string, { key: string; label: string }>();

    data.forEach((point) => {
      const key = point.display_topic_id || point.cluster_id || '-1';
      const mapKey = makePointCacheKey(point.unique_key, point.creation_date, point.modification_date);
      map.set(mapKey, {
        key,
        label: point.cluster_label || clusterNameById.get(key) || '',
      });
    });

    // Backend-opened modal rows are treated as authoritative overrides.
    recentClusterInfoCache.current.forEach((info, compositeKey) => {
      map.set(compositeKey, {
        key: info.clusterId || '-1',
        label: info.clusterLabel || clusterNameById.get(info.clusterId || '-1') || '',
      });
    });

    if (selectedNode) {
      const key = selectedNode.display_topic_id || selectedNode.cluster_id || '-1';
      const mapKey = makePointCacheKey(
        selectedNode.unique_key,
        selectedNode.creation_date,
        selectedNode.modification_date,
      );
      map.set(mapKey, {
        key,
        label: selectedNode.cluster_label || clusterNameById.get(key) || '',
      });
    }

    return map;
  }, [data, selectedNode, clusterNameById, makePointCacheKey]);

  const { buckets, pointLookup } = useMemo(() => {
    const bucketMap = new Map<string, PointBucket>();
    const lookup = new Map<string, VisualPoint>();
    const hasSearchHits = searchResults.length > 0;

    visibleLabels.forEach((label) => {
      const group = clusterGroups[label];
      const clusterColorBase = clusterColors[label] || '#4b5563';
      const clusterHasSearchHit = group.customdata.some((pointData) => searchScoreMap.has(pointData.unique_key));

      group.customdata.forEach((meta, index) => {
        const isHit = searchScoreMap.has(meta.unique_key);
        const pointKey = makePointCacheKey(meta.unique_key, meta.creation_date, meta.modification_date);
        const authoritative = authoritativeClusterByKey.get(pointKey);
        const pointClusterKey = authoritative?.key || label;
        const pointClusterLabel = authoritative?.label || clusterNameById.get(pointClusterKey) || meta.cluster_label;
        const pointClusterColorBase = clusterColors[pointClusterKey] || clusterColorBase;

        // Slightly larger baseline for guaranteed visibility.
        const size = 0.028;

        // Determine per-point color when a search is active.
        // Only the exact matching chunks (isHit) should be bright; all others should be much dimmer.
        let dotColor = pointClusterColorBase;
        if (searchResults.length > 0) {
          if (isHit) {
            // Keep hit color close to cluster base but slightly lifted for visibility.
            const c = new THREE.Color(pointClusterColorBase);
            const hsl: { h: number; s: number; l: number } = { h: 0, s: 0, l: 0 };
            c.getHSL(hsl);
            // increase lightness a touch for emphasis
            c.setHSL(hsl.h, Math.min(1, hsl.s * 1.0), Math.min(1, Math.max(hsl.l, hsl.l * 1.05)));
            dotColor = c.getStyle();
          } else {
            // Non-hit chunks: desaturate and darken strongly so they recede visually.
            const c = new THREE.Color(pointClusterColorBase);
            const hsl: { h: number; s: number; l: number } = { h: 0, s: 0, l: 0 };
            c.getHSL(hsl);
            c.setHSL(hsl.h, Math.max(0, hsl.s * 0.22), Math.max(0, hsl.l * 0.16));
            dotColor = c.getStyle();
          }
        }

        // Glow: strong for hits, very subtle for non-hits when searching.
        const glowOpacity = searchResults.length > 0 ? (isHit ? 0.36 : 0.02) : 0.24;
        const quantizedSize = Math.max(0.012, Math.round(size * 1000) / 1000);
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

        const positionData = pointPositionMap.get(meta.unique_key);
        const targetPos = positionData
          ? positionData.log
          : new THREE.Vector3(group.x[index], group.y[index], group.z[index]);

        const visualPoint: VisualPoint = {
          ...meta,
          cluster_id: pointClusterKey,
          display_topic_id: pointClusterKey,
          cluster_label: pointClusterLabel,
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

    return {
      buckets: Array.from(bucketMap.values()).sort((a, b) => a.sizeMetric - b.sizeMetric),
      pointLookup: lookup,
    };
  }, [
    authoritativeClusterByKey,
    clusterNameById,
    clusterColors,
    clusterGroups,
    makePointCacheKey,
    pointPositionMap,
    searchResults.length,
    searchScoreMap,
    visibleLabels,
  ]);

  const hoveredPoint = hoveredId ? pointLookup.get(hoveredId) || null : null;
  const hoveredClusterColor = hoveredPoint ? getDotSurfaceTint(hoveredPoint.dotColor) : '#ffffff';

  const renderedClusterCenters = useMemo(() => {
    const centers = new Map<string, THREE.Vector3>();
    Object.keys(clusterGroups).forEach((label) => {
      const group = clusterGroups[label];
      if (!group || group.customdata.length === 0) return;

      let sx = 0;
      let sy = 0;
      let sz = 0;
      let count = 0;
      group.customdata.forEach((meta) => {
        const positioned = pointPositionMap.get(meta.unique_key);
        if (!positioned) return;
        sx += positioned.log.x;
        sy += positioned.log.y;
        sz += positioned.log.z;
        count += 1;
      });

      if (count > 0) {
        centers.set(label, new THREE.Vector3(sx / count, sy / count, sz / count));
      }
    });
    return centers;
  }, [clusterGroups, pointPositionMap]);

  const ClusterHeaderCard = ({
    baseCenter,
    color,
    text,
    sceneRadius,
  }: {
    baseCenter: THREE.Vector3;
    color: string;
    text: string;
    sceneRadius: number;
  }) => {
    const groupRef = useRef<THREE.Group>(null);
    const camera = useThree((state) => state.camera);

    useFrame(() => {
      if (!groupRef.current || !groupRef.current.parent) return;
      const parent = groupRef.current.parent;
      const worldBase = parent.localToWorld(baseCenter.clone());
      const cameraDir = new THREE.Vector3().subVectors(camera.position, worldBase).normalize();
      const frontOffset = Math.max(sceneRadius * 0.055, 1.2);
      const verticalOffset = Math.max(sceneRadius * 0.03, 0.9);
      const worldTarget = new THREE.Vector3(
        worldBase.x + cameraDir.x * frontOffset,
        worldBase.y + verticalOffset,
        worldBase.z + cameraDir.z * frontOffset,
      );
      const localTarget = parent.worldToLocal(worldTarget);
      groupRef.current.position.copy(localTarget);
    });

    const brightColor = mixColorWithWhite(color, 0.58);
    const borderColor = mixColorWithWhite(color, 0.52);
    const maxWidth = Math.min(2.8, Math.max(0.95, sceneRadius * 0.038));
    const widthFactor = 0.11;
    const estimatedLines = Math.max(1, Math.ceil((text.length * widthFactor) / maxWidth));
    const panelHeight = 0.2 + estimatedLines * 0.115;

    return (
      <group ref={groupRef}>
        <Billboard>
          <group>
            {/* very subtle backing panel for readability without a heavy gray slab */}
            <mesh>
              <planeGeometry args={[maxWidth + 0.16, panelHeight]} />
              <meshBasicMaterial color="#071022" transparent opacity={0.18} depthWrite={false} />
            </mesh>

            {/* clean border using 4 strips (avoids wireframe center-line artifacts) */}
            <mesh position={[0, panelHeight / 2 + 0.006, 0.001]}>
              <planeGeometry args={[maxWidth + 0.18, 0.02]} />
              <meshBasicMaterial color={borderColor} transparent opacity={0.98} depthWrite={false} />
            </mesh>
            <mesh position={[0, -panelHeight / 2 - 0.006, 0.001]}>
              <planeGeometry args={[maxWidth + 0.18, 0.02]} />
              <meshBasicMaterial color={borderColor} transparent opacity={0.98} depthWrite={false} />
            </mesh>
            <mesh position={[-(maxWidth + 0.18) / 2, 0, 0.001]}>
              <planeGeometry args={[0.02, panelHeight + 0.012]} />
              <meshBasicMaterial color={borderColor} transparent opacity={0.98} depthWrite={false} />
            </mesh>
            <mesh position={[(maxWidth + 0.18) / 2, 0, 0.001]}>
              <planeGeometry args={[0.02, panelHeight + 0.012]} />
              <meshBasicMaterial color={borderColor} transparent opacity={0.98} depthWrite={false} />
            </mesh>

            <Text
              maxWidth={maxWidth}
              textAlign="center"
              anchorX="center"
              anchorY="middle"
              color={brightColor}
              fontSize={0.115}
              lineHeight={1.12}
              outlineWidth={0.0025}
              outlineColor="#01030a"
            >
              {text}
            </Text>
          </group>
        </Billboard>
      </group>
    );
  };

  // 3D cluster headers (billboarded)
  const ClusterHeaders = ({ labels }: { labels: string[] }) => {
    return (
      <group>
        {labels.map((label) => {
          const pos = renderedClusterCenters.get(label) || new THREE.Vector3(0, 0, 0);
          const color = clusterColors[label] || '#dddddd';
          const text = clusterGroups[label]?.clusterLabel || label;
          return (
            <ClusterHeaderCard
              key={`hdr-${label}`}
              baseCenter={pos}
              color={color}
              text={text}
              sceneRadius={sceneBounds.radius * GLOBAL_LAYOUT_SPREAD}
            />
          );
        })}
      </group>
    );
  };

  // Auto-rotate group helper
  const AutoRotateGroup = ({ children, speed = 0 }: { children: React.ReactNode; speed?: number }) => {
    const ref = useRef<THREE.Group>(null);
    useFrame(() => {
      if (ref.current && speed !== 0) {
        ref.current.rotation.y += speed;
      }
    });
    return <group ref={ref}>{children}</group>;
  };

  const selectedClusterSummaries = useMemo(() => {
    if (selectedClusters.size > 0) {
      return Array.from(selectedClusters)
        .map((clusterId) => {
          const group = clusterGroups[clusterId];
          return {
            clusterId: clusterId || '?',
            clusterLabel: group?.clusterLabel || clusterId || 'Unknown cluster',
          };
        })
        .sort((a, b) => a.clusterId.localeCompare(b.clusterId, undefined, { numeric: true }));
    }

    if (selectedNode) {
      return [
        {
          clusterId: selectedNode.cluster_id && selectedNode.cluster_id !== '-1' ? selectedNode.cluster_id : '?',
          clusterLabel: selectedNode.cluster_label || 'Unknown cluster',
        },
      ];
    }

    return [] as Array<{ clusterId: string; clusterLabel: string }>;
  }, [clusterGroups, selectedClusters, selectedNode]);

  const resolveClusterForUniqueKey = useCallback(
    (uniqueKey: string, creationDate?: string, modificationDate?: string) => {
      // If the selected (open) modal corresponds to this unique key, prefer its authoritative cluster info
      if (selectedNode && selectedNode.unique_key === uniqueKey) {
        const key = selectedNode.display_topic_id || selectedNode.cluster_id || '-1';
        return { key, label: selectedNode.cluster_label || clusterNameById.get(key) || '' };
      }

      // Check cache for recently loaded backend cluster info (from successfully opened modals)
      // This ensures we use fresh backend data even for points not currently in selectedNode
      const compositeKey = makePointCacheKey(uniqueKey, creationDate, modificationDate);
      const cached = recentClusterInfoCache.current.get(compositeKey);
      if (cached) {
        return { key: cached.clusterId, label: cached.clusterLabel || clusterNameById.get(cached.clusterId) || '' };
      }

      // Otherwise, consult the initial `data` set (best-effort local authority)
      const row = data.find((d) => d.unique_key === uniqueKey);
      if (row) {
        const key = row.display_topic_id || row.cluster_id || '-1';
        return { key, label: row.cluster_label || clusterNameById.get(key) || '' };
      }

      // Fallback: empty values
      return { key: '-1', label: '' };
    },
    [selectedNode, data, clusterNameById, makePointCacheKey],
  );

  const selectedNodeColor = useMemo(() => {
    if (!selectedNode) return '#ffffff';
    const selectedClusterKey = selectedNode.display_topic_id || selectedNode.cluster_id || '';
    const dotColor = clusterColors[selectedClusterKey] || '#ffffff';
    return getDotSurfaceTint(dotColor);
  }, [clusterColors, selectedNode]);

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
    },
    [],
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
    const spreadRadius = radius * GLOBAL_LAYOUT_SPREAD;
    return [
      center.x + spreadRadius * 1.7,
      center.y + spreadRadius * 1.3,
      center.z + spreadRadius * 1.7,
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
    // Select this chunk's cluster as the sole selection
    setSelectedClusters(new Set([chunk.cluster_id]));
    // Ensure the sidebar will scroll to this note after cluster load
    setPendingScrollNoteKey(note.note_key);
    setPendingScrollNoteTitle(note.title);
    setPendingScrollTargetCluster(chunk.cluster_id);
    openSidebarNote(note, chunk.chunk_index);
  }, [openSidebarNote]);

  const handleInactiveRailClick = useCallback(
    (note: SidebarNoteData, chunk: SidebarChunkData, e?: React.MouseEvent) => {
      // Immediately scroll the note to the top of the notes list
      const leftContainer = notesListRef.current;
      const cardEl = sidebarCardRefs.current[note.note_key];
      if (leftContainer && cardEl) {
        const cardRect = cardEl.getBoundingClientRect();
        const containerRect = leftContainer.getBoundingClientRect();
        const offsetFromTop = cardRect.top - containerRect.top;
        leftContainer.scrollTop += offsetFromTop;
      }

      const shift = !!(e && e.shiftKey);
      if (shift) {
        // add to selection (toggle)
        setSelectedClusters((prev) => {
          const next = new Set(prev);
          if (next.has(chunk.cluster_id)) next.delete(chunk.cluster_id);
          else next.add(chunk.cluster_id);
          return next;
        });
      } else {
        // normal click -> select sole cluster
        setSelectedClusters(new Set([chunk.cluster_id]));
      }

      setPendingScrollNoteKey(note.note_key);
      setPendingScrollNoteTitle(note.title);
      setPendingScrollTargetCluster(chunk.cluster_id);

      // After state updates, scroll legend cluster into view (centered)
      setTimeout(() => {
        const legendEl = legendClusterRefs.current[chunk.cluster_id];
        const legendContainer = legendContainerRef.current;
        if (legendEl && legendContainer) {
          const parentRect = legendContainer.getBoundingClientRect();
          const elRect = legendEl.getBoundingClientRect();
          const offset = elRect.top - parentRect.top;
          const target = offset - legendContainer.clientHeight / 2 + elRect.height / 2;
          try {
            legendContainer.scrollTo({ top: target, behavior: 'smooth' });
          } catch (err) {
            legendContainer.scrollTop = target;
          }
        }
      }, 60);
    },
    [],
  );

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

    if (notesSortMetric === 'search' && searchResults.length > 0) {
      const firstSeenRank = new Map<string, number>();
      searchResults.forEach((r, idx) => {
        const key = `${r.title}`;
        if (!firstSeenRank.has(key)) firstSeenRank.set(key, idx);
      });

      sorted.sort((a, b) => {
        const ra = firstSeenRank.get(a.title);
        const rb = firstSeenRank.get(b.title);
        if (ra !== undefined && rb !== undefined) return ra - rb;
        if (ra !== undefined) return -1;
        if (rb !== undefined) return 1;
        return notesSortDirection === 'asc'
          ? a.title.localeCompare(b.title, undefined, { numeric: true })
          : b.title.localeCompare(a.title, undefined, { numeric: true });
      });
    } else {
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
    }

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
        cluster_name:
          clusterNameById.get(row.display_topic_id || row.cluster_id || '-1')
          || row.cluster_label
          || 'Unclustered',
        in_cluster: (row.display_topic_id || row.cluster_id || '') === selectedClusterId,
        text: null,
      }));
  }, [clusterNameById, data, selectedNode]);

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
            <div style={{ display: 'flex', gap: 10, marginBottom: 10, alignItems: 'center' }}>
              <button
                type="button"
                title="Neural Mapping"
                style={{
                  width: 36,
                  height: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 8,
                  border: 'none',
                  cursor: 'default',
                  background: '#dbf4ff',
                }}
              >
                <ZoomOutMapIcon style={{ color: '#0ea5e9' }} />
              </button>

              <button
                type="button"
                onClick={() => setHideOtherClusters((value) => !value)}
                title={hideOtherClusters ? 'Show All Clusters' : 'Show Selected (Hide Others)'}
                style={{
                  width: 36,
                  height: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 8,
                  border: 'none',
                  cursor: 'pointer',
                  background: hideOtherClusters ? '#fff1f2' : '#eef6ec',
                }}
              >
                {hideOtherClusters ? (
                  <VisibilityOffIcon style={{ color: '#dc2626' }} />
                ) : (
                  <VisibilityIcon style={{ color: '#16a34a' }} />
                )}
              </button>
            </div>

            {/* removed Peak Recency / Momentum UI as requested */}

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
                  <option value="search">Search Order</option>
                </select>
              </div>
              <div style={{ width: '130px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: '#4b5563', marginBottom: '4px' }}>
                  Direction
                </label>
                <button
                  type="button"
                  onClick={() => setNotesSortDirection((dir) => (dir === 'asc' ? 'desc' : 'asc'))}
                  title={notesSortDirection === 'asc' ? 'Increasing' : 'Decreasing'}
                  style={{
                    width: '100%',
                    padding: '8px',
                    borderRadius: '6px',
                    border: '1px solid #ccc',
                    background: '#fff',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {notesSortDirection === 'asc' ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />}
                </button>
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
              {selectedClusterSummaries.length > 0 && (
                <div
                  style={{
                    marginTop: '8px',
                    padding: '8px 10px',
                    borderRadius: '6px',
                    backgroundColor: '#f3f4f6',
                    border: '1px solid #e5e7eb',
                    fontSize: '12px',
                    lineHeight: 1.4,
                    color: '#1f2937',
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: '6px' }}>Selected Clusters</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {selectedClusterSummaries.map((summary) => {
                      const tint = clusterTints[summary.clusterId] || '#f3f4f6';
                      const border = clusterOpaqueTints[summary.clusterId] || '#e5e7eb';
                      return (
                        <div
                          key={summary.clusterId}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            backgroundColor: tint,
                            border: `1px solid ${border}`,
                            padding: '6px 8px',
                            borderRadius: '6px',
                          }}
                        >
                          <div style={{ fontWeight: 700 }}>#{summary.clusterId}</div>
                          <div style={{ color: '#4b5563', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{summary.clusterLabel}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div ref={notesListRef} style={{ flex: 1, overflowY: 'auto' }}>
              {searchResults.length === 0 && searchQuery && (
                <div style={{ color: '#666', fontStyle: 'italic', padding: '10px' }}>No results found.</div>
              )}
              {searchResults.length === 0 && !searchQuery && selectedClusters.size === 0 && (
                <div style={{ color: '#666', fontStyle: 'italic', padding: '10px' }}>
                  Select clusters to view their notes.
                </div>
              )}
              {displayedSearchResults.map((result) => {
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
                      position: 'relative',
                    }}
                  >
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedSearchNotes((prev) => {
                          const next = new Set(prev);
                          if (next.has(result.unique_key)) next.delete(result.unique_key);
                          else next.add(result.unique_key);
                          return next;
                        });
                      }}
                      aria-pressed={selectedSearchNotes.has(result.unique_key)}
                      style={{
                        position: 'absolute',
                        left: 8,
                        top: 8,
                        width: 18,
                        height: 18,
                        borderRadius: 4,
                        border: '1px solid rgba(0,0,0,0.12)',
                        background: selectedSearchNotes.has(result.unique_key) ? '#fde68a' : 'transparent',
                        cursor: 'pointer',
                      }}
                      title={selectedSearchNotes.has(result.unique_key) ? 'Deselect for similarity' : 'Select for similarity'}
                    />
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
                      {pointMatch?.modification_date && (
                        <span style={{ float: 'right', fontSize: '0.8em', color: '#666' }}>{formatDateMMDDYYYY(pointMatch.modification_date)}</span>
                      )}
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
                    <span>{note.title}</span>
                    {note.modification_date && (
                      <span style={{ marginLeft: 8, fontWeight: 500, color: '#6b7280', fontSize: '0.85em' }}>
                        {formatDateMMDDYYYY(note.modification_date)}
                      </span>
                    )}
                  </div>

                  {note.chunks.length > 1 && (
                    (() => {
                      const isSameAsSelected = selectedNode && selectedNode.title === note.title;
                      const activeClusterIds = isSameAsSelected
                        ? new Set([selectedNode!.display_topic_id || selectedNode!.cluster_id || ''])
                        : selectedClusters;
                      const currentChunkIndex = isSameAsSelected ? selectedNode!.chunk_index : null;
                      return (
                        <SegmentedRail
                          chunks={note.chunks}
                          activeClusterIds={activeClusterIds}
                          currentChunkIndex={currentChunkIndex}
                          getClusterColor={getClusterColor}
                          onActiveDotClick={(chunk, _e) => handleActiveRailClick(note, chunk)}
                          onInactiveDashClick={(chunk, e) => handleInactiveRailClick(note, chunk, e)}
                        />
                      );
                    })()
                  )}

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {(() => {
                      const rows: React.ReactNode[] = [];
                      let i = 0;
                      while (i < note.chunks.length) {
                        const chunk = note.chunks[i];
                        if (chunk.in_cluster) {
                          const preview = (chunk.text || '').trim() || '(Empty chunk)';
                          const chunkClusterColor = clusterColors[chunk.cluster_id] || '#f3f4f6';
                          const lightenedColor = new THREE.Color(chunkClusterColor).lerp(new THREE.Color('#ffffff'), 0.75).getStyle();
                          rows.push(
                            <button
                              type="button"
                              key={`snippet-${note.note_key}-${chunk.chunk_index}`}
                              onClick={() => openSidebarNote(note, chunk.chunk_index)}
                              style={{
                                border: `1px solid ${chunkClusterColor}`,
                                background: lightenedColor,
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

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <div style={{ fontSize: '0.9em', color: '#555', textAlign: 'left' }}>
                    Cluster {selectedNode.display_topic_id || (selectedNode.cluster_id && selectedNode.cluster_id !== '-1' ? selectedNode.cluster_id : '?')}:{' '}
                    {selectedNode.cluster_label}
                  </div>
                  <div style={{ fontSize: '0.85em', color: '#666' }}>{formatDateMMDDYYYY(selectedNode.modification_date)}</div>
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
                <>
                  <div style={{ fontWeight: 700 }}>{hoveredPoint.title}</div>
                  {/* Use authoritative cluster key (display_topic_id if present) and lookup label from clusterNameById */}
                  {(() => {
                      const resolved = resolveClusterForUniqueKey(
                        hoveredPoint.unique_key,
                        hoveredPoint.creation_date,
                        hoveredPoint.modification_date,
                      );
                      const displayId = resolved.key && resolved.key !== '-1' ? resolved.key : '?';
                      const displayLabel = resolved.label || '';
                      return (
                        <div style={{ color: '#333' }}>
                          Cluster {displayId}: {displayLabel}
                        </div>
                      );
                    })()}
                  <div style={{ color: '#333' }}>
                    Chunk {hoveredPoint.chunk_index + 1} of {hoveredPoint.total_chunks || '?'}
                  </div>
                </>
              </div>
            )}

            <Canvas
              camera={{
                fov: 45,
                near: 0.1,
                far: Math.max(sceneBounds.radius * GLOBAL_LAYOUT_SPREAD * 22, 180),
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
              style={{ width: '100%', height: '100%', background: '#030313' }}
            >
              <ambientLight intensity={0.9} />
              <OrbitControls
                makeDefault
                enablePan
                enableZoom
                enableRotate
                target={[sceneBounds.center.x, sceneBounds.center.y, sceneBounds.center.z]}
              />
              <AutoRotateGroup>
                {/* When hovering a point, show only that cluster's header to avoid misleading overlaps */}
                <ClusterHeaders
                  labels={hoveredPoint
                    ? [
                      resolveClusterForUniqueKey(
                        hoveredPoint.unique_key,
                        hoveredPoint.creation_date,
                        hoveredPoint.modification_date,
                      ).key,
                    ]
                    : visibleLabels}
                />
                {buckets.map((bucket) => {
                  const adaptiveBase = Math.max(DOT_RADIUS_BASE, sceneBounds.radius * 0.0018);
                  const sphereRadius = (bucket.sizeMetric / 0.02) * adaptiveBase;
                  return (
                    <DotInstances
                      key={`bucket-${bucket.key}-${bucket.points.length}`}
                      bucket={bucket}
                      sphereRadius={sphereRadius}
                      hoveredId={hoveredId}
                      highlightedId={highlightedNodeId}
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
              </AutoRotateGroup>
            </Canvas>
          </div>

          <div
            style={{
              width: '440px',
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
              userSelect: 'none',
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
                animation: marquee-scroll linear infinite;
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
            <h3 style={{ margin: '0 0 10px 0', color: '#333', userSelect: 'none' }}>
              Showing {visibleLabels.length} cluster{visibleLabels.length === 1 ? '' : 's'}
            </h3>
            <div style={{ minHeight: '30px', marginBottom: '8px', userSelect: 'none' }}>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center', fontSize: 11, flexWrap: 'wrap' }}>
                  {([
                    { key: 'recency', label: 'Recency' },
                    { key: 'momentum', label: 'Momentum' },
                    { key: 'az', label: 'A–Z' },
                    { key: 'size', label: 'Size' },
                    ...(searchResults.length > 0 ? [{ key: 'search', label: 'Search Relevance' }] : []),
                    ...(selectedClusters.size > 0 ? [{ key: 'similarity', label: 'Cluster Similarity' }] : []),
                  ] as Array<{ key: ClusterSortMetric; label: string }>).map((opt, idx, arr) => (
                    <span key={opt.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <button
                        type="button"
                        onClick={() => setClusterSortMetric(opt.key)}
                        style={{
                          background: clusterSortMetric === opt.key ? '#eef2ff' : 'transparent',
                          border: clusterSortMetric === opt.key ? '1px solid #c7d2fe' : 'none',
                          padding: '4px 6px',
                          borderRadius: 6,
                          cursor: 'pointer',
                          fontWeight: clusterSortMetric === opt.key ? 700 : 600,
                          color: '#111827',
                          fontSize: 11,
                        }}
                      >
                        {opt.label}
                      </button>
                      {idx < arr.length - 1 && <span style={{ color: '#9ca3af' }}>•</span>}
                    </span>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <button
                    type="button"
                    onClick={() => setClusterSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'))}
                    title={clusterSortDirection === 'asc' ? 'Ascending' : 'Descending'}
                    style={{ border: 'none', background: 'transparent', cursor: 'pointer', padding: 6 }}
                  >
                    {clusterSortDirection === 'asc' ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />}
                  </button>
                </div>
              </div>
            </div>
            <div ref={legendContainerRef} style={{ flex: 1, overflowY: 'auto', userSelect: 'none' }}>
              {displayedClusterLabels.map((label, idx) => {
                const group = clusterGroups[label];
                const cid = group.clusterId || label;
                const displayLabel = group.clusterLabel || label;
                const hasHits = group.customdata.some((pointData) =>
                  searchScoreMap.has(pointData.unique_key),
                );
                const color = clusterColors[label];
                const isSelected = selectedClusters.has(label);
                const isSearchDimmed = searchResults.length > 0 && !hasHits;
                // Don't let the hide-other filter dim clusters that actually have search hits
                const isFilterDimmed = hideOtherClusters && hasActiveClusterFilter && !isSelected && !hasHits;
                const isHardDimmed = isFilterDimmed || (isSearchDimmed && !isSelected);
                const isSoftDimmed = isSearchDimmed && isSelected;
                const isDimmed = isHardDimmed || isSoftDimmed;
                const rowOpacity = isHardDimmed ? 0.5 : isSoftDimmed ? 0.76 : 1;
                const dotOpacity = isHardDimmed ? 0.55 : isSoftDimmed ? 0.78 : 1;
                const idColor = isHardDimmed ? '#8a8a8a' : isSoftDimmed ? '#666' : '#555';
                const items: React.ReactNode[] = [];
                if (clusterSortMetric === 'az') {
                  const prev = displayedClusterLabels[idx - 1];
                  const prevLetter = prev ? (clusterGroups[prev]?.clusterLabel || prev).charAt(0).toUpperCase() : '';
                  const curLetter = (displayLabel || '').charAt(0).toUpperCase();
                  if (!prev || prevLetter !== curLetter) {
                    items.push(
                      <div key={`divider-${label}`} style={{ padding: '6px 6px', color: '#6b7280', fontWeight: 700 }}>
                        {curLetter}
                      </div>,
                    );
                  }
                } else if (clusterSortMetric === 'recency' || clusterSortMetric === 'momentum') {
                  const prev = displayedClusterLabels[idx - 1];
                  const currentSection = getClusterSectionTitle(label);
                  const prevSection = prev ? getClusterSectionTitle(prev) : '';
                  if (!prev || currentSection !== prevSection) {
                    items.push(
                      <div key={`divider-${label}`} style={{ padding: '6px 6px', color: '#6b7280', fontWeight: 700 }}>
                        {currentSection}
                      </div>,
                    );
                  }
                }

                items.push(
                  <div
                    key={label}
                    ref={(el) => {
                      legendClusterRefs.current[label] = el;
                    }}
                    className={`cluster-row ${isSelected ? 'cluster-row-selected' : ''}`}
                    onClick={(e) => {
                      const isShiftClick = e.shiftKey;
                      if (isShiftClick) {
                        // Shift+click: Add/toggle cluster to selection
                        setSelectedClusters((prev) => {
                          const next = new Set(prev);
                          if (next.has(label)) {
                            next.delete(label);
                          } else {
                            next.add(label);
                          }
                          return next;
                        });
                      } else {
                        // Regular click: Select only this cluster
                        setSelectedClusters(new Set([label]));
                      }
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
                      {cid}
                    </div>
                    <ClusterLabel
                      label={displayLabel}
                      hasHits={hasHits}
                      isSelected={isSelected}
                      isDimmed={isDimmed}
                    />
                    {clusterSortMetric === 'size' && (
                      <span style={{ marginLeft: '8px', color: '#6b7280', fontWeight: 600, fontSize: '0.85em' }}>
                        {group.customdata.length} chunk{group.customdata.length === 1 ? '' : 's'}
                      </span>
                    )}
                  </div>,
                );

                return items;
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
