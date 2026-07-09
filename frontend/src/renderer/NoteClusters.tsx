import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Canvas, type ThreeEvent, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import LanguageIcon from '@mui/icons-material/Language';
import HistoryIcon from '@mui/icons-material/History';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import { RadialSimilarityHub } from './RadialSimilarityHub';
import { MetaClusterTree } from './MetaClusterTree';

interface NotePoint {
  unique_key: string;
  title: string;
  chunk_index: number;
  total_chunks?: number;
  cluster_id?: string;
  base_topic_id?: string;
  display_topic_id?: string;
  cluster_label: string;
  cluster_color?: string;
  dot_color?: string;
  umap_x: number;
  umap_y: number;
  umap_z: number;
  display_x?: number;   // ← ADD
  display_y?: number;   // ← ADD
  display_z?: number;   // ← ADD
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
  dot_color?: string;
  cluster_color?: string;
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

interface HistorySidebarNoteData extends SidebarNoteData {
  opened_at?: string;
  last_opened_at?: string;
  opened_count?: number;
}

type ClusterOrderMode = 'spike' | 'momentum';
type SidebarMode = 'global' | 'history';
type SearchLegendOrderMode = 'results' | 'similarity';
type NotesSortMetric = 'modified' | 'size' | 'search';
type SortDirection = 'desc' | 'asc';
type ClusterSortMetric = 'recency' | 'momentum' | 'az' | 'size' | 'search' | 'history' | 'similarity';

interface ClusterPointMeta {
  unique_key: string;
  title: string;
  chunk_index: number;
  total_chunks?: number;
  cluster_id: string;
  base_topic_id?: string;
  display_topic_id?: string;
  cluster_label: string;
  cluster_color?: string;
  dot_color?: string;
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
  clusterColor?: string;
  chunkCount?: number;
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
  opacity: number;
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

const mixColorWithDark = (baseColor: string, darkMix: number) => {
  const mixed = new THREE.Color(baseColor).lerp(new THREE.Color('#101827'), darkMix);
  return mixed.getStyle();
};

const getDotSurfaceTint = (dotColor: string) => mixColorWithWhite(dotColor, 0.35);
const NOTE_SURFACE_SATURATION = 0.5;

const DOT_RADIUS_BASE = 0.016;
const GLOBAL_LAYOUT_SPREAD = 2.45;
const CLUSTER_CENTROID_SPREAD = 2.4;
const CLUSTER_POINT_SPREAD = 6.0;

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
        // only show glow on the hovered/highlighted point, no glow on others
        let glowScale = 0;
        if (isHovered) {
          glowScale = scale * 3.0;
        } else if (isHighlighted) {
          glowScale = scale * 2.2;
        } else if (hoveredId || highlightedId) {
          glowScale = scale * 0.08;
        }
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
        <meshBasicMaterial color={bucket.color} transparent opacity={bucket.opacity} toneMapped={false} />
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
  const [viewMode, setViewMode] = useState<SidebarMode>('global');
  const [historyDates, setHistoryDates] = useState<string[]>([]);
  const [selectedHistoryDateIndex, setSelectedHistoryDateIndex] = useState<number>(-1);
  const [historySidebarNotes, setHistorySidebarNotes] = useState<HistorySidebarNoteData[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
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
   const [clusterColorsFromAPI, setClusterColorsFromAPI] = useState<Record<string, string>>({});
  const plotAreaRef = useRef<HTMLDivElement>(null);
  const sidebarCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const legendClusterRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const legendContainerRef = useRef<HTMLDivElement | null>(null);
  const notesListRef = useRef<HTMLDivElement | null>(null);
  const recentClusterInfoCache = useRef<Map<string, { clusterId: string; clusterLabel: string; color?: string }>>(new Map());
  // Orbit controls ref and camera tween state for focus animation
  const controlsRef = useRef<any>(null);

  const cameraTweenRef = useRef<{
    active: boolean;
    startTime: number;
    duration: number;
    target: THREE.Vector3;
    fromAzimuth: number;
    toAzimuth: number;
    fromPolar: number;
    toPolar: number;
    fromRadius: number;
    toRadius: number;
  } | null>(null);

  const makePointCacheKey = useCallback(
    (uniqueKey: string, _creationDate?: string | null, _modificationDate?: string | null) => uniqueKey, // Drop dates to ensure hits match
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

       // Compute cluster_color from the resolved display_topic_id / cluster_id
       const resolvedClusterKey = display_topic_id || cluster_id || '-1';
       const computedClusterColor = clusterColors[resolvedClusterKey]
         || response.data?.dot_color
         || response.data?.cluster_color
         || null;

       setSelectedNode({
          ...response.data,
         cluster_id,
         display_topic_id,
         base_topic_id,
         cluster_label,
         cluster_color: computedClusterColor,
         creation_date: creationDate,
         modification_date: modificationDate,
         unique_key: newUniqueKey,
        });
      // Cache the authoritative cluster info from backend for this unique_key
      const cacheClusId = display_topic_id || cluster_id || '-1';
      const color = response.data?.dot_color || response.data?.cluster_color;
      const cacheKey = makePointCacheKey(newUniqueKey);
      recentClusterInfoCache.current.set(cacheKey, {
        clusterId: cacheClusId,
        clusterLabel: cluster_label || '',
        color: color, // Store the color from the backend response
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

  const formatTimeHHMM = (iso?: string | number | null) => {
    if (!iso) return '';
    const t = typeof iso === 'number' ? new Date(iso) : new Date(String(iso));
    if (Number.isNaN(t.getTime())) return '';
    return new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
    }).format(t);
  };

   useEffect(() => {
     const fetchData = async () => {
       try {
          // Try shaped points first, fall back to raw points if unavailable
        let response = null;
        try {
          response = await axios.get('http://127.0.0.1:8000/points_shaped');
         } catch (err) {
          console.warn('points_shaped fetch failed, will try /points', err);
         }

        if (!response || !Array.isArray(response.data) || response.data.length === 0) {
           // fallback to legacy endpoint
          try {
            response = await axios.get('http://127.0.0.1:8000/points');
           } catch (err) {
            console.error('Fallback /points fetch failed', err);
            response = null;
           }
         }

        const points = response && Array.isArray(response.data) ? response.data : [];
        console.info(`Fetched ${points.length} points from backend`);
        setData(points);
       } catch (error) {
        console.error('Error fetching data:', error);
       } finally {
        setLoading(false);
       }
     };

     fetchData();
   }, []);

   // Fetch cluster colors once from the single source of truth
   useEffect(() => {
     let active = true;
     (async () => {
       try {
         const res = await axios.get('http://127.0.0.1:8000/cluster_colors');
         if (active && res.data) {
           setClusterColorsFromAPI(res.data as Record<string, string>);
         }
       } catch (err) {
         console.warn('Failed to fetch cluster_colors:', err);
       }
     })();
     return () => { active = false; };
   }, []);

  useEffect(() => {
    let active = true;

    const fetchHistoryDates = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/history/dates');
        const dates = Array.isArray(response.data?.dates) ? response.data.dates.filter((value: unknown) => typeof value === 'string') : [];
        if (!active) return;

        setHistoryDates(dates);
        setSelectedHistoryDateIndex((current) => {
          if (dates.length === 0) return -1;
          if (current >= 0 && current < dates.length) return current;
          return dates.length - 1;
        });
      } catch (error) {
        console.error('Error loading history dates:', error);
        if (active) {
          setHistoryDates([]);
          setSelectedHistoryDateIndex(-1);
        }
      }
    };

    fetchHistoryDates();

    return () => {
      active = false;
    };
  }, []);

  const selectedHistoryDate = selectedHistoryDateIndex >= 0 ? historyDates[selectedHistoryDateIndex] || null : null;
  const isHistoryMode = viewMode === 'history';
  const historyTitleSet = useMemo(() => new Set(historySidebarNotes.map((note) => note.title)), [historySidebarNotes]);
  const canGoBackHistory = historyDates.length > 0 && selectedHistoryDateIndex > 0;
  const canGoForwardHistory = historyDates.length > 0 && selectedHistoryDateIndex >= 0 && selectedHistoryDateIndex < historyDates.length - 1;
  const historyClusterFirstSeenRank = useMemo(() => {
    const rankMap = new Map<string, number>();
    historySidebarNotes.forEach((note, noteIndex) => {
      note.chunks.forEach((chunk) => {
        if (!rankMap.has(chunk.cluster_id)) {
          rankMap.set(chunk.cluster_id, noteIndex);
        }
      });
    });
    return rankMap;
  }, [historySidebarNotes]);

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
    if (viewMode !== 'global') {
      if (savedClustersBeforeSearch.size > 0) {
        setSelectedClusters(new Set(savedClustersBeforeSearch));
        setSavedClustersBeforeSearch(new Set());
      }
      return;
    }

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

  // Camera focus helpers: animate orbit to bring a cluster to the front
  function normalizeAngle(angle: number) {
    while (angle > Math.PI) angle -= Math.PI * 2;
    while (angle < -Math.PI) angle += Math.PI * 2;
    return angle;
  }

  function shortestAngleDelta(from: number, to: number) {
    return normalizeAngle(to - from);
  }

  function focusClusterByOrbit(clusterId: string) {
    const controls = controlsRef.current;
    const focusMetrics = getClusterFocusMetrics(clusterId);
    if (!controls || !focusMetrics) return;
    if (cameraTweenRef.current?.active) return;

    const { center: centroid, radius: clusterRadius } = focusMetrics;
    const center = sceneBounds.center;
    const dir = new THREE.Vector3(centroid.x - center.x, centroid.y - center.y, centroid.z - center.z);
    if (dir.lengthSq() < 1e-8) return;

    const currentAzimuth = controls.getAzimuthalAngle();
    const currentPolar = controls.getPolarAngle();
    const currentRadius = controls.getDistance();

    const desiredAzimuth = Math.atan2(dir.x, dir.z);
    const desiredPolar = Math.atan2(Math.sqrt(dir.x * dir.x + dir.z * dir.z), dir.y);

    // Keep the focus distance tied to the cluster's own rendered size.
    // Smaller clusters are shown more tightly; larger clusters get a bit more space.
    const minFocusDistance = visualSceneRadius * 0.06;
    const clusterFocusDistance = clusterRadius * 1.1;
    const desiredRadius = Math.max(minFocusDistance, clusterFocusDistance);

    cameraTweenRef.current = {
      active: true,
      startTime: performance.now(),
      duration: 900,
      target: centroid.clone(),
      fromAzimuth: currentAzimuth,
      toAzimuth: currentAzimuth + shortestAngleDelta(currentAzimuth, desiredAzimuth),
      fromPolar: currentPolar,
      toPolar: desiredPolar,
      fromRadius: currentRadius,
      toRadius: desiredRadius,
    };
  }

  const CameraOrbitAnimator = () => {
    const { camera } = useThree();
    useFrame(() => {
      const controls = controlsRef.current;
      const tween = cameraTweenRef.current;
      if (!controls || !tween || !tween.active) return;

      const now = performance.now();
      const rawT = (now - tween.startTime) / tween.duration;
      const t = Math.min(Math.max(rawT, 0), 1);
      const eased = 1 - Math.pow(1 - t, 3);

      const azimuth = THREE.MathUtils.lerp(tween.fromAzimuth, tween.toAzimuth, eased);
      const polar = THREE.MathUtils.lerp(tween.fromPolar, tween.toPolar, eased);
      const radius = THREE.MathUtils.lerp(tween.fromRadius, tween.toRadius, eased);

      const sinPolar = Math.sin(polar);
      camera.position.set(
        tween.target.x + radius * sinPolar * Math.sin(azimuth),
        tween.target.y + radius * Math.cos(polar),
        tween.target.z + radius * sinPolar * Math.cos(azimuth),
      );

      controls.target.copy(tween.target);
      controls.update();

      if (t >= 1 && cameraTweenRef.current) cameraTweenRef.current.active = false;
    });
    return null;
  };











  const { clusterGroups, clusterColors, clusterTints, clusterHoverTints, clusterOpaqueTints } = useMemo(() => {
    const processingGroups: Record<string, ClusterGroup> = {};

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
            clusterColor: point.cluster_color || point.dot_color,
          };
        }
        if (!processingGroups[clusterKey].clusterColor && (point.cluster_color || point.dot_color)) {
          processingGroups[clusterKey].clusterColor = point.cluster_color || point.dot_color;
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
      });
    }

    const processingColors: Record<string, string> = {};
    const processingTints: Record<string, string> = {};
    const processingHoverTints: Record<string, string> = {};
    const processingOpaqueTints: Record<string, string> = {};

    Object.keys(processingGroups).forEach((label) => {
      const points = processingGroups[label];
      if (points.clusterColor) {
        processingColors[label] = points.clusterColor;
        return;
      }
      processingColors[label] = '#6b7280';
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
    } else if (clusterSortMetric === 'history') {
      list = list.slice().sort((a, b) => {
        const rankA = historyClusterFirstSeenRank.get(a);
        const rankB = historyClusterFirstSeenRank.get(b);

        if (rankA !== undefined && rankB !== undefined) {
          return clusterSortDirection === 'asc' ? rankB - rankA : rankA - rankB;
        }
        if (rankA !== undefined) return -1;
        if (rankB !== undefined) return 1;

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
  }, [sortedLabels, clusterGroups, clusterSortMetric, clusterSortDirection, clusterOrderScores, clusterAverageRelevance, clusterCentroids, selectedClustersCentroid, historyClusterFirstSeenRank]);

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
    if (clusterSortMetric === 'history' && historySidebarNotes.length === 0) {
      setClusterSortMetric('momentum');
    }
    if (clusterSortMetric === 'similarity' && selectedClusters.size === 0) {
      setClusterSortMetric('momentum');
    }
  }, [clusterSortMetric, historySidebarNotes.length, searchResults.length, selectedClusters.size]);


  const hasActiveClusterFilter = selectedClusters.size > 0;
  const visibleLabels = displayedClusterLabels;

  useEffect(() => {
    let active = true;

    const fetchSidebar = async () => {
      if (viewMode === 'history') {
        if (active) {
          setSidebarNotes([]);
          setLoadedSidebarCluster(null);
        }
        return;
      }

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
  }, [selectedClusters, viewMode]);

  useEffect(() => {
    let active = true;

    const fetchHistorySidebar = async () => {
      if (!isHistoryMode || !selectedHistoryDate) {
        if (active) {
          setHistorySidebarNotes([]);
          setIsLoadingHistory(false);
        }
        return;
      }

      setIsLoadingHistory(true);
      try {
        const response = await axios.get(`http://127.0.0.1:8000/history/day/${encodeURIComponent(selectedHistoryDate)}`);
        const notes = Array.isArray(response.data?.notes) ? response.data.notes : [];

        if (active) {
          setHistorySidebarNotes(notes);
        }
      } catch (error) {
        console.error('Error loading day history:', error);
        if (active) {
          setHistorySidebarNotes([]);
        }
      } finally {
        if (active) setIsLoadingHistory(false);
      }
    };

    fetchHistorySidebar();

    return () => {
      active = false;
    };
  }, [isHistoryMode, selectedHistoryDate]);

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
      const x = point.display_x as number;
      const y = point.display_y as number;
      const z = point.display_z as number;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    });

    const center = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const dx = maxX - minX;
    const dy = maxY - minY;
    const dz = maxZ - minZ;
    const radius = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5, 1);

    return { center, radius };
  }, [data]);

  const pointPositionMap = useMemo(() => {
    const map = new Map<string, { log: THREE.Vector3 }>();
    data.forEach((point) => {
      if (point.display_x === undefined || point.display_y === undefined || point.display_z === undefined) {
        return;
      }
      map.set(point.unique_key, { log: new THREE.Vector3(point.display_x, point.display_y, point.display_z) });
    });
    return map;
  }, [data]);

  const displayPointPositionMap = useMemo(() => {
    const map = new Map<string, { log: THREE.Vector3 }>();
    const sceneCenter = sceneBounds.center;

    Object.keys(clusterGroups).forEach((label) => {
      const centroid = clusterCentroids.get(label);
      const group = clusterGroups[label];
      if (!group || group.customdata.length === 0) return;

      const centroidVector = centroid ? new THREE.Vector3(centroid.x, centroid.y, centroid.z) : null;

      group.customdata.forEach((meta) => {
        const positioned = pointPositionMap.get(meta.unique_key);
        if (!positioned) return;

        if (!centroidVector) {
          map.set(meta.unique_key, positioned);
          return;
        }

        const scaledCentroid = sceneCenter.clone().lerp(centroidVector, CLUSTER_CENTROID_SPREAD);
        const offset = positioned.log.clone().sub(centroidVector);
        map.set(meta.unique_key, {
          log: scaledCentroid.clone().add(offset.multiplyScalar(CLUSTER_POINT_SPREAD)),
        });
      });
    });

    return map;
  }, [clusterCentroids, clusterGroups, pointPositionMap, sceneBounds.center]);

  const getClusterFocusMetrics = useCallback(
    (clusterId: string) => {
      const group = clusterGroups[clusterId];
      if (!group || group.customdata.length === 0) return null;

      const points = group.customdata
        .map((meta) => displayPointPositionMap.get(meta.unique_key)?.log)
        .filter((point): point is THREE.Vector3 => !!point);

      if (points.length === 0) return null;
      if (points.length === 1) {
        return { center: points[0].clone(), radius: 0 };
      }

      const initialCenter = points.reduce(
        (acc, point) => acc.add(point),
        new THREE.Vector3(0, 0, 0),
      ).divideScalar(points.length);

      const rankedPoints = points
        .map((point) => ({ point, distance: point.distanceTo(initialCenter) }))
        .sort((a, b) => a.distance - b.distance);

      const keepCount = Math.max(3, Math.ceil(rankedPoints.length * 0.8));
      const focusPoints = rankedPoints.slice(0, keepCount).map((entry) => entry.point);

      const center = focusPoints
        .reduce((acc, point) => acc.add(point), new THREE.Vector3(0, 0, 0))
        .divideScalar(focusPoints.length);

      const radius = focusPoints.reduce((max, point) => Math.max(max, point.distanceTo(center)), 0);

      return { center, radius };
    },
    [clusterGroups, displayPointPositionMap],
  );

  const visualSceneRadius = sceneBounds.radius * CLUSTER_POINT_SPREAD;

  const authoritativeClusterByKey = useMemo(() => {
    const map = new Map<string, { key: string; label: string; color?: string }>();

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
        color: info.color,
      });
    });

    if (selectedNode) {
      const key = selectedNode.display_topic_id || selectedNode.cluster_id || '-1';
      const mapKey = makePointCacheKey(selectedNode.unique_key);
      map.set(mapKey, {
        key,
        label: selectedNode.cluster_label || clusterNameById.get(key) || '',
        color: selectedNode.dot_color || selectedNode.cluster_color,
      });
    }

    return map;
  }, [data, selectedNode, clusterNameById, makePointCacheKey]);

  const { buckets, pointLookup } = useMemo(() => {
    const bucketMap = new Map<string, PointBucket>();
    const lookup = new Map<string, VisualPoint>();
    const isSearchMode = !isHistoryMode && debouncedQuery.trim().length > 0;
    const hasSearchHits = !isHistoryMode && searchResults.length > 0;

    visibleLabels.forEach((label) => {
      const group = clusterGroups[label];
      const clusterColorBase = clusterColors[label] || '#4b5563';

      group.customdata.forEach((meta, index) => {
        const isHistoryHit = isHistoryMode && historyTitleSet.has(meta.title);
        const isHit = isHistoryMode ? isHistoryHit : hasSearchHits && searchScoreMap.has(meta.unique_key);
        const pointKey = makePointCacheKey(meta.unique_key);
        const authoritative = authoritativeClusterByKey.get(pointKey);
        const pointClusterKey = authoritative?.key || label;
        const pointClusterLabel = authoritative?.label || clusterNameById.get(pointClusterKey) || meta.cluster_label;

        // Search/history hits get a larger, brighter glyph; misses get a smaller, muted one.
        const size = isSearchMode || isHistoryMode ? (isHit ? 0.05 : 0.015) : 0.028;

        // SOURCE OF TRUTH: Use the color the server calculated for this specific row.
        // Fallback only if it's missing (which it shouldn't be now).
        const dotColor = authoritative?.color || meta.dot_color || clusterColors[label] || '#6b7280';
        const visualColor = (isSearchMode || isHistoryMode)
          ? (isHit ? mixColorWithWhite(dotColor, 0.12) : mixColorWithDark(dotColor, 0.72))
          : dotColor;

        // Glow: strong for hits in search/history mode, otherwise no glow by default (only on hover).
        const glowOpacity = (isSearchMode || isHistoryMode) ? (isHit ? 0.42 : 0.02) : 0.0;
        const opacity = (isSearchMode || isHistoryMode) ? (isHit ? 0.98 : 0.18) : 0.9;
        const quantizedSize = Math.max(0.012, Math.round(size * 1000) / 1000);
        const bucketKey = `${quantizedSize}|${visualColor}|${glowOpacity}|${opacity}`;
        if (!bucketMap.has(bucketKey)) {
          bucketMap.set(bucketKey, {
            key: bucketKey,
            sizeMetric: quantizedSize,
            color: visualColor,
            glowOpacity,
            opacity,
            points: [],
          });
        }

        const positionData = displayPointPositionMap.get(meta.unique_key);
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
    debouncedQuery,
    makePointCacheKey,
    displayPointPositionMap,
    historyTitleSet,
    isHistoryMode,
    searchResults.length,
    searchScoreMap,
    visibleLabels,
  ]);

  const hoveredPoint = hoveredId ? pointLookup.get(hoveredId) || null : null;
  const hoveredClusterColor = hoveredPoint ? hoveredPoint.dotColor : '#ffffff';
  const hoveredClusterSurfaceColor = hoveredPoint
    ? mixColorWithWhite(hoveredClusterColor, 1 - NOTE_SURFACE_SATURATION)
    : hoveredClusterColor;

  const renderedClusterCenters = useMemo(() => {
    const centers = new Map<string, THREE.Vector3>();
    Object.keys(clusterGroups).forEach((label) => {
      const focusMetrics = getClusterFocusMetrics(label);
      if (focusMetrics) {
        centers.set(label, focusMetrics.center);
      }
    });
    return centers;
  }, [clusterGroups, getClusterFocusMetrics]);

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
            chunkCount: group?.customdata?.length ?? group?.chunkCount ?? null,
          };
         })
         .sort((a, b) => a.clusterId.localeCompare(b.clusterId, undefined, { numeric: true }));
    }

    if (selectedNode) {
      return [
        {
          clusterId: selectedNode.cluster_id && selectedNode.cluster_id !== '-1' ? selectedNode.cluster_id : '?',
          clusterLabel: selectedNode.cluster_label || 'Unknown cluster',
          chunkCount: selectedNode.total_chunks ?? null,
        },
      ];
    }

    return [] as Array<{ clusterId: string; clusterLabel: string; chunkCount: number | null }>;
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
    return clusterColors[selectedClusterKey] || '#ffffff';
  }, [clusterColors, selectedNode]);

  const selectedNodeSurfaceColor = useMemo(() => {
    if (!selectedNode) return '#ffffff';
    return mixColorWithWhite(selectedNodeColor, 1 - NOTE_SURFACE_SATURATION);
  }, [selectedNode, selectedNodeColor]);

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
    const { center } = sceneBounds;
    const spreadRadius = visualSceneRadius * GLOBAL_LAYOUT_SPREAD;
    return [
      center.x + spreadRadius * 0.18,
      center.y + spreadRadius * 0.14,
      center.z + spreadRadius * 0.18,
    ] as [number, number, number];
  }, [sceneBounds, visualSceneRadius]);

  const openSidebarNote = useCallback(
    (note: SidebarNoteData, preferredChunkIndex?: number) => {
      const inClusterChunk = note.chunks.find((chunk) => chunk.in_cluster);
      const fallbackChunk = note.chunks[0];
      const chunkIndex = preferredChunkIndex ?? inClusterChunk?.chunk_index ?? fallbackChunk?.chunk_index ?? 0;

      // Use the first selected cluster or the note's cluster
      const targetCluster = isHistoryMode
        ? inClusterChunk?.cluster_id || fallbackChunk?.cluster_id
        : Array.from(selectedClusters)[0] || inClusterChunk?.cluster_id || fallbackChunk?.cluster_id;
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
    [clusterGroups, isHistoryMode, selectedClusters],
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

  const handleInactiveRailClickForHistory = useCallback(
    (chunk: SidebarChunkData) => {
      // In history mode, clicking a dash selects that cluster instead of opening the note
      setSelectedClusters(new Set([chunk.cluster_id]));
    },
    [],
  );

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

       // Always select sole cluster — no multi-select
       setSelectedClusters(new Set([chunk.cluster_id]));
       focusClusterByOrbit(chunk.cluster_id);

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
      return clusterColors[clusterId] || '#6b7280';
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

  const displayedHistorySidebarNotes = useMemo(() => {
    const deduped = historySidebarNotes.map((note) => {
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

    return deduped.slice().sort((a, b) => {
      const leftTs = Date.parse(String(a.opened_at || a.modification_date || ''));
      const rightTs = Date.parse(String(b.opened_at || b.modification_date || ''));
      const safeLeft = Number.isFinite(leftTs) ? leftTs : Number.NEGATIVE_INFINITY;
      const safeRight = Number.isFinite(rightTs) ? rightTs : Number.NEGATIVE_INFINITY;
      if (safeRight !== safeLeft) return safeRight - safeLeft;
      return a.title.localeCompare(b.title, undefined, { numeric: true });
    });
  }, [historySidebarNotes]);

  // Resolve authoritative cluster color from the API for header accent
  const modalClusterKey = selectedNode?.display_topic_id || selectedNode?.cluster_id || '';
  const modalApiColor = modalClusterKey
    ? (clusterColorsFromAPI[modalClusterKey] || selectedNode.cluster_color || selectedNode.dot_color || null)
    : null;

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
      (chunkIndexOrChunk: number | SidebarChunkData) => {
        if (!selectedNode) return;
        // Handle both number and chunk object for backward compatibility
        const idx = typeof chunkIndexOrChunk === 'number'
          ? chunkIndexOrChunk
          : chunkIndexOrChunk.chunk_index;
        fetchNoteContent(
          selectedNode.title,
          idx,
          selectedNode.cluster_id,
          selectedNode.cluster_label,
          selectedNode.display_topic_id,
          selectedNode.base_topic_id,
          selectedNode.creation_date,
          selectedNode.modification_date,
        );
      },
      [selectedNode, fetchNoteContent],
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
          {/* Left column: Notes list */}
          <div
            style={{
              width: '26%',
              minWidth: '240px',
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
            {/* removed Peak Recency / Momentum UI as requested */}

              <div
               style={{
                 display: 'none',
                 alignItems: 'center',
                 justifyContent: 'space-between',
                 gap: '8px',
                 marginBottom: '10px',
                 padding: '8px 10px',
                 borderRadius: '8px',
                 border: '1px solid #e5e7eb',
                 background: '#fff',
               }}
             >
               <button
                type="button"
                onClick={() => {
                  if (!canGoBackHistory) return;
                  setSelectedHistoryDateIndex((current) => Math.max(0, current - 1));
                  setViewMode('history');
                }}
                disabled={!canGoBackHistory}
                title="Previous day"
                style={{
                  width: '34px',
                  height: '34px',
                  borderRadius: '6px',
                  border: '1px solid #d1d5db',
                  background: '#f9fafb',
                  cursor: canGoBackHistory ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <ArrowForwardIcon style={{ transform: 'rotate(180deg)' }} />
              </button>
                {/* Day header hidden for now — will be dealt with later */}
                <div style={{ flex: 1, textAlign: 'center', minWidth: 0, display: 'none' }}>
                 <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '2px', display: 'none' }}>Day</div>
                 <div style={{ fontWeight: 700, color: '#111827', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'none' }}>
                   {selectedHistoryDate || 'No history dates'}
                 </div>
               </div>
              <button
                type="button"
                onClick={() => {
                  if (!canGoForwardHistory) return;
                  setSelectedHistoryDateIndex((current) => Math.min(historyDates.length - 1, Math.max(0, current + 1)));
                  setViewMode('history');
                }}
                disabled={!canGoForwardHistory}
                title="Next day"
                style={{
                  width: '34px',
                  height: '34px',
                  borderRadius: '6px',
                  border: '1px solid #d1d5db',
                  background: '#f9fafb',
                  cursor: canGoForwardHistory ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <ArrowForwardIcon />
              </button>
            </div>

            {!isHistoryMode && <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
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
            </div>}

            {!isHistoryMode && <div style={{ marginBottom: '15px' }}>
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
                             alignItems: 'flex-start',
                             gap: '8px',
                             backgroundColor: tint,
                             border: `1px solid ${border}`,
                             padding: '6px 8px',
                             borderRadius: '6px',
                            }}
                           >
                            <div style={{ fontWeight: 700, flexShrink: 0 }}>#{summary.clusterId}</div>
                            <div style={{ flex: 1, color: '#4b5563', wordBreak: 'break-word', whiteSpace: 'normal', lineHeight: 1.3, minWidth: 0 }}>{summary.clusterLabel}</div>
                            {summary.chunkCount != null && (
                              <div style={{ flexShrink: 0, marginLeft: 'auto', color: '#9ca3af', fontSize: '11px', whiteSpace: 'nowrap' }}>
                                {summary.chunkCount} chunks
                              </div>
                            )}
                          </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>}

            {isHistoryMode && (
              <div ref={notesListRef} style={{ flex: 1, overflowY: 'auto' }}>
                <div style={{ color: '#6b7280', fontSize: '12px', marginBottom: '10px', lineHeight: 1.4 }}>
                  {isLoadingHistory
                    ? 'Loading notes opened on this day...'
                    : selectedHistoryDate
                      ? `Notes opened on ${selectedHistoryDate}`
                      : 'No daily history available.'}
                </div>

                {displayedHistorySidebarNotes.length === 0 && !isLoadingHistory && (
                  <div style={{ color: '#666', fontStyle: 'italic', padding: '10px' }}>
                    No notes were opened on this day.
                  </div>
                )}

                {displayedHistorySidebarNotes.map((note) => {
                  const openLabel = note.opened_at ? formatTimeHHMM(note.opened_at) : '';
                  // Use selectedClusters for dots/dashes logic (same as regular mode)
                  // If no clusters selected, show all chunks as dots
                  const historyActiveClusterIds = selectedClusters.size > 0
                    ? selectedClusters
                    : new Set(note.chunks.map((chunk) => chunk.cluster_id));
                  const isSelectedNote = !!selectedNode
                    && selectedNode.title === note.title
                    && selectedNode.creation_date === note.creation_date
                    && selectedNode.modification_date === note.modification_date;
                  const currentChunkIndex = isSelectedNote ? selectedNode!.chunk_index : null;
                  return (
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
                        {note.creation_date && (
                          <span style={{ marginLeft: 8, fontWeight: 500, color: '#6b7280', fontSize: '0.85em' }}>
                            {formatDateMMDDYYYY(note.creation_date)}
                          </span>
                        )}
                        {openLabel && (
                          <span style={{ marginLeft: 8, fontWeight: 600, color: '#92400e', fontSize: '0.85em' }}>
                            Open at {openLabel}
                          </span>
                        )}
                        {note.opened_count ? (
                          <span style={{ marginLeft: 8, fontWeight: 500, color: '#a16207', fontSize: '0.82em' }}>
                            {note.opened_count}x
                          </span>
                        ) : null}
                      </div>

                      {note.chunks.length >= 1 && (
                        <SegmentedRail
                          chunks={note.chunks}
                          activeClusterIds={historyActiveClusterIds}
                          currentChunkIndex={currentChunkIndex}
                          getClusterColor={getClusterColor}
                          onActiveDotClick={(chunk, _e) => openSidebarNote(note, chunk.chunk_index)}
                          onInactiveDashClick={(chunk, _e) => {
                            // In history mode, clicking a dash selects that cluster instead of opening the note
                            handleInactiveRailClickForHistory(chunk);
                          }}
                        />
                      )}

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {(() => {
                          const rows: React.ReactNode[] = [];
                          let i = 0;
                          while (i < note.chunks.length) {
                            const chunk = note.chunks[i];
                            const isActiveChunk = historyActiveClusterIds.has(chunk.cluster_id);

                            if (isActiveChunk) {
                               // Show dot chunk - only if we have a selection, or show all if no selection
                              if (selectedClusters.size > 0) {
                                const preview = (chunk.text || '').trim() || '(Empty chunk)';
                                 // Resolve cluster color from API to match right column
                                const apiChunkColor = clusterColorsFromAPI[chunk.cluster_id];
                                const resolvedColor = apiChunkColor || clusterColors[chunk.cluster_id] || '#f3f4f6';
                                const lightenedColor = new THREE.Color(resolvedColor).lerp(new THREE.Color('#ffffff'), 0.75).getStyle();
                                rows.push(
                                  <button
                                    type="button"
                                    key={`history-snippet-${note.note_key}-${chunk.chunk_index}`}
                                    onClick={() => openSidebarNote(note, chunk.chunk_index)}
                                    style={{
                                      border: `1px solid ${resolvedColor}`,
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
                               } else {
                                 // No selection - show all chunks
                                const preview = (chunk.text || '').trim() || '(Empty chunk)';
                                 // Resolve cluster color from API to match right column
                                const apiChunkColor = clusterColorsFromAPI[chunk.cluster_id];
                                const resolvedColor = apiChunkColor || clusterColors[chunk.cluster_id] || '#f3f4f6';
                                const lightenedColor = new THREE.Color(resolvedColor).lerp(new THREE.Color('#ffffff'), 0.75).getStyle();
                                rows.push(
                                  <button
                                    type="button"
                                    key={`history-snippet-${note.note_key}-${chunk.chunk_index}`}
                                    onClick={() => openSidebarNote(note, chunk.chunk_index)}
                                    style={{
                                      border: `1px solid ${resolvedColor}`,
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
                              }
                              i += 1;
                              continue;
                            }

                            // Chunk is not in selected cluster - show as gap
                            let gapCount = 0;
                            while (i < note.chunks.length && !historyActiveClusterIds.has(note.chunks[i].cluster_id)) {
                              gapCount += 1;
                              i += 1;
                            }

                            rows.push(
                              <div
                                key={`history-gap-${note.note_key}-${i}-${gapCount}`}
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
                  );
                })}
              </div>
            )}

            <div ref={notesListRef} style={{ flex: 1, overflowY: 'auto', display: isHistoryMode ? 'none' : 'block' }}>
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
                // Use API colors for search snippet backgrounds in the left column
                const apiColor = clusterColorsFromAPI[resultClusterKey];
                const apiTint = apiColor ? mixColorWithWhite(apiColor, 0.82) : undefined;
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
                      backgroundColor: apiTint || clusterTints[resultClusterKey] || 'white',
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
                      const isSameAsSelected = !!selectedNode
                        && selectedNode.title === note.title
                        && selectedNode.creation_date === note.creation_date
                        && selectedNode.modification_date === note.modification_date;
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
                           // Resolve cluster color from the API so it matches the right column
                           const apiChunkColor = clusterColorsFromAPI[chunk.cluster_id];
                           const resolvedColor = apiChunkColor || clusterColors[chunk.cluster_id] || '#f3f4f6';
                           const lightenedColor = new THREE.Color(resolvedColor).lerp(new THREE.Color('#ffffff'), 0.75).getStyle();
                          rows.push(
                            <button
                              type="button"
                              key={`snippet-${note.note_key}-${chunk.chunk_index}`}
                              onClick={() => openSidebarNote(note, chunk.chunk_index)}
                              style={{
                                border: `1px solid ${resolvedColor}`,
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


{/* Center column: RadialSimilarityHub */}
<div style={{
  flex: 1,
  minWidth: '200px',
  display: 'flex',
  flexDirection: 'column',
  minHeight: 0,
  borderLeft: '1px solid #e0e0e0',
  borderRight: '1px solid #e0e0e0',
  backgroundColor: '#f9f9f9',
  padding: '10px',
  boxSizing: 'border-box',
}}>
  <div style={{
    fontSize: '13px',
    fontWeight: 700,
    color: '#1f2937',
    marginBottom: '8px',
    textAlign: 'center',
    flexShrink: 0,
  }}>
    Similar Clusters
  </div>
  <div style={{
    flex: 1,
    minHeight: 0,
    minWidth: 0,
    display: 'flex',
  }}>
    <RadialSimilarityHub
      selectedClusterId={selectedClusters.size > 0 ? Array.from(selectedClusters)[0] : null}
      onNodeClick={(clusterId) => {
        setSelectedClusters(new Set([clusterId]));
        try { focusClusterByOrbit(clusterId); } catch (_ignore) {}
      }}
      clusterColors={clusterColorsFromAPI}
    />
  </div>
</div>

              {/* Right column: MetaClusterTree */}
              <div style={{
             width: '26%',
             flexShrink: 0,
             display: 'flex',
             flexDirection: 'column',
             borderLeft: '1px solid #e0e0e0',
             backgroundColor: '#f9f9f9',
             padding: '10px',
             boxSizing: 'border-box',
             fontSize: '11px',
             fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
              }}>
               <MetaClusterTree
               onClusterSelect={(clusterId) => {
                 setSelectedClusters(new Set([clusterId]));
                 try { focusClusterByOrbit(clusterId); } catch (_ignore) { }
                }}
               selectedClusterId={selectedClusters.size > 0 ? Array.from(selectedClusters)[0] : null}
               sortMetric={clusterSortMetric}
               clusterColors={clusterColorsFromAPI}
               />
           </div>

           {/* Modal Popup for Note Content */}
           {selectedNode && (
             <div
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(0, 0, 0, 0.4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 9999,
                pointerEvents: 'auto',
              }}
              onClick={closePopup}
             >
               <div
                style={{
                  position: 'relative',
                  width: 'min(520px, 90vw)',
                  maxHeight: '80vh',
                  backgroundColor: '#fff',
                  borderRadius: '12px',
                  boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  pointerEvents: 'auto',
                }}
                 onClick={(e) => e.stopPropagation()}
                 >
                   {/* Close button (X) in upper right */}
                  <button
                  type="button"
                  onClick={closePopup}
                  style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    width: '32px',
                    height: '32px',
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: 'transparent',
                    color: '#6b7280',
                    fontSize: '18px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: 0,
                    zIndex: 10,
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#f3f4f6'; e.currentTarget.style.color = '#111827'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#6b7280'; }}
                  title="Close"
                 >
                   ×
                </button>

                  {/* Header with cluster color accent */}
                  <div style={{
                    padding: '16px 16px 12px 16px',
                   borderBottom: `1px solid ${modalApiColor ? mixColorWithWhite(modalApiColor, 0.85) : '#e5e7eb'}`,
                   backgroundColor: modalApiColor
                        ? mixColorWithWhite(modalApiColor, 0.88)
                        : '#f9fafb',
                  }}>
                      <div style={{
                        fontWeight: 700,
                        fontSize: '15px',
                        color: '#111827',
                        marginBottom: '4px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {selectedNode.title}
                      </div>

                      {/* Thin colored accent bar at top of header */}
                      {modalApiColor && (
                        <div style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: '3px',
                          backgroundColor: modalApiColor,
                        }} />
                      )}

                      <div style={{
                     fontSize: '12px',
                     color: '#6b7280',
                     display: 'flex',
                     alignItems: 'center',
                     gap: '8px',
                    }}>
                      {selectedNode.cluster_label && (
                        <span style={{
                         color: selectedNode.cluster_color || '#4b5563',
                         fontWeight: 600,
                        }}>
                          {selectedNode.cluster_label}
                        </span>
                      )}
                      <span>Chunk {selectedNode.chunk_index + 1} of {selectedNode.total_chunks}</span>
                    </div>

                    {/* Dot/Dash Segmented Rail for cluster navigation in modal */}
                    {modalRailChunks.length > 0 && (
                      <SegmentedRail
                       chunks={modalRailChunks}
                       activeClusterIds={new Set([selectedNode.display_topic_id || selectedNode.cluster_id || ''])}
                       currentChunkIndex={selectedNode.chunk_index}
                       getClusterColor={(clusterId) => clusterColors[clusterId] || '#6b7280'}
                      onActiveDotClick={(chunk) => handleModalChunkJump(chunk.chunk_index)}
                        onInactiveDashClick={(chunk) => {
                           // Clicking a dash: select that cluster, navigate to the chunk
                          setSelectedClusters(new Set([chunk.cluster_id]));
                          fetchNoteContent(
                            selectedNode.title,
                            chunk.chunk_index,
                            selectedNode.cluster_id,
                            selectedNode.cluster_label,
                            chunk.cluster_id,
                            selectedNode.base_topic_id,
                            selectedNode.creation_date,
                            selectedNode.modification_date,
                           );
                         }}
                      />
                    )}
                  </div>

                 {/* Navigation and Content */}
                 <div style={{
                   flex: 1,
                   overflowY: 'auto',
                   padding: '16px',
                   display: 'flex',
                   flexDirection: 'column',
                   gap: '12px',
                 }}>
                   {/* Navigation buttons */}
                   <div style={{
                     display: 'flex',
                     justifyContent: 'space-between',
                     alignItems: 'center',
                   }}>
                     <button
                      type="button"
                      onClick={handlePrevChunk}
                      style={{
                        padding: '8px 16px',
                        borderRadius: '6px',
                        border: '1px solid #d1d5db',
                        backgroundColor: '#fff',
                        cursor: 'pointer',
                        fontSize: '13px',
                        fontWeight: 600,
                      }}
                     >
                       ← Previous
                     </button>
                     <span style={{ fontSize: '12px', color: '#6b7280' }}>
                       {selectedNode.chunk_index + 1} / {selectedNode.total_chunks}
                     </span>
                     <button
                      type="button"
                      onClick={handleNextChunk}
                      style={{
                        padding: '8px 16px',
                        borderRadius: '6px',
                        border: '1px solid #d1d5db',
                        backgroundColor: '#fff',
                        cursor: 'pointer',
                        fontSize: '13px',
                        fontWeight: 600,
                      }}
                     >
                       Next →
                     </button>
                   </div>

                   {/* Content area */}
                   <div style={{
                     flex: 1,
                     overflowY: 'auto',
                     fontSize: '13px',
                     lineHeight: '1.6',
                     color: '#374151',
                     whiteSpace: 'pre-wrap',
                     wordBreak: 'break-word',
                   }}>
                     {isLoadingContent ? (
                       <div style={{ textAlign: 'center', padding: '20px', color: '#9ca3af' }}>
                         Loading...
                       </div>
                     ) : (
                       selectedNode.content || '(No content)'
                     )}
                   </div>
                 </div>
               </div>
             </div>
           )}

         </div>
      )}
    </div>
  );
}
