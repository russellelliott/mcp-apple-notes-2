/**
 * Shared types for the 2D spatial cluster canvas.
 *
 * The canvas renders the full metacluster hierarchy as a pan/zoomable web of
 * circles, with three modes:
 *  - base:   all metaclusters at their deterministic 2D UMAP positions
 *  - focus:  selected cluster's metacluster centered, others orbiting by
 *            semantic distance (cosine distance of embedding centroids)
 *  - search: all clusters visible, matches highlighted, non-matches dimmed
 */

/** Raw shape of GET /meta_clusters (backend: server.py MetaClusterInfo) */
export interface ApiMetaChildCluster {
  cluster_id: string;
  label: string;
  chunk_count: number;
  color?: string | null;
  centroid?: number[] | null;
}

export interface ApiMetaCluster {
  meta_cluster_id: string;
  label: string;
  child_clusters: ApiMetaChildCluster[];
  last_modified?: string | null;
}

/** Minimal point shape needed for 2D positioning (from /points|/points_shaped) */
export interface SpatialPoint {
  unique_key: string;
  display_topic_id?: string;
  cluster_id?: string;
  umap_x: number;
  umap_y: number;
  /** Optional enrichment fields present in the real /points response */
  cluster_label?: string;
  cluster_color?: string;
  dot_color?: string;
}

export type SpatialViewMode = 'base' | 'focus' | 'search';

export interface XY {
  x: number;
  y: number;
}

export interface SpatialColorPalette {
  bg: string;
  border: string;
  accent: string;
  subtle: string;
  text: string;
}

export interface SpatialCluster {
  id: string;
  metaClusterId: string;
  title: string;
  totalChunks: number;
  /** Base color (hex/rgb/hsl string) for this cluster */
  color: string;
  /** Raw embedding centroid from the backend (any dimensionality), or null */
  centroid: number[] | null;
  /** Deterministic 2D position in world coordinates (UMAP-based) */
  basePosition: XY;
}

export interface SpatialMetaCluster {
  id: string;
  title: string;
  color: SpatialColorPalette;
  /** Deterministic 2D position in world coordinates (mean of children) */
  basePosition: XY;
  /** Base radius of the metacluster circle in world pixels */
  radius: number;
  clusters: SpatialCluster[];
  totalChunks: number;
}

/** Similarity edge between two metacluster circles */
export interface MetaLink {
  sourceId: string;
  targetId: string;
  similarity: number;
  sourceCenter: XY;
  targetCenter: XY;
}

export interface SpatialLayoutResult {
  metaCircleCenters: Record<string, XY & { radius: number }>;
  clusterCoordinates: Record<string, XY>;
  metaLinks: MetaLink[];
}

/** clusterId -> number of search results landing in that cluster */
export type SearchMatchCounts = Record<string, number>;
