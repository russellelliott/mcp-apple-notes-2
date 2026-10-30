/**
 * Pure layout & similarity math for the spatial cluster canvas.
 *
 * Mirrors the reference "proximity.ts" utilities, generalized to work with
 * the real backend shapes:
 *  - centroids are raw embedding vectors of arbitrary dimensionality
 *  - 2D positions come from the 2D UMAP projection of the points
 */

import {
  ApiMetaCluster,
  ApiMetaChildCluster,
  MetaLink,
  SpatialCluster,
  SpatialColorPalette,
  SpatialLayoutResult,
  SpatialMetaCluster,
  SpatialPoint,
  SpatialViewMode,
  XY,
} from './types';

// ── World constants ────────────────────────────────────────────────────────
export const WORLD_WIDTH = 1500;
export const WORLD_HEIGHT = 960;
export const WORLD_CENTER: XY = { x: WORLD_WIDTH / 2, y: WORLD_HEIGHT / 2 };

/** Similarity threshold above which a connective web link is drawn */
export const META_LINK_THRESHOLD = 0.6;

// ── Vector math ────────────────────────────────────────────────────────────

/**
 * Cosine similarity between two vectors of arbitrary dimensionality.
 * Returns 0 when either vector is missing, empty, or a zero vector.
 */
export function cosineSimilarity(a: number[] | null, b: number[] | null): number {
  if (!a || !b || a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number.isFinite(a[i]) ? a[i] : 0;
    const bv = Number.isFinite(b[i]) ? b[i] : 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return Math.min(1, Math.max(-1, dot / denom));
}

/**
 * Normalized semantic distance [0..1] — 0 = identical semantics, 1 = far.
 */
export function semanticDistance(a: number[] | null, b: number[] | null): number {
  const sim = cosineSimilarity(a, b);
  return Math.max(0, Math.min(1, (1 - sim) / 2));
}

/** Centroid of a metacluster: mean of its children's centroids (or null). */
export function getMetaClusterCentroid(meta: SpatialMetaCluster): number[] | null {
  const valid = meta.clusters
    .map((c) => c.centroid)
    .filter((c): c is number[] => !!c && c.length > 0);
  if (valid.length === 0) return null;
  const dim = Math.min(...valid.map((v) => v.length));
  const out = new Array<number>(dim).fill(0);
  for (const v of valid) {
    for (let i = 0; i < dim; i++) out[i] += v[i];
  }
  for (let i = 0; i < dim; i++) out[i] /= valid.length;
  return out;
}

// ── Color helpers (no external deps) ───────────────────────────────────────

interface Rgb {
  r: number;
  g: number;
  b: number;
}

const clamp255 = (v: number) => Math.max(0, Math.min(255, Math.round(v)));

function hslToRgb(h: number, s: number, l: number): Rgb {
  const hue = (((h % 360) + 360) % 360) / 360;
  const sat = Math.max(0, Math.min(1, s));
  const light = Math.max(0, Math.min(1, l));
  if (sat === 0) {
    const v = light * 255;
    return { r: v, g: v, b: v };
  }
  const q = light < 0.5 ? light * (1 + sat) : light + sat - light * sat;
  const p = 2 * light - q;
  const conv = (t: number) => {
    let tt = t;
    if (tt < 0) tt += 1;
    if (tt > 1) tt -= 1;
    if (tt < 1 / 6) return p + (q - p) * 6 * tt;
    if (tt < 1 / 2) return q;
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
    return p;
  };
  return {
    r: conv(hue + 1 / 3) * 255,
    g: conv(hue) * 255,
    b: conv(hue - 1 / 3) * 255,
  };
}

/** Parse a CSS color string into an RGB triplet (or null). */
export function parseColor(input: string | null | undefined): Rgb | null {
  if (!input || typeof input !== 'string') return null;
  const s = input.trim().toLowerCase();

  const hex = s.match(/^#?([0-9a-f]{3}|[0-9a-f]{6})$/);
  if (hex) {
    let h = hex[1];
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    return {
      r: parseInt(h.slice(0, 2), 16),
      g: parseInt(h.slice(2, 4), 16),
      b: parseInt(h.slice(4, 6), 16),
    };
  }

  const rgb = s.match(/^rgba?\(\s*([\d.]+)\s*[, ]+\s*([\d.]+)\s*[, ]+\s*([\d.]+)/);
  if (rgb) {
    return { r: Number(rgb[1]), g: Number(rgb[2]), b: Number(rgb[3]) };
  }

  const hsl = s.match(/^hsla?\(\s*([\d.]+)\s*[, ]+\s*([\d.]+)%\s*[, ]+\s*([\d.]+)%/);
  if (hsl) {
    return hslToRgb(Number(hsl[1]), Number(hsl[2]) / 100, Number(hsl[3]) / 100);
  }

  return null;
}

const FALLBACK_RGB: Rgb = parseColor('#6b7280') as Rgb;

export function toHex(c: Rgb): string {
  const h = (v: number) => clamp255(v).toString(16).padStart(2, '0');
  return `#${h(c.r)}${h(c.g)}${h(c.b)}`;
}

/** Linear mix of a color toward a target color; returns a hex string. */
export function mixColor(base: string, target: string, t: number): string {
  const a = parseColor(base) ?? FALLBACK_RGB;
  const b = parseColor(target) ?? FALLBACK_RGB;
  const k = Math.max(0, Math.min(1, t));
  return toHex({
    r: a.r + (b.r - a.r) * k,
    g: a.g + (b.g - a.g) * k,
    b: a.b + (b.b - a.b) * k,
  });
}

export const mixWithWhite = (base: string, t: number) => mixColor(base, '#ffffff', t);
export const mixWithBlack = (base: string, t: number) => mixColor(base, '#000000', t);

/** Average of a list of colors; falls back to a neutral grey. */
export function averageColors(colors: Array<string | null | undefined>): string {
  const valid = colors.map((c) => parseColor(c)).filter((c): c is Rgb => !!c);
  if (valid.length === 0) return toHex(FALLBACK_RGB);
  const sum = valid.reduce(
    (acc, c) => ({ r: acc.r + c.r, g: acc.g + c.g, b: acc.b + c.b }),
    { r: 0, g: 0, b: 0 },
  );
  return toHex({
    r: sum.r / valid.length,
    g: sum.g / valid.length,
    b: sum.b / valid.length,
  });
}

/** Derive the metacluster circle palette from a single base color. */
export function buildMetaPalette(baseColor: string | null | undefined): SpatialColorPalette {
  const base = parseColor(baseColor) ? baseColor! : toHex(FALLBACK_RGB);
  return {
    bg: mixWithWhite(base, 0.93),
    border: mixWithWhite(base, 0.5),
    accent: base,
    subtle: mixWithWhite(base, 0.78),
    text: mixWithBlack(base, 0.15),
  };
}
// ── Geometry helpers ───────────────────────────────────────────────────────

/** Base radius of a metacluster circle, scaled by content volume. */
export function metaCircleRadius(totalChunks: number, clusterCount: number): number {
  const r = 60 + Math.sqrt(Math.max(1, totalChunks)) * 5 + Math.min(clusterCount, 40) * 2.5;
  return Math.max(64, Math.min(150, r));
}

/** Disc size for cluster nodes, shrunk for crowded metaclusters. */
export function clusterDiscSize(clusterCount: number): number {
  if (clusterCount <= 6) return 54;
  if (clusterCount <= 12) return 44;
  if (clusterCount <= 24) return 38;
  return 32;
}

/** Deterministic golden-angle (phyllotaxis) fallback position. */
function spiralPosition(index: number): XY {
  const goldenAngle = 2.399963229728653;
  const r = 34 * Math.sqrt(index + 1);
  const a = index * goldenAngle;
  return {
    x: WORLD_CENTER.x + r * Math.cos(a),
    y: WORLD_CENTER.y + r * Math.sin(a),
  };
}

/**
 * Map raw 2D (UMAP) coordinates into the padded world box. Falls back to a
 * phyllotaxis spiral when the data is degenerate (e.g. UMAP unavailable).
 */
function normalizeTo2D(points: SpatialPoint[], clusterKeys: string[]): Map<string, XY> {
  const result = new Map<string, XY>();

  const sums = new Map<string, { x: number; y: number; n: number }>();
  for (const p of points) {
    if (!Number.isFinite(p.umap_x) || !Number.isFinite(p.umap_y)) continue;
    const key = p.display_topic_id || p.cluster_id || '-1';
    const acc = sums.get(key) ?? { x: 0, y: 0, n: 0 };
    acc.x += p.umap_x;
    acc.y += p.umap_y;
    acc.n += 1;
    sums.set(key, acc);
  }

  const means = new Map<string, XY>();
  for (const key of clusterKeys) {
    const acc = sums.get(key);
    if (acc && acc.n > 0) means.set(key, { x: acc.x / acc.n, y: acc.y / acc.n });
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const m of means.values()) {
    minX = Math.min(minX, m.x);
    minY = Math.min(minY, m.y);
    maxX = Math.max(maxX, m.x);
    maxY = Math.max(maxY, m.y);
  }

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const degenerate = means.size < 2 || spanX < 1e-6 || spanY < 1e-6;

  const margin = 120;
  const usableW = WORLD_WIDTH - margin * 2;
  const usableH = WORLD_HEIGHT - margin * 2;

  let fallbackIndex = 0;
  for (const key of clusterKeys) {
    const m = means.get(key);
    if (!degenerate && m) {
      result.set(key, {
        x: margin + ((m.x - minX) / spanX) * usableW,
        y: margin + ((m.y - minY) / spanY) * usableH,
      });
    } else {
      result.set(key, spiralPosition(fallbackIndex++));
    }
  }

  return result;
}

/**
 * Build the full spatial hierarchy from the backend metacluster response
 * and the point cloud.
 *
 * @param apiMetas       Raw GET /meta_clusters response
 * @param points         The point cloud from /points (for 2D UMAP positions)
 * @param fallbackColors Map of clusterId -> color used when the endpoint
 *                       did not provide a per-cluster color
 */
export function buildSpatialMetaClusters(
  apiMetas: ApiMetaCluster[],
  points: SpatialPoint[],
  fallbackColors: Record<string, string>,
): SpatialMetaCluster[] {
  const allClusterKeys: string[] = [];
  const seen = new Set<string>();
  for (const meta of apiMetas) {
    for (const child of meta.child_clusters) {
      if (!seen.has(child.cluster_id)) {
        seen.add(child.cluster_id);
        allClusterKeys.push(child.cluster_id);
      }
    }
  }

  const positions = normalizeTo2D(points, allClusterKeys);

  // Per-cluster label/color fallbacks from the point cloud
  const labelByCluster = new Map<string, string>();
  const colorByCluster = new Map<string, string>();
  const pointCountByCluster = new Map<string, number>();
  for (const p of points) {
    const key = p.display_topic_id || p.cluster_id || '-1';
    if (!labelByCluster.has(key) && p.cluster_label && p.cluster_label !== 'Unclustered') {
      labelByCluster.set(key, p.cluster_label);
    }
    if (!colorByCluster.has(key) && (p.cluster_color || p.dot_color)) {
      colorByCluster.set(key, (p.cluster_color || p.dot_color)!);
    }
    pointCountByCluster.set(key, (pointCountByCluster.get(key) ?? 0) + 1);
  }

  const metas: SpatialMetaCluster[] = [];

  for (const meta of apiMetas) {
    const clusters: SpatialCluster[] = meta.child_clusters.map((child: ApiMetaChildCluster) => {
      const baseColor =
        child.color ||
        fallbackColors[child.cluster_id] ||
        colorByCluster.get(child.cluster_id) ||
        '#6b7280';
      return {
        id: child.cluster_id,
        metaClusterId: meta.meta_cluster_id,
        title: child.label || labelByCluster.get(child.cluster_id) || child.cluster_id,
        totalChunks: child.chunk_count || pointCountByCluster.get(child.cluster_id) || 0,
        color: baseColor,
        centroid: child.centroid ?? null,
        basePosition: positions.get(child.cluster_id) ?? spiralPosition(0),
      };
    });

    // Meta 2D position = mean of its children's positions
    const posSum = clusters.reduce(
      (acc, c) => ({ x: acc.x + c.basePosition.x, y: acc.y + c.basePosition.y }),
      { x: 0, y: 0 },
    );
    const basePosition: XY =
      clusters.length > 0
        ? { x: posSum.x / clusters.length, y: posSum.y / clusters.length }
        : { ...WORLD_CENTER };

    const totalChunks = clusters.reduce((sum, c) => sum + c.totalChunks, 0);
    const accent = averageColors(clusters.map((c) => c.color));

    metas.push({
      id: meta.meta_cluster_id,
      title: meta.label || `Meta ${meta.meta_cluster_id}`,
      color: buildMetaPalette(accent),
      basePosition,
      radius: metaCircleRadius(totalChunks, clusters.length),
      clusters,
      totalChunks,
    });
  }

  return metas;
}
/**
 * Positions of clusters distributed inside a metacluster circle, ensuring
 * discs stay within the boundary. Ported from the reference implementation.
 */
export function getClusterPositionsInsideMetaCircle(
  meta: SpatialMetaCluster,
  metaCenter: XY,
  metaRadius: number,
): Record<string, XY> {
  const positions: Record<string, XY> = {};
  const clusters = meta.clusters;
  const n = clusters.length;
  if (n === 0) return positions;

  if (n === 1) {
    positions[clusters[0].id] = { x: metaCenter.x, y: metaCenter.y + 10 };
    return positions;
  }

  // Leave room for the title pill at the top of the circle
  const availableRadius = Math.max(24, metaRadius - 55);

  clusters.forEach((cluster, idx) => {
    // When crowded, park the first cluster at the center of the circle
    if (n >= 5 && idx === 0) {
      positions[cluster.id] = { x: metaCenter.x, y: metaCenter.y + 12 };
      return;
    }

    const outerCount = n >= 5 ? n - 1 : n;
    const adjustedIdx = n >= 5 && idx > 0 ? idx - 1 : idx;

    // Distribute along a circle starting near the top-right
    const angleOffset = -Math.PI / 2 + 0.35;
    const angle = angleOffset + (adjustedIdx / outerCount) * Math.PI * 2;
    const orbitalRadius = availableRadius * 0.72;

    positions[cluster.id] = {
      x: metaCenter.x + orbitalRadius * Math.cos(angle),
      y: metaCenter.y + 12 + orbitalRadius * Math.sin(angle),
    };
  });

  return positions;
}

/**
 * Compute the full canvas layout for a given view mode:
 *  - base/search: metacluster circles at their deterministic 2D positions
 *  - focus: the selected cluster's metacluster is centered, the others orbit
 *           around it at a radius proportional to their semantic distance,
 *           and connective web links are drawn between similar metaclusters.
 */
export function calculateDynamicCircleWebLayout(
  selectedCluster: SpatialCluster | null,
  metaClusters: SpatialMetaCluster[],
  viewMode: SpatialViewMode,
  canvasCenter: XY = WORLD_CENTER,
): SpatialLayoutResult {
  const metaCircleCenters: Record<string, XY & { radius: number }> = {};
  const clusterCoordinates: Record<string, XY> = {};
  const metaLinks: MetaLink[] = [];

  const centroids: Record<string, number[] | null> = {};
  for (const mc of metaClusters) centroids[mc.id] = getMetaClusterCentroid(mc);

  if (viewMode === 'focus' && selectedCluster) {
    const activeMeta =
      metaClusters.find((mc) => mc.id === selectedCluster.metaClusterId) || metaClusters[0];

    // Sort other metaclusters by semantic distance to the selected cluster's
    // centroid. Metaclusters without a centroid fall to the outer ring.
    const otherMetas = metaClusters
      .filter((mc) => mc.id !== activeMeta?.id)
      .map((mc) => {
        const hasCentroid = !!centroids[mc.id] && !!selectedCluster.centroid;
        const dist = hasCentroid
          ? semanticDistance(centroids[mc.id], selectedCluster.centroid)
          : 0.75;
        return { mc, dist, hasCentroid };
      })
      .sort((a, b) => a.dist - b.dist);

    if (activeMeta) {
      metaCircleCenters[activeMeta.id] = {
        x: canvasCenter.x,
        y: canvasCenter.y,
        radius: activeMeta.radius,
      };
    }

    const numOthers = otherMetas.length;
    otherMetas.forEach((item, index) => {
      const orbitRadius = 340 + item.dist * 640;
      const angle = (index / Math.max(1, numOthers)) * Math.PI * 2 - Math.PI / 2;
      metaCircleCenters[item.mc.id] = {
        x: canvasCenter.x + orbitRadius * Math.cos(angle),
        y: canvasCenter.y + orbitRadius * Math.sin(angle),
        radius: item.mc.radius * 0.92,
      };
    });
  } else {
    for (const mc of metaClusters) {
      metaCircleCenters[mc.id] = {
        x: mc.basePosition.x,
        y: mc.basePosition.y,
        radius: mc.radius,
      };
    }
  }

  // Nest cluster discs inside each metacluster circle
  for (const mc of metaClusters) {
    const metaCircle = metaCircleCenters[mc.id];
    if (!metaCircle) continue;
    const nested = getClusterPositionsInsideMetaCircle(
      mc,
      { x: metaCircle.x, y: metaCircle.y },
      metaCircle.radius,
    );
    for (const [clusterId, pos] of Object.entries(nested)) {
      clusterCoordinates[clusterId] = pos;
    }
  }

  // Connective web links between similar metaclusters (centroid cosine)
  for (let i = 0; i < metaClusters.length; i++) {
    for (let j = i + 1; j < metaClusters.length; j++) {
      const mcA = metaClusters[i];
      const mcB = metaClusters[j];
      const sim = cosineSimilarity(centroids[mcA.id], centroids[mcB.id]);
      if (sim > META_LINK_THRESHOLD) {
        metaLinks.push({
          sourceId: mcA.id,
          targetId: mcB.id,
          similarity: Number(sim.toFixed(3)),
          sourceCenter: { x: metaCircleCenters[mcA.id].x, y: metaCircleCenters[mcA.id].y },
          targetCenter: { x: metaCircleCenters[mcB.id].x, y: metaCircleCenters[mcB.id].y },
        });
      }
    }
  }

  return { metaCircleCenters, clusterCoordinates, metaLinks };
}
