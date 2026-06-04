/**
 * Build Google / Apple Maps URLs that follow the app's route (Chicago only).
 * Waypoints are subsampled so URLs stay within practical limits.
 */

function buildPointList(route, source, destination) {
  const pts = [];
  if (route?.length >= 2) {
    for (const p of route) {
      const lat = typeof p.latitude === "number" ? p.latitude : parseFloat(p.latitude);
      const lng = typeof p.longitude === "number" ? p.longitude : parseFloat(p.longitude);
      if (Number.isFinite(lat) && Number.isFinite(lng)) pts.push([lat, lng]);
    }
  }
  if (pts.length < 2) {
    return [
      [source.lat, source.lon],
      [destination.lat, destination.lon],
    ];
  }
  return pts;
}

function dedupeAdjacent(points) {
  const out = [];
  for (const p of points) {
    const prev = out[out.length - 1];
    if (!prev || prev[0] !== p[0] || prev[1] !== p[1]) out.push(p);
  }
  return out;
}

export function sampleWaypoints(points, max = 22) {
  const clean = dedupeAdjacent(points);
  if (clean.length <= max) return clean;
  const out = [];
  const n = clean.length;
  for (let i = 0; i < max; i++) {
    const idx = Math.round((i * (n - 1)) / (max - 1));
    out.push(clean[Math.min(idx, n - 1)]);
  }
  return dedupeAdjacent(out);
}

function fmt(ll) {
  return `${Number(ll[0]).toFixed(5)},${Number(ll[1]).toFixed(5)}`;
}

/**
 * Google Maps: multi-stop /dir path (follows given order; Google may re-optimize between stops).
 */
export function buildGoogleMapsPathUrl(route, source, destination) {
  const sampled = sampleWaypoints(buildPointList(route, source, destination), 23);
  const path = sampled.map(fmt).join("/");
  return `https://www.google.com/maps/dir/${path}`;
}

/**
 * Apple Maps: saddr + daddr with +to: chain (documented pattern for multiple stops).
 */
export function buildAppleMapsPathUrl(route, source, destination) {
  const sampled = sampleWaypoints(buildPointList(route, source, destination), 18);
  if (sampled.length < 2) {
    return `http://maps.apple.com/?ll=${fmt([source.lat, source.lon])}`;
  }
  const [first, ...rest] = sampled;
  const daddr = rest.map(fmt).join("+to:");
  return `http://maps.apple.com/?saddr=${fmt(first)}&daddr=${daddr}&dirflg=d`;
}
