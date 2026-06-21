import React, { useMemo } from "react";
import { View, Text, StyleSheet, ScrollView, Platform, WebView } from "react-native";

export default function SafetyRouteMap({ mapRegion, source, destination, zones = [], route = [], fastestRoute = [], riskToColor }) {
  const srcCoord = { latitude: Number(source?.lat ?? source?.latitude ?? 0), longitude: Number(source?.lon ?? source?.longitude ?? 0) };
  const dstCoord = { latitude: Number(destination?.lat ?? destination?.latitude ?? 0), longitude: Number(destination?.lon ?? destination?.longitude ?? 0) };

  const routeCoords = (route || []).map(p => ({ latitude: Number(p.latitude ?? p.lat), longitude: Number(p.longitude ?? p.lon) }));
  const fastestCoords = (fastestRoute || []).map(p => ({ latitude: Number(p.latitude ?? p.lat), longitude: Number(p.longitude ?? p.lon) }));

  // Generate Leaflet HTML map for both web and native (via WebView)
  const mapHtml = useMemo(() => {
    const center = mapRegion || { latitude: srcCoord.latitude || 19.124, longitude: srcCoord.longitude || 72.825 };
    const zoom = 13;

    const routeGeoJSON = routeCoords.length > 1 ? routeCoords.map(p => [p.latitude, p.longitude]) : [];
    const fastestGeoJSON = fastestCoords.length > 1 ? fastestCoords.map(p => [p.latitude, p.longitude]) : [];

    const routeJson = JSON.stringify(routeGeoJSON);
    const fastestJson = JSON.stringify(fastestGeoJSON);
    const srcJson = JSON.stringify([srcCoord.latitude, srcCoord.longitude]);
    const dstJson = JSON.stringify([dstCoord.latitude, dstCoord.longitude]);

    // Limit to 500 zones for performance
    const zonesJson = JSON.stringify(
      (zones || []).slice(0, 500).map(z => ({
        lat: Number(z.latitude ?? z.lat ?? 0),
        lon: Number(z.longitude ?? z.lon ?? 0),
        risk: Number(z.risk ?? 0),
      }))
    );

    return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Safety Route Map</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { height: 100vh; width: 100vw; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0B1220; }
    #map { width: 100%; height: 100%; }
    .legend {
      position: absolute;
      bottom: 18px;
      right: 10px;
      z-index: 1000;
      background: rgba(12, 12, 18, 0.88);
      border-radius: 10px;
      padding: 8px 12px;
      border: 1px solid rgba(255,255,255,0.1);
      pointer-events: none;
    }
    .legend-title { color: #888; font-size: 9px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
    .legend-row { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; font-size: 11px; color: #ccc; font-weight: 500; }
    .legend-row:last-child { margin-bottom: 0; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 5px currentColor; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <div class="legend-title">Safety Zones</div>
    <div class="legend-row"><div class="legend-dot" style="background:#30D158;color:#30D158"></div>Safe</div>
    <div class="legend-row"><div class="legend-dot" style="background:#FF9F0A;color:#FF9F0A"></div>Moderate</div>
    <div class="legend-row"><div class="legend-dot" style="background:#FF453A;color:#FF453A"></div>Unsafe</div>
  </div>
  <script>
    var map = L.map('map', { zoomControl: true, attributionControl: false })
               .setView([${center.latitude}, ${center.longitude}], ${zoom});

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19
    }).addTo(map);

    var src     = ${srcJson};
    var dst     = ${dstJson};
    var route   = ${routeJson};
    var fastest = ${fastestJson};
    var zones   = ${zonesJson};

    // --- Zone colour helpers ---
    function zoneColor(risk) {
      if (risk < 0.33) return '#30D158';
      if (risk < 0.66) return '#FF9F0A';
      return '#FF453A';
    }
    function zoneLabel(risk) {
      if (risk < 0.33) return 'Safe';
      if (risk < 0.66) return 'Moderate';
      return 'Unsafe';
    }

    // --- Draw zones: outer glow ring + inner solid circle ---
    zones.forEach(function(z) {
      if (!z.lat || !z.lon) return;
      var col = zoneColor(z.risk);
      var pct = (z.risk * 100).toFixed(0);

      // Outer glow (large, very transparent)
      L.circle([z.lat, z.lon], {
        radius: 1400,
        fillColor: col,
        fillOpacity: 0.07,
        stroke: false,
      }).addTo(map);

      // Inner solid circle with coloured border
      L.circle([z.lat, z.lon], {
        radius: 700,
        fillColor: col,
        fillOpacity: 0.42,
        color: col,
        weight: 1.5,
        opacity: 0.65,
      })
      .bindTooltip(
        '<span style="font-size:12px;font-weight:700;color:' + col + '">' + zoneLabel(z.risk) + '</span>' +
        '&nbsp;<span style="color:#aaa;font-size:11px">' + pct + '% risk</span>',
        { sticky: true, opacity: 0.95 }
      )
      .addTo(map);
    });

    // --- Start marker ---
    if (src[0] !== 0) {
      L.circleMarker(src, { radius: 10, fillColor: '#30D158', color: '#fff', weight: 2.5, fillOpacity: 1 })
        .bindPopup('<b style="color:#30D158">&#9654; Start</b>')
        .addTo(map);
    }

    // --- End marker ---
    if (dst[0] !== 0) {
      L.circleMarker(dst, { radius: 10, fillColor: '#0A84FF', color: '#fff', weight: 2.5, fillOpacity: 1 })
        .bindPopup('<b style="color:#0A84FF">&#11035; Destination</b>')
        .addTo(map);
    }

    // --- Safe route: glow layer + core polyline ---
    if (route.length > 1) {
      L.polyline(route, { color: '#0A84FF', weight: 14, opacity: 0.12 }).addTo(map);
      L.polyline(route, { color: '#0A84FF', weight: 5, opacity: 0.95 })
        .bindPopup('Safest Route (' + route.length + ' pts)')
        .addTo(map);
    }

    // --- Fastest route (orange dashed) ---
    if (fastest.length > 1) {
      L.polyline(fastest, { color: '#FF9F0A', weight: 3.5, opacity: 0.85, dashArray: '8, 7' })
        .bindPopup('Fastest Route (' + fastest.length + ' pts)')
        .addTo(map);
    }
  </script>
</body>
</html>
    `;
  }, [mapRegion, srcCoord.latitude, srcCoord.longitude, dstCoord.latitude, dstCoord.longitude,
    routeCoords.length, fastestCoords.length, zones]);

  // Render iframe on web, WebView on native
  if (Platform.OS === 'web') {
    return (
      <View style={styles.container}>
        <iframe
          srcDoc={mapHtml}
          style={{ width: '100%', height: 400, borderRadius: 16, border: 'none', marginBottom: 12 }}
          title="Safety Route Map"
        />
        <ScrollView style={styles.list} contentContainerStyle={{ padding: 8 }}>
          <Text style={styles.sectionTitle}>Safe Route ({route.length} pts)</Text>
          {route.slice(0, 20).map((p, i) => (
            <View key={`r-${i}`} style={styles.pointRow}>
              <View style={[styles.colorSwatch, { backgroundColor: riskToColor ? riskToColor(p.risk ?? 0) : 'rgba(0,0,0,0.1)' }]} />
              <Text style={styles.pointText}>{i + 1}. {Number(p.latitude || p.lat).toFixed(5)}, {Number(p.longitude || p.lon).toFixed(5)}</Text>
            </View>
          ))}
        </ScrollView>
      </View>
    );
  }

  // Native: use WebView
  return (
    <View style={styles.container}>
      <WebView
        source={{ html: mapHtml }}
        style={styles.mapBox}
        scrollEnabled={false}
        scalesPageToFit={true}
      />
      <ScrollView style={styles.list} contentContainerStyle={{ padding: 8 }}>
        <Text style={styles.sectionTitle}>Safe Route ({route.length} pts)</Text>
        {route.slice(0, 20).map((p, i) => (
          <View key={`r-${i}`} style={styles.pointRow}>
            <View style={[styles.colorSwatch, { backgroundColor: riskToColor ? riskToColor(p.risk ?? 0) : 'rgba(0,0,0,0.1)' }]} />
            <Text style={styles.pointText}>{i + 1}. {Number(p.latitude || p.lat).toFixed(5)}, {Number(p.longitude || p.lon).toFixed(5)}</Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: '100%' },
  mapBox: {
    height: 400,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 12,
  },
  list: { maxHeight: 300 },
  sectionTitle: { color: '#fff', fontWeight: '700', marginBottom: 6 },
  pointRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  colorSwatch: { width: 16, height: 16, borderRadius: 4, marginRight: 8 },
  pointText: { color: '#ddd', fontSize: 12 },
});
