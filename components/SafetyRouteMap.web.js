import React, { useEffect, useRef } from "react";
import { View, Text, StyleSheet, ScrollView, Platform } from "react-native";
import L from "leaflet";

export default function SafetyRouteMap({ mapRegion, source, destination, zones = [], route = [], fastestRoute = [], riskToColor }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const layersRef = useRef({
    startMarker: null,
    endMarker: null,
    safePolyline: null,
    fastestPolyline: null,
    circles: []
  });

  const srcCoord = { latitude: Number(source?.lat ?? source?.latitude ?? 0), longitude: Number(source?.lon ?? source?.longitude ?? 0) };
  const dstCoord = { latitude: Number(destination?.lat ?? destination?.latitude ?? 0), longitude: Number(destination?.lon ?? destination?.longitude ?? 0) };

  const routeCoords = (route || []).map(p => [Number(p.latitude ?? p.lat), Number(p.longitude ?? p.lon)]);
  const fastestCoords = (fastestRoute || []).map(p => [Number(p.latitude ?? p.lat), Number(p.longitude ?? p.lon)]);

  const centerLat = mapRegion?.latitude || srcCoord.latitude || 19.1240;
  const centerLon = mapRegion?.longitude || srcCoord.longitude || 72.8254;

  // 1. Initialize Map and Load Leaflet CSS
  useEffect(() => {
    if (Platform.OS === 'web' && mapContainerRef.current && !mapInstanceRef.current) {
      // Inject Leaflet CSS
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.id = 'leaflet-css';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(link);

      // Create Leaflet map instance
      const map = L.map(mapContainerRef.current, {
        center: [centerLat, centerLon],
        zoom: 13,
      });

      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CARTO',
        maxZoom: 19
      }).addTo(map);

      mapInstanceRef.current = map;

      return () => {
        const existingLink = document.getElementById('leaflet-css');
        if (existingLink) {
          document.head.removeChild(existingLink);
        }
        if (mapInstanceRef.current) {
          mapInstanceRef.current.remove();
          mapInstanceRef.current = null;
        }
      };
    }
  }, []);

  // 2. Sync Map Center/Zoom when region changes
  useEffect(() => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([centerLat, centerLon], 13);
    }
  }, [centerLat, centerLon]);

  // 3. Render markers, routes, and safety circles dynamically
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const layers = layersRef.current;

    // Clear previous layers
    if (layers.startMarker) map.removeLayer(layers.startMarker);
    if (layers.endMarker) map.removeLayer(layers.endMarker);
    if (layers.safePolyline) map.removeLayer(layers.safePolyline);
    if (layers.fastestPolyline) map.removeLayer(layers.fastestPolyline);
    layers.circles.forEach(c => map.removeLayer(c));
    layers.circles = [];

    // Custom icons for start and end
    const greenIcon = L.icon({
      iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowSize: [41, 41]
    });

    const redIcon = L.icon({
      iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowSize: [41, 41]
    });

    // Add Start Marker
    if (srcCoord.latitude !== 0) {
      layers.startMarker = L.marker([srcCoord.latitude, srcCoord.longitude], { icon: greenIcon })
        .bindPopup("Start Location")
        .addTo(map);
    }

    // Add End Marker
    if (dstCoord.latitude !== 0) {
      layers.endMarker = L.marker([dstCoord.latitude, dstCoord.longitude], { icon: redIcon })
        .bindPopup("End Destination")
        .addTo(map);
    }

    // Add Safe Route Polyline (Blue)
    if (routeCoords.length > 1) {
      layers.safePolyline = L.polyline(routeCoords, { color: '#0A84FF', weight: 5, opacity: 0.9 })
        .addTo(map);
    }

    // Add Fastest Route Polyline (Orange, Dashed)
    if (fastestCoords.length > 1) {
      layers.fastestPolyline = L.polyline(fastestCoords, { color: '#FF9F0A', weight: 3, opacity: 0.8, dashArray: '5, 10' })
        .addTo(map);
    }

    // Add Safety Circles — outer glow ring + inner solid circle
    const displayZones = (zones || []).slice(0, 500);
    displayZones.forEach(z => {
      const risk = Number(z.risk ?? 0);
      let col;
      if (risk < 0.33) col = '#30D158';
      else if (risk < 0.66) col = '#FF9F0A';
      else col = '#FF453A';

      // Outer glow
      const glow = L.circle([Number(z.latitude), Number(z.longitude)], {
        radius: 1400,
        fillColor: col,
        fillOpacity: 0.07,
        stroke: false,
      }).addTo(map);
      layers.circles.push(glow);

      // Inner solid
      const circle = L.circle([Number(z.latitude), Number(z.longitude)], {
        radius: 700,
        fillColor: col,
        fillOpacity: 0.42,
        color: col,
        weight: 1.5,
        opacity: 0.6,
      })
        .bindTooltip(
          (risk < 0.33 ? 'Safe' : risk < 0.66 ? 'Moderate' : 'Unsafe') +
          ' — ' + (risk * 100).toFixed(0) + '% risk',
          { sticky: true }
        )
        .addTo(map);
      layers.circles.push(circle);
    });

  }, [srcCoord.latitude, srcCoord.longitude, dstCoord.latitude, dstCoord.longitude, routeCoords.length, fastestCoords.length, zones]);

  return (
    <View style={styles.container}>
      {/* Map rendering container for Leaflet */}
      <div ref={mapContainerRef} style={{ height: 400, width: '100%', borderRadius: 16, overflow: 'hidden', marginBottom: 12 }} />

      <ScrollView style={styles.list} contentContainerStyle={{ padding: 8 }}>
        <Text style={styles.sectionTitle}>Safe Route ({route.length} points)</Text>
        {route.slice(0, 20).map((p, i) => (
          <View key={`r-${i}`} style={styles.pointRow}>
            <View style={[styles.colorSwatch, { backgroundColor: riskToColor ? riskToColor(p.risk ?? p.risk) : 'rgba(0,0,0,0.1)' }]} />
            <Text style={styles.pointText}>{i + 1}. {Number(p.latitude || p.lat).toFixed(5)}, {Number(p.longitude || p.lon).toFixed(5)}</Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: '100%' },
  list: { maxHeight: 300 },
  sectionTitle: { color: '#fff', fontWeight: '700', marginBottom: 6 },
  pointRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  colorSwatch: { width: 16, height: 16, borderRadius: 4, marginRight: 8 },
  pointText: { color: '#ddd', fontSize: 12 },
});
