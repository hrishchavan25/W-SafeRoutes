import React from "react";
import { View, Text, StyleSheet, ScrollView, Platform } from "react-native";

// Try to load react-native-maps dynamically (so web or missing native module won't crash)
let MapView = null;
let Marker = null;
let Polyline = null;
try {
  if (Platform.OS !== 'web') {
    const RNMaps = require('react-native-maps');
    MapView = RNMaps && (RNMaps.default || RNMaps);
    Marker = RNMaps && RNMaps.Marker;
    Polyline = RNMaps && RNMaps.Polyline;
  }
} catch (e) {
  MapView = null;
}

export default function SafetyRouteMap({ mapRegion, source, destination, zones = [], route = [], fastestRoute = [], riskToColor }) {
  const srcCoord = { latitude: Number(source?.lat ?? source?.latitude ?? 0), longitude: Number(source?.lon ?? source?.longitude ?? 0) };
  const dstCoord = { latitude: Number(destination?.lat ?? destination?.latitude ?? 0), longitude: Number(destination?.lon ?? destination?.longitude ?? 0) };

  const routeCoords = (route || []).map(p => ({ latitude: Number(p.latitude ?? p.lat), longitude: Number(p.longitude ?? p.lon) }));
  const fastestCoords = (fastestRoute || []).map(p => ({ latitude: Number(p.latitude ?? p.lat), longitude: Number(p.longitude ?? p.lon) }));

  // If native MapView exists, render interactive map
  if (MapView) {
    const region = mapRegion || { latitude: srcCoord.latitude || 0, longitude: srcCoord.longitude || 0, latitudeDelta: 0.05, longitudeDelta: 0.05 };
    return (
      <View style={styles.container}>
        <MapView style={styles.mapBox} initialRegion={region} region={region}>
          {srcCoord.latitude !== 0 && (
            <Marker coordinate={srcCoord} title="Start" />
          )}
          {dstCoord.latitude !== 0 && (
            <Marker coordinate={dstCoord} pinColor="blue" title="End" />
          )}
          {routeCoords.length > 1 && (
            <Polyline coordinates={routeCoords} strokeColor="rgba(10,132,255,0.9)" strokeWidth={4} />
          )}
          {fastestCoords.length > 1 && (
            <Polyline coordinates={fastestCoords} strokeColor="rgba(255,149,0,0.8)" strokeWidth={2} lineDashPattern={[4,8]} />
          )}
        </MapView>

        <ScrollView style={styles.list} contentContainerStyle={{ padding: 8 }}>
          <Text style={styles.sectionTitle}>Safe Route ({route.length})</Text>
          {route.slice(0, 20).map((p, i) => (
            <View key={`r-${i}`} style={styles.pointRow}>
              <View style={[styles.colorSwatch, { backgroundColor: riskToColor ? riskToColor(p.risk ?? p.risk) : 'rgba(0,0,0,0.1)'}]} />
              <Text style={styles.pointText}>{i+1}. {Number(p.latitude || p.lat).toFixed(5)}, {Number(p.longitude || p.lon).toFixed(5)}</Text>
            </View>
          ))}
        </ScrollView>
      </View>
    );
  }

  // Fallback placeholder (web or missing native module)
  return (
    <View style={styles.container}>
      <View style={styles.mapBoxPlaceholder}>
        <Text style={styles.mapTitle}>Map placeholder (react-native-maps not available)</Text>
        <Text style={styles.small}>Center: {mapRegion?.latitude?.toFixed?.(4)},{mapRegion?.longitude?.toFixed?.(4)}</Text>
        <Text style={styles.small}>Source: {srcCoord.latitude.toFixed(5)},{srcCoord.longitude.toFixed(5)}</Text>
        <Text style={styles.small}>Destination: {dstCoord.latitude.toFixed(5)},{dstCoord.longitude.toFixed(5)}</Text>
      </View>

      <ScrollView style={styles.list} contentContainerStyle={{ padding: 8 }}>
        <Text style={styles.sectionTitle}>Safe Route ({route.length})</Text>
        {route.slice(0, 20).map((p, i) => (
          <View key={`r-${i}`} style={styles.pointRow}>
            <View style={[styles.colorSwatch, { backgroundColor: riskToColor ? riskToColor(p.risk ?? p.risk) : 'rgba(0,0,0,0.1)'}]} />
            <Text style={styles.pointText}>{i+1}. {Number(p.latitude || p.lat).toFixed(5)}, {Number(p.longitude || p.lon).toFixed(5)}</Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: '100%' },
  mapBox: {
    height: 360,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 12,
  },
  mapBoxPlaceholder: {
    height: 300,
    backgroundColor: '#0B1220',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  mapTitle: { color: '#fff', fontWeight: '700', marginBottom: 6 },
  small: { color: '#ddd', fontSize: 12 },
  list: { maxHeight: 300 },
  sectionTitle: { color: '#fff', fontWeight: '700', marginBottom: 6 },
  pointRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  colorSwatch: { width: 16, height: 16, borderRadius: 4, marginRight: 8 },
  pointText: { color: '#ddd', fontSize: 12 },
});
