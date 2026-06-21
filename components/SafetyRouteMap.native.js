import React from "react";
import { View, Text, StyleSheet, ScrollView } from "react-native";
import MapView, { Marker, Polyline, Circle } from "react-native-maps";

export default function SafetyRouteMap({ mapRegion, source, destination, zones = [], route = [], fastestRoute = [], riskToColor }) {
  const srcCoord = { latitude: Number(source?.lat ?? source?.latitude ?? 0), longitude: Number(source?.lon ?? source?.longitude ?? 0) };
  const dstCoord = { latitude: Number(destination?.lat ?? destination?.latitude ?? 0), longitude: Number(destination?.lon ?? destination?.longitude ?? 0) };

  const routeCoords = (route || []).map(p => ({ latitude: Number(p.latitude ?? p.lat), longitude: Number(p.longitude ?? p.lon) }));
  const fastestCoords = (fastestRoute || []).map(p => ({ latitude: Number(p.latitude ?? p.lat), longitude: Number(p.longitude ?? p.lon) }));

  const region = mapRegion || {
    latitude: srcCoord.latitude || 19.1240,
    longitude: srcCoord.longitude || 72.8254,
    latitudeDelta: 0.05,
    longitudeDelta: 0.05,
  };

  // Show up to 400 zones; render each as two Circles (glow + fill)
  const displayZones = (zones || []).slice(0, 400);

  function zoneFillColor(risk) {
    if (risk < 0.33) return 'rgba(48, 209, 88,  0.40)';   // green
    if (risk < 0.66) return 'rgba(255, 159, 10, 0.38)';   // orange
    return 'rgba(255, 69,  58,  0.42)';        // red
  }
  function zoneGlowColor(risk) {
    if (risk < 0.33) return 'rgba(48, 209, 88,  0.08)';
    if (risk < 0.66) return 'rgba(255, 159, 10, 0.08)';
    return 'rgba(255, 69,  58,  0.08)';
  }
  function zoneStrokeColor(risk) {
    if (risk < 0.33) return 'rgba(48, 209, 88,  0.55)';
    if (risk < 0.66) return 'rgba(255, 159, 10, 0.55)';
    return 'rgba(255, 69,  58,  0.55)';
  }

  return (
    <View style={styles.container}>
      <MapView style={styles.mapBox} initialRegion={region} region={region}>

        {/* Draw safety zones: outer glow ring + inner solid circle */}
        {displayZones.map((z, idx) => {
          const lat = Number(z.latitude);
          const lon = Number(z.longitude);
          const risk = Number(z.risk ?? 0);
          return (
            <React.Fragment key={`zone-${idx}`}>
              {/* Outer glow */}
              <Circle
                center={{ latitude: lat, longitude: lon }}
                radius={1400}
                fillColor={zoneGlowColor(risk)}
                strokeColor="transparent"
                strokeWidth={0}
              />
              {/* Inner solid */}
              <Circle
                center={{ latitude: lat, longitude: lon }}
                radius={700}
                fillColor={zoneFillColor(risk)}
                strokeColor={zoneStrokeColor(risk)}
                strokeWidth={1.5}
              />
            </React.Fragment>
          );
        })}

        {/* Start Marker */}
        {srcCoord.latitude !== 0 && (
          <Marker coordinate={srcCoord} title="Start" pinColor="green" />
        )}

        {/* End Marker */}
        {dstCoord.latitude !== 0 && (
          <Marker coordinate={dstCoord} title="End" pinColor="red" />
        )}

        {/* Safe Route: glow layer + core polyline */}
        {routeCoords.length > 1 && (
          <>
            <Polyline coordinates={routeCoords} strokeColor="rgba(10,132,255,0.15)" strokeWidth={14} />
            <Polyline coordinates={routeCoords} strokeColor="rgba(10,132,255,0.95)" strokeWidth={5} />
          </>
        )}

        {/* Fastest Route (orange dashed) */}
        {fastestCoords.length > 1 && (
          <Polyline coordinates={fastestCoords} strokeColor="rgba(255,149,0,0.85)" strokeWidth={3.5} lineDashPattern={[8, 7]} />
        )}
      </MapView>

      <ScrollView style={styles.list} contentContainerStyle={{ padding: 8 }}>
        <Text style={styles.sectionTitle}>Safe Route ({route.length} points)</Text>
        {route.slice(0, 20).map((p, i) => (
          <View key={`r-${i}`} style={styles.pointRow}>
            <View style={[styles.colorSwatch, { backgroundColor: riskToColor ? riskToColor(p.risk ?? 0) : 'rgba(128,128,128,0.3)' }]} />
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
