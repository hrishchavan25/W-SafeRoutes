import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Dimensions,
  TextInput,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Animated,
  Platform,
  Linking,
} from "react-native";
import { Picker } from "@react-native-picker/picker";
import { Ionicons } from "@expo/vector-icons";
import MLService from "./MLServices";
import SafetyRouteMap from "./components/SafetyRouteMap";
import { BACKEND_URL } from "./config";
import { ANDHERI_REGIONS, getRegionById } from "./data/andheri_regions";

const { width, height } = Dimensions.get("window");

const CITY_PLACE_DEFAULTS = {
  chicago: { from: "Millennium Park", to: "Lincoln Park Zoo" },
  mumbai: { from: "Andheri Railway Station", to: "Versova Beach" },
  andheri: { from: "Andheri Railway Station", to: "Versova Beach" },
};

function regionFromPoints(lat1, lon1, lat2, lon2) {
  const midLat = (lat1 + lat2) / 2;
  const midLon = (lon1 + lon2) / 2;
  const dLat = Math.abs(lat1 - lat2) * 2.2 + 0.03;
  const dLon = Math.abs(lon1 - lon2) * 2.2 + 0.03;
  return {
    latitude: midLat,
    longitude: midLon,
    latitudeDelta: Math.max(dLat, 0.06),
    longitudeDelta: Math.max(dLon, 0.06),
  };
}

export default function MLDashboard({ userName = "User", onLogout, onBack }) {
  const CHICAGO_REGION = { latitude: 41.8781, longitude: -87.6298, latitudeDelta: 0.12, longitudeDelta: 0.12 };
  const CHICAGO_SOURCE = { lat: 41.8810, lon: -87.6278 };
  const CHICAGO_DEST = { lat: 41.7914, lon: -87.6005 };
  const MUMBAI_REGION = { latitude: 19.124, longitude: 72.825, latitudeDelta: 0.12, longitudeDelta: 0.12 };
  const MUMBAI_SOURCE = { lat: 19.1248, lon: 72.8254 };
  const MUMBAI_DEST = { lat: 19.1350, lon: 72.8150 };
  const ANDHERI_REGION = { latitude: 19.124, longitude: 72.825, latitudeDelta: 0.04, longitudeDelta: 0.04 };
  const ANDHERI_SOURCE = { lat: 19.1248, lon: 72.8254 };
  const ANDHERI_DEST = { lat: 19.1350, lon: 72.8150 };

  const [selectedCity, setSelectedCity] = useState("mumbai");
  const [apiHost, setApiHost] = useState(BACKEND_URL);
  const [loading, setLoading] = useState(false);
  const [zonesLoading, setZonesLoading] = useState(false);
  const [route, setRoute] = useState([]);
  const [fastestRoute, setFastestRoute] = useState([]);
  const [zones, setZones] = useState([]);
  const [networkError, setNetworkError] = useState(null);

  // Stats
  const [trafficDelay, setTrafficDelay] = useState(null);
  const [avgRisk, setAvgRisk] = useState(null);
  const [routeDistance, setRouteDistance] = useState(null);

  // External Maps Navigation Redirect URLs
  const [googleMapsUrl, setGoogleMapsUrl] = useState(null);
  const [appleMapsUrl, setAppleMapsUrl] = useState(null);

  // Map settings
  const [mapRegion, setMapRegion] = useState({
    ...MUMBAI_REGION,
  });

  // Locations (filled from place names via /geocode)
  const [source, setSource] = useState(MUMBAI_SOURCE);
  const [destination, setDestination] = useState(MUMBAI_DEST);
  const [fromPlace, setFromPlace] = useState(CITY_PLACE_DEFAULTS.mumbai.from);
  const [toPlace, setToPlace] = useState(CITY_PLACE_DEFAULTS.mumbai.to);
  const [fromRegionId, setFromRegionId] = useState("railway");
  const [toRegionId, setToRegionId] = useState("versova");

  // Animation
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    loadSettings();
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 1000,
      useNativeDriver: true,
    }).start();
  }, []);

  const loadSettings = async () => {
    const host = await MLService.getApiHost();
    setApiHost(host);
    refreshData(host);
  };

  const refreshData = async (host, cityOverride) => {
    setZonesLoading(true);
    setNetworkError(null);
    try {
      const city = cityOverride || selectedCity;
      const data = await MLService.getZones({ city, limit: 1000 });
      if (data.zones) setZones(data.zones);
    } catch (err) {
      setNetworkError("Cannot connect to ML Backend");
      console.log(err);
    } finally {
      setZonesLoading(false);
    }
  };

  const handleRefresh = () => {
    refreshData(apiHost);
  };

  const switchCity = (city) => {
    setSelectedCity(city);
    setRoute([]);
    setFastestRoute([]);
    setGoogleMapsUrl(null);
    setAppleMapsUrl(null);
    const defaults = CITY_PLACE_DEFAULTS[city] || CITY_PLACE_DEFAULTS.mumbai;
    setFromPlace(defaults.from);
    setToPlace(defaults.to);
    if (city === "chicago") {
      setSource(CHICAGO_SOURCE);
      setDestination(CHICAGO_DEST);
      setMapRegion(CHICAGO_REGION);
      refreshData(apiHost, city);
      return;
    }
    if (city === "mumbai") {
      setSource(MUMBAI_SOURCE);
      setDestination(MUMBAI_DEST);
      setMapRegion(MUMBAI_REGION);
      refreshData(apiHost, city);
      return;
    }
    setFromRegionId("railway");
    setToRegionId("versova");
    const rFrom = getRegionById("railway");
    const rTo = getRegionById("versova");
    setSource({ lat: rFrom.lat, lon: rFrom.lon });
    setDestination({ lat: rTo.lat, lon: rTo.lon });
    setFromPlace(rFrom.name);
    setToPlace(rTo.name);
    setMapRegion(regionFromPoints(rFrom.lat, rFrom.lon, rTo.lat, rTo.lon));
    refreshData(apiHost, city);
  };

  const resolvePlacesToCoordinates = async () => {
    if (selectedCity === "andheri") {
      if (fromRegionId === toRegionId) {
        Alert.alert("Same region", "Choose two different Andheri regions for start and end.");
        return null;
      }
      const rFrom = getRegionById(fromRegionId);
      const rTo = getRegionById(toRegionId);
      const nextSource = { lat: rFrom.lat, lon: rFrom.lon };
      const nextDestination = { lat: rTo.lat, lon: rTo.lon };
      setSource(nextSource);
      setDestination(nextDestination);
      setFromPlace(rFrom.name);
      setToPlace(rTo.name);
      setMapRegion(regionFromPoints(rFrom.lat, rFrom.lon, rTo.lat, rTo.lon));
      return { source: nextSource, destination: nextDestination };
    }

    const from = fromPlace.trim();
    const to = toPlace.trim();
    if (!from || !to) {
      Alert.alert("Missing places", "Enter both the starting area and where you are going.");
      return null;
    }
    try {
      const g1 = await MLService.geocode(from, selectedCity);
      await new Promise((r) => setTimeout(r, 1100));
      const g2 = await MLService.geocode(to, selectedCity);
      const nextSource = { lat: g1.lat, lon: g1.lon };
      const nextDestination = { lat: g2.lat, lon: g2.lon };
      setSource(nextSource);
      setDestination(nextDestination);
      setMapRegion(regionFromPoints(g1.lat, g1.lon, g2.lat, g2.lon));
      return { source: nextSource, destination: nextDestination };
    } catch (err) {
      const msg = err?.message || String(err);
      Alert.alert("Could not find a place", msg);
      return null;
    }
  };

  const runPrediction = async () => {
    if (loading) return;
    setLoading(true);
    setNetworkError(null);
    setGoogleMapsUrl(null);
    setAppleMapsUrl(null);

    const resolved = await resolvePlacesToCoordinates();
    if (!resolved) {
      setLoading(false);
      return;
    }

    const params = {
      start_lat: resolved.source.lat,
      start_lon: resolved.source.lon,
      end_lat: resolved.destination.lat,
      end_lon: resolved.destination.lon,
      current_lat: resolved.source.lat,
      current_lon: resolved.source.lon,
      city: selectedCity,
      data_limit: 1000,
    };

    try {
      // Fetch both routes independently to prevent one failure from blocking the other
      const safePromise = MLService.getSafeRoute(params)
        .then(data => {
          if (data.route) {
            setRoute(data.route);
            setTrafficDelay(data.traffic_delay_minutes);
            setAvgRisk(data.average_risk);
            if (data.google_maps_url) setGoogleMapsUrl(data.google_maps_url);
            if (data.apple_maps_url) setAppleMapsUrl(data.apple_maps_url);

            // Center map on the safest route
            const midIdx = Math.floor(data.route.length / 2);
            setMapRegion(prev => ({
              ...prev,
              latitude: data.route[midIdx].latitude,
              longitude: data.route[midIdx].longitude,
            }));
          }
        })
        .catch(err => console.log("Safe route error:", err));

      const astarPromise = MLService.getAStarRoute(params)
        .then(data => {
          if (data.path) {
            setFastestRoute(data.path);
          }
        })
        .catch(err => {
          console.log("A* route error:", err);
        });

      await Promise.all([safePromise, astarPromise]);

    } catch (err) {
      const host = await MLService.getApiHost();
      setNetworkError(`Network Error: Ensure backend is running at ${host} and your phone can reach this computer's IP.`);
      console.log(err);
    } finally {
      setLoading(false);
    }
  };

  const testConnection = async () => {
    setZonesLoading(true);
    setNetworkError(null);
    try {
      const status = await MLService.healthCheck();
      if (status.status === "ok") {
        Alert.alert("Success", "AI Backend is connected and running!");
        refreshData(apiHost);
      }
    } catch (err) {
      setNetworkError(`Connection Failed to ${apiHost}. Check your IP and Firewall.`);
      Alert.alert("Connection Failed", "Ensure uvicorn is running on your computer.");
    } finally {
      setZonesLoading(false);
    }
  };



  const riskToColor = (risk) => {
    if (risk >= 0.7) return 'rgba(255, 69, 58, 0.4)'; // Red
    if (risk >= 0.3) return 'rgba(255, 159, 10, 0.35)'; // Orange
    return 'rgba(48, 209, 88, 0.25)'; // Green
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* Header */}
      <View style={styles.header}>
        <View style={{ flexDirection: "row", alignItems: "center" }}>
          {onBack && (
            <TouchableOpacity onPress={onBack} style={{ marginRight: 12 }}>
              <Ionicons name="chevron-back" size={28} color="#0A84FF" />
            </TouchableOpacity>
          )}
          <View>
            <Text style={styles.headerTitle}>SECURE<Text style={{ color: '#0A84FF' }}>ROUTES</Text></Text>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Text style={styles.headerSubtitle}>AI Analysis Agent • </Text>
              <Text style={[styles.headerSubtitle, { color: '#FFF' }]}>{userName}</Text>
            </View>
          </View>
        </View>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <TouchableOpacity style={[styles.refreshBadge, { marginRight: 10 }]} onPress={() => refreshData(apiHost)}>
            {zonesLoading ? (
              <ActivityIndicator size="small" color="#0A84FF" />
            ) : (
              <Text style={styles.refreshText}>Live active</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity onPress={onLogout} style={styles.logoutBtn}>
            <Text style={styles.logoutBtnText}>Logout</Text>
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40 }}>

        {/* API Config Panel */}
        <View style={styles.glassCard}>
          <Text style={styles.cardLabel}>AI BACKEND (FROM CONFIG.JS)</Text>
          <View style={[styles.inputRow, { marginBottom: 10 }]}>
            <TouchableOpacity
              style={[styles.connectBtn, { flex: 1, marginRight: 6 }, selectedCity === "chicago" ? null : { backgroundColor: "#2C2C2E" }]}
              onPress={() => switchCity("chicago")}
            >
              <Text style={[styles.connectBtnText, { textAlign: 'center' }]}>Chicago</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.connectBtn, { flex: 1, marginRight: 6 }, selectedCity === "mumbai" ? null : { backgroundColor: "#2C2C2E" }]}
              onPress={() => switchCity("mumbai")}
            >
              <Text style={[styles.connectBtnText, { textAlign: 'center' }]}>Mumbai</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.connectBtn, { flex: 1 }, selectedCity === "andheri" ? null : { backgroundColor: "#2C2C2E" }]}
              onPress={() => switchCity("andheri")}
            >
              <Text style={[styles.connectBtnText, { textAlign: 'center' }]}>Andheri</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.inputRow}>
            <TextInput
              style={[styles.apiInput, { opacity: 0.8 }]}
              value={apiHost}
              editable={false}
              placeholder="https://...ngrok-free.app"
              placeholderTextColor="#636366"
            />
            <TouchableOpacity style={styles.connectBtn} onPress={handleRefresh}>
              <Text style={styles.connectBtnText}>Refresh</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.helperText}>
            City: {selectedCity}
            {selectedCity === "chicago" ? " | Geocode + maps path (Chicago)" :
              selectedCity === "mumbai" ? " | Geocode + maps path (Mumbai)" : " | Region dropdowns (Andheri)"}
          </Text>

          <TouchableOpacity style={styles.testBtn} onPress={testConnection}>
            <Text style={styles.testBtnText}>Check Connection Status</Text>
          </TouchableOpacity>
          {networkError && <Text style={styles.errorText}>{networkError}</Text>}
        </View>

        {/* Map Container (native: MapView; web: SafetyRouteMap.web.js — no react-native-maps) */}
        <View style={styles.mapContainer}>
          <SafetyRouteMap
            styles={styles}
            selectedCity={selectedCity}
            mapRegion={mapRegion}
            source={source}
            destination={destination}
            zones={zones}
            route={route}
            fastestRoute={fastestRoute}
            riskToColor={riskToColor}
          />
        </View>

        {/* Route Planning */}
        <View style={styles.glassCard}>
          <Text style={styles.cardLabel}>
            {selectedCity === "chicago" ? "ROUTE (CHICAGO — SEARCH)" :
              selectedCity === "mumbai" ? "ROUTE (MUMBAI — SEARCH)" : "ROUTE (ANDHERI — REGIONS)"}
          </Text>
          {selectedCity === "chicago" || selectedCity === "mumbai" ? (
            <Text style={styles.placeHint}>
              Type start and end areas. The server geocodes them (OpenStreetMap). Open in Google / Apple Maps uses your safest-route waypoints ({selectedCity === "chicago" ? "Chicago" : "Mumbai"} only).
            </Text>
          ) : (
            <Text style={styles.placeHint}>
              Pick Andheri West regions from your project data. Each option shows coordinates in brackets.
            </Text>
          )}

          {selectedCity === "chicago" || selectedCity === "mumbai" ? (
            <>
              <View style={styles.coordGroup}>
                <Text style={styles.smallLabel}>FROM (AREA OR LANDMARK)</Text>
                <TextInput
                  style={styles.placeInput}
                  value={fromPlace}
                  onChangeText={setFromPlace}
                  placeholder="e.g. Hyde Park"
                  placeholderTextColor="#636366"
                />
              </View>
              <View style={styles.coordGroup}>
                <Text style={styles.smallLabel}>TO (AREA OR LANDMARK)</Text>
                <TextInput
                  style={styles.placeInput}
                  value={toPlace}
                  onChangeText={setToPlace}
                  placeholder="e.g. The Loop"
                  placeholderTextColor="#636366"
                />
              </View>
              <TouchableOpacity
                style={[styles.secondaryBtn, loading && styles.disabledBtn]}
                onPress={async () => {
                  if (loading) return;
                  setLoading(true);
                  try {
                    const resolved = await resolvePlacesToCoordinates();
                    if (resolved) {
                      Alert.alert("Locations set", "Map updated. Tap analyze to run safety routing.");
                    }
                  } finally {
                    setLoading(false);
                  }
                }}
                disabled={loading}
              >
                <Text style={styles.secondaryBtnText}>LOOKUP ONLY (SET MAP)</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <View style={styles.coordGroup}>
                <Text style={styles.smallLabel}>FROM (REGION)</Text>
                <View style={styles.pickerWrap}>
                  <Picker
                    mode="dropdown"
                    selectedValue={fromRegionId}
                    onValueChange={(id) => {
                      setFromRegionId(id);
                      const r = getRegionById(id);
                      setSource({ lat: r.lat, lon: r.lon });
                      setFromPlace(r.name);
                    }}
                    style={styles.picker}
                    itemStyle={styles.pickerItemIOS}
                  >
                    {ANDHERI_REGIONS.map((r) => (
                      <Picker.Item
                        key={r.id}
                        label={`${r.name} (${r.lat}, ${r.lon})`}
                        value={r.id}
                        color="#fff"
                      />
                    ))}
                  </Picker>
                </View>
              </View>
              <View style={styles.coordGroup}>
                <Text style={styles.smallLabel}>TO (REGION)</Text>
                <View style={styles.pickerWrap}>
                  <Picker
                    mode="dropdown"
                    selectedValue={toRegionId}
                    onValueChange={(id) => {
                      setToRegionId(id);
                      const r = getRegionById(id);
                      setDestination({ lat: r.lat, lon: r.lon });
                      setToPlace(r.name);
                    }}
                    style={styles.picker}
                    itemStyle={styles.pickerItemIOS}
                  >
                    {ANDHERI_REGIONS.map((r) => (
                      <Picker.Item
                        key={r.id}
                        label={`${r.name} (${r.lat}, ${r.lon})`}
                        value={r.id}
                        color="#fff"
                      />
                    ))}
                  </Picker>
                </View>
              </View>
            </>
          )}

          {/* Stats Display */}
          {(trafficDelay !== null || avgRisk !== null) && (
            <Animated.View style={[styles.statsContainer, { opacity: fadeAnim }]}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{trafficDelay}m</Text>
                <Text style={styles.statLabel}>TRAFFIC DELAY</Text>
              </View>
              <View style={[styles.statItem, { borderLeftWidth: 1, borderRightWidth: 1, borderColor: '#38383A' }]}>
                <Text style={[styles.statValue, { color: avgRisk < 0.3 ? '#30D158' : (avgRisk < 0.7 ? '#FF9F0A' : '#FF453A') }]}>
                  {(avgRisk * 100).toFixed(1)}%
                </Text>
                <Text style={styles.statLabel}>AVG RISK</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>RL Agent</Text>
                <Text style={styles.statLabel}>MODEL TYPE</Text>
              </View>
            </Animated.View>
          )}

          {/* External Map Redirection Buttons */}
          {(googleMapsUrl || appleMapsUrl) && (
            <View style={styles.redirectContainer}>
              <Text style={styles.redirectTitle}>🚀 EXTERNAL NAVIGATION SHORTCUTS</Text>
              <View style={styles.redirectRow}>
                {googleMapsUrl ? (
                  <TouchableOpacity
                    style={[styles.redirectBtn, styles.googleBtn]}
                    onPress={() => Linking.openURL(googleMapsUrl).catch(err => Alert.alert("Error", "Could not open Google Maps"))}
                  >
                    <Text style={styles.redirectBtnEmoji}>🟢</Text>
                    <Text style={styles.redirectBtnText}>Google Maps</Text>
                  </TouchableOpacity>
                ) : null}
                {appleMapsUrl && (Platform.OS === 'ios' || Platform.OS === 'web') ? (
                  <TouchableOpacity
                    style={[styles.redirectBtn, styles.appleBtn]}
                    onPress={() => Linking.openURL(appleMapsUrl).catch(err => Alert.alert("Error", "Could not open Apple Maps"))}
                  >
                    <Text style={styles.redirectBtnEmoji}>🍎</Text>
                    <Text style={styles.redirectBtnText}>Apple Maps</Text>
                  </TouchableOpacity>
                ) : null}
              </View>
            </View>
          )}

          <TouchableOpacity
            style={[styles.actionBtn, loading && styles.disabledBtn]}
            onPress={runPrediction}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.actionBtnText}>FIND AND ANALYZE SAFEST ROUTE</Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Info Card */}
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>💡 Safety Intelligence</Text>
          <Text style={styles.infoText}>
            Chicago/Mumbai: search by name and open Google or Apple Maps with your safest route as waypoints. Andheri: choose regions from the list (coordinates shown), then analyze.
          </Text>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#1C1C1E',
    borderBottomWidth: 1,
    borderColor: '#38383A'
  },
  headerTitle: { fontSize: 22, fontWeight: "900", color: "#FFF", letterSpacing: 1 },
  headerSubtitle: { fontSize: 10, color: "#8E8E93", marginTop: 2, letterSpacing: 1.5 },
  refreshBadge: {
    backgroundColor: '#2C2C2E',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center'
  },
  refreshText: { color: '#30D158', fontSize: 10, fontWeight: 'bold' },

  glassCard: {
    margin: 15,
    padding: 20,
    backgroundColor: '#1C1C1E',
    borderRadius: 24,
    borderWidth: 1,
    borderColor: '#38383A',
  },
  cardLabel: { fontSize: 10, fontWeight: "bold", color: "#8E8E93", marginBottom: 15, letterSpacing: 1.2 },
  smallLabel: { fontSize: 9, fontWeight: "bold", color: "#636366", marginBottom: 5 },

  inputRow: { flexDirection: "row", alignItems: "center" },
  apiInput: {
    flex: 1,
    backgroundColor: '#2C2C2E',
    borderRadius: 12,
    padding: 12,
    color: '#FFF',
    fontSize: 14,
  },
  connectBtn: { marginLeft: 10, backgroundColor: "#0A84FF", paddingHorizontal: 15, paddingVertical: 12, borderRadius: 12 },
  connectBtnText: { color: "#FFF", fontSize: 12, fontWeight: "bold" },

  mapContainer: {
    marginHorizontal: 15,
    height: 380,
    borderRadius: 30,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: '#38383A',
    backgroundColor: '#1C1C1E'
  },
  map: { width: "100%", height: "100%" },
  mapControls: { position: 'absolute', bottom: 15, right: 15 },
  legendBox: { backgroundColor: 'rgba(28, 28, 30, 0.9)', padding: 10, borderRadius: 15 },
  legendItem: { flexDirection: 'row', alignItems: 'center', marginVertical: 2 },
  legendDot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  legendText: { color: '#FFF', fontSize: 9 },

  coordInputs: { marginBottom: 20 },
  coordGroup: { marginBottom: 12 },
  coordRow: { flexDirection: 'row', justifyContent: 'space-between' },
  coordInput: {
    width: '48%',
    backgroundColor: '#2C2C2E',
    padding: 10,
    borderRadius: 10,
    color: '#FFF',
    fontSize: 13
  },
  placeHint: {
    color: "#8E8E93",
    fontSize: 11,
    lineHeight: 16,
    marginBottom: 14,
  },
  placeInput: {
    width: "100%",
    backgroundColor: "#2C2C2E",
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 12,
    color: "#FFF",
    fontSize: 15,
    borderWidth: 1,
    borderColor: "#38383A",
  },
  secondaryBtn: {
    marginBottom: 14,
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
    backgroundColor: "#2C2C2E",
    borderWidth: 1,
    borderColor: "#0A84FF",
  },
  secondaryBtnText: {
    color: "#0A84FF",
    fontWeight: "800",
    fontSize: 12,
    letterSpacing: 0.5,
  },
  pickerWrap: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#38383A",
    backgroundColor: "#2C2C2E",
    overflow: "hidden",
  },
  picker: {
    color: "#fff",
    width: "100%",
  },
  pickerItemIOS: {
    color: "#fff",
    fontSize: 16,
  },

  statsContainer: {
    flexDirection: "row",
    backgroundColor: "#2C2C2E",
    borderRadius: 16,
    marginBottom: 20,
    paddingVertical: 15,
  },
  statItem: { flex: 1, alignItems: 'center' },
  statValue: { fontSize: 18, fontWeight: 'bold', color: '#FFF' },
  statLabel: { fontSize: 8, color: '#8E8E93', marginTop: 4, fontWeight: 'bold' },
  redirectContainer: {
    backgroundColor: "#1C1C1E",
    borderRadius: 16,
    padding: 15,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "#38383A",
  },
  redirectTitle: {
    fontSize: 10,
    fontWeight: "bold",
    color: "#8E8E93",
    marginBottom: 10,
    letterSpacing: 1.0,
  },
  redirectRow: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  redirectBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    borderRadius: 12,
    marginHorizontal: 4,
  },
  googleBtn: {
    backgroundColor: "#34A853",
  },
  appleBtn: {
    backgroundColor: "#2C2C2E",
    borderWidth: 1,
    borderColor: "#38383A",
  },
  redirectBtnEmoji: {
    marginRight: 6,
    fontSize: 14,
  },
  redirectBtnText: {
    color: "#FFF",
    fontSize: 12,
    fontWeight: "bold",
  },

  actionBtn: {
    backgroundColor: "#0A84FF",
    padding: 18,
    borderRadius: 16,
    alignItems: "center",
    shadowColor: "#0A84FF",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5
  },
  disabledBtn: { backgroundColor: "#3A3A3C" },
  actionBtnText: { color: "#FFF", fontWeight: "900", fontSize: 15, letterSpacing: 1 },

  infoCard: { margin: 15, padding: 20, backgroundColor: '#1C1C1E', borderRadius: 20, borderLeftWidth: 4, borderLeftColor: '#0A84FF' },
  infoTitle: { color: '#FFF', fontWeight: 'bold', fontSize: 14, marginBottom: 8 },
  infoText: { color: '#8E8E93', fontSize: 12, lineHeight: 18 },

  errorText: { color: "#FF453A", fontSize: 11, marginTop: 10, textAlign: "center", fontWeight: 'bold' },
  helperText: { color: "#8E8E93", fontSize: 10, marginTop: 8, textAlign: "center", fontStyle: 'italic' },
  testBtn: { marginTop: 12, padding: 8, alignItems: 'center', borderTopWidth: 1, borderColor: '#2C2C2E' },
  testBtnText: { color: '#0A84FF', fontSize: 11, fontWeight: '600' },
  markerContainer: { width: 24, height: 24, borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  markerDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 2, borderColor: '#FFF' },
  logoutBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 69, 58, 0.2)',
    borderWidth: 1,
    borderColor: 'rgba(255, 69, 58, 0.5)',
  },
  logoutBtnText: { color: '#FF453A', fontSize: 10, fontWeight: 'bold' }
});
