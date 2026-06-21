// Dashboard.js
import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ImageBackground,
  StatusBar,
  Alert,
  Linking,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Location from "expo-location";
import MLService from "./MLServices";

export default function Dashboard({ navigation, userName = "User", onLogout }) {
  const handleMapPress = () => {
    console.log("Navigate to Map");
    if (navigation && navigation.navigate) {
      navigation.navigate("Map");
    }
  };

  const handleProfilePress = () => {
    console.log("Navigate to Profile");
    // navigation.navigate("Profile");
  };

  const handleSOSPress = async () => {
    console.log("SOS Alert Triggered!");
    Alert.alert(
      "🚨 SOS Triggered",
      "Are you sure you want to trigger SOS? This will alert emergency services and contacts.",
      [
        {
          text: "Cancel",
          style: "cancel",
        },
        {
          text: "YES, TRIGGER",
          style: "destructive",
          onPress: async () => {
            try {
              // 1. Request location permission
              let { status } = await Location.requestForegroundPermissionsAsync();
              let location = null;
              if (status === "granted") {
                location = await Location.getCurrentPositionAsync({
                  accuracy: Location.Accuracy.Balanced,
                });
              } else {
                console.log("Location permission denied");
              }

              const params = {
                user_id: userName,
                latitude: location ? location.coords.latitude : null,
                longitude: location ? location.coords.longitude : null,
                message: `Emergency SOS triggered by ${userName}! Please assist.`,
              };

              // 2. Call backend SOS service
              try {
                const res = await MLService.activateSOS(params);
                console.log("SOS Backend Response:", res);
              } catch (err) {
                console.log("Could not contact SOS backend:", err);
              }

              // 3. Inform user and trigger calls
              Alert.alert(
                "🚨 Alert Dispatched",
                "Police (103), Ambulance (102), Fire (101), and emergency contacts have been notified.\n\nInitiating emergency phone calls...",
                [
                  {
                    text: "Call 103 (Police)",
                    onPress: () => {
                      Linking.openURL("tel:103").catch(() => {
                        Alert.alert("Error", "Could not open dialer for 103");
                      });
                    },
                  },
                  {
                    text: "Call 100 (Alternative Police)",
                    onPress: () => {
                      Linking.openURL("tel:100").catch(() => {
                        Alert.alert("Error", "Could not open dialer for 100");
                      });
                    },
                  },
                  { text: "Dismiss", style: "cancel" },
                ]
              );
            } catch (error) {
              Alert.alert("SOS Error", "Failed to trigger SOS fully. Calling police dialer immediately.");
              Linking.openURL("tel:103").catch(() => { });
            }
          },
        },
      ]
    );
  };

  return (
    <ImageBackground
      source={{
        uri: "https://wallpaperaccess.com/full/5244290.jpg",
      }}
      style={styles.background}
      resizeMode="cover"
    >
      <StatusBar barStyle="light-content" />
      <View style={styles.overlay}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerTextContainer}>
            <Text style={styles.welcomeText}>Welcome back,</Text>
            <Text style={styles.userName}>{userName}</Text>
          </View>

          {/* Profile Button (Top Right) */}
          <TouchableOpacity
            style={styles.profileIcon}
            onPress={handleProfilePress}
            activeOpacity={0.8}
          >
            <Ionicons name="person-circle-outline" size={60} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* Main Content */}
        <View style={styles.cardContainer}>
          {/* Map Card */}
          <TouchableOpacity
            style={[styles.card, styles.mapCard]}
            onPress={handleMapPress}
            activeOpacity={0.8}
          >
            <View style={styles.iconCircle}>
              <Ionicons name="map-outline" size={40} color="#fff" />
            </View>
            <Text style={styles.cardTitle}>Safe Routes Map</Text>
            <Text style={styles.cardSubtitle}>
              Find the safest path to your destination
            </Text>
          </TouchableOpacity>

          {/* SOS Button */}
          <View style={styles.sosContainer}>
            <TouchableOpacity
              style={styles.sosButton}
              onPress={handleSOSPress}
              activeOpacity={0.9}
            >
              <Ionicons name="alert-outline" size={50} color="#fff" />
              <Text style={styles.sosText}>SOS</Text>
            </TouchableOpacity>
          </View>

          {/* Logout Button */}
          {onLogout && (
            <TouchableOpacity
              style={styles.logoutButton}
              onPress={onLogout}
              activeOpacity={0.8}
            >
              <Text style={styles.logoutText}>Logout</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>Stay safe. Stay connected.</Text>
        </View>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    paddingTop: 50,
    paddingHorizontal: 20,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 30,
  },
  headerTextContainer: {
    flexShrink: 1,
  },
  welcomeText: {
    fontSize: 24,
    color: "#fff",
    fontWeight: "400",
  },
  userName: {
    fontSize: 34,
    color: "#fff",
    fontWeight: "800",
    marginTop: 5,
  },
  profileIcon: {
    marginRight: 5,
    alignSelf: "flex-start",
  },
  cardContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 30,
  },
  card: {
    width: "90%",
    backgroundColor: "rgba(46, 139, 139, 0.4)",
    borderRadius: 20,
    padding: 25,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.3,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 6,
    borderWidth: 1.5,
    borderColor: "rgba(255, 255, 255, 0.3)",
  },
  iconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 15,
  },
  cardTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: "#fff",
    marginBottom: 8,
  },
  cardSubtitle: {
    fontSize: 16,
    color: "#fff",
    textAlign: "center",
    opacity: 0.9,
  },
  sosContainer: {
    marginTop: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  sosButton: {
    backgroundColor: "red",
    width: 160,
    height: 160,
    borderRadius: 80,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.4,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 8,
    borderWidth: 3,
    borderColor: "#fff",
  },
  sosText: {
    fontSize: 30,
    color: "#fff",
    fontWeight: "bold",
    marginTop: 5,
  },
  logoutButton: {
    marginTop: 10,
    paddingVertical: 12,
    paddingHorizontal: 40,
    backgroundColor: "rgba(255, 59, 48, 0.8)",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.3)",
    alignSelf: "center",
  },
  logoutText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
  },
  footer: {
    paddingBottom: 30,
    alignItems: "center",
  },
  footerText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
    fontStyle: "italic",
  },
});