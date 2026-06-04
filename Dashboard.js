// Dashboard.js
import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ImageBackground,
  StatusBar,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

export default function Dashboard({ navigation, userName = "User" }) {
  const handleMapPress = () => {
    console.log("Navigate to Map");
    // navigation.navigate("Map");
  };

  const handleProfilePress = () => {
    console.log("Navigate to Profile");
    // navigation.navigate("Profile");
  };

  const handleSOSPress = () => {
    console.log("SOS Alert Triggered!");
    // You can trigger SOS alert logic here
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