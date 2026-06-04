// App.js
import React, { useState } from "react";
import MLDashboard from "./MLDashboard";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  StatusBar,
  ImageBackground,
} from "react-native";

// 🔹 Database setup placeholder (connect later via Firebase or firewall)
const connectToDatabase = () => {
  console.log("Database connection setup will go here");
};

// Dashboard Component
function Dashboard({ userName, onLogout }) {
  const handleMapPress = () => {
    console.log("Navigate to Map");
    Alert.alert("Map", "Opening Safe Routes Map...");
  };

  const handleProfilePress = () => {
    console.log("Navigate to Profile");
    Alert.alert("Profile", "Opening Profile...");
  };

  const handleSOSPress = () => {
    console.log("SOS Alert Triggered!");
    Alert.alert("🚨 SOS Alert", "Your emergency alert has been triggered!");
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
        <View style={styles.header}>
          <Text style={styles.welcomeText}>Welcome back,</Text>
          <Text style={styles.userName}>{userName}</Text>
        </View>

        <View style={styles.cardContainer}>
          {/* Map Card */}
          <TouchableOpacity
            style={[styles.card, styles.mapCard]}
            onPress={handleMapPress}
            activeOpacity={0.8}
          >
            <View style={styles.iconCircle}>
              <Text style={styles.iconEmoji}>🗺️</Text>
            </View>
            <Text style={styles.cardTitle}>Safe Routes Map</Text>
            <Text style={styles.cardSubtitle}>
              Find the safest path to your destination
            </Text>
          </TouchableOpacity>

          {/* Profile Card */}
          <TouchableOpacity
            style={[styles.card, styles.profileCard]}
            onPress={handleProfilePress}
            activeOpacity={0.8}
          >
            <View style={styles.iconCircle}>
              <Text style={styles.iconEmoji}>👤</Text>
            </View>
            <Text style={styles.cardTitle}>Profile</Text>
            <Text style={styles.cardSubtitle}>
              View and edit your personal information
            </Text>
          </TouchableOpacity>
        </View>

        {/* SOS BUTTON */}
        <View style={{ alignItems: "center", marginVertical: 30 }}>
          <TouchableOpacity
            style={styles.sosButton}
            onPress={handleSOSPress}
            activeOpacity={0.9}
          >
            <Text style={styles.sosIcon}>🚨</Text>
            <Text style={styles.sosText}>SOS</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.logoutButton} onPress={onLogout}>
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>

        <View style={styles.footer}>
          <Text style={styles.footerText}>Stay safe. Stay connected.</Text>
        </View>
      </View>
    </ImageBackground>
  );
}

// Main App Component
export default function App() {
  const [isLogin, setIsLogin] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [userName, setUserName] = useState("");

  const handleSubmit = () => {
    if (isLogin) {
      if (!email && !password) {
        Alert.alert("Missing Details", "Please enter your email and password.");
        return;
      } else if (!email) {
        Alert.alert("Missing Email", "Please enter your email address.");
        return;
      } else if (!password) {
        Alert.alert("Missing Password", "Please enter your password.");
        return;
      }

      connectToDatabase();
      setUserName(email.split("@")[0]);
      setIsLoggedIn(true);
      Alert.alert("Welcome Back!", `Logged in as ${email}`);
    } else {
      if (!name && !email && !password && !confirmPassword) {
        Alert.alert("Missing Details", "Please fill in all fields to sign up.");
        return;
      } else if (!name) {
        Alert.alert("Missing Name", "Please enter your full name.");
        return;
      } else if (!email) {
        Alert.alert("Missing Email", "Please enter your email address.");
        return;
      } else if (!password) {
        Alert.alert("Missing Password", "Please set a password.");
        return;
      } else if (!confirmPassword) {
        Alert.alert("Missing Confirmation", "Please confirm your password.");
        return;
      } else if (password !== confirmPassword) {
        Alert.alert("Password Mismatch", "Passwords do not match!");
        return;
      }

      connectToDatabase();
      setUserName(name);
      setIsLoggedIn(true);
      Alert.alert("Account Created", `Welcome, ${name}!`);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setName("");
    setEmail("");
    setPassword("");
    setConfirmPassword("");
    setUserName("");
    Alert.alert("Logged Out", "You have been logged out successfully.");
  };

  if (isLoggedIn) {
  return <MLDashboard userName={userName} onLogout={handleLogout} />;
}

  return (
    <ImageBackground
      source={{
        uri: "https://tse2.mm.bing.net/th/id/OIP.Cf9Cu9n5hoZcghMu_ds3IwHaPP?rs=1&pid=ImgDetMain&o=7&rm=3",
      }}
      style={styles.background}
      resizeMode="cover"
    >
      <StatusBar barStyle="light-content" />
      <View style={styles.overlay}>
        <View style={styles.container}>
          <Text style={styles.title}>W-SafeRoutes</Text>
          <Text style={styles.subtitle}>
            {isLogin
              ? "Welcome back! Sign in to stay protected."
              : "Join us in making the world safer for women."}
          </Text>

          {!isLogin && (
            <TextInput
              style={styles.input}
              placeholder="Full Name"
              placeholderTextColor="#fff"
              value={name}
              onChangeText={setName}
            />
          )}

          <TextInput
            style={styles.input}
            placeholder="Email"
            placeholderTextColor="#fff"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
          />
          <TextInput
            style={styles.input}
            placeholder="Password"
            placeholderTextColor="#fff"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          {!isLogin && (
            <TextInput
              style={styles.input}
              placeholder="Confirm Password"
              placeholderTextColor="#fff"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry
            />
          )}

          <TouchableOpacity style={styles.button} onPress={handleSubmit}>
            <Text style={styles.buttonText}>
              {isLogin ? "Login" : "Sign Up"}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={() => setIsLogin(!isLogin)}>
            <Text style={styles.switchText}>
              {isLogin
                ? "Don't have an account? Sign up"
                : "Already have an account? Log in"}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: { flex: 1 },
  overlay: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 40,
  },
  container: {
    backgroundColor: "rgba(255, 255, 255, 0.25)",
    padding: 30,
    borderRadius: 20,
    width: "85%",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 4,
  },
  title: { fontSize: 34, fontWeight: "800", color: "#fff", marginBottom: 10 },
  subtitle: {
    color: "#fff",
    textAlign: "center",
    fontSize: 20,
    marginBottom: 25,
  },
  input: {
    width: "100%",
    padding: 15,
    borderWidth: 1.5,
    borderColor: "#fff",
    borderRadius: 8,
    marginBottom: 18,
    color: "#fff",
    fontSize: 20,
    fontWeight: "600",
  },
  button: {
    backgroundColor: "#2E8B8B",
    paddingVertical: 12,
    paddingHorizontal: 50,
    borderRadius: 8,
    marginTop: 10,
  },
  buttonText: { color: "#fff", fontWeight: "700", fontSize: 18 },
  switchText: {
    color: "#fff",
    marginTop: 20,
    fontSize: 18,
    textDecorationLine: "underline",
  },
  header: { marginBottom: 40, paddingTop: 60 },
  welcomeText: { fontSize: 24, color: "#fff", fontWeight: "400" },
  userName: { fontSize: 36, color: "#fff", fontWeight: "800", marginTop: 5 },
  cardContainer: { width: "90%", gap: 20 },
  card: {
    backgroundColor: "rgba(255, 255, 255, 0.25)",
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
  mapCard: { backgroundColor: "rgba(46, 139, 139, 0.4)" },
  profileCard: { backgroundColor: "rgba(218, 112, 37, 0.4)" },
  iconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 15,
  },
  iconEmoji: { fontSize: 40 },
  cardTitle: { fontSize: 26, fontWeight: "800", color: "#fff", marginBottom: 8 },
  cardSubtitle: { fontSize: 16, color: "#fff", textAlign: "center", opacity: 0.9 },
  sosButton: {
    backgroundColor: "red",
    width: 150,
    height: 150,
    borderRadius: 75,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.4,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 8,
    borderWidth: 3,
    borderColor: "#fff",
  },
  sosIcon: { fontSize: 50 },
  sosText: { fontSize: 28, color: "#fff", fontWeight: "bold", marginTop: 5 },
  logoutButton: {
    marginTop: 10,
    paddingVertical: 10,
    paddingHorizontal: 30,
    backgroundColor: "rgba(255, 59, 48, 0.7)",
    borderRadius: 8,
  },
  logoutText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  footer: { marginTop: 20, marginBottom: 30, alignItems: "center" },
  footerText: { color: "#fff", fontSize: 16, fontWeight: "600", fontStyle: "italic" },
});