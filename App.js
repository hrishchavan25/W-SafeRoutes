// App.js
import React, { useState, useEffect } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import MLDashboard from "./MLDashboard";
import Dashboard from "./Dashboard";
import { AUTH_BACKEND_URL } from "./config";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  StatusBar,
  ImageBackground,
  Modal,
  ScrollView,
} from "react-native";

// 🔹 Database setup placeholder (connect later via Firebase or firewall)
const connectToDatabase = () => {
  console.log("Database connection setup will go here");
};

const LEGAL_DOCS = {
  terms: {
    title: "Terms and Conditions",
    body: [
      "Welcome to W-SafeRoutes. By creating an account, logging in, or using the app, you agree to use the service responsibly and only for lawful personal safety and route-planning purposes.",
      "W-SafeRoutes provides safety insights, route suggestions, emergency contact tools, and map-based information. These features are provided for assistance only and should not replace your personal judgment, local emergency services, official safety guidance, or real-time conditions around you.",
      "You are responsible for keeping your account details secure and for entering accurate information when using route, SOS, or location-based features.",
      "The app may depend on third-party services for maps, routing, geocoding, hosting, authentication, and other technical features. Availability and accuracy may vary based on network access, device settings, location permissions, and third-party service performance.",
      "We may update app features, safety logic, and these terms from time to time. Continued use of the app means you accept the updated terms.",
    ],
  },
  privacy: {
    title: "Privacy Policy",
    body: [
      "W-SafeRoutes collects the information needed to provide account access and safety-route features, such as your name, email address, password authentication data, selected start and destination areas, route requests, and location data when you allow location-based features.",
      "Location and route information is used to generate safety analysis, route suggestions, maps, and SOS-related functionality. Account information is used for login, signup, session handling, and user identification inside the app.",
      "Your information may be processed by backend hosting, database, authentication, maps, routing, geocoding, and analytics or infrastructure providers that help the app function. We do not sell your personal information.",
      "You can deny device location permission, but some route and safety features may not work fully. You can log out from the app to remove the saved local session from your device.",
      "For account deletion, privacy questions, or data requests, contact the app owner or developer using the support contact listed on the app store listing.",
    ],
  },
};

// Main App Component
export default function App() {
  const [isLogin, setIsLogin] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [userName, setUserName] = useState("");
  const [currentScreen, setCurrentScreen] = useState("dashboard");
  const [acceptedLegal, setAcceptedLegal] = useState(false);
  const [legalDoc, setLegalDoc] = useState(null);

  useEffect(() => {
    checkSavedSession();
  }, []);

  const checkSavedSession = async () => {
    try {
      const session = await AsyncStorage.getItem("user_session");
      if (session) {
        const parsed = JSON.parse(session);
        if (parsed && parsed.name) {
          setUserName(parsed.name);
          setIsLoggedIn(true);
        }
      }
    } catch (e) {
      console.log("Failed to load session:", e);
    }
  };

  const handleSubmit = async () => {
    if (!acceptedLegal) {
      Alert.alert(
        "Agreement Required",
        "Please accept the Terms and Conditions and Privacy Policy to continue."
      );
      return;
    }

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

      try {
        const response = await fetch(`${AUTH_BACKEND_URL}/api/auth/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ email: email.toLowerCase().trim(), password }),
        });
        const data = await response.json();

        if (response.ok) {
          const loggedInName = data.name || email.split("@")[0];
          setUserName(loggedInName);
          setIsLoggedIn(true);

          // Save session to device database (AsyncStorage)
          await AsyncStorage.setItem(
            "user_session",
            JSON.stringify({ name: loggedInName, email: email.toLowerCase().trim() })
          );

          Alert.alert("Welcome Back!", `Logged in as ${loggedInName}`);
        } else {
          Alert.alert("Login Failed", data.message || "Invalid credentials.");
        }
      } catch (err) {
        console.error("Login connection error:", err);
        Alert.alert(
          "Connection Error",
          "Could not connect to the authentication server. Please check if backend is running on port 5001."
        );
      }
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

      try {
        const response = await fetch(`${AUTH_BACKEND_URL}/api/auth/signup`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name: name.trim(),
            email: email.toLowerCase().trim(),
            password,
          }),
        });
        const data = await response.json();

        if (response.ok) {
          Alert.alert(
            "Account Created",
            "Signup successful! Please log in with your new credentials."
          );
          setIsLogin(true);
          setPassword("");
          setConfirmPassword("");
        } else {
          Alert.alert("Signup Failed", data.message || "Could not register user.");
        }
      } catch (err) {
        console.error("Signup connection error:", err);
        Alert.alert(
          "Connection Error",
          "Could not connect to the authentication server. Please check if backend is running on port 5001."
        );
      }
    }
  };

  const handleLogout = async () => {
    try {
      await AsyncStorage.removeItem("user_session");
    } catch (e) {
      console.log("Error clearing session:", e);
    }
    setIsLoggedIn(false);
    setName("");
    setEmail("");
    setPassword("");
    setConfirmPassword("");
    setUserName("");
    setCurrentScreen("dashboard");
    Alert.alert("Logged Out", "You have been logged out successfully.");
  };

  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
    setAcceptedLegal(false);
  };

  const activeLegalDoc = legalDoc ? LEGAL_DOCS[legalDoc] : null;

  if (isLoggedIn) {
    if (currentScreen === "mldashboard") {
      return (
        <MLDashboard
          userName={userName}
          onLogout={handleLogout}
          onBack={() => setCurrentScreen("dashboard")}
        />
      );
    }
    return (
      <Dashboard
        userName={userName}
        navigation={{
          navigate: (screen) => {
            if (screen === "Map" || screen === "MLDashboard") {
              setCurrentScreen("mldashboard");
            }
          },
        }}
        onLogout={handleLogout}
      />
    );
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

          <View style={styles.legalRow}>
            <TouchableOpacity
              style={[styles.checkbox, acceptedLegal && styles.checkboxChecked]}
              onPress={() => setAcceptedLegal(!acceptedLegal)}
              accessibilityRole="checkbox"
              accessibilityState={{ checked: acceptedLegal }}
            >
              {acceptedLegal ? <Text style={styles.checkboxMark}>✓</Text> : null}
            </TouchableOpacity>
            <Text style={styles.legalText}>
              I agree to the{" "}
              <Text style={styles.legalLink} onPress={() => setLegalDoc("terms")}>
                Terms and Conditions
              </Text>
              {" "}and{" "}
              <Text style={styles.legalLink} onPress={() => setLegalDoc("privacy")}>
                Privacy Policy
              </Text>
              .
            </Text>
          </View>

          <TouchableOpacity style={styles.button} onPress={handleSubmit}>
            <Text style={styles.buttonText}>
              {isLogin ? "Login" : "Sign Up"}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={toggleAuthMode}>
            <Text style={styles.switchText}>
              {isLogin
                ? "Don't have an account? Sign up"
                : "Already have an account? Log in"}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      <Modal
        visible={!!activeLegalDoc}
        transparent
        animationType="slide"
        onRequestClose={() => setLegalDoc(null)}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.legalModal}>
            <Text style={styles.legalTitle}>{activeLegalDoc?.title}</Text>
            <ScrollView style={styles.legalScroll} showsVerticalScrollIndicator>
              {activeLegalDoc?.body.map((paragraph, index) => (
                <Text key={index} style={styles.legalParagraph}>
                  {paragraph}
                </Text>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.modalButton} onPress={() => setLegalDoc(null)}>
              <Text style={styles.modalButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
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
  legalRow: {
    width: "100%",
    flexDirection: "row",
    alignItems: "flex-start",
    marginTop: 2,
    marginBottom: 6,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: "#fff",
    marginRight: 10,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.12)",
  },
  checkboxChecked: {
    backgroundColor: "#2E8B8B",
  },
  checkboxMark: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "900",
    lineHeight: 18,
  },
  legalText: {
    flex: 1,
    color: "#fff",
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "600",
  },
  legalLink: {
    color: "#E8FFFF",
    textDecorationLine: "underline",
    fontWeight: "900",
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.65)",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  legalModal: {
    width: "100%",
    maxWidth: 520,
    maxHeight: "82%",
    backgroundColor: "#fff",
    borderRadius: 18,
    padding: 20,
  },
  legalTitle: {
    fontSize: 22,
    fontWeight: "800",
    color: "#173B3B",
    marginBottom: 12,
  },
  legalScroll: {
    marginBottom: 16,
  },
  legalParagraph: {
    color: "#243333",
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 12,
  },
  modalButton: {
    backgroundColor: "#2E8B8B",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  modalButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "800",
  },
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
