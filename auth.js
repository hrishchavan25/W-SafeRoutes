const express = require("express");
const bcrypt = require("bcryptjs");
const mongoose = require("mongoose");
const fs = require("fs");
const path = require("path");
const User = require("./User");

const router = express.Router();
const USERS_FILE = path.join(__dirname, "users.json");

// Helper: Check if MongoDB is connected
const isMongoConnected = () => {
  return mongoose.connection.readyState === 1;
};

// Helper: Get local JSON users
const getLocalUsers = () => {
  if (!fs.existsSync(USERS_FILE)) {
    return [];
  }
  try {
    return JSON.parse(fs.readFileSync(USERS_FILE, "utf8"));
  } catch (e) {
    return [];
  }
};

// Helper: Save user to local JSON
const saveLocalUser = (user) => {
  const users = getLocalUsers();
  users.push(user);
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2), "utf8");
};

/* SIGNUP */
router.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;
  try {
    const cleanEmail = email.toLowerCase().trim();
    if (isMongoConnected()) {
      const existingUser = await User.findOne({ email: cleanEmail });
      if (existingUser) {
        return res.status(400).json({ message: "Email already registered" });
      }
      const hashedPassword = await bcrypt.hash(password, 10);
      await new User({ name, email: cleanEmail, password: hashedPassword }).save();
      res.json({ message: "Signup Successful" });
    } else {
      console.log("⚠️ MongoDB offline. Using users.json database fallback.");
      const users = getLocalUsers();
      const existingUser = users.find((u) => u.email === cleanEmail);
      if (existingUser) {
        return res.status(400).json({ message: "Email already registered" });
      }
      const hashedPassword = await bcrypt.hash(password, 10);
      saveLocalUser({
        name: name.trim(),
        email: cleanEmail,
        password: hashedPassword,
      });
      res.json({ message: "Signup Successful (Local Database Fallback)" });
    }
  } catch (err) {
    res.status(500).json({ message: "Error during signup", error: err.message });
  }
});

/* LOGIN */
router.post("/login", async (req, res) => {
  const { email, password } = req.body;
  try {
    const cleanEmail = email.toLowerCase().trim();
    if (isMongoConnected()) {
      const user = await User.findOne({ email: cleanEmail });
      if (!user) {
        return res.status(400).json({ message: "Invalid email or password" });
      }
      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        return res.status(400).json({ message: "Invalid email or password" });
      }
      res.json({ message: "Login Successful", name: user.name });
    } else {
      console.log("⚠️ MongoDB offline. Using users.json database fallback.");
      const users = getLocalUsers();
      const user = users.find((u) => u.email === cleanEmail);
      if (!user) {
        return res.status(400).json({ message: "Invalid email or password" });
      }
      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        return res.status(400).json({ message: "Invalid email or password" });
      }
      res.json({ message: "Login Successful (Local Database Fallback)", name: user.name });
    }
  } catch (err) {
    res.status(500).json({ message: "Error during login", error: err.message });
  }
});

module.exports = router;