const express = require("express");
const cors = require("cors");

require("./db"); // database connection

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
    res.send("Backend is running");
});

app.use("/api/auth", require("./routes/auth"));

app.listen(5000, () => {
    console.log("✅ Server running on port 5000");
});