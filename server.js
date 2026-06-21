const express = require("express");
const cors = require("cors");

process.on("unhandledRejection", (reason) => {
    console.warn("Unhandled rejection detected:", reason);
});

process.on("uncaughtException", (err) => {
    console.error("Uncaught exception thrown:", err);
});

require("./db");

const app = express();
const port = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
    res.send("Backend is running");
});

app.use("/api/auth", require("./auth"));

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
