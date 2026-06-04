const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

config.transpilePackages = [
  ...(config.transpilePackages || []),
  "leaflet",
  "react-leaflet",
];

module.exports = config;
