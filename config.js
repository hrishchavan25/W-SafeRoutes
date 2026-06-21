import Constants from "expo-constants";

const env = typeof process !== "undefined" ? process.env || {} : {};

const PROD_ML_URL =
    env.EXPO_PUBLIC_ML_API_URL || "https://w-saferoutes-ml-api.onrender.com";
const PROD_SOS_URL =
    env.EXPO_PUBLIC_SOS_API_URL || "https://w-saferoutes-auth-api.onrender.com";
const PROD_AUTH_URL =
    env.EXPO_PUBLIC_AUTH_API_URL || "https://w-saferoutes-auth-api.onrender.com";

let resolvedMlUrl = "http://localhost:8100";
let resolvedSosUrl = "http://localhost:5000";
let resolvedAuthUrl = "http://localhost:5001";

const hostUri = Constants.expoConfig?.hostUri || "";
const ipAddress = hostUri.split(":")[0];

if (__DEV__ && ipAddress) {
    resolvedMlUrl = `http://${ipAddress}:8100`;
    resolvedSosUrl = `http://${ipAddress}:5000`;
    resolvedAuthUrl = `http://${ipAddress}:5001`;
    console.log(`[W-SafeRoutes] Dynamically resolved dev endpoints using host IP ${ipAddress}:`);
    console.log(`  - ML Backend: ${resolvedMlUrl}`);
    console.log(`  - SOS Backend: ${resolvedSosUrl}`);
    console.log(`  - Auth Backend: ${resolvedAuthUrl}`);
} else {
    resolvedMlUrl = PROD_ML_URL;
    resolvedSosUrl = PROD_SOS_URL;
    resolvedAuthUrl = PROD_AUTH_URL;
}

export const BACKEND_URL = resolvedMlUrl;
export const SOS_BACKEND_URL = resolvedSosUrl;
export const AUTH_BACKEND_URL = resolvedAuthUrl;
export const LOCAL_BACKEND_URL = resolvedMlUrl;
