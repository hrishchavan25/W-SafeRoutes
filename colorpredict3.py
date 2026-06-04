# ===============================
# colorpredict3.py
# ===============================

import pandas as pd
import numpy as np
import math
import heapq
import time
import requests
import json
import logging
import os
import traceback
from datetime import datetime

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chicago_ml_shared import records_to_row_dicts, rows_to_feature_frame

# --- GLOBAL CONFIG & API KEYS ---
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "") # Or set manually here if desired

# Optional helper module
try:
    import datapipe3 as dp
except Exception:
    dp = None
# ===============================
# APP SETUP
# ===============================
app = FastAPI(title="Women Safety Route Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Expo friendly
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logging setup: console only to avoid OneDrive lock issues ---
logger = logging.getLogger('colorpredict3')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

# ===============================
# LOAD DATASET
# ===============================
DATASET = "andheri_west_feedback_cleaned_full.csv"
CHICAGO_CRIME_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "W-SecureRoutes/1.0 (research project; local dev)"
CHICAGO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chicago_safety_pipeline.joblib")
_chicago_pipeline = None
_chicago_pipeline_attempted = False
grid_df = None
chicago_grid_df = pd.DataFrame()
try:
    # Prefer datapipe3 loader if present (keeps behavior consistent)
    try:
        import datapipe3 as dp
    except Exception:
        dp = None

    if dp is not None:
        grid_df = dp.load_dataset()
    else:
        grid_df = pd.read_csv(DATASET)
except Exception:
    # If dataset missing or fails to load, synthesize a small grid so server stays up
    try:
        lat_center, lon_center = 19.1240, 72.8254
        lats = np.linspace(lat_center - 0.01, lat_center + 0.01, 25)
        lons = np.linspace(lon_center - 0.01, lon_center + 0.01, 25)
        gl, go = np.meshgrid(lats, lons)
        n = len(gl.ravel())
        grid_df = pd.DataFrame({
            "latitude": gl.ravel(),
            "longitude": go.ravel(),
            "unsafe_prob": np.clip(np.random.beta(2,5,n), 0.0, 1.0)
        })
        grid_df["unsafe"] = (grid_df["unsafe_prob"] > 0.5).astype(int)
        print("[colorpredict3] Warning: dataset load failed, using synthetic grid.")
    except Exception:
        grid_df = pd.DataFrame(columns=["latitude", "longitude", "unsafe_prob", "unsafe"])

# ===============================
# NORMALIZE LABELS
# ===============================
def _find_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

# Ensure latitude/longitude column names exist
if grid_df is not None and not grid_df.empty:
    lat_col = _find_column(grid_df, ['latitude', 'lat', 'Latitude', 'LAT'])
    lon_col = _find_column(grid_df, ['longitude', 'lon', 'Longitude', 'LON'])
    if lat_col and lon_col:
        if lat_col != 'latitude':
            grid_df['latitude'] = grid_df[lat_col]
        if lon_col != 'longitude':
            grid_df['longitude'] = grid_df[lon_col]
    else:
        # No coordinates! Synthesize a default grid around Andheri West
        print("[colorpredict3] No lat/lon found in dataset, synthesizing grid...")
        lat_center, lon_center = 19.1240, 72.8254
        if dp is not None and hasattr(dp, 'generate_safety_grid'):
            grid_df = dp.generate_safety_grid(lat_center, lon_center, grid_size=25)
        else:
            lats = np.linspace(lat_center - 0.01, lat_center + 0.01, 25)
            lons = np.linspace(lon_center - 0.01, lon_center + 0.01, 25)
            gl, go = np.meshgrid(lats, lons)
            n = len(gl.ravel())
            grid_df = pd.DataFrame({
                "latitude": gl.ravel(),
                "longitude": go.ravel(),
                "unsafe_prob": np.clip(np.random.beta(2,5,n), 0.0, 1.0)
            })
    # Map unsafe prob synonyms
    up_col = _find_column(grid_df, ['unsafe_prob', 'unsafe_probability', 'unsafeprob', 'unsafe'])
    if up_col and up_col != 'unsafe_prob':
        grid_df['unsafe_prob'] = grid_df[up_col]

if "unsafe" not in grid_df.columns:
    if "unsafe_prob" in grid_df.columns:
        grid_df["unsafe"] = (grid_df["unsafe_prob"] > 0.5).astype(int)
    else:
        if set(["crime_density","lighting_score","pop_density","night_activity"]).issubset(set(grid_df.columns)):
            vals = grid_df[["crime_density","lighting_score","pop_density","night_activity"]].fillna(0.0)
            score = (
                0.25 * vals["crime_density"]
                - 0.6 * vals["lighting_score"]
                + 0.008 * vals["pop_density"]
                + 0.15 * vals["night_activity"]
            )
            prob = 1 / (1 + np.exp(-(score - 1.0)))
            grid_df["unsafe_prob"] = prob
            grid_df["unsafe"] = (prob > 0.5).astype(int)
        else:
            grid_df["unsafe"] = 0
            grid_df["unsafe_prob"] = 0.0

if grid_df is not None and "unsafe_prob" not in grid_df.columns:
    if 'unsafe' in grid_df.columns:
        grid_df["unsafe_prob"] = grid_df["unsafe"].astype(float)
    else:
        grid_df["unsafe_prob"] = 0.0


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def _parse_hour(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(str(date_str).replace("Z", "")).hour
    except Exception:
        return None


def _crime_risk_score(primary_type):
    crime = (primary_type or "").strip().upper()
    high = {"CRIM SEXUAL ASSAULT", "SEX OFFENSE", "KIDNAPPING", "HOMICIDE"}
    medium_high = {
        "BATTERY",
        "ASSAULT",
        "ROBBERY",
        "WEAPONS VIOLATION",
        "HUMAN TRAFFICKING",
        "STALKING",
    }
    medium = {"BURGLARY", "MOTOR VEHICLE THEFT", "THEFT", "CRIMINAL DAMAGE", "NARCOTICS"}
    if crime in high:
        return 0.9
    if crime in medium_high:
        return 0.7
    if crime in medium:
        return 0.5
    return 0.35


def get_chicago_pipeline():
    """Lazy-load sklearn pipeline trained by train_chicago_safety.py (optional)."""
    global _chicago_pipeline, _chicago_pipeline_attempted
    if _chicago_pipeline_attempted:
        return _chicago_pipeline
    _chicago_pipeline_attempted = True
    if not os.path.isfile(CHICAGO_MODEL_PATH):
        logger.info("Chicago ML model not found (%s); using heuristic risk.", CHICAGO_MODEL_PATH)
        _chicago_pipeline = None
        return None
    try:
        _chicago_pipeline = joblib.load(CHICAGO_MODEL_PATH)
        logger.info("Loaded Chicago ML model from %s", CHICAGO_MODEL_PATH)
    except Exception as exc:
        logger.warning("Failed to load Chicago model: %s", exc)
        _chicago_pipeline = None
    return _chicago_pipeline


def load_chicago_grid(limit=1000):
    try:
        safe_limit = max(1, min(int(limit or 1000), 1000))
    except Exception:
        safe_limit = 1000

    for attempt in range(2):
        try:
            response = requests.get(
                CHICAGO_CRIME_API,
                params={"$limit": safe_limit},
                timeout=(15, 90),
            )
            response.raise_for_status()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == 0:
                logger.warning("Chicago API attempt 1 failed (%s), retrying...", e)
                continue
            logger.error("Chicago API failed after retry: %s", e)
            raise

    records = response.json()
    pipeline = get_chicago_pipeline()
    row_dicts = records_to_row_dicts(records)
    if not row_dicts:
        return pd.DataFrame(columns=["latitude", "longitude", "unsafe_prob", "unsafe"])

    if pipeline is not None and row_dicts:
        X = rows_to_feature_frame(row_dicts)
        try:
            probs = pipeline.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.warning("Chicago model predict failed (%s); falling back to heuristic.", exc)
            pipeline = None

    if pipeline is not None and row_dicts:
        rows = []
        for rd, p in zip(row_dicts, probs):
            p = float(max(0.0, min(1.0, float(p))))
            rows.append(
                {
                    "latitude": rd["latitude"],
                    "longitude": rd["longitude"],
                    "unsafe_prob": p,
                    "unsafe": int(p >= 0.5),
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for item in records:
        try:
            lat = float(item.get("latitude"))
            lon = float(item.get("longitude"))
        except Exception:
            continue

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        risk = _crime_risk_score(item.get("primary_type"))
        if _to_bool(item.get("domestic")):
            risk += 0.15
        if not _to_bool(item.get("arrest")):
            risk += 0.05

        hour = _parse_hour(item.get("date"))
        if hour is not None and (hour >= 20 or hour <= 5):
            risk += 0.08

        location_desc = (item.get("location_description") or "").upper()
        if "STREET" in location_desc or "ALLEY" in location_desc:
            risk += 0.04

        risk = float(max(0.0, min(risk, 1.0)))
        rows.append(
            {
                "latitude": lat,
                "longitude": lon,
                "unsafe_prob": risk,
                "unsafe": int(risk >= 0.5),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["latitude", "longitude", "unsafe_prob", "unsafe"])
    return pd.DataFrame(rows)

# ===============================
# COLOR MAPPING (IMPORTANT)
# ===============================
def get_zone_color(prob):
    if prob < 0.33:
        return "green"
    elif prob < 0.66:
        return "yellow"
    else:
        return "red"

# ===============================
# REQUEST MODEL
# ===============================
class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    current_lat: float
    current_lon: float
    city: str = "andheri"
    data_limit: int = 1000
    use_traffic: bool = True
    tomtom_key: str = ""

# ===============================
# TRAFFIC (OSRM)
# ===============================
def get_traffic_delay(start, end, tomtom_key=""):
    """Get traffic delay from TomTom if key provided, else OSRM heuristic."""
    key = tomtom_key or TOMTOM_API_KEY
    if key:
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{start[0]},{start[1]}:{end[0]},{end[1]}/json"
        try:
            r = requests.get(url, params={'key': key, 'traffic': 'true'}, timeout=10)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get('routes'):
                first = payload['routes'][0]
                summary = first.get('summary', {})
                delay = summary.get('trafficDelayInSeconds', 0) / 60.0
                return delay
            # fall through to fallback heuristics if payload unexpected
        except Exception:
            pass

    # Fallback to OSRM + Time-of-day heuristic
    url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=false"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and payload.get('routes'):
            duration = payload['routes'][0].get('duration', None)  # seconds
        else:
            duration = None
        hour = time.localtime().tm_hour
        
        # Rush hour penalty (matching colorpredict2 logic)
        if duration is None:
            return 5.0
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            return (duration * 0.3) / 60
        return (duration * 0.1) / 60
    except:
        return 5.0

# ===============================
# A* ROUTING
# ===============================
def heuristic(a, b):
    return math.dist(a, b)

def build_graph(df, max_edge_deg: float = 0.004):
    """Build safety-aware graph. Aligns with colorpredict2's 60/40 distance/safety split."""
    graph = {}
    points = list(zip(df["latitude"].values, df["longitude"].values, df["unsafe_prob"].values))

    for lat, lon, risk in points:
        graph[(lat, lon)] = []
        for lat2, lon2, risk2 in points:
            d = math.dist((lat, lon), (lat2, lon2))
            if 0 < d < max_edge_deg:
                # Combined cost: 60% distance + 40% safety weight
                # Adjusting risk2 factor to match colorpredict2's influence
                cost = (d * 0.6) + (risk2 * 0.4)
                graph[(lat, lon)].append(((lat2, lon2), cost))
    return graph

def astar(graph, start, goal):
    pq = [(0, start)]
    came = {}
    g = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)

        if cur == goal:
            break

        # defensively handle missing nodes in the graph
        for nxt, cost in graph.get(cur, []):
            t = g[cur] + cost
            if t < g.get(nxt, 1e9):
                came[nxt] = cur
                g[nxt] = t
                heapq.heappush(pq, (t + heuristic(nxt, goal), nxt))

    path = []
    while goal in came:
        path.append(goal)
        goal = came[goal]

    path.append(start)
    return path[::-1]


def find_closest_node(df, point):
    # df must have 'latitude' and 'longitude'
    lat = point[0]
    lon = point[1]
    if df is None or df.empty:
        raise ValueError("Grid dataframe is empty or unavailable")
    dists = (df['latitude'] - lat)**2 + (df['longitude'] - lon)**2
    idx = dists.idxmin()
    return (float(df.loc[idx, 'latitude']), float(df.loc[idx, 'longitude']))


def _route_grid_for_request(req: RouteRequest):
    """Shared grid selection for safe route and A* (same graph semantics)."""
    city = str(getattr(req, "city", "andheri")).strip().lower()
    data_limit = getattr(req, "data_limit", 1000)
    
    # Ensure Andheri grid is refreshed from CSV if needed
    base_df = grid_df
    if (base_df is None or base_df.empty) and city == "andheri":
        try:
            base_df = dp.load_dataset()
        except Exception:
            pass

    if city == "chicago":
        base_df = load_chicago_grid(limit=data_limit)

    # If base dataframe is missing or doesn't contain coordinates, synthesize a grid
    if base_df is None or base_df.empty:
        lat_center = (req.start_lat + req.end_lat) / 2.0
        lon_center = (req.start_lon + req.end_lon) / 2.0
        lats = np.linspace(lat_center - 0.01, lat_center + 0.01, 30)
        lons = np.linspace(lon_center - 0.01, lon_center + 0.01, 30)
        gl, go = np.meshgrid(lats, lons)
        pts = np.vstack([gl.ravel(), go.ravel()]).T
        dists = np.sqrt((pts[:, 0] - lat_center) ** 2 + (pts[:, 1] - lon_center) ** 2)
        probs = 1 / (1 + np.exp((dists - 0.005) * 200))
        grid_use = pd.DataFrame({"latitude": pts[:, 0], "longitude": pts[:, 1], "unsafe_prob": probs})
        grid_use["unsafe"] = (grid_use["unsafe_prob"] > 0.5).astype(int)
    elif "latitude" not in base_df.columns or "longitude" not in base_df.columns:
        lat_center = (req.start_lat + req.end_lat) / 2.0
        lon_center = (req.start_lon + req.end_lon) / 2.0
        try:
            if dp is not None and hasattr(dp, "generate_safety_grid"):
                grid_use = dp.generate_safety_grid(lat_center, lon_center, grid_size=30)
            else:
                lats = np.linspace(lat_center - 0.01, lat_center + 0.01, 30)
                lons = np.linspace(lon_center - 0.01, lon_center + 0.01, 30)
                gl, go = np.meshgrid(lats, lons)
                center = np.array([lat_center, lon_center])
                pts = np.vstack([gl.ravel(), go.ravel()]).T
                dists = np.sqrt((pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2)
                probs = 1 / (1 + np.exp((dists - 0.005) * 200))
                grid_use = pd.DataFrame(
                    {"latitude": pts[:, 0], "longitude": pts[:, 1], "unsafe_prob": probs}
                )
                grid_use["unsafe"] = (grid_use["unsafe_prob"] > 0.5).astype(int)
        except Exception:
            grid_use = base_df.copy()
    else:
        grid_use = base_df

    # Andheri dataset might be sparser, so we use a larger edge distance (0.015 approx 1.6km)
    edge = 0.022 if city == "chicago" else 0.015
    return grid_use, edge


# ===============================
# ROOT (browser sanity check)
# ===============================
@app.get("/")
def root():
    """Visiting http://127.0.0.1:8100/ in a browser otherwise returns 404 from FastAPI."""
    return {
        "service": "Women Safety Route Engine (colorpredict3)",
        "endpoints": {
            "health": "GET /health",
            "zones": "GET /zones?city=chicago&limit=1000  (or city=andheri)",
            "safe_route": "POST /get-safe-route  (JSON body includes city, data_limit)",
            "astar": "POST /api/astar-route",
            "geocode": "GET /geocode?q=...&city=chicago|andheri",
        },
        "try_in_browser": [
            "/health",
            "/zones?city=chicago&limit=100",
        ],
        "train_chicago_model": "python train_chicago_safety.py   (writes chicago_safety_pipeline.joblib)",
    }


# ===============================
# ZONES API (FOR MAP COLORS)
# ===============================
@app.get("/zones")
def get_zones(city: str = "andheri", limit: int = 1000):
    global chicago_grid_df
    zones = []
    use_df = grid_df

    if city.strip().lower() == "chicago":
        chicago_grid_df = load_chicago_grid(limit=limit)
        use_df = chicago_grid_df

    for _, row in use_df.iterrows():
        zones.append({
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "risk": float(row["unsafe_prob"]),
            "color": get_zone_color(row["unsafe_prob"]),
        })

    return {"zones": zones}


@app.get("/geocode")
def geocode(q: str = "", city: str = "chicago"):
    """Forward geocode (OpenStreetMap Nominatim). Proxied so web clients avoid CORS."""
    text = (q or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing query parameter q")

    c = city.strip().lower()
    if c == "chicago":
        location_query = f"{text}, Chicago, IL, USA"
    else:
        location_query = f"{text}, Andheri, Mumbai, Maharashtra, India"

    try:
        r = requests.get(
            NOMINATIM_SEARCH,
            params={"q": location_query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": NOMINATIM_UA, "Accept": "application/json"},
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("Geocode request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Geocoding service error: {exc}") from exc

    if not data:
        raise HTTPException(
            status_code=404,
            detail="No results for that place name. Try a landmark or neighborhood.",
        )
    hit = data[0]
    return {
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
        "display_name": hit.get("display_name", ""),
    }


# ===============================
# SAFE ROUTE API
# ===============================
@app.post("/get-safe-route")
def get_safe_route(req: RouteRequest):
    # Log incoming request
    try:
        req_payload = {
            'start_lat': req.start_lat,
            'start_lon': req.start_lon,
            'end_lat': req.end_lat,
            'end_lon': req.end_lon,
            'current_lat': req.current_lat,
            'current_lon': req.current_lon
        }
    except Exception:
        req_payload = {}
    logger.info('INCOMING /get-safe-route %s', json.dumps(req_payload, ensure_ascii=False))

    try:
        grid_use, edge = _route_grid_for_request(req)
        graph = build_graph(grid_use, max_edge_deg=edge)

        start = (req.start_lat, req.start_lon)
        end = (req.end_lat, req.end_lon)

        # Ensure start/end are nodes in the graph: pick closest grid nodes
        try:
            start_node = find_closest_node(grid_use, start)
            end_node = find_closest_node(grid_use, end)
        except ValueError as ve:
            logger.error("Grid node resolution failed: %s", ve)
            raise HTTPException(status_code=400, detail=str(ve))

        route = astar(graph, start_node, end_node)

        # Get live traffic delay using the updated engine
        traffic_delay = get_traffic_delay(start, end, req.tomtom_key)

        colored_route = []
        risks = []

        for i, (lat, lon) in enumerate(route):
            idx = ((grid_use["latitude"] - lat) ** 2 + (grid_use["longitude"] - lon) ** 2).idxmin()
            prob = float(grid_use.loc[idx, "unsafe_prob"])
            risks.append(prob)

            colored_route.append({
                "latitude": lat,
                "longitude": lon,
                "risk": prob,
                "color": get_zone_color(prob),
                "is_live_sample": i % 5 == 0  # To simulate live location movement points
            })

        avg_risk = float(np.mean(risks)) if risks else 0.0

        response_summary = {
            'route_points': len(colored_route),
            'avg_risk': round(avg_risk, 3),
            'traffic_delay_minutes': round(traffic_delay, 2)
        }
        
        # --- External Map Links ---
        # Format: origin=lat,lon&destination=lat,lon&waypoints=lat|lon
        gmaps_base = "https://www.google.com/maps/dir/?api=1"
        gmaps_url = f"{gmaps_base}&origin={req.start_lat},{req.start_lon}&destination={req.end_lat},{req.end_lon}"
        
        # Sample waypoints for gmaps (max 23, we take a few)
        if len(route) > 5:
            wps = route[1:-1:max(1, len(route)//10)]
            wp_str = "|".join([f"{lat},{lon}" for lat, lon in wps])
            gmaps_url += f"&waypoints={wp_str}"

        apple_maps_url = f"maps://?saddr={req.start_lat},{req.start_lon}&daddr={req.end_lat},{req.end_lon}&dirflg=d"

        logger.info('RESPONSE /get-safe-route %s', json.dumps(response_summary, ensure_ascii=False))

        return {
            "route": colored_route,
            "traffic_delay_minutes": round(traffic_delay, 2),
            "average_risk": round(avg_risk, 3),
            "google_maps_url": gmaps_url,
            "apple_maps_url": apple_maps_url,
            "city_context": req.city
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error('ERROR /get-safe-route %s', tb)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post("/api/astar-route")
def astar_route(req: RouteRequest):
    """Fast path on same safety graph as /get-safe-route (for Expo map overlay)."""
    try:
        grid_use, edge = _route_grid_for_request(req)
        graph = build_graph(grid_use, max_edge_deg=edge)
        start = (req.start_lat, req.start_lon)
        end = (req.end_lat, req.end_lon)
        try:
            start_node = find_closest_node(grid_use, start)
            end_node = find_closest_node(grid_use, end)
        except ValueError as ve:
            logger.error("Grid node resolution failed: %s", ve)
            raise HTTPException(status_code=400, detail=str(ve))

        path = astar(graph, start_node, end_node)
        formatted = [{"latitude": float(lat), "longitude": float(lon)} for lat, lon in path]
        return {"path": formatted}
    except Exception as e:
        logger.error("ERROR /api/astar-route %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    if grid_df is None or grid_df.empty:
        return {"status": "degraded", "message": "no grid data"}
    return {
        "status": "ok",
        "rows": int(len(grid_df)),
        "chicago_model_present": os.path.isfile(CHICAGO_MODEL_PATH),
    }

# ===============================
# MAP GENERATION (Back-compatibility with colorpredict2.py)
# ===============================
import folium

@app.get("/generate-map/{city}")
def generate_city_map(city: str = "andheri"):
    """
    Saves a Folium HTML map equivalent to colorpredict2.py's output.
    Accessible at: http://127.0.0.1:8200/generate-map/andheri
    """
    map_df = grid_df
    center = [19.1240, 72.8254]
    if city.lower() == "chicago":
        map_df = load_chicago_grid()
        center = [41.8781, -87.6298]

    m = folium.Map(location=center, zoom_start=13)
    for _, row in map_df.iterrows():
        prob = row["unsafe_prob"]
        color = get_zone_color(prob)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fillOpacity=0.4,
            stroke=False
        ).add_to(m)
    
    filename = f"{city}_safety_map_gen.html"
    m.save(filename)
    return {"status": "success", "file": filename, "message": f"Map saved as {filename}"}


if __name__ == "__main__":
    port = int(os.getenv("COLORPREDICT3_PORT", "4040"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")