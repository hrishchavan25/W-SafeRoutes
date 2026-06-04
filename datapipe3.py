# =====================================
# datapipe3.py
# =====================================

import pandas as pd
import numpy as np
import joblib
import time
import requests

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================================
# CONSTANTS
# =====================================
DATASET_PATH = "andheri_west_feedback_cleaned_full.csv"

FEATURES = [
    "crime_density",
    "lighting_score",
    "pop_density",
    "night_activity"
]

_model = None
_grid_df = None

# =====================================
# COLOR LOGIC (USED BY FRONTEND)
# =====================================
def get_zone_color(prob):
    if prob < 0.33:
        return "green"
    elif prob < 0.66:
        return "yellow"
    else:
        return "red"

# =====================================
# LOAD + NORMALIZE DATASET
# =====================================
def load_dataset():
    global _grid_df

    if _grid_df is None:
        _grid_df = pd.read_csv(DATASET_PATH)

        # Ensure unsafe + unsafe_prob exist
        if "unsafe" not in _grid_df.columns:
            if "unsafe_prob" in _grid_df.columns:
                _grid_df["unsafe"] = (_grid_df["unsafe_prob"] > 0.5).astype(int)
            else:
                if set(FEATURES).issubset(set(_grid_df.columns)):
                    vals = _grid_df[FEATURES].fillna(0.0)
                    score = (
                        0.25 * vals["crime_density"]
                        - 0.6 * vals["lighting_score"]
                        + 0.008 * vals["pop_density"]
                        + 0.15 * vals["night_activity"]
                    )
                    prob = 1 / (1 + np.exp(-(score - 1.0)))
                    _grid_df["unsafe_prob"] = prob
                    _grid_df["unsafe"] = (prob > 0.5).astype(int)
                else:
                    _grid_df["unsafe"] = 0
                    _grid_df["unsafe_prob"] = 0.0

        if "unsafe_prob" not in _grid_df.columns:
            _grid_df["unsafe_prob"] = _grid_df["unsafe"].astype(float)

    return _grid_df

# =====================================
# LOAD / TRAIN MODEL
# =====================================
def load_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load("andheri_west_safety_pipeline.joblib")
        except FileNotFoundError:
            _model = train_model()
    return _model

def train_model():
    df = load_dataset()

    X = df[FEATURES]
    y = df["unsafe"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "andheri_west_safety_pipeline.joblib")
    return pipeline

# =====================================
# SAFETY GRID GENERATION (FOR MAP)
# =====================================
def generate_safety_grid(lat_center, lon_center, grid_size=40):
    model = load_model()

    lat_range = np.linspace(lat_center - 0.03, lat_center + 0.03, grid_size)
    lon_range = np.linspace(lon_center - 0.03, lon_center + 0.03, grid_size)

    grid_lats, grid_lons = np.meshgrid(lat_range, lon_range)
    n = len(grid_lats.ravel())

    # Synthetic environment features
    grid_features = pd.DataFrame({
        "crime_density": np.random.exponential(4, n),
        "lighting_score": np.random.uniform(3, 9, n),
        "pop_density": np.random.gamma(3, 80, n),
        "night_activity": np.random.uniform(2, 7, n)
    })

    unsafe_probs = model.predict_proba(grid_features)[:, 1]

    grid_df = pd.DataFrame({
        "latitude": grid_lats.ravel(),
        "longitude": grid_lons.ravel(),
        "unsafe_prob": unsafe_probs
    })

    # ADD COLOR (CRUCIAL FOR DASHBOARD)
    grid_df["color"] = grid_df["unsafe_prob"].apply(get_zone_color)

    return grid_df

# =====================================
# TRAFFIC DELAY (OSRM)
# =====================================
def get_traffic_delay(start_lat, start_lon, end_lat, end_lon):
    url = (
        "http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}?overview=false"
    )

    try:
        data = requests.get(url, timeout=10).json()
        base_time = data["routes"][0]["duration"]

        hour = time.localtime().tm_hour
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            return base_time * 0.3 / 60
        elif 11 <= hour <= 16:
            return base_time * 0.15 / 60
        else:
            return base_time * 0.05 / 60

    except Exception:
        return 5.0

# =====================================
# ROUTE SAFETY SCORE
# =====================================
def calculate_route_safety(route_points, grid_df, traffic_delay=0):
    total_risk = 0

    for lat, lon in route_points:
        distances = np.sqrt(
            (grid_df.latitude - lat) ** 2 +
            (grid_df.longitude - lon) ** 2
        )
        idx = distances.idxmin()
        total_risk += grid_df.loc[idx, "unsafe_prob"]

    avg_risk = total_risk / len(route_points)
    traffic_penalty = (traffic_delay / 60) * 0.1

    return avg_risk + traffic_penalty

# =====================================
# EXPORTS (USED BY colorpredict3.py)
# =====================================
__all__ = [
    "load_dataset",
    "load_model",
    "generate_safety_grid",
    "get_zone_color",
    "get_traffic_delay",
    "calculate_route_safety"
]