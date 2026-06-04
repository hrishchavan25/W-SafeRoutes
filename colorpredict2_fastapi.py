import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import colorpredict2 as cp2
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ColorPredict2API")

app = FastAPI(title="ColorPredict2 API Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'andheri_west_safety_pipeline.joblib')
CSV_PATH = os.path.join(BASE_DIR, 'standardized_dataset.csv')

pipeline = None

def get_pipeline():
    """Lazy-load the model to ensure server stability."""
    global pipeline
    if pipeline is not None:
        return pipeline
        
    try:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Loaded model successfully from {MODEL_PATH}")
        else:
            logger.info(f"Model not found at {MODEL_PATH}. Checking dataset...")
            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
                logger.info(f"Loaded dataset: {len(df)} rows")
                
                # Robust Mapping for standardized_dataset.csv
                # Fill missing columns to avoid KeyError
                if 'police_patrolling_std' not in df.columns: df['police_patrolling_std'] = False
                if 'cctv_present_std' not in df.columns: df['cctv_present_std'] = False
                if 'harassment_free_std' not in df.columns: df['harassment_free_std'] = True
                if 'safety_std' not in df.columns: df['safety_std'] = 'safe'
                if 'unsafe_incident_std' not in df.columns: df['unsafe_incident_std'] = 'no'

                # Data Cleaning & Feature Engineering
                df['crime_density'] = df['unsafe_incident_std'].apply(lambda x: 8.5 if str(x).lower() == 'yes' else 2.5)
                df.loc[df['police_patrolling_std'] == True, 'crime_density'] -= 1.5
                df['lighting_score'] = df['cctv_present_std'].apply(lambda x: 8.0 if x == True else 3.5)
                df['pop_density'] = 450.0  # Constant for Andheri
                df['night_activity'] = df['harassment_free_std'].apply(lambda x: 7.5 if x == True else 2.0)
                df['unsafe'] = df['safety_std'].apply(lambda x: 1 if str(x).lower() == 'unsafe' else 0)

                # Location Synthesis (MANDATORY for mapping)
                if 'latitude' not in df.columns:
                    logger.info("Synthesizing missing coordinates...")
                    df['latitude'] = 19.1240 + np.random.normal(0, 0.015, len(df))
                    df['longitude'] = 72.8254 + np.random.normal(0, 0.015, len(df))

                features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
                X = df[features].fillna(0)
                y = df['unsafe'].fillna(0)
                
                pipeline = cp2.create_ml_pipeline(features)
                pipeline.fit(X, y)
                joblib.dump(pipeline, MODEL_PATH)
                logger.info("Successfully trained and saved new model.")
            else:
                logger.error(f"CRITICAL: Both model and CSV missing ({CSV_PATH})")
    except Exception as e:
        logger.error(f"ERROR during model initialization: {e}")
        logger.error(traceback.format_exc())
    return pipeline

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    current_lat: float
    current_lon: float

@app.get("/zones")
def get_zones():
    try:
        model = get_pipeline()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        grid_df = cp2.predict_grid(model)
        zones = []
        for _, row in grid_df.iterrows():
            prob = float(row["unsafe_prob"])
            color = "green" if prob < 0.3 else ("yellow" if prob < 0.7 else "red")
            zones.append({
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "risk": prob,
                "color": color
            })
        logger.info(f"✅ Served {len(zones)} safety zones")
        return {"zones": zones}
    except Exception as e:
        logger.error(f"Error in /zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-safe-route")
def get_safe_route(req: RouteRequest):
    logger.info(f"📥 POST /get-safe-route: {req.start_lat}, {req.start_lon} -> {req.end_lat}, {req.end_lon}")
    try:
        model = get_pipeline()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        grid_df = cp2.predict_grid(model)
        agent = cp2.LiveRouteSafetyAgent(grid_df, learning_rate=0.1, discount=0.9, epsilon=0.1)
        
        route_points = agent.find_safest_fastest_route(
            req.start_lat, req.start_lon, req.end_lat, req.end_lon,
            episodes=20, use_traffic=False
        )
        
        traffic_delay = 0
        if not route_points:
            route_points, dist, dur, traffic_delay = cp2.get_route_with_traffic(
                req.start_lat, req.start_lon, req.end_lat, req.end_lon
            )
        
        if not route_points:
            raise HTTPException(status_code=404, detail="Safe route not found")
            
        colored_route = []
        risks = []
        for lat, lon in route_points:
            dists = (grid_df['latitude'] - lat)**2 + (grid_df['longitude'] - lon)**2
            idx = dists.idxmin()
            prob = float(grid_df.loc[idx, 'unsafe_prob'])
            risks.append(prob)
            
            color = "green" if prob < 0.3 else ("yellow" if prob < 0.7 else "red")
            colored_route.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "risk": prob,
                "color": color
            })
            
        avg_risk = float(np.mean(risks)) if risks else 0.0
        
        return {
            "route": colored_route,
            "traffic_delay_minutes": round(traffic_delay / 60, 2),
            "average_risk": round(avg_risk, 3)
        }
    except Exception as e:
        logger.error(f"Error in /get-safe-route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/astar-route")
def get_astar_route(req: RouteRequest):
    logger.info(f"📥 POST /api/astar-route: {req.start_lat}, {req.start_lon}")
    try:
        model = get_pipeline()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        grid_df = cp2.predict_grid(model)
        path = cp2.astar_pathfinding(
            grid_df, req.start_lat, req.start_lon, req.end_lat, req.end_lon
        )
        
        if not path:
            raise HTTPException(status_code=404, detail="A* Route not found")
            
        formatted_path = [{"latitude": lat, "longitude": lon} for lat, lon in path]
        return {"path": formatted_path}
    except Exception as e:
        logger.error(f"Error in /api/astar-route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    model = get_pipeline()
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 ColorPredict2 API is initializing...")
    # Pre-load pipeline to catch issues before server starts
    try:
        get_pipeline()
    except Exception as e:
        logger.error(f"Failed to load pipeline during startup: {e}")
        
    logger.info("🚀 Starting Uvicorn server on port 8100...")
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
