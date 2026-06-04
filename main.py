import sys
import os
import importlib

# Prevent local `fastapi.py` from shadowing the installed `fastapi` package.
# Temporarily remove current working directory from sys.path, import, then restore.
_cwd = os.path.abspath(os.getcwd())
_removed = False
if _cwd in map(os.path.abspath, sys.path):
    try:
        sys.path.remove(_cwd)
        _removed = True
    except ValueError:
        _removed = False

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

# Restore cwd to sys.path so local imports work
if _removed:
    sys.path.insert(0, _cwd)
import os
import traceback

# ✅ Use shared ML helpers to avoid heavy imports on startup
from ml_helpers import process_data, predict_color, load_model

app = FastAPI(title="ML Prediction API", version="1.0")

# ✅ CORS (for Expo / React Native)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    model_status = False
    model_msg = "Model not loaded"

    try:
        load_model()
        model_status = True
        model_msg = "Model loaded"
    except Exception as e:
        model_msg = str(e)

    return {
        "status": "healthy",
        "message": "ML API is running",
        "model_loaded": model_status,
        "model_message": model_msg
    }

@app.post("/predict")
def predict(data: dict = Body(...)):
    try:
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")

        # Process input
        processed_data = process_data(data)

        # Ensure model is loaded
        try:
            load_model()
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Model file not found. Train or place the joblib file."
            )

        prediction = predict_color(processed_data)

        return {
            "success": True,
            "prediction": prediction,
            "message": "Prediction completed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        show_trace = os.environ.get("SHOW_TRACE", "0") == "1"
        payload = {
            "success": False,
            "error": str(e),
            "message": "Prediction failed"
        }
        if show_trace:
            payload["trace"] = traceback.format_exc()
        raise HTTPException(status_code=500, detail=payload)

@app.post("/batch_predict")
def batch_predict(data: dict = Body(...)):
    try:
        if not data or "batch_data" not in data:
            raise HTTPException(status_code=400, detail="No batch data provided")

        batch_data = data["batch_data"]
        predictions = []

        try:
            load_model()
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Model file not found. Train or place the joblib file."
            )

        for item in batch_data:
            processed_item = process_data(item)
            prediction = predict_color(processed_item)
            predictions.append(prediction)

        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "message": "Batch prediction completed"
        }

    except HTTPException:
        raise
    except Exception as e:
        show_trace = os.environ.get("SHOW_TRACE", "0") == "1"
        payload = {
            "success": False,
            "error": str(e),
            "message": "Batch prediction failed"
        }
        if show_trace:
            payload["trace"] = traceback.format_exc()
        raise HTTPException(status_code=500, detail=payload)