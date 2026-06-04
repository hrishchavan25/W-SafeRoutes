from fastapi import FastAPI, HTTPException
from ml_helpers import process_data, predict_color
import os

app = FastAPI()

MODEL_PATH = "andheri_west_safety_pipeline.joblib"

load_error = None
if not os.path.exists(MODEL_PATH):
    load_error = FileNotFoundError(f"Model file not found at {MODEL_PATH}")

@app.post("/predict")
def predict(data: dict):
    if load_error:
        raise HTTPException(status_code=500, detail=str(load_error))
    try:
        processed = process_data(data)
        pred = predict_color(processed, model_path=MODEL_PATH)
        return {"prediction": pred}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))