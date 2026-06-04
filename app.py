from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import traceback

# ✅ Use shared ML helpers
from ml_helpers import process_data, predict_color, load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native requests

@app.route('/health', methods=['GET'])
def health_check():
    model_status = False
    model_msg = 'Model not loaded'
    try:
        load_model()
        model_status = True
        model_msg = 'Model loaded'
    except Exception as e:
        model_msg = str(e)

    return jsonify({
        "status": "healthy",
        "message": "ML API is running",
        "model_loaded": model_status,
        "model_message": model_msg
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Process input data
        processed_data = process_data(data)

        # Ensure model is loaded
        try:
            load_model()
        except FileNotFoundError:
            return jsonify({
                "error": "Model file not found. Train or place the joblib file."
            }), 500

        # Prediction
        prediction = predict_color(processed_data)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "message": "Prediction completed successfully"
        })
    except Exception as e:
        show_trace = os.environ.get('SHOW_TRACE', '0') == '1'
        payload = {
            "success": False,
            "error": str(e),
            "message": "Prediction failed"
        }
        if show_trace:
            payload['trace'] = traceback.format_exc()
        return jsonify(payload), 500


@app.route('/get-safe-route', methods=['POST'])
def get_safe_route():
    """Compatibility endpoint for dashboard: returns a route with per-point risk.

    Expects JSON: { start_lat, start_lon, end_lat, end_lon, steps(optional) }
    Returns: { route: [{latitude, longitude, risk, color}], traffic_delay_minutes, average_risk }
    """
    try:
        data = request.get_json() or {}
        start_lat = float(data.get('start_lat'))
        start_lon = float(data.get('start_lon'))
        end_lat = float(data.get('end_lat'))
        end_lon = float(data.get('end_lon'))
    except Exception:
        return jsonify({"error": "Invalid or missing coordinates"}), 400

    steps = int(data.get('steps', 12))
    if steps <= 0:
        steps = 12

    lat_step = (end_lat - start_lat) / steps
    lon_step = (end_lon - start_lon) / steps

    route = []
    risks = []

    for i in range(steps + 1):
        lat = start_lat + lat_step * i
        lon = start_lon + lon_step * i
        # Build synthetic feature row
        try:
            processed = process_data({"latitude": lat, "longitude": lon})
            try:
                # try to predict using model
                load_model()
                pred = predict_color(processed)
                risk = float(pred if isinstance(pred, (int, float)) else pred[0])
            except FileNotFoundError:
                # model missing: fallback heuristic
                risk = min(1.0, max(0.0, (abs(lat - 19.1240) + abs(lon - 72.8254)) * 2.0))
        except Exception:
            # processing error fallback
            risk = 0.5

        color = 'green' if risk < 0.33 else (risk < 0.66 and 'orange') or 'red'
        route.append({"latitude": lat, "longitude": lon, "risk": risk, "color": color})
        risks.append(risk)

    avg = sum(risks) / len(risks) if risks else 0

    return jsonify({
        "route": route,
        "traffic_delay_minutes": 0,
        "average_risk": avg
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()

        if not data or 'batch_data' not in data:
            return jsonify({"error": "No batch data provided"}), 400

        batch_data = data['batch_data']
        predictions = []

        # Ensure model exists
        try:
            load_model()
        except FileNotFoundError:
            return jsonify({
                "error": "Model file not found. Train or place the joblib file."
            }), 500

        for item in batch_data:
            processed_item = process_data(item)
            prediction = predict_color(processed_item)
            predictions.append(prediction)

        return jsonify({
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "message": "Batch prediction completed"
        })

    except Exception as e:
        show_trace = os.environ.get('SHOW_TRACE', '0') == '1'
        payload = {
            "success": False,
            "error": str(e),
            "message": "Batch prediction failed"
        }
        if show_trace:
            payload['trace'] = traceback.format_exc()
        return jsonify(payload), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)