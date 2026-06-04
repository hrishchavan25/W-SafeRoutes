import os
import joblib
import pandas as pd
import numpy as np

# Expected features used by the ML pipeline
FEATURES = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']

_MODEL = None

def load_model(path='andheri_west_safety_pipeline.joblib'):
    global _MODEL
    if _MODEL is None:
        if os.path.exists(path):
            _MODEL = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
    return _MODEL

def process_data(data):
    """Convert incoming JSON-like data into a DataFrame with expected features.

    This function attempts to be flexible with input shapes commonly sent
    from mobile/front-end clients. Supported inputs:
    - dict with exact feature keys ('crime_density', 'lighting_score', ...)
    - dict with camelCase or alternative names (e.g. 'crimeDensity', 'population')
    - dict with nested payload like {'data': {...}} or {'body': {...}}
    - dict with 'latitude' and 'longitude' (creates simple synthetic features)
    - dict with 'features' key as list or dict
    - list/array-like of rows where columns include required features

    Returns a pandas.DataFrame with columns in the order of FEATURES.
    """

    def camel_to_snake(name: str) -> str:
        # simple camelCase to snake_case
        s = ''
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0 and name[i-1] != '_':
                s += '_'
            s += ch
        return s.replace('__', '_').lower()

    def normalize_key(k: str) -> str:
        s = camel_to_snake(k)
        s = s.replace('-', '_')
        return s.strip().lower()

    # synonyms mapping for flexibility
    SYNONYMS = {
        'crime_density': ['crime_density', 'crime', 'crime_density_score', 'crime_density_score', 'crime_density_value', 'crime_densityindex', 'crimeindex'],
        'lighting_score': ['lighting_score', 'lighting', 'lighting_score_value', 'lightingscore', 'light_score', 'lightingLevel', 'lightinglevel'],
        'pop_density': ['pop_density', 'population', 'popdensity', 'pop_density_value', 'population_density'],
        'night_activity': ['night_activity', 'nightactivity', 'night_active', 'activity_night', 'night_active_level']
    }

    def find_feature_value(flat: dict, feature_name: str):
        # look for exact key, then synonyms
        if feature_name in flat:
            return flat[feature_name]
        for syn in SYNONYMS.get(feature_name, []):
            nk = normalize_key(syn)
            if nk in flat:
                return flat[nk]
        return None

    # If data is a dict, try to flatten common wrapper keys
    if isinstance(data, dict):
        # unwrap common wrappers
        candidates = [data]
        for key in ('data', 'body', 'payload', 'params'):
            if key in data and isinstance(data[key], dict):
                candidates.append(data[key])

        # Build a flattened map (normalized keys -> values)
        flat = {}
        for cand in candidates:
            for k, v in cand.items():
                nk = normalize_key(k)
                flat[nk] = v

        # Direct features check
        if all(find_feature_value(flat, f) is not None for f in FEATURES):
            row = {f: float(find_feature_value(flat, f)) for f in FEATURES}
            return pd.DataFrame([row])

        # If 'features' provided as list or dict
        if 'features' in flat:
            feats = flat['features']
            if isinstance(feats, dict) and all(k in feats for k in FEATURES):
                row = {f: float(feats[f]) for f in FEATURES}
                return pd.DataFrame([row])
            if isinstance(feats, (list, tuple)) and len(feats) >= len(FEATURES):
                # assume same order
                vals = [float(feats[i]) for i in range(len(FEATURES))]
                row = dict(zip(FEATURES, vals))
                return pd.DataFrame([row])

        # If latitude/longitude exist, synthesize features
        if 'latitude' in flat and 'longitude' in flat:
            try:
                lat = float(flat['latitude'])
                lon = float(flat['longitude'])
            except Exception:
                lat = 0.0
                lon = 0.0

            crime_density = max(0.1, abs((lat - 19.1240) * 100) + np.random.rand() * 2)
            lighting_score = max(0.1, 5 - abs((lon - 72.8254) * 20) + np.random.rand() * 2)
            pop_density = max(1.0, (np.abs(lat - 19.1240) + np.abs(lon - 72.8254)) * 1000 * np.random.rand())
            night_activity = max(0.1, np.random.rand() * 8)

            df = pd.DataFrame([{
                'crime_density': crime_density,
                'lighting_score': lighting_score,
                'pop_density': pop_density,
                'night_activity': night_activity
            }])
            return df

    # If input is array-like, try to convert to DataFrame and select features
    try:
        arr = pd.DataFrame(data)
        lowered = {normalize_key(c): c for c in arr.columns}
        # If all features present in some normalized name
        if all(any(normalize_key(col) == f or normalize_key(col) in SYNONYMS.get(f, []) for col in arr.columns) for f in FEATURES):
            # build DataFrame with correct column mapping
            mapped = {}
            for f in FEATURES:
                # find the first column matching feature or synonyms
                for col in arr.columns:
                    if normalize_key(col) == f or normalize_key(col) in SYNONYMS.get(f, []):
                        mapped[f] = arr[col].astype(float)
                        break
            return pd.DataFrame(mapped)

        # If there are exactly len(FEATURES) columns, assume order matches
        if arr.shape[1] == len(FEATURES):
            return pd.DataFrame(arr.values, columns=FEATURES)
    except Exception:
        pass

    raise ValueError('Unable to process input data into required features')

def predict_color(processed_df, model_path='andheri_west_safety_pipeline.joblib'):
    """Return unsafe probability for the input DataFrame (one or more rows).

    Returns a list of probabilities (floats between 0 and 1) when multiple rows
    are provided, otherwise a single float.
    """
    model = load_model(model_path)

    # If model is a pipeline with predict_proba
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(processed_df)[..., 1]
        # convert numpy types to python floats
        results = [float(p) for p in np.asarray(probs).ravel()]
        return results[0] if len(results) == 1 else results

    # Fallback: use predict and return 1.0 for predicted unsafe, 0.0 otherwise
    if hasattr(model, 'predict'):
        preds = model.predict(processed_df)
        results = [float(p) for p in np.asarray(preds).ravel()]
        return results[0] if len(results) == 1 else results

    raise RuntimeError('Loaded model does not support prediction')
