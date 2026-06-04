"""
Fetch Chicago crime (Socrata) and train a sklearn pipeline saved as chicago_safety_pipeline.joblib.
Run from project root: python train_chicago_safety.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chicago_ml_shared import add_training_labels, records_to_row_dicts, rows_to_feature_frame

CHICAGO_CRIME_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chicago_safety_pipeline.joblib")


def fetch_chicago_crime(total_limit: int = 15000, page_size: int = 5000) -> list:
    rows: list = []
    offset = 0
    while len(rows) < total_limit:
        chunk = min(page_size, total_limit - len(rows))
        params = {"$limit": chunk, "$offset": offset, "$order": "date DESC"}
        last_err = None
        for attempt in range(2):
            try:
                r = requests.get(CHICAGO_CRIME_API, params=params, timeout=(15, 120))
                r.raise_for_status()
                batch = r.json()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt == 0:
                    print(f"Retrying fetch after: {e}")
                    continue
                raise
        else:
            raise last_err  # pragma: no cover
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < chunk:
            break
    return rows[:total_limit]


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", max_categories=40, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", max_categories=40, sparse=False)


def build_pipeline() -> Pipeline:
    numeric = ["latitude", "longitude", "hour", "arrest", "domestic"]
    categorical = ["primary_type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", _one_hot_encoder(), categorical),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    print(f"Fetching up to {total} Chicago records...")
    raw = fetch_chicago_crime(total_limit=total)
    print(f"Downloaded {len(raw)} raw records")
    row_dicts = records_to_row_dicts(raw)
    if len(row_dicts) < 200:
        print("Not enough rows with lat/lon to train; aborting.")
        sys.exit(1)
    X = rows_to_feature_frame(row_dicts)
    Xy = add_training_labels(X)
    y = Xy["unsafe"].values
    print(f"Training rows: {len(y)}, positive rate: {y.mean():.3f}")

    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"Holdout accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["lower_person_risk", "higher_person_risk"]))
    if accuracy_score(y_test, pred) >= 0.99:
        print(
            "(Note: labels are rule-based from crime type; metrics can look optimistic — "
            "the saved model is still useful for consistent risk scores on live API rows.)"
        )

    joblib.dump(pipe, MODEL_OUT)
    print(f"Saved pipeline to {MODEL_OUT}")


if __name__ == "__main__":
    main()
