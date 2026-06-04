import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

FEATURES = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']

np.random.seed(42)
X = pd.DataFrame({
    'crime_density': np.random.exponential(4, 200),
    'lighting_score': np.random.uniform(1, 10, 200),
    'pop_density': np.random.gamma(2, 50, 200),
    'night_activity': np.random.uniform(0, 8, 200)
})
y = np.random.binomial(1, 0.3, 200)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=20, random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, 'andheri_west_safety_pipeline.joblib')
print('Saved dummy model: andheri_west_safety_pipeline.joblib')
