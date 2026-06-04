import pandas as pd
import numpy as np
import folium
import folium.plugins  # Added this import for HeatMap
import geopandas as gpd
from shapely.geometry import Point
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_clean_data(csv_path='clean_safety_data_mumbai.csv'):
    """Load clean CSV dataset."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols.")
    print(df.head())
    return df

def create_ml_pipeline(features):
    """Create end-to-end sklearn Pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features)  # Scale numeric features
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    return pipeline

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """Train pipeline and print metrics."""
    pipeline.fit(X_train, y_train)
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    y_pred = pipeline.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))
    
    return pipeline

def predict_grid(pipeline, lat_center=19.1240, lon_center=72.8254, grid_size=50):
    """Predict on lat/lon grid for Andheri West."""
    lat_grid = np.linspace(lat_center - 0.03, lat_center + 0.03, grid_size)
    lon_grid = np.linspace(lon_center - 0.03, lon_center + 0.03, grid_size)
    grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
    
    n_grid = len(grid_lats.ravel())
    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    grid_data = pd.DataFrame({
        'crime_density': np.random.exponential(5, n_grid),
        'lighting_score': np.random.uniform(0, 10, n_grid),
        'pop_density': np.random.gamma(2, 50, n_grid),
        'night_activity': np.random.uniform(0, 8, n_grid)
    })
    
    unsafe_probs = pipeline.predict_proba(grid_data[features])[:, 1]
    grid_df = pd.DataFrame({
        'latitude': grid_lats.ravel(),
        'longitude': grid_lons.ravel(),
        'unsafe_prob': unsafe_probs
    })
    grid_df['risk_level'] = pd.cut(unsafe_probs, bins=[0, 0.3, 0.7, 1], 
                                   labels=['Safe 🟢', 'Moderate 🟡', 'Unsafe 🔴'])
    return grid_df

def save_safety_map(df, grid_df, output_path='andheri_west_safety_map.html'):
    """Save interactive Folium map for Andheri West with gradient zones."""
    lat_center, lon_center = 19.1240, 72.8254
    m = folium.Map(location=[lat_center, lon_center], zoom_start=14, tiles='OpenStreetMap')
    
    # Create heatmap data: [lat, lon, weight]
    heat_data = [[row['latitude'], row['longitude'], row['unsafe_prob']] 
                 for idx, row in grid_df.iterrows()]
    
    # Add gradient heatmap overlay
    folium.plugins.HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=25,
        blur=15,
        gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}
    ).add_to(m)
    
    # Original data points
    safe_pts = df[df['unsafe'] == 0][['latitude', 'longitude']].values
    unsafe_pts = df[df['unsafe'] == 1][['latitude', 'longitude']].values
    
    for pt in safe_pts:
        folium.CircleMarker(location=pt, radius=4, popup='Safe Area', 
                           color='darkgreen', fill=True, fillColor='lightgreen', fillOpacity=0.8, weight=0, stroke=False).add_to(m)
    for pt in unsafe_pts:
        folium.CircleMarker(location=pt, radius=4, popup='Unsafe Area', 
                           color='darkred', fill=True, fillColor='lightcoral', fillOpacity=0.8, weight=0, stroke=False).add_to(m)
    
    m.save(output_path)
    print(f"✅ Andheri West map saved: {output_path}")

# =====================================
# MAIN PIPELINE EXECUTION (ANDHERI WEST)
# =====================================
if __name__ == "__main__":
    # Step 1: Load or generate Mumbai-specific data
    csv_path = 'clean_safety_data_mumbai.csv'
    try:
        df = load_clean_data(csv_path)
    except FileNotFoundError:
        print("Generating synthetic data for Andheri West...")
        np.random.seed(42)
        n_samples = 1000
        lat_center, lon_center = 19.1240, 72.8254  # 🔄 CHANGED: Andheri West, Mumbai
        lats = np.random.normal(lat_center, 0.02, n_samples)  # Tighter for suburb
        lons = np.random.normal(lon_center, 0.02, n_samples)
        
        # Urban India tweaks: Higher pop, moderate crime
        crime_density = np.random.exponential(4, n_samples) + np.random.rand(n_samples) * 8
        lighting_score = np.random.uniform(3, 9, n_samples)  # Better street lights
        pop_density = np.random.gamma(3, 80, n_samples)  # Denser urban
        night_activity = np.random.uniform(2, 7, n_samples)
        
        unsafe_prob = 1 / (1 + np.exp(-(0.25*crime_density - 0.6*lighting_score + 0.008*pop_density + 0.15*night_activity - 4)))
        df = pd.DataFrame({
            'latitude': lats, 'longitude': lons,
            'crime_density': crime_density, 'lighting_score': lighting_score,
            'pop_density': pop_density, 'night_activity': night_activity,
            'unsafe': np.random.binomial(1, unsafe_prob, n_samples)
        })
        df.to_csv(csv_path, index=False)
        print(f"Synthetic Mumbai data saved: {csv_path}")

    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    X = df[features]
    y = df['unsafe']

    # Step 2-4: Train, grid predict, map
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = create_ml_pipeline(features)
    pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    grid_df = predict_grid(pipeline)
    save_safety_map(df, grid_df)

    # Step 5: Save
    joblib.dump(pipeline, 'andheri_west_safety_pipeline.joblib')
    print("✅ Mumbai pipeline model saved!")