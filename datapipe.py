import pandas as pd
import numpy as np
import folium
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
import os
warnings.filterwarnings('ignore')

def load_clean_data(csv_path='clean_safety_data_mumbai.csv'):
    """Load YOUR clean CSV dataset (Andheri West)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found! Place your file in workspace.")
    
    df = pd.read_csv(csv_path)
    required_cols = ['latitude', 'longitude', 'crime_density', 'lighting_score', 'pop_density', 'night_activity', 'unsafe']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected: {required_cols}")
    
    print(f"✅ Loaded YOUR data: {df.shape[0]} rows, {df.shape[1]} cols.")
    print(df[required_cols].describe())
    print(f"Andheri West bounds: Lat {df['latitude'].min():.4f}-{df['latitude'].max():.4f}, Lon {df['longitude'].min():.4f}-{df['longitude'].max():.4f}")
    return df

def create_ml_pipeline(features):
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), features)],
        remainder='passthrough'
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    y_pred = pipeline.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))
    return pipeline

def predict_grid(pipeline, df, grid_size=50):
    """Grid predict centered on YOUR data."""
    lat_center = df['latitude'].mean()
    lon_center = df['longitude'].mean()
    print(f"Grid centered on data mean: ({lat_center:.4f}, {lon_center:.4f})")
    
    lat_grid = np.linspace(lat_center - 0.015, lat_center + 0.015, grid_size)  # ~3km area
    lon_grid = np.linspace(lon_center - 0.015, lon_center + 0.015, grid_size)
    grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
    
    n_grid = len(grid_lats.ravel())
    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    # Create DataFrame with realistic features based on training data
    grid_data = pd.DataFrame({
        'crime_density': np.tile(df['crime_density'].mean(), n_grid) + np.random.normal(0, df['crime_density'].std(), n_grid),
        'lighting_score': np.tile(df['lighting_score'].mean(), n_grid) + np.random.normal(0, df['lighting_score'].std(), n_grid),
        'pop_density': np.tile(df['pop_density'].mean(), n_grid) + np.random.normal(0, df['pop_density'].std(), n_grid),
        'night_activity': np.tile(df['night_activity'].mean(), n_grid) + np.random.normal(0, df['night_activity'].std(), n_grid)
    })
    # Pass DataFrame directly to pipeline - it handles preprocessing
    unsafe_probs = pipeline.predict_proba(grid_data[features])[:, 1]
    
    grid_df = pd.DataFrame({
        'latitude': grid_lats.ravel(),
        'longitude': grid_lons.ravel(),
        'unsafe_prob': unsafe_probs
    })
    grid_df['risk_level'] = pd.cut(unsafe_probs, bins=[0, 0.3, 0.7, 1], 
                                   labels=['Safe 🟢', 'Moderate 🟡', 'Unsafe 🔴'])
    return grid_df

def save_safety_map(df, grid_df, output_path='safe1.html'):
    """Save interactive map with gradient zones for safe/unsafe areas."""
    lat_center, lon_center = df['latitude'].mean(), df['longitude'].mean()
    
    m = folium.Map(location=[lat_center, lon_center], zoom_start=14, tiles='OpenStreetMap')
    
    # Create gradient zones from grid data
    for idx, row in grid_df.iterrows():
        unsafe_prob = row['unsafe_prob']
        
        # Color gradient: Green (safe) -> Yellow (moderate) -> Red (unsafe)
        if unsafe_prob < 0.3:
            color = 'green'
            opacity = 0.3 + (unsafe_prob * 0.4)  # 0.3 to 0.42
        elif unsafe_prob < 0.7:
            color = 'orange'
            opacity = 0.5 + ((unsafe_prob - 0.3) * 0.4)  # 0.5 to 0.66
        else:
            color = 'red'
            opacity = 0.6 + ((unsafe_prob - 0.7) * 0.4)  # 0.6 to 0.76
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=12,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=opacity,
            weight=1,
            popup=f"Risk: {unsafe_prob:.2%}",
            tooltip=f"Unsafe Probability: {unsafe_prob:.2%}"
        ).add_to(m)
    
    # Add actual data points as markers
    if 'unsafe' in df.columns:
        safe_data = df[df['unsafe'] == 0]
        unsafe_data = df[df['unsafe'] == 1]
        
        for idx, row in safe_data.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Safe Zone<br>Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}",
                icon=folium.Icon(color='green', icon='check-circle', prefix='fa')
            ).add_to(m)
        
        for idx, row in unsafe_data.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Unsafe Zone<br>Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}",
                icon=folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
            ).add_to(m)
    
    m.save(output_path)
    print(f"✅ Interactive safety map with gradient zones: {output_path} (open in browser!)")

# ===================================== MAIN EXECUTION =====================================
if __name__ == "__main__":
    # Load YOUR file
    df = load_clean_data('clean_safety_data_mumbai.csv')
    
    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    X = df[features]
    y = df['unsafe']
    
    # Train pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = create_ml_pipeline(features)
    pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    
    # Predict & visualize on your area
    grid_df = predict_grid(pipeline, df)
    save_safety_map(df, grid_df)
    
    # Save model
    joblib.dump(pipeline, 'safety_pipeline_model.joblib')
    print("\n🚀 All done! Model ready for predictions.")
    print("Example predict:\n```python\nmodel = joblib.load('andheri_west_feedback_pipeline.joblib')\nnew_X = ...  # Your features\nprint(model.predict_proba(new_X)[:,1])\n```")