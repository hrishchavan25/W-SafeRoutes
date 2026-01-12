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
warnings.filterwarnings('ignore')

def load_clean_data(csv_path='andheri_west_feedback_cleaned_full.csv'):
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
        remainder='passthrough'  # Ignore other cols like lat/lon if present
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    return pipeline

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """Train pipeline and print metrics."""
    pipeline.fit(X_train, y_train)
    
    # Cross-val
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test set
    y_pred = pipeline.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))
    
    return pipeline

def predict_grid(pipeline, X_train, lat_center=40.7128, lon_center=-74.0060, grid_size=50):
    """Predict on lat/lon grid using realistic spatial features."""
    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    lat_grid = np.linspace(lat_center - 0.03, lat_center + 0.03, grid_size)
    lon_grid = np.linspace(lon_center - 0.03, lon_center + 0.03, grid_size)
    grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
    
    # Generate realistic grid features based on spatial patterns
    n_grid = len(grid_lats.ravel())
    # Use training data statistics to generate realistic features
    grid_crime = np.random.exponential(X_train['crime_density'].mean(), n_grid) + \
                 2 * (np.abs(grid_lats.ravel() - lat_center) + np.abs(grid_lons.ravel() - lon_center)) * 100
    grid_lighting = X_train['lighting_score'].mean() - np.random.uniform(0, 3, n_grid)
    grid_pop = np.random.gamma(2, X_train['pop_density'].mean() / 2, n_grid)
    grid_night = np.random.uniform(X_train['night_activity'].min(), X_train['night_activity'].max(), n_grid)
    
    # Create grid dataframe with actual features
    grid_data = pd.DataFrame({
        'crime_density': grid_crime,
        'lighting_score': grid_lighting,
        'pop_density': grid_pop,
        'night_activity': grid_night
    })
    
    # Use pipeline to predict (includes preprocessing)
    grid_X = grid_data[features]
    unsafe_probs = pipeline.predict_proba(grid_X)[:, 1]
    
    # Build output with all features for traceability
    grid_df = pd.DataFrame({
        'latitude': grid_lats.ravel(),
        'longitude': grid_lons.ravel(),
        'crime_density': grid_crime,
        'lighting_score': grid_lighting,
        'pop_density': grid_pop,
        'night_activity': grid_night,
        'unsafe_prob': unsafe_probs
    })
    grid_df['risk_level'] = pd.cut(unsafe_probs, bins=[0, 0.3, 0.7, 1], 
                                   labels=['Safe 🟢', 'Moderate 🟡', 'Unsafe 🔴'])
    return grid_df

def save_safety_map(df, grid_df, output_path='safety_pipeline_map.html'):
    """Save interactive Folium map with heatmap and training data."""
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                   zoom_start=12, tiles='OpenStreetMap')
    
    # Function to determine color based on unsafe probability
    def get_color(prob):
        if prob < 0.33:
            return 'green'
        elif prob < 0.67:
            return 'orange'
        else:
            return 'red'
    
    # Add grid heatmap as colored circles
    for idx, row in grid_df.iterrows():
        color = get_color(row['unsafe_prob'])
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"Risk: {row['unsafe_prob']:.1%}<br>Crime: {row['crime_density']:.1f}<br>Lighting: {row['lighting_score']:.1f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.5,
            weight=0.5
        ).add_to(m)
    
    # Original training data points (larger, darker colors)
    for idx, row in df.iterrows():
        if row['unsafe'] == 0:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                popup='Training: Safe Area',
                color='darkgreen',
                fill=True,
                fillColor='lightgreen',
                fillOpacity=0.8,
                weight=1
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                popup='Training: Unsafe Area',
                color='darkred',
                fill=True,
                fillColor='lightcoral',
                fillOpacity=0.8,
                weight=1
            ).add_to(m)
    
    m.save(output_path)
    print(f"✅ Map saved: {output_path}")

# =====================================
# MAIN PIPELINE EXECUTION
# =====================================
if __name__ == "__main__":
    # Step 1: Load clean data (generate synthetic if no file)
    try:
        df = load_clean_data('clean_safety_data.csv')
    except FileNotFoundError:
        print("No CSV found. Generating synthetic clean data...")
        np.random.seed(42)
        n_samples = 1000
        lat_center, lon_center = 40.7128, -74.0060
        lats = np.random.normal(lat_center, 0.05, n_samples)
        lons = np.random.normal(lon_center, 0.05, n_samples)
        crime_density = np.random.exponential(5, n_samples)
        lighting_score = np.random.uniform(0, 10, n_samples)
        pop_density = np.random.gamma(2, 50, n_samples)
        night_activity = np.random.uniform(0, 8, n_samples)
        unsafe_prob = 1 / (1 + np.exp(-(0.3*crime_density - 0.5*lighting_score + 0.01*pop_density + 0.2*night_activity - 5)))
        df = pd.DataFrame({
            'latitude': lats, 'longitude': lons,
            'crime_density': crime_density, 'lighting_score': lighting_score,
            'pop_density': pop_density, 'night_activity': night_activity,
            'unsafe': np.random.binomial(1, unsafe_prob, n_samples)
        })
        df.to_csv('clean_safety_data.csv', index=False)
        print("Synthetic data saved to 'clean_safety_data.csv'")

    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    X = df[features]
    y = df['unsafe']

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Build & train pipeline
    pipeline = create_ml_pipeline(features)
    pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    # Step 4: Grid prediction & map
    grid_df = predict_grid(pipeline, X_train)
    save_safety_map(df, grid_df)

    # Step 5: Save model
    joblib.dump(pipeline, 'safety_pipeline_model.joblib')
    print("✅ Pipeline model saved: 'safety_pipeline_model.joblib'")
    print("\nTo predict new data:\n```python\nmodel = joblib.load('safety_pipeline_model.joblib')\nnew_pred = model.predict(new_X)\n```")