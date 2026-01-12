import pandas as pd
import numpy as np
import folium
from folium import Choropleth
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Step 1: Generate synthetic dataset (1000 samples)
np.random.seed(42)
n_samples = 1000
lat_center, lon_center = 40.7128, -74.0060  # NYC-like area
lats = np.random.normal(lat_center, 0.05, n_samples)
lons = np.random.normal(lon_center, 0.05, n_samples)

# Simulate correlated features for realism
crime_density = np.random.exponential(5, n_samples) + (np.random.rand(n_samples) * 10)
lighting_score = np.random.uniform(0, 10, n_samples)
pop_density = np.random.gamma(2, 50, n_samples)
night_activity = np.random.uniform(0, 8, n_samples)

# Unsafe areas: higher crime, low lighting, high pop + activity
unsafe_prob = 1 / (1 + np.exp(-(0.3*crime_density - 0.5*lighting_score + 0.01*pop_density + 0.2*night_activity - 5)))
labels = np.random.binomial(1, unsafe_prob, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'latitude': lats,
    'longitude': lons,
    'crime_density': crime_density,
    'lighting_score': lighting_score,
    'pop_density': pop_density,
    'night_activity': night_activity,
    'unsafe': labels
})

print("Dataset preview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Unsafe areas: {df['unsafe'].sum()} ({df['unsafe'].mean()*100:.1f}%)")

# Step 2: Prepare features and split data
features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
X = df[features]
y = df['unsafe']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(importances)

# Step 4: Predict safety on a fine lat/lon grid for heatmap (5km x 5km area)
lat_grid = np.linspace(lat_center - 0.03, lat_center + 0.03, 50)
lon_grid = np.linspace(lon_center - 0.03, lon_center + 0.03, 50)
grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
grid_points = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])

# Simulate features on grid (correlated with position for demo)
grid_crime = np.random.exponential(5, len(grid_points)) + 2 * (np.abs(grid_lats.ravel() - lat_center) + np.abs(grid_lons.ravel() - lon_center)) * 100
grid_lighting = 8 - np.random.uniform(0, 4, len(grid_points))
grid_pop = np.random.gamma(2, 40, len(grid_points))
grid_night = np.random.uniform(1, 6, len(grid_points))

grid_df = pd.DataFrame({
    'crime_density': grid_crime,
    'lighting_score': grid_lighting,
    'pop_density': grid_pop,
    'night_activity': grid_night
})
grid_X = scaler.transform(grid_df[features])
grid_predictions = model.predict_proba(grid_X)[:, 1]  # Probability of unsafe
grid_df['unsafe_prob'] = grid_predictions
grid_df['latitude'] = grid_lats.ravel()
grid_df['longitude'] = grid_lons.ravel()

# Step 5: Create interactive Folium map with heatmap overlay
m = folium.Map(location=[lat_center, lon_center], zoom_start=12, tiles='OpenStreetMap')

# Function to determine color based on unsafe probability
def get_color(prob):
    if prob < 0.33:
        return 'green'
    elif prob < 0.67:
        return 'orange'
    else:
        return 'red'

# Add grid points as colored circles (heatmap effect)
for idx, row in grid_df.iterrows():
    color = get_color(row['unsafe_prob'])
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        popup=f"Risk: {row['unsafe_prob']:.1%}",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.6,
        weight=1
    ).add_to(m)

# Add original training data points with larger markers
for idx, row in df.iterrows():
    if row['unsafe'] == 0:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup='Safe Area (Training)',
            color='darkgreen',
            fill=True,
            fillColor='lightgreen',
            fillOpacity=0.5,
            weight=1
        ).add_to(m)
    else:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup='Unsafe Area (Training)',
            color='darkred',
            fill=True,
            fillColor='lightcoral',
            fillOpacity=0.5,
            weight=1
        ).add_to(m)

m.save('safety_map.html')
print("\n✅ Map saved as 'safety_map.html' – Open in browser for interactive view!")
print("Safe areas (🟢): Low crime + good lighting + low density.")
print("Unsafe areas (🔴): High crime + poor lighting + high activity.")

# Optional: Predict on new data example
new_area = scaler.transform([[10, 2, 200, 7]])  # High crime, poor lighting
pred_prob = model.predict_proba(new_area)[0, 1]
print(f"\nExample prediction for [crime=10, lighting=2, pop=200, night=7]: {pred_prob:.2%} unsafe")
