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

# Choropleth overlay for unsafe probability (red=unsafe, green=safe)
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=[Point(xy) for xy in zip(grid_df.longitude, grid_df.latitude)])

# Bin probabilities for coloring
grid_gdf['risk_level'] = pd.cut(grid_df['unsafe_prob'], bins=[0, 0.3, 0.7, 1], labels=['Safe 🟢', 'Moderate 🟡', 'Unsafe 🔴'])

folium.Choropleth(
    geo_data=grid_gdf.set_index('risk_level').geometry.unary_union,  # Simplified for demo
    data=grid_df,
    columns=['latitude', 'unsafe_prob'],  # Use lat as fake 'region' for scatter-like effect
    key_on='feature.geometry',  # Note: Simplified; use plugins for true heatmap
    fill_color='RdYlGn_r',  # Red-Yellow-Green reverse (high prob = red)
    fill_opacity=0.6,
    line_opacity=0.1,
    legend_name='Unsafe Probability',
    nan_fill_color='white',
    bins=5
).add_to(m)

# Add original data points
safe_points = folium.features.CircleMarker(
    location=df[df['unsafe']==0][['latitude', 'longitude']].values,
    radius=3, popup='Safe Area', color='green', fill=True, fillColor='green', fillOpacity=0.7
)
unsafe_points = folium.features.CircleMarker(
    location=df[df['unsafe']==1][['latitude', 'longitude']].values,
    radius=4, popup='Unsafe Area', color='red', fill=True, fillColor='red', fillOpacity=0.7
)

for point in safe_points:
    point.add_to(m)
for point in unsafe_points:
    point.add_to(m)

m.save('safety_map.html')
print("\n✅ Map saved as 'safety_map.html' – Open in browser for interactive view!")
print("Safe areas (🟢): Low crime + good lighting + low density.")
print("Unsafe areas (🔴): High crime + poor lighting + high activity.")

# Optional: Predict on new data example
new_area = scaler.transform([[10, 2, 200, 7]])  # High crime, poor lighting
pred_prob = model.predict_proba(new_area)[0, 1]
print(f"\nExample prediction for [crime=10, lighting=2, pop=200, night=7]: {pred_prob:.2%} unsafe")