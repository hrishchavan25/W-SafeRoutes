import pandas as pd
import numpy as np
import folium
import folium.plugins
import joblib
import requests
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
import time
from fastapi import FastAPI
from pydantic import BaseModel
warnings.filterwarnings('ignore')
app = FastAPI()

PIPELINE = None
GRID_DF = None

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
            ('num', StandardScaler(), features)
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

def save_safety_map(df, grid_df, output_path='safe1.html'):
    """Save interactive map with gradient zones for safe/unsafe areas."""
    lat_center = (df['latitude'].mean() + df['latitude'].mean()) / 2
    lon_center = (df['longitude'].mean() + df['longitude'].mean()) / 2
    m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles='OpenStreetMap')
    
    # Create gradient zones from grid data
    grid_size = int(np.sqrt(len(grid_df)))
    
    # Add circular markers for each grid point with color gradient
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
    
    m.save(output_path)
    print(f"✅ Interactive safety map: {output_path}")

# ==================== ROUTING WITH TRAFFIC DATA ====================
def get_traffic_data_tomtom(start_lat, start_lon, end_lat, end_lon, api_key=None):
    """Get route with traffic data using TomTom API (free tier available)."""
    if not api_key:
        print("⚠️ TomTom API key not provided. Using OSRM without traffic.")
        return get_route_from_osrm(start_lat, start_lon, end_lat, end_lon)
    
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
    params = {
        'key': api_key,
        'traffic': 'true',
        'travelMode': 'car',
        'routeType': 'fastest'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]
            # Extract coordinates
            legs = route['legs'][0]
            points = legs['points']
            route_points = [[p['latitude'], p['longitude']] for p in points]
            
            distance = route['summary']['lengthInMeters']
            duration = route['summary']['travelTimeInSeconds']
            traffic_delay = route['summary'].get('trafficDelayInSeconds', 0)
            
            print(f"🚦 Traffic delay: {traffic_delay}s ({traffic_delay/60:.1f} min)")
            return route_points, distance, duration, traffic_delay
        else:
            print(f"TomTom Error: {data.get('error', 'Unknown error')}")
            return None, None, None, None
    except Exception as e:
        print(f"Error fetching traffic data: {e}")
        return None, None, None, None

def get_route_from_osrm(start_lat, start_lon, end_lat, end_lon):
    """Get route using Open Source Routing Machine (OSRM) API - No traffic."""
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&annotations=true"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data['code'] == 'Ok':
            route = data['routes'][0]
            coordinates = route['geometry']['coordinates']  # [lon, lat] pairs
            distance = route['distance']  # meters
            duration = route['duration']  # seconds
            
            # Convert to [lat, lon] for our use
            route_points = [[coord[1], coord[0]] for coord in coordinates]
            
            # Simulate traffic based on time of day (since OSRM doesn't provide it)
            current_hour = time.localtime().tm_hour
            if 8 <= current_hour <= 10 or 17 <= current_hour <= 20:  # Rush hours
                traffic_delay = duration * 0.3  # 30% delay
            elif 11 <= current_hour <= 16:  # Moderate traffic
                traffic_delay = duration * 0.15  # 15% delay
            else:  # Light traffic
                traffic_delay = duration * 0.05  # 5% delay
            
            print(f"🚦 Estimated traffic delay: {traffic_delay}s ({traffic_delay/60:.1f} min)")
            return route_points, distance, duration, traffic_delay
        else:
            print(f"OSRM Error: {data.get('message', 'Unknown error')}")
            return None, None, None, None
    except Exception as e:
        print(f"Error fetching route: {e}")
        return None, None, None, None

def get_route_with_traffic(start_lat, start_lon, end_lat, end_lon, use_tomtom=False, tomtom_key=None):
    """Wrapper to get route with traffic - uses TomTom if key provided, else OSRM."""
    if use_tomtom and tomtom_key:
        return get_traffic_data_tomtom(start_lat, start_lon, end_lat, end_lon, tomtom_key)
    else:
        return get_route_from_osrm(start_lat, start_lon, end_lat, end_lon)

def get_safety_score_for_route(route_points, grid_df, traffic_delay=0):
    """Calculate safety score for a given route including traffic penalty."""
    if not route_points:
        return float('inf')
    
    total_risk = 0
    for point in route_points:
        lat, lon = point
        # Find closest grid point
        distances = np.sqrt((grid_df['latitude'] - lat)**2 + (grid_df['longitude'] - lon)**2)
        closest_idx = distances.idxmin()
        risk = grid_df.loc[closest_idx, 'unsafe_prob']
        total_risk += risk
    
    avg_risk = total_risk / len(route_points)
    
    # Add traffic penalty (more time in traffic = more exposure to risk)
    traffic_penalty = (traffic_delay / 60) * 0.1  # Traffic adds to overall risk
    
    return avg_risk + traffic_penalty

# ==================== RL AGENT FOR ROUTE OPTIMIZATION ====================
class LiveRouteSafetyAgent:
    """RL Agent that learns to optimize routes based on safety and speed."""
    
    def __init__(self, grid_df, learning_rate=0.1, discount=0.9, epsilon=0.2):
        self.grid_df = grid_df
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = {}  # State-action values
        
    def get_state_key(self, lat, lon):
        """Convert lat/lon to discrete state."""
        return (round(lat, 4), round(lon, 4))
    
    def get_nearby_waypoints(self, current_lat, current_lon, destination_lat, destination_lon):
        """Generate possible waypoints to explore."""
        # Generate waypoints in direction of destination
        waypoints = []
        steps = 5
        
        for i in range(1, steps + 1):
            ratio = i / steps
            # Direct path points
            lat = current_lat + ratio * (destination_lat - current_lat)
            lon = current_lon + ratio * (destination_lon - current_lon)
            waypoints.append((lat, lon))
            
            # Add variations (explore alternative paths)
            offset = 0.005 * i
            waypoints.append((lat + offset, lon))
            waypoints.append((lat - offset, lon))
            waypoints.append((lat, lon + offset))
            waypoints.append((lat, lon - offset))
        
        return waypoints
    
    def choose_waypoint(self, current_lat, current_lon, destination_lat, destination_lon):
        """Choose next waypoint using epsilon-greedy."""
        state_key = self.get_state_key(current_lat, current_lon)
        waypoints = self.get_nearby_waypoints(current_lat, current_lon, destination_lat, destination_lon)
        
        if np.random.rand() < self.epsilon or state_key not in self.q_table:
            # Explore: random waypoint
            return waypoints[np.random.randint(len(waypoints))]
        else:
            # Exploit: best known waypoint
            q_values = [self.q_table[state_key].get(self.get_state_key(wp[0], wp[1]), 0) 
                       for wp in waypoints]
            best_idx = np.argmax(q_values)
            return waypoints[best_idx]
    
    def learn(self, current_state, next_state, reward):
        """Update Q-values."""
        curr_key = self.get_state_key(*current_state)
        next_key = self.get_state_key(*next_state)
        
        if curr_key not in self.q_table:
            self.q_table[curr_key] = {}
        
        current_q = self.q_table[curr_key].get(next_key, 0)
        next_max_q = max(self.q_table.get(next_key, {}).values(), default=0)
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[curr_key][next_key] = new_q
    
    def find_safest_fastest_route(self, start_lat, start_lon, end_lat, end_lon, episodes=50, use_traffic=True, tomtom_key=None):
        """Train RL agent to find optimal route considering traffic."""
        print(f"\n🤖 Training RL Agent for route optimization (with traffic data)...")
        
        best_route = None
        best_score = float('inf')
        
        for episode in range(episodes):
            current_lat, current_lon = start_lat, start_lon
            route = [(current_lat, current_lon)]
            
            for step in range(20):  # Max steps per episode
                # Choose next waypoint
                next_lat, next_lon = self.choose_waypoint(current_lat, current_lon, end_lat, end_lon)
                
                # Get route segment with traffic
                segment_route, distance, duration, traffic_delay = get_route_with_traffic(
                    current_lat, current_lon, next_lat, next_lon, use_traffic, tomtom_key
                )
                
                if segment_route:
                    # Calculate reward (negative cost) including traffic
                    safety_score = get_safety_score_for_route(segment_route, self.grid_df, traffic_delay)
                    time_cost = (duration + traffic_delay) / 60  # Total time in minutes
                    reward = -(safety_score * 10 + time_cost * 0.5)  # Balance safety & speed
                    
                    # Learn
                    self.learn((current_lat, current_lon), (next_lat, next_lon), reward)
                    
                    # Update position
                    current_lat, current_lon = next_lat, next_lon
                    route.append((current_lat, current_lon))
                    
                    # Check if reached destination
                    if abs(current_lat - end_lat) < 0.001 and abs(current_lon - end_lon) < 0.001:
                        break
            
            # Evaluate full route
            full_route, total_dist, total_time, total_traffic = get_route_with_traffic(
                start_lat, start_lon, end_lat, end_lon, use_traffic, tomtom_key
            )
            if full_route:
                score = get_safety_score_for_route(full_route, self.grid_df, total_traffic) * 10 + ((total_time + total_traffic) / 60)
                if score < best_score:
                    best_score = score
                    best_route = full_route
                    print(f"Episode {episode}: New best route! Score: {score:.2f}, Traffic: {total_traffic/60:.1f}min")
        
        return best_route

# ==================== LIVE LOCATION TRACKING ====================
def simulate_live_tracking(route_points, update_interval=2):
    """Simulate live location tracking along route."""
    print("\n📍 Starting live location tracking simulation...")
    for i, point in enumerate(route_points[::5]):  # Sample every 5th point
        lat, lon = point
        print(f"Current Location: Lat {lat:.6f}, Lon {lon:.6f} ({i*5}/{len(route_points)} points)")
        time.sleep(update_interval)
    print("✅ Destination reached!")

def save_live_route_map(df, grid_df, route_points, start, end, output_path='live_safety_route.html'):
    """Save interactive map with live route."""
    lat_center = (start[0] + end[0]) / 2
    lon_center = (start[1] + end[1]) / 2
    m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles='OpenStreetMap')
    
    # Add heatmap
    heat_data = [[row['latitude'], row['longitude'], row['unsafe_prob']] 
                 for idx, row in grid_df.iterrows()]
    folium.plugins.HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=25,
        blur=15,
        gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}
    ).add_to(m)
    
    # Add route
    if route_points:
        folium.PolyLine(
            route_points,
            color='blue',
            weight=5,
            opacity=0.8,
            popup='Safest & Fastest Route'
        ).add_to(m)
    
    # Start and End markers
    folium.Marker(start, popup='Start', icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(end, popup='Destination', icon=folium.Icon(color='red', icon='stop')).add_to(m)
    
    # Live location marker (simulated at start)
    folium.Marker(
        start,
        popup='Your Location',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    m.save(output_path)
    print(f"✅ Live route map saved: {output_path}")

def load_model_and_grid():
    global PIPELINE, GRID_DF
    PIPELINE = joblib.load("andheri_west_safety_pipeline.joblib")
    GRID_DF = predict_grid(PIPELINE)

class RouteInput(BaseModel):
    route_points: list  # [[lat, lon], ...]

@app.post("/route-risk")
def route_risk(data: RouteInput):
    if GRID_DF is None:
        load_model_and_grid()

    total_risk = 0

    for lat, lon in data.route_points:
        distances = np.sqrt(
            (GRID_DF['latitude'] - lat) ** 2 +
            (GRID_DF['longitude'] - lon) ** 2
        )
        idx = distances.idxmin()
        total_risk += GRID_DF.loc[idx, 'unsafe_prob']

    avg_risk = total_risk / len(data.route_points)

    if avg_risk < 0.3:
        level = "GREEN"
    elif avg_risk < 0.7:
        level = "YELLOW"
    else:
        level = "RED"

    return {
        "risk_score": round(float(avg_risk), 3),
        "risk_level": level
    }

# ===================================== MAIN EXECUTION =====================================
# =====================================
# MAIN PIPELINE EXECUTION
# =====================================
if __name__ == "__main__":
    # Step 1: Load or generate data
    csv_path = 'clean_safety_data_mumbai.csv'
    try:
        df = load_clean_data(csv_path)
    except FileNotFoundError:
        print("Generating synthetic data for Andheri West...")
        np.random.seed(42)
        n_samples = 1000
        lat_center, lon_center = 19.1240, 72.8254
        lats = np.random.normal(lat_center, 0.02, n_samples)
        lons = np.random.normal(lon_center, 0.02, n_samples)
        
        crime_density = np.random.exponential(4, n_samples) + np.random.rand(n_samples) * 8
        lighting_score = np.random.uniform(3, 9, n_samples)
        pop_density = np.random.gamma(3, 80, n_samples)
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

    # Step 2: Train ML model
    features = ['crime_density', 'lighting_score', 'pop_density', 'night_activity']
    X = df[features]
    y = df['unsafe']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = create_ml_pipeline(features)
    pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    
    # Step 3: Create safety grid
    grid_df = predict_grid(pipeline)
    
    # Step 4: LIVE ROUTE FINDING with RL
    print("\n" + "="*60)
    print("LIVE ROUTE SAFETY SYSTEM")
    print("="*60)
    
    # Example: Andheri Station to JVPD
    start_location = (19.1200, 72.8470)  # Andheri Station
    end_location = (19.1300, 72.8320)    # JVPD area
    
    print(f"\n📍 Start: Andheri Station ({start_location[0]:.4f}, {start_location[1]:.4f})")
    print(f"📍 Destination: JVPD ({end_location[0]:.4f}, {end_location[1]:.4f})")
    
    # Get baseline route with traffic
    print("\n🛣️ Fetching baseline route with traffic data...")
    baseline_route, baseline_dist, baseline_time, baseline_traffic = get_route_with_traffic(
        start_location[0], start_location[1],
        end_location[0], end_location[1],
        use_tomtom=False,  # Set to True and add key if you have TomTom API
        tomtom_key=None    # Add your TomTom API key here
    )
    
    if baseline_route:
        baseline_safety = get_safety_score_for_route(baseline_route, grid_df, baseline_traffic)
        total_time = (baseline_time + baseline_traffic) / 60
        print(f"Baseline route: {baseline_dist:.0f}m, {total_time:.1f}min (Traffic: {baseline_traffic/60:.1f}min), Safety: {baseline_safety:.3f}")
    
    # Train RL agent for optimal route (with traffic consideration)
    rl_agent = LiveRouteSafetyAgent(grid_df)
    optimal_route = rl_agent.find_safest_fastest_route(
        start_location[0], start_location[1],
        end_location[0], end_location[1],
        episodes=30,
        use_traffic=True,
        tomtom_key=None  # Add TomTom API key for real traffic data
    )
    
    # Use baseline if RL didn't improve
    final_route = optimal_route if optimal_route else baseline_route
    
    # Save map
    save_live_route_map(df, grid_df, final_route, start_location, end_location)
    
    # Simulate live tracking
    if final_route:
        simulate_live_tracking(final_route, update_interval=1)
    
    # Save model
    joblib.dump(pipeline, 'andheri_west_safety_pipeline.joblib')
    print("\n✅ All done! Model and route saved!")
    load_model_and_grid()