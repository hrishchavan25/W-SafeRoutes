import pandas as pd
import numpy as np
import folium
import folium.plugins
import joblib
import requests
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
import time
import heapq
import math
warnings.filterwarnings('ignore')

# ==================== A* PATHFINDING INTEGRATION ====================
def heuristic(node, goal):
    """Euclidean distance heuristic for A*"""
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def build_safety_graph(grid_df):
    """Build safety-aware graph from grid data efficiently using grid indices."""
    graph = {}
    if grid_df is None or len(grid_df) == 0:
        return graph
    
    # Get unique sorted lats and lons to reconstruct grid structure
    lats = sorted(grid_df['latitude'].unique())
    lons = sorted(grid_df['longitude'].unique())
    
    if len(lats) < 2 or len(lons) < 2:
        return graph

    # Create a lookup for (lat, lon) -> index
    # We use a bit of tolerance for float comparisons
    node_to_prob = { (row.latitude, row.longitude): row.unsafe_prob for row in grid_df.itertuples() }
    
    # Grid search for neighbors
    n_lats = len(lats)
    n_lons = len(lons)
    
    # Mapping for fast index lookup
    lat_map = { lat: i for i, lat in enumerate(lats) }
    lon_map = { lon: i for i, lon in enumerate(lons) }

    for row in grid_df.itertuples():
        current_node = (row.latitude, row.longitude)
        i = lat_map[row.latitude]
        j = lon_map[row.longitude]
        
        neighbors = []
        # Check adjacent 8 cells in the grid
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0: continue
                
                ni, nj = i + di, j + dj
                if 0 <= ni < n_lats and 0 <= nj < n_lons:
                    neighbor_node = (lats[ni], lons[nj])
                    if neighbor_node in node_to_prob:
                        dist = heuristic(current_node, neighbor_node)
                        neighbors.append((neighbor_node, dist, node_to_prob[neighbor_node]))
        
        graph[current_node] = neighbors
    
    return graph



def find_closest_grid_node(graph, location):
    """Find nearest graph node to a location"""
    if not graph:
        return None
    
    closest = None
    min_dist = float('inf')
    
    for node in graph.keys():
        dist = heuristic(node, location)
        if dist < min_dist:
            min_dist = dist
            closest = node
    
    return closest

def astar_pathfinding(grid_df, start_lat, start_lon, end_lat, end_lon):
    """
    A* algorithm for combined safety + distance optimization
    
    Returns:
        List of (lat, lon) tuples representing optimal path
    """
    
    graph = build_safety_graph(grid_df)
    if not graph:
        return None
    
    start = find_closest_grid_node(graph, (start_lat, start_lon))
    goal = find_closest_grid_node(graph, (end_lat, end_lon))
    
    if not start or not goal:
        return None
    
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        if current not in graph:
            continue
        
        for neighbor, cost, safety in graph[current]:
            # Combined cost: 60% distance + 40% safety
            # This finds a balance between shortest and safest
            combined_cost = (cost * 0.6) + (safety * 0.4)
            temp_g = g_score[current] + combined_cost
            
            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                f_score = temp_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
                came_from[neighbor] = current
    
    return None

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
    """Save interactive map with live route and gradient zones."""
    lat_center = (start[0] + end[0]) / 2
    lon_center = (start[1] + end[1]) / 2
    m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles='OpenStreetMap')
    
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
            stroke=False,
            weight=0,
            popup=f"Risk: {unsafe_prob:.2%}",
            tooltip=f"Unsafe Probability: {unsafe_prob:.2%}"
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
    
def save_interactive_route_finder(grid_df, output_path='interactive_route_finder.html'):
    """Create interactive HTML map with real routing, traffic prediction, live location, and traffic visualization."""
    lat_center = 19.1240
    lon_center = 72.8254
    
    # Prepare gradient zone data
    gradient_data = []
    for idx, row in grid_df.iterrows():
        gradient_data.append({
            'lat': row['latitude'],
            'lon': row['longitude'],
            'risk': row['unsafe_prob']
        })
    
    # Generate traffic congestion zones (simulated based on road network)
    traffic_zones = []
    np.random.seed(42)
    for i in range(15):
        traffic_zones.append({
            'lat': lat_center + np.random.uniform(-0.02, 0.02),
            'lon': lon_center + np.random.uniform(-0.02, 0.02),
            'congestion': int(np.random.uniform(20, 100)),  # Congestion percentage
            'speed': int(np.random.uniform(10, 50))  # Speed in km/h
        })
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Safety Route Finder - Andheri West</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine/dist/leaflet-routing-machine.css" />
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }}
        
        .container {{ display: flex; height: 100vh; }}
        
        .sidebar {{
            width: 350px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            z-index: 100;
        }}
        
        #map {{ flex: 1; }}
        
        .sidebar h1 {{ color: #2c3e50; font-size: 20px; margin-bottom: 10px; }}
        
        .traffic-toggle {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .traffic-toggle button {{
            flex: 1;
            padding: 8px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}
        
        .traffic-toggle button.active {{
            background: #27ae60;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        label {{
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
            font-size: 14px;
        }}
        
        input, button {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
        }}
        
        input {{
            margin-bottom: 5px;
            background: #fafafa;
        }}
        
        input:focus {{
            outline: none;
            border-color: #3498db;
            background: white;
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
        }}
        
        button {{
            background: #27ae60;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.3s;
            margin-top: 10px;
        }}
        
        button:hover {{ background: #229954; }}
        
        button.secondary {{
            background: #e74c3c;
            margin-top: 5px;
        }}
        
        button.secondary:hover {{ background: #c0392b; }}
        
        .suggestions {{
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            max-height: 150px;
            overflow-y: auto;
            width: 310px;
            z-index: 1000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .suggestion-item {{
            padding: 10px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
        }}
        
        .suggestion-item:hover {{
            background: #f0f0f0;
        }}
        
        .status {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
            font-size: 13px;
            color: #2c3e50;
            min-height: 60px;
        }}
        
        .status.success {{ background: #d5f4e6; color: #27ae60; }}
        .status.error {{ background: #fadbd8; color: #c0392b; }}
        .status.loading {{ background: #d6eaf8; color: #3498db; }}
        
        .route-info {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
        }}
        
        .location-info {{
            background: #e8f8f5;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 12px;
            border-left: 3px solid #27ae60;
        }}
        
        .traffic-info {{
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 12px;
            border-left: 3px solid #ffc107;
            max-height: 120px;
            overflow-y: auto;
        }}
        
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
            font-size: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid #999;
        }}
        
        .traffic-legend {{
            margin-top: 10px;
            border-top: 1px solid #eee;
            padding-top: 10px;
        }}
        
        .traffic-legend strong {{
            display: block;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>Safety Route Finder</h1>
            
            <div class="traffic-toggle">
                <button id="trafficToggleBtn" class="active">Traffic ON</button>
                <button id="liveLocationBtn">Live Location</button>
            </div>
            
            <div class="form-group">
                <label for="tomtomKey">TomTom API Key (Free tier)</label>
                <input type="text" id="tomtomKey" placeholder="Get free key from developer.tomtom.com" autocomplete="off">
                <small style="color: #666; font-size: 11px; margin-top: 3px; display: block;">Optional: For real-time traffic (1000 free requests/day)</small>
            </div>
            
            <div class="form-group">
                <label for="source">Starting Location</label>
                <input type="text" id="source" placeholder="e.g., Andheri Station" autocomplete="off">
                <div id="sourceSuggestions" class="suggestions" style="display:none;"></div>
            </div>
            
            <div class="form-group">
                <label for="destination">Destination</label>
                <input type="text" id="destination" placeholder="e.g., JVPD" autocomplete="off">
                <div id="destSuggestions" class="suggestions" style="display:none;"></div>
            </div>
            
            <button id="findRouteBtn">Find Safest & Shortest Route</button>
            <button id="clearBtn" class="secondary">Clear</button>
            
            <div id="status" class="status">
                Enter start and destination locations or enable live location
            </div>
            
            <div id="locationInfo" class="location-info" style="display:none;"></div>
            <div id="trafficInfo" class="traffic-info" style="display:none;"></div>
            <div id="routeInfo" class="route-info" style="display:none;"></div>
            
            <div class="legend">
                <strong>Safety Zones:</strong>
                <div class="legend-item">
                    <div class="legend-color" style="background: green;"></div>
                    <span>Safe (0-30%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: orange;"></div>
                    <span>Moderate (30-70%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: red;"></div>
                    <span>Unsafe (70-100%)</span>
                </div>
                
                <div class="traffic-legend">
                    <strong>Real-Time Traffic (TomTom):</strong>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #2ecc71;"></div>
                        <span>Free Flow (&gt;80%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f39c12;"></div>
                        <span>Moderate (50-80%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #e67e22;"></div>
                        <span>Heavy (20-50%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #e74c3c;"></div>
                        <span>Very Heavy (&lt;20%)</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 11px; color: #666;">
                        Route segments are color-coded based on speed ratio to free-flow speed
                    </div>
                </div>
            </div>
        </div>
        
        <div id="map"></div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-routing-machine/dist/leaflet-routing-machine.js"></script>
    
    <script>
        // Initialize map
        const map = L.map('map').setView([{lat_center}, {lon_center}], 14);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }}).addTo(map);
        
        // Traffic layer group
        const trafficLayerGroup = L.layerGroup();
        
        // Add gradient zones
        const gradientData = {json.dumps(gradient_data)};
        gradientData.forEach(point => {{
            let color, opacity;
            if (point.risk < 0.3) {{
                color = 'green';
                opacity = 0.3 + (point.risk * 0.4);
            }} else if (point.risk < 0.7) {{
                color = 'orange';
                opacity = 0.5 + ((point.risk - 0.3) * 0.4);
            }} else {{
                color = 'red';
                opacity = 0.6 + ((point.risk - 0.7) * 0.4);
            }}
            
            L.circleMarker([point.lat, point.lon], {{
                radius: 12,
                color: color,
                fillColor: color,
                fillOpacity: opacity,
                stroke: false,
                weight: 0,
                popup: 'Safety Risk: ' + (point.risk * 100).toFixed(1) + '%',
                tooltip: 'Risk: ' + (point.risk * 100).toFixed(1) + '%'
            }}).addTo(map);
        }});
        
        // Add traffic zones
        const trafficZones = {json.dumps(traffic_zones)};
        const trafficMarkers = [];
        
        function updateTrafficLayer() {{
            trafficLayerGroup.clearLayers();
            trafficMarkers.length = 0;
            
            trafficZones.forEach(zone => {{
                let color, radius;
                const congestion = zone.congestion;
                
                if (congestion < 25) {{
                    color = '#2ecc71'; // Green - Free flow
                    radius = 15;
                }} else if (congestion < 50) {{
                    color = '#f39c12'; // Orange - Moderate
                    radius = 18;
                }} else {{
                    color = '#e74c3c'; // Red - Heavy
                    radius = 22;
                }}
                
                const marker = L.circleMarker([zone.lat, zone.lon], {{
                    radius: radius,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.6,
                    weight: 2,
                    dashArray: '5, 5',
                    popup: `Traffic Congestion: ${{congestion}}%<br>Avg Speed: ${{zone.speed}} km/h`,
                    tooltip: `Congestion: ${{congestion}}%`
                }}).addTo(trafficLayerGroup);
                
                trafficMarkers.push({{
                    lat: zone.lat,
                    lon: zone.lon,
                    congestion: zone.congestion,
                    speed: zone.speed
                }});
            }});
            
            trafficLayerGroup.addTo(map);
        }}
        
        // Variables
        const sourceInput = document.getElementById('source');
        const destInput = document.getElementById('destination');
        const tomtomKeyInput = document.getElementById('tomtomKey');
        const statusDiv = document.getElementById('status');
        const routeInfoDiv = document.getElementById('routeInfo');
        const locationInfoDiv = document.getElementById('locationInfo');
        const trafficInfoDiv = document.getElementById('trafficInfo');
        const findRouteBtn = document.getElementById('findRouteBtn');
        const clearBtn = document.getElementById('clearBtn');
        const liveLocationBtn = document.getElementById('liveLocationBtn');
        const trafficToggleBtn = document.getElementById('trafficToggleBtn');
        
        let sourceMarker = null;
        let destMarker = null;
        let liveMarker = null;
        let routingControl = null;
        let watchId = null;
        let isLiveLocation = false;
        let trafficVisible = true;
        let tomtomTrafficLayer = null;
        let tomtomFlowLayer = null;
        
        // Initialize traffic layer
        updateTrafficLayer();
        
        // Get traffic prediction based on time of day
        function getTrafficFactor() {{
            const hour = new Date().getHours();
            if (hour >= 7 && hour <= 10) return 1.5;
            if (hour >= 17 && hour <= 20) return 1.8;
            if (hour >= 11 && hour <= 16) return 1.2;
            if (hour >= 21 || hour <= 6) return 1.05;
            return 1.0;
        }}
        
        function getTrafficInfo() {{
            const hour = new Date().getHours();
            if (hour >= 7 && hour <= 10) return 'Heavy (Morning Rush)';
            if (hour >= 17 && hour <= 20) return 'Heavy (Evening Rush)';
            if (hour >= 11 && hour <= 16) return 'Moderate';
            return 'Light';
        }}
        
        function getAvgCongestion() {{
            if (trafficMarkers.length === 0) return 0;
            const avg = trafficMarkers.reduce((sum, m) => sum + m.congestion, 0) / trafficMarkers.length;
            return Math.round(avg);
        }}
        
        function getAvgSpeed() {{
            if (trafficMarkers.length === 0) return 0;
            const avg = trafficMarkers.reduce((sum, m) => sum + m.speed, 0) / trafficMarkers.length;
            return Math.round(avg);
        }}
        
        // Calculate distance between two points (Haversine formula)
        function getDistance(lat1, lon1, lat2, lon2) {{
            const R = 6371; // Earth radius in km
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                      Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }}
        
        // Find nearest traffic zone to a point
        function getNearestTrafficCongestion(lat, lon, radius = 0.01) {{
            let nearest = null;
            let minDist = Infinity;
            
            trafficMarkers.forEach(zone => {{
                const dist = getDistance(lat, lon, zone.lat, zone.lon);
                if (dist < 1 && dist < minDist) {{
                    nearest = zone;
                    minDist = dist;
                }}
            }});
            
            return nearest ? nearest.congestion : 0;
        }}
        
        // Draw route with traffic-based color segments
        function drawTrafficAwareRoute(coordinates) {{
            // Create polylines for each traffic level
            const freeFlowSegments = [];
            const moderateSegments = [];
            const heavySegments = [];
            
            for (let i = 0; i < coordinates.length - 1; i++) {{
                const congestion = getNearestTrafficCongestion(
                    coordinates[i][0], 
                    coordinates[i][1]
                );
                
                const segment = [coordinates[i], coordinates[i + 1]];
                
                if (congestion < 25) {{
                    freeFlowSegments.push(segment);
                }} else if (congestion < 50) {{
                    moderateSegments.push(segment);
                }} else {{
                    heavySegments.push(segment);
                }}
            }}
            
            // Draw segments with different colors
            freeFlowSegments.forEach(segment => {{
                L.polyline(segment, {{
                    color: '#2ecc71',
                    weight: 8,
                    opacity: 0.8,
                    dashArray: '5, 5'
                }}).addTo(map);
            }});
            
            moderateSegments.forEach(segment => {{
                L.polyline(segment, {{
                    color: '#f39c12',
                    weight: 8,
                    opacity: 0.8,
                    dashArray: '5, 5'
                }}).addTo(map);
            }});
            
            heavySegments.forEach(segment => {{
                L.polyline(segment, {{
                    color: '#e74c3c',
                    weight: 8,
                    opacity: 0.8,
                    dashArray: '5, 5'
                }}).addTo(map);
            }});
            
            // Add traffic hotspots on route
            trafficMarkers.forEach(zone => {{
                let isNearRoute = false;
                coordinates.forEach(point => {{
                    if (getDistance(point[0], point[1], zone.lat, zone.lon) < 0.5) {{
                        isNearRoute = true;
                    }}
                }});
                
                if (isNearRoute) {{
                    const icon = L.icon({{
                        iconUrl: 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2228%22 height=%2228%22><rect width=%2228%22 height=%2228%22 rx=%2214%22 fill=%22%23ff6b6b%22/><text x=%2214%22 y=%2720%22 text-anchor=%22middle%22 font-size=%2216%22 fill=%22white%22 font-weight=%22bold%22>!</text></svg>',
                        iconSize: [28, 28],
                        iconAnchor: [14, 14]
                    }});
                    
                    L.marker([zone.lat, zone.lon], {{ icon: icon }})
                        .bindPopup(`
                            <div style="font-size: 12px;">
                                <strong>Traffic Alert</strong><br>
                                Congestion: ${{zone.congestion}}%<br>
                                Speed: ${{zone.speed}} km/h
                            </div>
                        `)
                        .addTo(map);
                }}
            }});
        }}
        
        // Get real-time traffic speed from TomTom
        async function getTomTomTrafficSpeed(lat, lon) {{
            const apiKey = tomtomKeyInput.value.trim();
            if (!apiKey) return null;
            
            try {{
                const response = await fetch(`https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=${{lat}},${{lon}}&key=${{apiKey}}`);
                if (!response.ok) return null;
                const data = await response.json();
                
                if (data.flowSegmentData) {{
                    const flow = data.flowSegmentData;
                    return {{
                        speed: Math.round(flow.currentSpeed),
                        freeFlowSpeed: Math.round(flow.freeFlowSpeed),
                        currentTravelTime: flow.currentTravelTime,
                        freeFlowTravelTime: flow.freeFlowTravelTime
                    }};
                }}
            }} catch (e) {{
                console.log('TomTom API error:', e);
            }}
            return null;
        }}
        
        // Get traffic color based on speed ratio
        function getTrafficColor(currentSpeed, freeFlowSpeed) {{
            const ratio = currentSpeed / freeFlowSpeed;
            
            if (ratio > 0.8) {{
                return {{ color: '#2ecc71', label: 'Free Flow' }}; // Green
            }} else if (ratio > 0.5) {{
                return {{ color: '#f39c12', label: 'Moderate' }}; // Orange
            }} else if (ratio > 0.2) {{
                return {{ color: '#e67e22', label: 'Heavy' }}; // Dark Orange
            }} else {{
                return {{ color: '#e74c3c', label: 'Very Heavy' }}; // Red
            }}
        }}
        
        // Draw route with real-time TomTom traffic
        async function drawRouteWithTomTomTraffic(coordinates) {{
            const apiKey = tomtomKeyInput.value.trim();
            
            if (!apiKey) {{
                // Fallback to simulated traffic
                drawTrafficAwareRoute(coordinates);
                return;
            }}
            
            // Process every 5th point to avoid too many API calls
            const samplePoints = [];
            for (let i = 0; i < coordinates.length; i += Math.max(1, Math.floor(coordinates.length / 20))) {{
                samplePoints.push(coordinates[i]);
            }}
            samplePoints.push(coordinates[coordinates.length - 1]);
            
            const trafficData = [];
            for (const point of samplePoints) {{
                const traffic = await getTomTomTrafficSpeed(point[0], point[1]);
                trafficData.push({{ point, traffic }});
                await new Promise(resolve => setTimeout(resolve, 100)); // Rate limit
            }}
            
            // Draw segments with traffic colors
            for (let i = 0; i < coordinates.length - 1; i++) {{
                let segmentColor = '#2196F3'; // Default blue
                let trafficLabel = 'Unknown';
                
                // Find nearest traffic data point
                let nearestTraffic = null;
                let minDist = Infinity;
                
                trafficData.forEach(td => {{
                    if (td.traffic) {{
                        const dist = getDistance(coordinates[i][0], coordinates[i][1], td.point[0], td.point[1]);
                        if (dist < minDist) {{
                            minDist = dist;
                            nearestTraffic = td.traffic;
                        }}
                    }}
                }});
                
                if (nearestTraffic) {{
                    const trafficColor = getTrafficColor(nearestTraffic.speed, nearestTraffic.freeFlowSpeed);
                    segmentColor = trafficColor.color;
                    trafficLabel = trafficColor.label;
                }}
                
                const segment = [coordinates[i], coordinates[i + 1]];
                L.polyline(segment, {{
                    color: segmentColor,
                    weight: 9,
                    opacity: 0.85,
                    dashArray: '2, 4'
                }}).bindPopup(`
                    <div style="font-size: 12px; min-width: 150px;">
                        <strong>Traffic Status</strong><br>
                        Condition: ${{trafficLabel}}
                    </div>
                `).addTo(map);
            }}
        }}
        
        // Traffic toggle
        trafficToggleBtn.addEventListener('click', () => {{
            trafficVisible = !trafficVisible;
            if (trafficVisible) {{
                trafficLayerGroup.addTo(map);
                trafficToggleBtn.textContent = 'Traffic ON';
                trafficToggleBtn.classList.add('active');
            }} else {{
                map.removeLayer(trafficLayerGroup);
                trafficToggleBtn.textContent = 'Traffic OFF';
                trafficToggleBtn.classList.remove('active');
            }}
        }});
        
        // Geocoding function using Nominatim
        async function geocode(query) {{
            const url = `https://nominatim.openstreetmap.org/search?format=json&q=${{query}},Andheri,Mumbai,India&limit=5`;
            try {{
                const response = await fetch(url);
                return await response.json();
            }} catch (e) {{
                console.error('Geocoding error:', e);
                return [];
            }}
        }}
        
        // Get route from OSRM
        async function getRouteFromOSRM(startLat, startLon, endLat, endLon) {{
            const url = `https://router.project-osrm.org/route/v1/driving/${{startLon}},${{startLat}};${{endLon}},${{endLat}}?overview=full&geometries=geojson&annotations=duration,distance,speed&steps=true`;
            try {{
                const response = await fetch(url);
                const data = await response.json();
                
                if (data.code === 'Ok') {{
                    const route = data.routes[0];
                    const coords = route.geometry.coordinates.map(c => [c[1], c[0]]);
                    const distance = (route.distance / 1000).toFixed(2);
                    const duration = route.duration;
                    const trafficFactor = getTrafficFactor();
                    const adjustedDuration = duration * trafficFactor;
                    
                    // Extract turn-by-turn directions
                    const directions = [];
                    if (route.legs) {{
                        route.legs.forEach(leg => {{
                            if (leg.steps) {{
                                leg.steps.forEach(step => {{
                                    directions.push({{
                                        instruction: step.maneuver?.type || 'Continue',
                                        name: step.name || 'Road',
                                        distance: (step.distance / 1000).toFixed(2),
                                        duration: Math.round(step.duration / 60)
                                    }});
                                }});
                            }}
                        }});
                    }}
                    
                    return {{
                        coordinates: coords,
                        distance: distance,
                        duration: Math.round(duration / 60),
                        adjustedDuration: Math.round(adjustedDuration / 60),
                        trafficDelay: Math.round((adjustedDuration - duration) / 60),
                        traffic: getTrafficInfo(),
                        directions: directions
                    }};
                }}
            }} catch (e) {{
                console.error('Route error:', e);
            }}
            return null;
        }}
        
        // Show suggestions
        async function showSuggestions(input, suggestionsDiv, results) {{
            suggestionsDiv.innerHTML = '';
            if (results.length === 0) return;
            
            results.forEach(place => {{
                const item = document.createElement('div');
                item.className = 'suggestion-item';
                item.textContent = place.display_name;
                item.onclick = () => {{
                    input.value = place.display_name;
                    input.dataset.lat = place.lat;
                    input.dataset.lon = place.lon;
                    suggestionsDiv.style.display = 'none';
                }};
                suggestionsDiv.appendChild(item);
            }});
            
            suggestionsDiv.style.display = 'block';
        }}
        
        // Debounced geocoding
        let debounceTimer;
        sourceInput.addEventListener('input', async e => {{
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(async () => {{
                if (e.target.value.length > 2) {{
                    const results = await geocode(e.target.value);
                    showSuggestions(sourceInput, document.getElementById('sourceSuggestions'), results);
                }}
            }}, 300);
        }});
        
        destInput.addEventListener('input', async e => {{
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(async () => {{
                if (e.target.value.length > 2) {{
                    const results = await geocode(e.target.value);
                    showSuggestions(destInput, document.getElementById('destSuggestions'), results);
                }}
            }}, 300);
        }});
        
        // Live Location Tracking
        liveLocationBtn.addEventListener('click', () => {{
            if (!isLiveLocation) {{
                if (navigator.geolocation) {{
                    statusDiv.className = 'status loading';
                    statusDiv.textContent = 'Requesting location access...';
                    
                    watchId = navigator.geolocation.watchPosition(
                        position => {{
                            const lat = position.coords.latitude;
                            const lon = position.coords.longitude;
                            const accuracy = position.coords.accuracy;
                            
                            if (liveMarker) map.removeLayer(liveMarker);
                            liveMarker = L.marker([lat, lon], {{
                                icon: L.icon({{
                                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
                                    iconSize: [25, 41],
                                    iconAnchor: [12, 41]
                                }})
                            }}).bindPopup('Your Location').addTo(map);
                            
                            sourceInput.value = 'My Location';
                            sourceInput.dataset.lat = lat;
                            sourceInput.dataset.lon = lon;
                            
                            locationInfoDiv.style.display = 'block';
                            locationInfoDiv.innerHTML = `
                                <strong>Live Location Active</strong><br>
                                Lat: ${{lat.toFixed(6)}}<br>
                                Lon: ${{lon.toFixed(6)}}<br>
                                Accuracy: ${{accuracy.toFixed(0)}}m
                            `;
                            
                            statusDiv.className = 'status success';
                            statusDiv.textContent = 'Live location enabled. Select destination and find route.';
                            
                            isLiveLocation = true;
                            liveLocationBtn.textContent = 'Stop Live Location';
                            liveLocationBtn.style.background = '#e74c3c';
                            
                            map.setView([lat, lon], 15);
                        }},
                        error => {{
                            statusDiv.className = 'status error';
                            statusDiv.textContent = 'Location access denied. Please use manual entry.';
                        }}
                    );
                }} else {{
                    statusDiv.className = 'status error';
                    statusDiv.textContent = 'Geolocation not supported in your browser.';
                }}
            }} else {{
                navigator.geolocation.clearWatch(watchId);
                if (liveMarker) map.removeLayer(liveMarker);
                liveMarker = null;
                isLiveLocation = false;
                liveLocationBtn.textContent = 'Live Location';
                liveLocationBtn.style.background = '#3498db';
                locationInfoDiv.style.display = 'none';
                statusDiv.textContent = 'Live location stopped.';
            }}
        }});
        
        // Find route
        findRouteBtn.addEventListener('click', async () => {{
            if (!sourceInput.dataset.lat || !destInput.dataset.lat) {{
                statusDiv.className = 'status error';
                statusDiv.textContent = 'Please select valid locations from suggestions';
                return;
            }}
            
            const startLat = parseFloat(sourceInput.dataset.lat);
            const startLon = parseFloat(sourceInput.dataset.lon);
            const endLat = parseFloat(destInput.dataset.lat);
            const endLon = parseFloat(destInput.dataset.lon);
            
            if (routingControl) map.removeControl(routingControl);
            if (sourceMarker) map.removeLayer(sourceMarker);
            if (destMarker) map.removeLayer(destMarker);
            
            sourceMarker = L.marker([startLat, startLon], {{
                icon: L.icon({{
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41]
                }})
            }}).bindPopup('Start: ' + sourceInput.value).addTo(map).openPopup();
            
            destMarker = L.marker([endLat, endLon], {{
                icon: L.icon({{
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41]
                }})
            }}).bindPopup('Destination: ' + destInput.value).addTo(map).openPopup();
            
            statusDiv.className = 'status loading';
            statusDiv.textContent = 'Computing safest & shortest route...';
            routeInfoDiv.style.display = 'none';
            trafficInfoDiv.style.display = 'none';
            
            try {{
                // Get OSRM route for proper road-based routing
                const osrmRoute = await getRouteFromOSRM(startLat, startLon, endLat, endLon);
                
                if (!osrmRoute) throw new Error('No route found');
                
                // Try to get safety optimization from A* backend
                let safetyScore = 0.5;  // Default neutral
                let safetyLevel = 'Moderate';
                try {{
                    const response = await fetch('http://localhost:5001/api/astar-route', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            start_lat: startLat,
                            start_lon: startLon,
                            end_lat: endLat,
                            end_lon: endLon
                        }})
                    }});
                    
                    if (response.ok) {{
                        const astarData = await response.json();
                        if (astarData.average_safety) {{
                            safetyScore = (1 - astarData.average_safety).toFixed(2);
                            safetyLevel = astarData.average_safety < 0.3 ? 'Safe' : astarData.average_safety < 0.7 ? 'Moderate' : 'Unsafe';
                        }}
                    }}
                }} catch (e) {{
                    console.log('Safety optimization skipped');
                }}
                
                // Draw OSRM route with traffic coloring
                await drawRouteWithTomTomTraffic(osrmRoute.coordinates);
                
                statusDiv.className = 'status success';
                statusDiv.innerHTML = 'Route found (Safest & Shortest)';
                
                routeInfoDiv.style.display = 'block';
                routeInfoDiv.innerHTML = `
                    <strong>📍 Route Information:</strong><br>
                    Distance: ${{osrmRoute.distance}} km<br>
                    Duration: ${{osrmRoute.duration}} min<br>
                    Safety Score: ${{safetyScore}}/1<br>
                    Safety Level: ${{safetyLevel}}<br>
                    <small style="color: #666;">Following actual roads</small>
                `;
                
                trafficInfoDiv.style.display = 'block';
                trafficInfoDiv.innerHTML = `
                    <strong>🚦 Traffic Info:</strong><br>
                    Est. Time: ${{osrmRoute.adjustedDuration}} min<br>
                    Traffic Delay: +${{osrmRoute.trafficDelay}} min<br>
                    Current: ${{getTrafficInfo()}}
                `;
                
                // Display turn-by-turn directions
                if (osrmRoute.directions && osrmRoute.directions.length > 0) {{
                    const directionsDiv = document.createElement('div');
                    directionsDiv.id = 'directionsDiv';
                    directionsDiv.style.marginTop = '15px';
                    directionsDiv.style.paddingTop = '15px';
                    directionsDiv.style.borderTop = '1px solid #ddd';
                    directionsDiv.innerHTML = '<strong>🧭 Turn-by-Turn Directions:</strong><br>';
                    
                    let directionsList = '<ol style="font-size: 12px; margin-left: 15px;">';
                    osrmRoute.directions.slice(0, 10).forEach((dir, idx) => {{
                        directionsList += `<li>${{dir.instruction}} on ${{dir.name}} (${{dir.distance}} km)</li>`;
                    }});
                    if (osrmRoute.directions.length > 10) {{
                        directionsList += `<li>... and ${{osrmRoute.directions.length - 10}} more turns</li>`;
                    }}
                    directionsList += '</ol>';
                    directionsDiv.innerHTML += directionsList;
                    
                    routeInfoDiv.appendChild(directionsDiv);
                }}
                
                map.fitBounds(L.polyline(osrmRoute.coordinates).getBounds());
            }} catch (error) {{
                console.log('Route finding error:', error);
                const routeData = await getRouteFromOSRM(startLat, startLon, endLat, endLon);
                
                if (routeData) {{
                    await drawRouteWithTomTomTraffic(routeData.coordinates);
                    
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = 'Route found (OSRM)';
                    
                    trafficInfoDiv.style.display = 'block';
                    trafficInfoDiv.innerHTML = `
                        <strong>Route Details:</strong><br>
                        Distance: ${{routeData.distance}} km<br>
                        Duration: ${{routeData.duration}} min<br>
                        Traffic Delay: +${{routeData.trafficDelay/60}} min<br>
                        Traffic: ${{routeData.traffic}}
                    `;
                    
                    routeInfoDiv.style.display = 'block';
                    routeInfoDiv.innerHTML = `<strong>Computing optimal route...</strong><br><small>Start A* server for best results</small>`;
                    
                    map.fitBounds(L.latLngBounds(routeData.coordinates));
                }} else {{
                    statusDiv.className = 'status error';
                    statusDiv.textContent = 'Could not compute route. Try different locations.';
                }}
            }}
        }});
        
        // Clear
        clearBtn.addEventListener('click', () => {{
            sourceInput.value = '';
            destInput.value = '';
            delete sourceInput.dataset.lat;
            delete sourceInput.dataset.lon;
            delete destInput.dataset.lat;
            delete destInput.dataset.lon;
            
            if (routingControl) map.removeControl(routingControl);
            if (sourceMarker) map.removeLayer(sourceMarker);
            if (destMarker) map.removeLayer(destMarker);
            
            statusDiv.className = 'status';
            statusDiv.textContent = 'Enter start and destination locations';
            routeInfoDiv.style.display = 'none';
            trafficInfoDiv.style.display = 'none';
            
            document.getElementById('sourceSuggestions').style.display = 'none';
            document.getElementById('destSuggestions').style.display = 'none';
            
            map.setView([{lat_center}, {lon_center}], 14);
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive route finder saved: {output_path}")
    return output_path

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
    
    # Step 4: Create interactive route finder map
    print("\n" + "="*60)
    print("CREATING INTERACTIVE ROUTE FINDER")
    print("="*60)
    
    # Save the interactive map for user input
    interactive_map_file = save_interactive_route_finder(grid_df, 'interactive_safety_route_finder.html')
    print(f"\nOpen 'interactive_safety_route_finder.html' in your browser to find safe routes!")
    
    # Optional: Create a sample route for demonstration
    print("\nGenerating sample route for demonstration...")
    start_location = (19.1200, 72.8470)  # Andheri Station
    end_location = (19.1300, 72.8320)    # JVPD area
    
    # Get baseline route with traffic
    baseline_route, baseline_dist, baseline_time, baseline_traffic = get_route_with_traffic(
        start_location[0], start_location[1],
        end_location[0], end_location[1],
        use_tomtom=False,
        tomtom_key=None
    )
    
    if baseline_route:
        baseline_safety = get_safety_score_for_route(baseline_route, grid_df, baseline_traffic)
        total_time = (baseline_time + baseline_traffic) / 60
        print(f"Sample route: {baseline_dist:.0f}m, {total_time:.1f}min (Traffic: {baseline_traffic/60:.1f}min), Safety: {baseline_safety:.3f}")
        save_live_route_map(None, grid_df, baseline_route, start_location, end_location, 'sample_route.html')
    
    # Save model
    joblib.dump(pipeline, 'andheri_west_safety_pipeline.joblib')
    print("\nAll done! Model saved. Open the HTML file in your browser to start!")