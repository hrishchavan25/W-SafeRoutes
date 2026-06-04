import heapq
import math
import json
import numpy as np
from datetime import datetime

# Import ML model and safety prediction
import joblib
from pathlib import Path

# -------------------------------
# Load Safety Model
# -------------------------------
MODEL_PATH = "andheri_west_safety_pipeline.joblib"

def load_safety_model():
    """Load the trained safety prediction model"""
    try:
        if Path(MODEL_PATH).exists():
            return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
    return None

# -------------------------------
# Heuristic Function (Distance)
# -------------------------------
def heuristic(node, goal):
    """Euclidean distance heuristic"""
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

# -------------------------------
# Safety-Aware Heuristic
# -------------------------------
def safety_heuristic(node, goal, safety_score=0):
    """Heuristic combining distance and safety"""
    distance = heuristic(node, goal)
    # Penalize unsafe areas (safety_score close to 1 = unsafe)
    safety_penalty = safety_score * 10  # Increase cost for unsafe routes
    return distance + safety_penalty

# -------------------------------
# A* Algorithm Function (Basic)
# -------------------------------
def astar(graph, start, goal):
    """Basic A* pathfinding algorithm"""
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
            
        for neighbor, cost in graph[current]:
            temp_g = g_score[current] + cost

            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                f_score = temp_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
                came_from[neighbor] = current

    return None

# -------------------------------
# Safety-Aware A* Algorithm
# -------------------------------
def astar_safety(grid_data, start, goal, model=None):
    """
    A* algorithm that considers safety scores
    
    Args:
        grid_data: DataFrame with latitude, longitude, unsafe_prob columns
        start: (lat, lon) tuple for start location
        goal: (lat, lon) tuple for goal location
        model: Trained safety prediction model
    
    Returns:
        List of (lat, lon) tuples representing the safest route
    """
    
    # Build grid graph with safety awareness
    graph = build_safety_aware_graph(grid_data, model)
    
    # Find nodes closest to start and goal
    start_node = find_closest_node(graph, start)
    goal_node = find_closest_node(graph, goal)
    
    if not start_node or not goal_node:
        return None
    
    # Run A* with safety consideration
    open_list = []
    heapq.heappush(open_list, (0, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    safety_scores = {}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal_node:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            
            # Add metadata
            route_info = {
                'path': path[::-1],
                'safety_scores': [safety_scores.get(node, 0.5) for node in path[::-1]],
                'total_distance': calculate_total_distance(path[::-1]),
                'average_safety': np.mean([safety_scores.get(node, 0.5) for node in path[::-1]])
            }
            return route_info
        
        if current not in graph:
            continue
        
        for neighbor, cost, safety in graph[current]:
            temp_g = g_score[current] + cost
            
            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                safety_scores[neighbor] = safety
                # Combine distance and safety in heuristic
                f_score = temp_g + safety_heuristic(neighbor, goal_node, safety)
                heapq.heappush(open_list, (f_score, neighbor))
                came_from[neighbor] = current
    
    return None

# -------------------------------
# Build Safety-Aware Graph
# -------------------------------
def build_safety_aware_graph(grid_data, model=None):
    """
    Build a graph from grid data considering safety
    
    Args:
        grid_data: DataFrame with location and safety data
        model: ML model for predictions
    
    Returns:
        Dictionary representing the graph with safety awareness
    """
    graph = {}
    
    if grid_data is None or len(grid_data) == 0:
        return graph
    
    # Create nodes from grid data
    nodes = list(zip(grid_data['latitude'].values, grid_data['longitude'].values))
    
    for i, node in enumerate(nodes):
        neighbors = []
        safety_score = grid_data.iloc[i].get('unsafe_prob', 0.5)
        
        # Connect to nearby nodes (8-connectivity or distance-based)
        for j, other_node in enumerate(nodes):
            if i != j:
                dist = heuristic(node, other_node)
                # Only connect nearby nodes (within 0.01 degrees)
                if dist < 0.01 and dist > 0:
                    neighbors.append((other_node, dist, safety_score))
        
        graph[node] = neighbors
    
    return graph

# -------------------------------
# Find Closest Node to Location
# -------------------------------
def find_closest_node(graph, location):
    """Find the graph node closest to a given location"""
    if not graph:
        return None
    
    closest_node = None
    min_distance = float('inf')
    
    for node in graph.keys():
        dist = heuristic(node, location)
        if dist < min_distance:
            min_distance = dist
            closest_node = node
    
    return closest_node

# -------------------------------
# Calculate Total Distance
# -------------------------------
def calculate_total_distance(path):
    """Calculate total distance for a path"""
    total = 0
    for i in range(len(path) - 1):
        total += heuristic(path[i], path[i + 1])
    return total

# -------------------------------
# Generate Safety Report
# -------------------------------
def generate_route_safety_report(route_info):
    """Generate a detailed safety report for the route"""
    if not route_info:
        return None
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_distance': round(route_info['total_distance'], 4),
        'average_safety_score': round(route_info['average_safety'], 2),
        'safety_level': 'Safe' if route_info['average_safety'] < 0.3 else 'Moderate' if route_info['average_safety'] < 0.7 else 'Unsafe',
        'waypoints_count': len(route_info['path']),
        'route_coordinates': route_info['path'],
        'segment_safety_scores': [round(score, 2) for score in route_info['safety_scores']]
    }

# -------------------------------
# MAIN DRIVER CODE
# -------------------------------
if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 60)
    print("A* SAFETY-AWARE PATHFINDING")
    print("=" * 60)
    
    # Load safety model
    model = load_safety_model()
    print(f"Model loaded: {model is not None}")
    
    # Example 1: Basic A* with simple graph
    print("\n1. BASIC A* PATHFINDING")
    print("-" * 60)
    
    graph = {
        (0, 0): [((1, 0), 1), ((0, 1), 1)],
        (1, 0): [((2, 0), 1)],
        (0, 1): [((1, 1), 1)],
        (1, 1): [((2, 1), 1)],
        (2, 0): [((2, 1), 1)],  
        (2, 1): [((3, 1), 1)],
        (3, 1): []
    }
    
    start = (0, 0)
    goal = (3, 1)
    path = astar(graph, start, goal)
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Path found: {path}")
    
    # Example 2: Safety-Aware A* with real data
    print("\n2. SAFETY-AWARE A* WITH ML PREDICTIONS")
    print("-" * 60)
    
    try:
        # Try to load real data
        data_files = ['clean_safety_data_mumbai.csv', 'clean_safety_data.csv']
        grid_df = None
        
        for file in data_files:
            try:
                grid_df = pd.read_csv(file)
                print(f"Loaded data from: {file}")
                print(f"Data shape: {grid_df.shape}")
                break
            except:
                continue
        
        if grid_df is not None and len(grid_df) > 0:
            # Define start and goal
            start_lat, start_lon = 19.1240, 72.8254  # Andheri
            goal_lat, goal_lon = 19.1620, 72.8294    # Different location
            
            start_loc = (start_lat, start_lon)
            goal_loc = (goal_lat, goal_lon)
            
            print(f"\nStart Location: {start_loc}")
            print(f"Goal Location: {goal_loc}")
            
            # Run safety-aware A*
            route_info = astar_safety(grid_df, start_loc, goal_loc, model)
            
            if route_info:
                # Generate safety report
                report = generate_route_safety_report(route_info)
                
                print(f"\nRoute Found Successfully!")
                print(f"Total Distance: {report['total_distance']:.4f} degrees")
                print(f"Number of Waypoints: {report['waypoints_count']}")
                print(f"Average Safety Score: {report['average_safety_score']:.2f}")
                print(f"Safety Level: {report['safety_level']}")
                print(f"Route Coordinates (first 5): {report['route_coordinates'][:5]}")
            else:
                print("No route found between start and goal")
        else:
            print("Could not load data - running basic A* only")
    
    except Exception as e:
        print(f"Error in safety-aware pathfinding: {e}")
    
    print("\n" + "=" * 60)
    print("A* Pathfinding Complete")
    print("=" * 60)