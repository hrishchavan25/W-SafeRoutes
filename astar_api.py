# A* Pathfinding API Server
# Flask REST API for A* safety-aware routing
#
# Install required packages:
# pip install flask flask-cors pandas numpy scikit-learn joblib
#
# Run server:
# python astar_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import json
import heapq
import math
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "andheri_west_safety_pipeline.joblib"
DATA_PATH = "clean_safety_data_mumbai.csv"

# Global variables
model = None
grid_data = None

def load_model():
    """Load the trained safety prediction model"""
    global model
    try:
        if Path(MODEL_PATH).exists():
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
            return True
    except Exception as e:
        print(f"Error loading model: {e}")
    return False

def load_grid_data():
    """Load grid data for pathfinding"""
    global grid_data
    try:
        if Path(DATA_PATH).exists():
            grid_data = pd.read_csv(DATA_PATH)
            print(f"Grid data loaded: {grid_data.shape}")
            return True
    except Exception as e:
        print(f"Error loading grid data: {e}")
    return False

def heuristic(node, goal):
    """Euclidean distance heuristic"""
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def build_safety_graph(grid_df):
    """Build safety-aware graph from grid data"""
    graph = {}
    if grid_df is None or len(grid_df) == 0:
        return graph
    
    nodes = list(zip(grid_df['latitude'].values, grid_df['longitude'].values))
    safety_scores = dict(zip(nodes, grid_df.get('unsafe_prob', pd.Series([0.5]*len(grid_df))).values))
    
    for i, node in enumerate(nodes):
        neighbors = []
        safety_score = grid_df.iloc[i].get('unsafe_prob', 0.5) if isinstance(grid_df.iloc[i].get('unsafe_prob'), (int, float)) else 0.5
        
        # Connect to nearby nodes
        for j, other_node in enumerate(nodes):
            if i != j:
                dist = heuristic(node, other_node)
                # Connect nodes within reasonable distance (0.01 degrees ~ 1 km)
                if 0 < dist < 0.015:
                    other_safety = safety_scores.get(other_node, 0.5)
                    neighbors.append((other_node, dist, other_safety))
        
        graph[node] = neighbors
    
    return graph

def find_closest_node(graph, location):
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

def astar_pathfinding(graph, start_lat, start_lon, end_lat, end_lon):
    """
    A* algorithm for combined safety + distance optimization
    Balances shortest path with safest path
    
    Returns:
        Dictionary with path, safety scores, and metadata
    """
    
    if not graph:
        return None
    
    start = find_closest_node(graph, (start_lat, start_lon))
    goal = find_closest_node(graph, (end_lat, end_lon))
    
    if not start or not goal:
        return None
    
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    safety_scores = {start: 0}
    
    iterations = 0
    max_iterations = 10000
    
    while open_list and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            
            path = path[::-1]
            avg_safety = np.mean([safety_scores.get(node, 0.5) for node in path])
            
            return {
                'path': path,
                'safety_scores': [safety_scores.get(node, 0.5) for node in path],
                'average_safety': float(avg_safety),
                'waypoints': len(path),
                'iterations': iterations
            }
        
        if current not in graph:
            continue
        
        for neighbor, cost, neighbor_safety in graph[current]:
            # Combined optimization: 60% distance + 40% safety
            # This finds the balance between shortest and safest
            combined_cost = (cost * 0.6) + (neighbor_safety * 0.4)
            temp_g = g_score[current] + combined_cost
            
            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                safety_scores[neighbor] = neighbor_safety
                f_score = temp_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
                came_from[neighbor] = current
    
    return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "data_loaded": grid_data is not None
    }), 200

@app.route('/api/astar-route', methods=['POST'])
def compute_astar_route():
    """Compute A* safety-aware route"""
    try:
        if grid_data is None:
            return jsonify({
                "success": False,
                "error": "Grid data not loaded"
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        start_lat = float(data.get('start_lat'))
        start_lon = float(data.get('start_lon'))
        end_lat = float(data.get('end_lat'))
        end_lon = float(data.get('end_lon'))
        
        if not all([start_lat, start_lon, end_lat, end_lon]):
            return jsonify({"error": "Missing coordinates"}), 400
        
        # Build graph and compute A* route
        graph = build_safety_graph(grid_data)
        
        result = astar_pathfinding(graph, start_lat, start_lon, end_lat, end_lon)
        
        if result:
            return jsonify({
                "success": True,
                "path": result['path'],
                "safety_scores": result['safety_scores'],
                "average_safety": result['average_safety'],
                "waypoints": result['waypoints'],
                "iterations": result['iterations'],
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Could not compute A* path"
            }), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/graph-stats', methods=['GET'])
def get_graph_stats():
    """Get statistics about the pathfinding graph"""
    try:
        if grid_data is None:
            return jsonify({
                "success": False,
                "error": "Grid data not loaded"
            }), 500
        
        graph = build_safety_graph(grid_data)
        
        return jsonify({
            "success": True,
            "nodes": len(graph),
            "edges": sum(len(neighbors) for neighbors in graph.values()),
            "data_points": len(grid_data),
            "avg_safety": float(grid_data['unsafe_prob'].mean()) if 'unsafe_prob' in grid_data.columns else 0.5
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("A* PATHFINDING API SERVER")
    print("=" * 60)
    
    # Load data and model
    model_loaded = load_model()
    data_loaded = load_grid_data()
    
    print(f"Model Status: {'Loaded' if model_loaded else 'Not loaded'}")
    print(f"Data Status: {'Loaded' if data_loaded else 'Not loaded'}")
    
    print("\nServer running on http://localhost:5001")
    print("\nAvailable endpoints:")
    print("   GET  /api/health")
    print("   POST /api/astar-route")
    print("   GET  /api/graph-stats")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
