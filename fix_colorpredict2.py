#!/usr/bin/env python3
# Script to fix the corrupted colorpredict2.py

with open('colorpredict2_corrupted.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Keep lines 1-1440 (indices 0-1439)
good_lines = lines[:1440]

# Main pipeline code
main_code = '''
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
    print("\\n" + "="*60)
    print("CREATING INTERACTIVE ROUTE FINDER")
    print("="*60)
    
    # Save the interactive map for user input
    interactive_map_file = save_interactive_route_finder(grid_df, 'interactive_safety_route_finder.html')
    print(f"\\nOpen 'interactive_safety_route_finder.html' in your browser to find safe routes!")
    
    # Optional: Create a sample route for demonstration
    print("\\nGenerating sample route for demonstration...")
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
    print("\\nAll done! Model saved. Open the HTML file in your browser to start!")
'''

# Write the fixed file
with open('colorpredict2.py', 'w', encoding='utf-8') as f:
    f.writelines(good_lines)
    f.write(main_code)

print("Fixed colorpredict2.py has been recreated!")
