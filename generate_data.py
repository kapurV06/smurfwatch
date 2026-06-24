import numpy as np
import pandas as pd
import os

def generate_player_session(player_type, timesteps=60):
    """
    Generates a 60-second continuous telemetry sequence for a single player match-window.
    Each timestep represents 1 second of aggregated high-frequency match telemetry.
    """
    # Using None to ensure true randomness on every function call invocation
    np.random.seed(None)
    
    # --- FEATURE 1 & 2: SPATIAL TRAJECTORY VECTOR MECHANICS (X, Y Coordinates on a 1000x1000 Map) ---
    if player_type == 'smurf':
        # Smurfs travel in highly optimized, direct, low-variance geometric paths (excellent map routing)
        start_pos = np.random.uniform(100, 900, size=2)
        end_pos = start_pos + np.random.uniform(-150, 150, size=2)
        x_path = np.linspace(start_pos[0], end_pos[0], timesteps) + np.random.normal(0, 2, timesteps)
        y_path = np.linspace(start_pos[1], end_pos[1], timesteps) + np.random.normal(0, 2, timesteps)
        
        # --- FEATURE 3: ACTIONS PER MINUTE (APM) CADENCE ---
        # Smurfs maintain an elite, high-throughput, tightly clustered APM distribution
        apm = np.random.normal(320, 25, timesteps) 
        
        # --- FEATURE 4: MOUSE JITTER (Crosshair Micro-Adjustment Variance) ---
        # Smurfs exhibit incredibly steady, razor-sharp crosshair placement (low jitter floor)
        mouse_jitter = np.random.exponential(1.5, timesteps)
        
        # --- FEATURE 5: STIMULUS REACTION LATENCY (Milliseconds) ---
        # Smurfs hit human-limit reflex spikes with hyper-consistent, narrow variance
        reaction_time = np.random.normal(160, 12, timesteps)
    else:
        # Legitimate low-rank players exhibit high spatial path variance, backtracking, and erratic lines
        start_pos = np.random.uniform(100, 900, size=2)
        mid_pos = start_pos + np.random.uniform(-200, 200, size=2)
        end_pos = mid_pos + np.random.uniform(-200, 200, size=2)
        x_path = np.concatenate([np.linspace(start_pos[0], mid_pos[0], timesteps//2), np.linspace(mid_pos[0], end_pos[0], timesteps//2)]) + np.random.normal(0, 15, timesteps)
        y_path = np.concatenate([np.linspace(start_pos[1], mid_pos[1], timesteps//2), np.linspace(mid_pos[1], end_pos[1], timesteps//2)]) + np.random.normal(0, 15, timesteps)
        
        # Low-ranks show lower, highly fluctuating mechanical APM bursts
        apm = np.random.normal(110, 40, timesteps)
        
        # Massive crosshair over-correction/under-correction adjustments (high jitter profile)
        mouse_jitter = np.random.exponential(8.0, timesteps)
        
        # Sluggish, highly volatile visual processing reaction cycles
        reaction_time = np.random.normal(280, 55, timesteps)

    # Consolidate raw multi-dimensional arrays into a structured dataframe
    session_df = pd.DataFrame({
        'player_x': x_path,
        'player_y': y_path,
        'mouse_jitter': mouse_jitter,
        'apm': np.clip(apm, 10, 500), # Bound metrics to realistic game physics limits
        'reaction_time': np.clip(reaction_time, 100, 600)
    })
    
    return session_df

def build_telemetry_dataset(num_samples_per_class=250):
    """
    Executes structural simulation loop and packages sessions into a 
    production-ready database CSV format.
    """
    all_sessions = []
    
    print(f"Executing Generation Engine: Simulating {num_samples_per_class} Smurf & {num_samples_per_class} Legit Sessions...")
    
    # 1. Synthesize Smurf Telemetry Cohort
    for i in range(num_samples_per_class):
        df = generate_player_session('smurf')
        df['session_id'] = f'session_smurf_{i:03d}'
        df['is_smurf'] = 1
        df['timestep'] = np.arange(len(df))
        all_sessions.append(df)
        
    # 2. Synthesize Legitimate Player Cohort
    for i in range(num_samples_per_class):
        df = generate_player_session('legit')
        df['session_id'] = f'session_legit_{i:03d}'
        df['is_smurf'] = 0
        df['timestep'] = np.arange(len(df))
        all_sessions.append(df)
        
    # 3. Concatenate and Enforce Pipeline Layout Order
    dataset = pd.concat(all_sessions, ignore_index=True)
    columns_ordering = ['session_id', 'timestep', 'player_x', 'player_y', 'mouse_jitter', 'apm', 'reaction_time', 'is_smurf']
    dataset = dataset[columns_ordering]
    
    # 4. Save cleanly to local file storage architecture
    os.makedirs('data', exist_ok=True)
    output_path = 'data/smurf_telemetry.csv'
    dataset.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("DAY 1 PIPELINE COMPLETION AUDIT:")
    print(f" -> Output Path:       {output_path}")
    print(f" -> Total Data Rows:   {len(dataset)} records")
    print(f" -> Tracked Features:  {list(dataset.columns[2:-1])}")
    print(f" -> Target Classes:    0 (Legit Log) | 1 (Smurf Flag)")
    print("="*60)

if __name__ == "__main__":
    build_telemetry_dataset(num_samples_per_class=250)