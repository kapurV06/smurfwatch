import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def extract_kinematics(df):
    """
    Computes position-invariant physics derivatives (Velocity, Acceleration, Jerk)
    separately for each individual game session to prevent sequence cross-contamination.
    """
    processed_sessions = []
    
    # Group by unique session to ensure data boundaries are respected
    for session_id, group in df.groupby('session_id'):
        sorted_group = group.sort_values('timestep').copy()
        
        # 1. Delta calculations (dt = 1 second)
        dx = sorted_group['player_x'].diff().fillna(0)
        dy = sorted_group['player_y'].diff().fillna(0)
        
        # 2. Derive Kinematic Features
        velocity = np.sqrt(dx**2 + dy**2)
        acceleration = velocity.diff().fillna(0)
        jerk = acceleration.diff().fillna(0)
        
        # 3. Inject new engineering columns back into session matrix
        sorted_group['velocity'] = velocity
        sorted_group['acceleration'] = acceleration
        sorted_group['jerk'] = jerk
        
        processed_sessions.append(sorted_group)
        
    return pd.concat(processed_sessions, ignore_index=True)

def load_and_prepare_sequences(csv_path="data/smurf_telemetry.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data source: {csv_path}. Run the Day 1 script first!")

    # Ingest raw tabular logs
    raw_df = pd.read_csv(csv_path)
    print(f"Extracted raw telemetry logs. Engineering kinematic derivatives...")
    
    # Execute physics-based feature layer extraction
    df = extract_kinematics(raw_df)
    
    # Isolate unique session IDs to partition data securely without intra-match leakage
    unique_sessions = df['session_id'].unique()
    labels_dict = df.groupby('session_id')['is_smurf'].first().to_dict()
    
    session_ids = list(labels_dict.keys())
    labels = list(labels_dict.values())
    
    # Split matches cleanly: 80% Train, 20% Test
    train_ids, test_ids = train_test_split(session_ids, test_size=0.2, stratify=labels, random_state=42)
    print(f"Split Layout Confirmed: {len(train_ids)} Train Matches | {len(test_ids)} Test Matches")
    
    train_df = df[df['session_id'].isin(train_ids)].copy()
    test_df = df[df['session_id'].isin(test_ids)].copy()
    
    # Define our updated feature columns (Notice player_x and player_y are removed)
    # The neural net will train on pure mechanics, keeping the model position-invariant!
    feature_cols = ['velocity', 'acceleration', 'jerk', 'mouse_jitter', 'apm', 'reaction_time']
    
    # Standardize data distributions (Fit ONLY on training data to lock out leakage)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save the scaler; our Day 5 live streaming engine will require it to normalize real-time ticks
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/telemetry_scaler.pkl')
    print("Serialized feature normalization state to 'models/telemetry_scaler.pkl'")
    
    # Restructure flat rows into 3D Recurrent Tensors: (Samples, Timesteps, Features)
    def reshape_to_tensor(target_df, target_ids):
        tensor_data = []
        tensor_labels = []
        
        for idx in target_ids:
            session_slice = target_df[target_df['session_id'] == idx].sort_values('timestep')
            
            # Form shape matrix dimension layout: (60 timesteps, 6 features)
            feature_matrix = session_slice[feature_cols].values
            tensor_data.append(feature_matrix)
            
            tensor_labels.append(session_slice['is_smurf'].iloc[0])
            
        return np.array(tensor_data), np.array(tensor_labels)

    X_train, y_train = reshape_to_tensor(train_df, train_ids)
    X_test, y_test = reshape_to_tensor(test_df, test_ids)
    
    print("\n" + "="*60)
    print("DAY 2 COMPONENT MATRIX VERIFICATION:")
    print(f" -> X_train Shape: {X_train.shape} (Matches, Seconds, Kinematic Features)")
    print(f" -> X_test Shape:  {X_test.shape}")
    print(f" -> Tracked Inputs: {feature_cols}")
    print("="*60)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_sequences()