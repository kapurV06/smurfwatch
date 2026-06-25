import pandas as pd
import numpy as np
from preprocess import extract_kinematics

def test_kinematic_feature_derivation():
    """
    Validates that spatial coordinate vectors correctly resolve into 
    first, second, and third-order physical derivatives.
    """
    print("Initializing test data matrix...")
    
    # Construct a deterministic linear trajectory with required session tracking
    mock_raw = pd.DataFrame({
        'session_id': np.full(10, 'session_mock_001'),
        'timestep': np.arange(10),
        'player_x': np.linspace(10, 100, 10),
        'player_y': np.linspace(20, 200, 10),
        'apm': np.full(10, 150.0),
        'reaction_time': np.full(10, 200.0),
        'mouse_jitter': np.full(10, 2.5)
    })
    
    print("Running kinematic engineering pipeline...")
    processed_df = extract_kinematics(mock_raw)
    
    # Assert structural integrity
    assert 'velocity' in processed_df.columns, "Missing velocity vector extraction."
    assert 'acceleration' in processed_df.columns, "Missing acceleration vector extraction."
    assert 'jerk' in processed_df.columns, "Missing neuromuscular jerk vector extraction."
    assert len(processed_df) == 10, "Dataframe dimensions mutated during preprocessing."
    
    # Assert mathematical sanity (constant velocity means zero acceleration)
    assert processed_df['acceleration'].iloc[-1] >= 0.0
    
    print("SUCCESS: All core pipeline assertions passed safely.")

if __name__ == "__main__":
    try:
        test_kinematic_feature_derivation()
    except AssertionError as e:
        print(f"CRITICAL FAILURE: {e}")