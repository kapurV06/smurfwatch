import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import joblib
import os

# Import custom components from the processing and moderation layers
from preprocess import extract_kinematics
from moderator import TelemetryModeratorAgent

# Enforce professional widescreen telemetry layout
st.set_page_config(page_title="SmurfWatch Telemetry Platform", layout="wide")

@st.cache_resource
def load_production_assets():
    """
    Safely cache weights and scaling variables to keep live inference high-throughput.
    """
    model = tf.keras.models.load_model('models/smurfwatch_net.keras')
    scaler = joblib.load('models/telemetry_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_production_assets()
    agent = TelemetryModeratorAgent()
except Exception as e:
    st.error(f"Initialization Error: Core assets could not be loaded. Verify upstream pipeline execution. Details: {e}")
    st.stop()

st.title("SmurfWatch: Kinematic Telemetry Analytics Platform")
st.markdown("Evaluating streaming spatial coordinate vectors invariant of map positioning via deep recurrent GRU sequencing and automated evaluation networks.")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("Stream Configuration")
raw_data_path = "data/smurf_telemetry.csv"

if not os.path.exists(raw_data_path):
    st.sidebar.error("Data Source Error: File 'data/smurf_telemetry.csv' not found.")
    st.stop()

# Load telemetry cache and append the Adversarial Stress Testing Profiles
full_df = pd.read_csv(raw_data_path)
base_sessions = list(full_df['session_id'].unique())
stress_profiles = [
    "STRESS_Network_Rubberbanding",
    "STRESS_Hardware_Macro",
    "STRESS_Closet_Smurf",
    "STRESS_Rage_Hacker"
]
available_sessions = base_sessions + stress_profiles

selected_session = st.sidebar.selectbox("Target Session ID", available_sessions)
stream_speed = st.sidebar.slider("Ingestion Latency Delay (Seconds)", 0.01, 0.3, 0.02)

start_stream = st.sidebar.button("Establish Live Telemetry Stream")

# --- CORE PERFORMANCE METRICS REGISTER ---
metric_row = st.columns(4)
with metric_row[0]:
    v_metric = st.empty()
with metric_row[1]:
    a_metric = st.empty()
with metric_row[2]:
    apm_metric = st.empty()
with metric_row[3]:
    nn_metric = st.empty()

# --- TEMPORAL AND SPATIAL DATA LAYOUT CONTAINERS ---
graph_container = st.empty()
agent_container = st.container()

# --- LIVE TELEMETRY PROCESSING LOOP ---
if start_stream:
    timesteps = 60
    
    # Check if a live data stream or an adversarial simulation profile was selected
    if selected_session.startswith("STRESS_"):
        df_mock = pd.DataFrame()
        df_mock['timestep'] = np.arange(timesteps)
        
        # Set low-rank baseline defaults
        df_mock['velocity'] = 4.0
        df_mock['acceleration'] = 0.4
        df_mock['jerk'] = 0.1
        df_mock['mouse_jitter'] = 7.5
        df_mock['apm'] = 110.0
        df_mock['reaction_time'] = 250.0
        df_mock['player_x'] = np.linspace(200, 400, timesteps)
        df_mock['player_y'] = np.linspace(200, 400, timesteps)
        
        if selected_session == 'STRESS_Network_Rubberbanding':
            # Low skill profile, but heavily broken by severe packet loss spikes
            for spike in [15, 30, 45]:
                df_mock.loc[spike, 'velocity'] = 900.0
                df_mock.loc[spike, 'acceleration'] = 300.0
                df_mock.loc[spike, 'jerk'] = 200.0
                df_mock.loc[spike, 'player_x'] = 850.0  # Visual teleportation line
                df_mock.loc[spike, 'player_y'] = 850.0
                
        elif selected_session == 'STRESS_Hardware_Macro':
            # Flawless automation script setup
            df_mock['velocity'] = 10.0
            df_mock['acceleration'] = 1.0
            df_mock['jerk'] = 0.2
            df_mock['mouse_jitter'] = 0.0  # Absolute zero variance (biologically impossible)
            df_mock['apm'] = 450.0
            df_mock['reaction_time'] = 1.0
            df_mock['player_x'] = np.linspace(100, 800, timesteps)  # Perfect linear tracking
            df_mock['player_y'] = np.linspace(100, 800, timesteps)
            
        elif selected_session == 'STRESS_Closet_Smurf':
            # Intentional skill masking profile
            df_mock['velocity'] = 6.0
            df_mock['acceleration'] = 0.6
            df_mock['jerk'] = 0.1
            df_mock['mouse_jitter'] = 1.05  # Elite precision mouse control
            df_mock['apm'] = 120.0          # Depressed input speed to hide profile
            df_mock['reaction_time'] = 140.0  # Unmaskable raw cognitive speed
            # Visual idle/containment circle path
            df_mock['player_x'] = 500 + 30 * np.sin(np.linspace(0, 15, timesteps))
            df_mock['player_y'] = 500 + 30 * np.cos(np.linspace(0, 15, timesteps))
            
        elif selected_session == 'STRESS_Rage_Hacker':
            # Blatant engine saturation profile
            df_mock['velocity'] = 500.0
            df_mock['acceleration'] = 120.0
            df_mock['jerk'] = 60.0
            df_mock['mouse_jitter'] = 30.0
            df_mock['apm'] = 700.0
            df_mock['reaction_time'] = 5.0
            # Visual spinbot chaos pattern
            df_mock['player_x'] = np.random.randint(100, 900, timesteps)
            df_mock['player_y'] = np.random.randint(100, 900, timesteps)
            
        session_processed = df_mock
    else:
        session_raw = full_df[full_df['session_id'] == selected_session].sort_values('timestep').copy()
        session_processed = extract_kinematics(session_raw)
    
    streamed_ticks = []
    st.sidebar.success(f"Connected to Stream: {selected_session}")
    
    # Simulate network ingestion ticks
    for idx, current_row in session_processed.iterrows():
        streamed_ticks.append(current_row)
        current_stream_df = pd.DataFrame(streamed_ticks)
        
        latest_apm = current_row['apm']
        
        # Update live telemetry metrics
        v_metric.metric("Mean Velocity", f"{current_stream_df['velocity'].mean():.2f} units/s")
        a_metric.metric("Mean Acceleration", f"{current_stream_df['acceleration'].mean():.2f} units/s²")
        apm_metric.metric("Rolling APM", f"{int(latest_apm)}")
        nn_metric.metric("Analysis Engine Status", "EVALUATING...")
        
        # Generate dual-axis evaluation graphs
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.4, 0.6],
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Plot Spatial Path (Positional Minimap)
        fig.add_trace(
            go.Scatter(
                x=current_stream_df['player_x'], 
                y=current_stream_df['player_y'],
                mode='lines+markers',
                marker=dict(size=3, color='cyan'),
                line=dict(width=1.5, color='royalblue'),
                name='Spatial Trajectory'
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[current_row['player_x']], 
                y=[current_row['player_y']],
                mode='markers',
                marker=dict(size=10, color='red', symbol='cross'),
                name='Current Coordinate'
            ), row=1, col=1
        )
        
        # Plot Kinematic Profiles Over Time
        fig.add_trace(
            go.Scatter(x=current_stream_df['timestep'], y=current_stream_df['velocity'], mode='lines', name='Velocity', line=dict(color='#00CC96', width=1.5)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=current_stream_df['timestep'], y=current_stream_df['acceleration'], mode='lines', name='Acceleration', line=dict(color='#EF553B', width=1.5)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=current_stream_df['timestep'], y=current_stream_df['jerk'], mode='lines', name='Jerk (Neuromuscular Shock)', line=dict(color='#AB63FA', width=1.5)),
            row=1, col=2
        )
        
        # Apply dark mode styling configuration
        fig.update_layout(
            height=450, 
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Map X Boundary", range=[0, 1000], row=1, col=1)
        fig.update_yaxes(title_text="Map Y Boundary", range=[0, 1000], row=1, col=1)
        fig.update_xaxes(title_text="Time Vector (Seconds)", row=1, col=2)
        fig.update_yaxes(title_text="Magnitude Metrics", row=1, col=2)
        
        graph_container.plotly_chart(fig, use_container_width=True)
        time.sleep(stream_speed)
        
    # --- DEEP INFERENCE BLOCK ---
    st.sidebar.info("Sequence window captured. Executing tensor evaluation...")
    
    feature_cols = ['velocity', 'acceleration', 'jerk', 'mouse_jitter', 'apm', 'reaction_time']
    final_features = session_processed[feature_cols].values
    
    scaled_features = scaler.transform(final_features)
    input_tensor = np.expand_dims(scaled_features, axis=0)
    
    prediction_confidence = float(model.predict(input_tensor)[0][0])
    
    # Update state variables using hardened industry classifications
    if prediction_confidence > 0.5:
        nn_metric.metric("Analysis Engine Status", f"ANOMALOUS PROFILE ({prediction_confidence*100:.1f}%)")
        st.error(f"Security Alert: Structural behavior exception triggered for {selected_session}.")
    else:
        nn_metric.metric("Analysis Engine Status", f"VERIFIED LEGITIMATE ({(1 - prediction_confidence)*100:.1f}%)")
        st.success(f"Session Cleared: Telemetry profile conforms to targeted match parameters.")
        
    # --- AUTOMATED COMPLIANCE GENERATOR ---
    with agent_container:
        st.markdown("---")
        st.subheader("Forensic Integrity Evaluation")
        with st.spinner("Compiling structural data packet and executing verification protocol..."):
            
            aggregated_metrics = {
                'mean_velocity': float(session_processed['velocity'].mean()),
                'mean_acceleration': float(session_processed['acceleration'].mean()),
                'mean_jerk': float(session_processed['jerk'].mean()),
                'mean_jitter': float(session_processed['mouse_jitter'].mean()),
                'mean_apm': float(session_processed['apm'].mean()),
                'mean_reaction': float(session_processed['reaction_time'].mean())
            }
            
            report_output = agent.generate_incident_brief(
                session_id=selected_session,
                metrics=aggregated_metrics,
                model_confidence=prediction_confidence
            )
            
            st.markdown(report_output)
else:
    st.info("System Status: Standby. Select an active session matrix from the configuration control panel to begin evaluation.")