import os
import json
from groq import Groq

class TelemetryModeratorAgent:
    def __init__(self):
        # Gracefully handle API authentication from the environment variables
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("[⚠️ WARNING]: GROQ_API_KEY environment variable not detected.")
            print("The agent will operate in MOCK mode. Export your key to activate production inference.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            
        # Meta's 70-billion parameter reasoning model via Groq's low-latency LPU architecture
        self.model_id = "llama-3.3-70b-versatile"

    def compile_forensic_prompt(self, session_id, metrics, model_confidence):
        """
        Translates raw physics-based kinematics and mechanical features into an
        authoritative evidentiary context file for LLM evaluation.
        """
        prompt = f"""
        [COMPETITIVE INTEGRITY ENFORCEMENT PROTOCOL - INCIDENT REPORT]
        
        Target Session ID: {session_id}
        Neural Network Flag Confidence: {model_confidence * 100:.2f}%
        
        Aggregated Kinematic & Mechanical Vectors (60-Second Sequence Window):
        - Mean Velocity (Map Units/sec): {metrics['mean_velocity']:.2f}
        - Mean Acceleration (Rate of Velocity Change): {metrics['mean_acceleration']:.2f}
        - Mean Jerk Vector (Abrupt Kinematic Shocks): {metrics['mean_jerk']:.2f}
        - Crosshair Micro-Adjustment Error (Mouse Jitter): {metrics['mean_jitter']:.4f}
        - Actions-Per-Minute (APM) Throughput: {metrics['mean_apm']:.1f}
        - Average Visual Processing Stimulus Latency: {metrics['mean_reaction']:.1f} ms
        
        CRITICAL ASSESSMENT PARADIGM:
        The deep learning core uses position-invariant kinematics. This means the layout evaluates 
        neuromuscular capacity independent of map coordinates. 
        Low-Rank Baseline Profiles typically exhibit low velocity, erratic acceleration bounds, 
        chaotic jerk lines, high jitter (>6.0), low APM (<150), and sluggish response times (>240ms).
        
        INSTRUCTIONS FOR EXAMINER:
        Generate a professional, structured Matchmaking Integrity Brief in clean Markdown format. 
        The generated text MUST explicitly include:
        1. EXECUTIVE SUMMARY (Final integrity verdict and severe risk level rating)
        2. KINEMATIC & NEUROMUSCULAR PATTERN DECONSTRUCTION (Analyze how the velocity, acceleration spikes, and hyper-low reaction latency match elite cognitive thresholds vs low-tier mechanics)
        3. AUTOMATED ANTI-CHEAT RECOMMENDATION (Enforcement action vector: Permanent Account Ban, Immediate Competitive Rank Calibration, or Latent Passive Tracking)
        
        Maintain an objective, technical, and analytical tone. Do not declare your identity as an AI.
        """
        return prompt

    def generate_incident_brief(self, session_id, metrics, model_confidence):
        # Anchor system rules into the engine
        system_instruction = (
            "You are an automated, high-throughput E-Sports Competitive Integrity anti-cheat agent. "
            "Your structural purpose is to ingest kinematic telemetry violations and compile strict, "
            "forensic, and definitive assessment files for review boards."
        )
        
        prompt = self.compile_forensic_prompt(session_id, metrics, model_confidence)
        
        # Fallback mechanism if operating without a live key
        if not self.client:
            return f"""
            ### [MOCK INTEGRITY BRIEF - EXPORT GROQ_API_KEY TO UNLOCK LIVE LLM]
            **Verdict:** SUSPECTED POSITION-INVARIANT MECHANIC ANOMALY (SMURFING)
            - **Session Tracking ID:** {session_id}
            - **Classifier Neural Confidence:** {model_confidence*100:.2f}%
            - **Kinematic Profile:** Mean APM of {metrics['mean_apm']:.1f} and Reaction Time of {metrics['mean_reaction']:.1f}ms cleanly breach the structural thresholds of this matchmaking lobby.
            """
            
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.15, # Hardened constraint to ensure structured, non-hallucinatory legalistic reporting
                max_tokens=1200
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to execute Agent inference link: {str(e)}"

if __name__ == "__main__":
    # Simulate a highly precise kinematic anomaly alert captured from our Day 3 GRU model weights
    agent = TelemetryModeratorAgent()
    
    sample_anomalous_metrics = {
        'mean_velocity': 148.3,
        'mean_acceleration': 12.4,
        'mean_jerk': 4.1,
        'mean_jitter': 1.08,
        'mean_apm': 345.2,
        'mean_reaction': 152.1
    }
    
    print("Simulating live agent inference loop for kinematic structural flag...")
    report = agent.generate_incident_brief(
        session_id="session_smurf_084",
        metrics=sample_anomalous_metrics,
        model_confidence=0.9987
    )
    
    print("\n" + "="*25 + " GENERATED AGENT OUTPUT " + "="*25)
    print(report)