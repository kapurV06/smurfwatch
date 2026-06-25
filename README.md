# SmurfWatch

SmurfWatch is a behavioral anomaly detection platform designed to identify high-skill smurf accounts and automated scripts by analyzing player kinematics. Rather than relying on client-side memory scanning or intrusive anti-cheat hooks, the platform evaluates behavioral patterns using deep sequence modeling and generates human-readable forensic logs via an agentic LLM layer.

---

## Repository Architecture

```text
smurfwatch/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── smurf_telemetry.csv
├── models/
│   ├── smurfwatch_net.keras
│   └── telemetry_scaler.pkl
├── dashboard.py
├── preprocess.py
├── moderator.py
├── stress_test.py
├── test_pipeline.py
└── requirements.txt

```

---

## Core Pipeline & Features

* **Kinematic Telemetry Extraction:** Computes spatial coordinate vectors into discrete physical derivatives (velocity, acceleration, and neuromuscular jerk) to map behavioral intent.
* **Temporal Sequence Modeling:** Leverages a 3-layer Gated Recurrent Unit (GRU) network stabilized with Batch Normalization and Dropout layers to process continuous input streams without vanishing gradient constraints.
* **Automated Forensics:** Integrates a clinical, data-backed LLM assessment layer via the Groq API (LLaMA) to synthesize metrics into structured security briefs.
* **Live Telemetry Dashboard:** Built with Streamlit to visualize rolling real-time player telemetry, tracking probability thresholds along side-by-side coordinate plots.

---

## Dataset & Training Profile

The underlying neural network is optimized for high-frequency temporal sequences tracking positional tracking variables:

* **Dataset Scale:** 1.42 million rows spanning 4,800 distinct behavioral sequence segments.
* **Input Tensor Shape:** `(Batch_Size, 600, 6)` representing a 60-second temporal window sampled at 10Hz across 6 physical features.
* **Hardware Baseline:** Trained on a single NVIDIA RTX 4060 (8GB VRAM); inference profiled on an x86 AMD Ryzen 5 CPU execution layer.

---

## Model Performance & Validation Framework

### Methodological Note on Calibration Baselines

Current performance indices are derived using high-fidelity synthetic data matrices and adversarial simulation profiles. While synthetic environments yield pristine accuracy markers, real-world deployment targets expect higher environmental noise, network jitter fluctuations, and non-standard input drivers.

Phase 1 demonstrates architectural viability via deep temporal sequence modeling. Phase 2 targets shadow deployment data collection to address real-world domain adaptation.

| Evaluation Metric | Simulation Target Baseline | Operational Focus |
| --- | --- | --- |
| **Global Accuracy** | 98.42% | Synthetic validation matrix baseline |
| **F1-Score** | 0.976 | Macro and absolute zero-variance execution profiles |
| **False Positive Rate (FPR)** | < 0.04% | Preservation of edge-case high-skill human inputs (e.g., high-DPI flicks) |
| **Inference Step Latency** | 2.1 ms | Rolling single-tick pipeline evaluation on CPU environments |

### Edge-Case Hardening

* **Network Desync Invariance:** The pipeline handles single-frame telemetry drops up to 30% synthetic packet loss. The GRU reset gates preserve long-term temporal intent, suppressing false positives during network rubberbanding.
* **Micro-Adjustment Detection:** Disguised capability profiles are flagged via high-precision variance analysis within localized mouse jitter metrics and sub-150ms reaction windows, bypassing attempts to depress global Actions Per Minute (APM).

---

## Forensic Layer & Persona Specification

The LLM layer contained in `moderator.py` functions as a neutral, data-driven utility rather than a conversational agent. It is explicitly configured via system instructions to act as a transparent software layer, evaluating telemetry data strictly through empirical thresholds without adopting a human identity or simulating human reasoning patterns.

---

## Automated Verification (CI/CD)

The project implements automated unit testing via a localized verification script (`test_pipeline.py`) that checks kinematic feature derivation consistency across data frames.

The repository utilizes GitHub Actions (`.github/workflows/ci.yml`) to execute these test suites on a virtualized Ubuntu runner on every push or pull request to the `main` branch, ensuring stability across dependencies.

---

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/kapurV06/smurfwatch.git
cd smurfwatch

```


2. Install runtime dependencies:
```bash
pip install -r requirements.txt

```


3. Run the local automated test suite:
```bash
python test_pipeline.py

```


4. Launch the live telemetry dashboard:
```bash
streamlit run dashboard.py

```
