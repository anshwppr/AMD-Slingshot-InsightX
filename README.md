# Digital Hygiene — Post-Quantum Cybersecurity Defense System

> **AMD AI Cybersecurity Hackathon** · Research Prototype · 2026

---

## The Story

It started with a conversation most people never think about until it's too late.

Imagine a first-year student call him Cipher sitting in a university computer lab, clicking what looks like a routine email from the registrar's office. The link looks right. The logo looks right. Forty-eight hours later, his login credentials are circulating on a paste site he's never heard of. His academic record has been accessed. His scholarship application is gone. He didn't do anything wrong. He just didn't know.

Now multiply Cipher by the tens of thousands of students who walk into labs, click on links, share files on public Wi-Fi, and assume that if a connection is encrypted, it's safe. They're not careless. They're just not equipped.

That's the gap we set out to close.

---

## The Problem

Phishing, malware, spoofing, and data misuse are not abstract threats. They hit students, labs, and institutions every day and the tools designed to stop them are either invisible to the people who need them most, or so technical that they generate alerts nobody reads.

There's a second layer most people don't talk about yet: the **post-quantum threat**. Nation-state actors and sophisticated adversaries are already running "Harvest Now, Decrypt Later" campaigns collecting encrypted traffic today with the plan to decrypt it once quantum computers are capable enough. For institutions holding long-term research data, medical records, or student PII, the threat isn't hypothetical. It's already in motion.

The specific pain points:
- No plain-language, real-time explanation of *why* something is dangerous
- No adaptive system that gets better the more threats it sees
- No awareness layer built for students and first-year users, not just security teams
- No readiness for post-quantum cryptographic attacks quietly accumulating data right now

---

## The Solution: Digital Hygiene

Digital Hygiene is an AI-powered, explainable post-quantum cybersecurity defense system built for educational institutions first
and extensible to any organization that handles sensitive data.

**Layer one** is a ResNet-style threat detector trained on 12 threat classes spanning both classical network attacks (DDoS, DoS, brute force, spoofing, Mirai/IoT botnet, web-based attacks, reconnaissance, and benign traffic) and four post-quantum threat categories (PQ-Downgrade, PQ-HNDL, PQ-SideChannel, and PQ-Hybrid). It runs in real time against live network traffic and returns a confidence score alongside a structured explanation.

**Layer two** is a reinforcement learning response agent trained with a Deep Q-Network and prioritized experience replay. It learns, across thousands of simulated threat events, how to select the right response action from "Quick Scan" to "Emergency Crypto Rotate" based on the detected threat class, confidence level, system health, and cryptographic strength.

**Layer three (NEW)** is a LoRA fine-tuned version of the detector same architecture, same accuracy, but only **7.9% of parameters trained**. The adapter file is 13× smaller than the full model, making it ideal for domain adaptation across institutions without retraining from scratch.

---

## Results

### Baseline Model

| Metric | Score |
|--------|-------|
| Overall Detection Accuracy | 89.20% |
| F1-Score (weighted) | 0.890 |
| Response Accuracy | 88.10% |
| PQ Detection Accuracy | 82.93% |
| PQ Response Accuracy | 96.65% |
| Average Detection Confidence | 79.9% |
| Average RL Reward | 49.71 |
| Total Parameters | 185,970 |
| Model Size | 0.709 MB |

### LoRA Fine-tuned Model

| Metric | Baseline | LoRA | Δ |
|--------|----------|------|---|
| Detection Accuracy | 89.20% | **90.80%** | +1.60% |
| PQ Detection Accuracy | 82.93% | **89.64%** | +6.72% |
| Response Accuracy | 88.10% | **90.20%** | +2.10% |
| PQ Response Accuracy | 96.65% | **96.65%** | — |
| F1-Score | 0.890 | **0.906** | +0.016 |
| Confidence Score | 79.9% | **83.2%** | +3.3% |
| Trainable Parameters | 185,970 (100%) | **14,744 (7.9%)** | 92.1% frozen |
| Adapter Size | 0.709 MB | **0.056 MB** | 13× smaller |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/anshwppr/AMD-Slingshot-InsightX
cd digital-hygiene
```

### 2. Set up virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model weights

Place your trained `.pth` files inside the `models/` folder:

```
models/
├── detector.pth                   ← Baseline detector weights (0.709 MB)
├── lora_adapter.pth               ← LoRA adapter weights (0.056 MB)
├── response_agent.pth             ← Baseline response agent weights
└── response_agent_lora.pth        ← LoRA response agent weights
```

> The models folder is intentionally excluded from version control (see `.gitignore`).
> Train your own using the notebooks, or download pre-trained weights separately.

### 5. Run the Flask backend

```bash
python app.py
```

The server starts at `http://localhost:5000`

### 6. Open the dashboard

Open `index.html` directly in your browser, **or** navigate to:

```
http://localhost:5000
```

> The Flask server serves `index.html` automatically at the root route.  
> Make sure `app.py` is running before opening the dashboard so the `/predict` and `/model_info` endpoints are available.

---

## Dataset

This project uses the **CIC-IoT Dataset 2023 (Updated 2024)**:

🔗 [https://www.kaggle.com/datasets/mdabdulalemo/cic-iot-dataset2023-updated-2024-10-08](https://www.kaggle.com/datasets/mdabdulalemo/cic-iot-dataset2023-updated-2024-10-08)

Download and place the CSV files in a folder, then update `INPUT_FOLDER` in the notebooks before training.

Post-quantum threat classes (PQ-Downgrade, PQ-HNDL, PQ-SideChannel, PQ-Hybrid) are **synthetically generated** from the classical samples using class-specific feature transformations.

---

## Technical Architecture

### System Pipeline

![System Architecture](system.jpeg)

### Detection Model — EffectiveThreatDetector

- **Architecture:** ResNet-style MLP with skip connections
- **Input:** 47-feature normalized network flow vector (StandardScaler)
- **Layers:** 256 → 256 (residual) → 192 → 128 → 64 → 12 (softmax)
- **Regularization:** BatchNorm1d, Dropout (0.15–0.3), gradient clipping (1.0)
- **Loss:** Weighted Focal Loss (γ=2.0) for class imbalance
- **Optimizer:** AdamW (lr=0.001, weight_decay=1e-4)
- **Training:** 100 epochs, batch 256, early stopping (patience=35)
- **Total Parameters:** 185,970
- **Model Size:** 0.709 MB

### LoRA Threat Detector — LoRAThreatDetector

- **Base:** Loads pretrained weights from `detector.pth`
- **Frozen:** Input BN + Encoder block (low-level features preserved)
- **Trainable:** Deep features (192→128→64), classifier head (→12), LoRA adapters on all linear layers
- **LoRA Rank:** 4 (A×B decomposition, 14,744 params total)
- **Param Efficiency:** 92.1% frozen (171,226 params), 7.9% trainable (14,744 params)
- **Adapter Size:** 0.056 MB (13× smaller than full model)
- **Training:** 100 epochs, same optimizer as baseline

### Threat Classes

| ID | Class | Category |
|----|-------|----------|
| 0 | Benign | Normal Traffic |
| 1 | DDoS | Distributed Denial of Service |
| 2 | DoS | Denial of Service |
| 3 | Recon | Reconnaissance/Scanning |
| 4 | Web-based | Web Application Attack |
| 5 | BruteForce | Credential Brute Force |
| 6 | Spoofing | Identity Spoofing |
| 7 | Mirai | IoT Botnet Activity |
| 8 | PQ-Downgrade | Post-Quantum Cryptographic Downgrade |
| 9 | PQ-HNDL | Harvest Now, Decrypt Later |
| 10 | PQ-SideChannel | Post-Quantum Side-Channel Attack |
| 11 | PQ-Hybrid | Hybrid Classical-Quantum Attack |

### Response Agent — ImprovedResponseDQN

- **Architecture:** DQN with LayerNorm: 256 → 128 → 64 → 8 (actions)
- **State space (15 dimensions):** threat class, confidence, detection correctness, PQ flags, system health, crypto strength, uncertainty proxy, categorical threat range indicators
- **Action space (8 actions):** Ignore, Quick Scan, Full Scan, Quarantine, Delete, Network Isolate, PQ-Crypto Upgrade, Emergency Crypto Rotate
- **Training:** Double DQN, Prioritized Experience Replay (capacity 30,000), epsilon decay 1.0→0.02, γ=0.99, **8,000 episodes**
- **Optimizer:** AdamW (lr=0.0003, weight_decay=1e-5), gradient clipping (10.0)

### LoRA Configuration

```python
LORA_CONFIG = {
    'rank': 4,           # Low-rank dimension
    'alpha': 8,          # Scaling factor (2× rank)
    'dropout': 0.1,      # LoRA dropout
    'use_lora': True,
    'lora_layers': 'all'
}
```

### Response Actions and Optimal Mappings

| Threat | Optimal Action |
|--------|---------------|
| Benign | Ignore |
| DDoS, Mirai | Network Isolate |
| DoS | Full Scan |
| Recon | Quick Scan |
| Web-based, BruteForce, Spoofing | Full Scan / Quarantine / Delete |
| PQ-Downgrade, PQ-SideChannel | PQ-Crypto Upgrade |
| PQ-HNDL, PQ-Hybrid | Emergency Crypto Rotate |

---

## File Structure

```
digital_hygiene/
├── app.py                          # Flask backend — loads .pth models, serves API
├── index.html                      # Interactive threat simulation dashboard
├── baseline_detector.ipynb         # Baseline training notebook (EffectiveThreatDetector)
├── lora_enhanced_detector.ipynb    # LoRA fine-tuning notebook (LoRAThreatDetector)
├── training_lora.png               # LoRA training visualizations
├── training_baseline.png           # Baseline training visualizations
├── lora_results_summary.json       # LoRA metrics summary
├── results_summarybaseline.json    # Baseline metrics summary
├── requirements.txt                # Python dependencies
├── README.md                       # This document
└── models/
    ├── detector.pth                # Baseline detector weights (0.709 MB)
    ├── lora_adapter.pth            # LoRA adapter weights (0.056 MB)
    ├── response_agent.pth          # Baseline response agent weights
    └── response_agent_lora.pth     # LoRA response agent weights
```

---

## API Endpoints

Once `app.py` is running:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves `index.html` |
| `/predict` | POST | Run threat detection + response |
| `/model_info` | GET | Parameter counts and model info |

**POST `/predict` example:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'  # 47 float values
```

---

## Training From Scratch

### Baseline model

Open and run `baseline_detector.ipynb`trains `EffectiveThreatDetector` and saves `detector.pth` and `response_agent.pth`.

### LoRA fine-tuning

Open and run `lora_enhanced_detector.ipynb` loads `detector.pth`, applies LoRA adapters (rank=4), saves `lora_adapter.pth` and `response_agent_lora.pth`.

---

## Infrastructure Requirements

| Component | Requirement |
|-----------|-------------|
| Detection inference | CPU: 4 cores, 8 GB RAM (GPU optional) |
| Training (full pipeline) | GPU recommended: NVIDIA RTX 3080+ |
| Storage | 10 GB for dataset, 500 MB for models |
| OS | Ubuntu 22.04 LTS recommended |
| Python | 3.8+ |

---

## Future Work

1. **Adversarial Multi-Agent RL** — Red-team agent generates evasive signatures; defender adapts in response (MADDPG framework)
2. **K-Fold Cross-Validation** — Statistically grounded confidence intervals on all metrics
3. **Federated Learning** — Multi-institution training without sharing raw traffic data
4. **Real-Time PCAP Integration** — Direct Scapy/DPDK stream ingestion at line rate
5. **NIST PQC Signature Updates** — Threat taxonomy updated for FIPS 203/204/205 (ML-KEM, ML-DSA, SLH-DSA)
6. **Student-Facing Companion** — Browser extension that surfaces plain-English threat explanations at the moment of risk

---

## License

Submitted as a prototype for the AMD AI Cybersecurity Hackathon. Research use only.
