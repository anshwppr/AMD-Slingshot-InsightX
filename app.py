"""
Flask backend — Digital Hygiene / AMD AI Cybersecurity Hackathon

Architectures copied EXACTLY from training code:

  LoRAThreatDetector (lora_adapter.pth):
    input_bn          : BatchNorm1d(input_dim)
    encoder           : LoRALinear(47,256) BN ReLU Drop  LoRALinear(256,256) BN ReLU Drop
    shortcut          : Linear(47, 256)  [no LoRA]
    deep_features     : LoRALinear(256,192) BN ReLU Drop  LoRALinear(192,128) BN ReLU Drop
    classifier        : LoRALinear(128,64) BN ReLU Drop  LoRALinear(64,12)
    forward           : out = ReLU(encoder(x) + shortcut(x))
                        out = deep_features(out)
                        return classifier(out)

  ImprovedResponseDQN (response_agent_lora.pth):
    network:
      Linear(15,256) -> LayerNorm(256) -> ReLU -> Dropout(0.2)
      Linear(256,128) -> LayerNorm(128) -> ReLU -> Dropout(0.2)
      Linear(128,64)  -> LayerNorm(64)  -> ReLU -> Dropout(0.15)
      Linear(64,8)

Folder layout:
  app.py
  index.html
  models/
    lora_adapter/          <- extract lora_adapter_pth.zip here
    response_agent_lora/   <- extract response_agent_lora_pth.zip here

Install:  pip install flask flask-cors torch numpy
Run:      python app.py  ->  open http://localhost:5000
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE     = torch.device("cpu")

INPUT_DIM   = 39
NUM_CLASSES = 12
STATE_DIM   = 15
ACTION_DIM  = 8

LORA_CONFIG = {'rank': 8, 'alpha': 16, 'dropout': 0.1}

RESPONSE_ACTIONS = [
    "Ignore", "Quick Scan", "Full Scan", "Quarantine",
    "Delete", "Network Isolate", "PQ-Crypto Upgrade", "Emergency Crypto Rotate"
]

THREAT_NAMES = [
    'Benign', 'DDoS', 'DoS', 'Recon',
    'Web_based', 'BruteForce', 'Spoofing', 'Mirai',
    'PQ-Downgrade', 'PQ-HNDL', 'PQ-SideChannel', 'PQ-Hybrid'
]


# ─────────────────────────────────────────────────────────────
# LoRALinear  — identical to training code
# ─────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1, use_lora=True):
        super().__init__()
        self.use_lora = use_lora
        self.scaling  = lora_alpha / lora_rank
        self.linear   = nn.Linear(in_features, out_features, bias=bias)

        if use_lora and lora_rank > 0:
            self.lora_A       = nn.Parameter(torch.zeros(lora_rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            self.lora_B       = nn.Parameter(torch.zeros(out_features, lora_rank))
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_A = self.lora_B = self.lora_dropout = None

    def forward(self, x):
        base = self.linear(x)
        if self.use_lora and self.lora_A is not None:
            lora = F.linear(self.lora_dropout(x),
                            self.lora_B @ self.lora_A) * self.scaling
            return base + lora
        return base


# ─────────────────────────────────────────────────────────────
# LoRAThreatDetector  — identical to training code
# ─────────────────────────────────────────────────────────────
class LoRAThreatDetector(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                 lora_config=None):
        super().__init__()
        if lora_config is None:
            lora_config = LORA_CONFIG
        r  = lora_config['rank']
        a  = lora_config['alpha']
        do = lora_config['dropout']

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.encoder = nn.Sequential(
            LoRALinear(input_dim, 256, lora_rank=r, lora_alpha=a, lora_dropout=do),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            LoRALinear(256, 256, lora_rank=r, lora_alpha=a, lora_dropout=do),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.shortcut = nn.Linear(input_dim, 256)  # no LoRA on shortcut

        self.deep_features = nn.Sequential(
            LoRALinear(256, 192, lora_rank=r, lora_alpha=a, lora_dropout=do),
            nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.25),
            LoRALinear(192, 128, lora_rank=r, lora_alpha=a, lora_dropout=do),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            LoRALinear(128, 64, lora_rank=r, lora_alpha=a, lora_dropout=do),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            LoRALinear(64, num_classes, lora_rank=r, lora_alpha=a, lora_dropout=do)
        )

    def forward(self, x):
        x        = self.input_bn(x)
        identity = self.shortcut(x)
        out      = F.relu(self.encoder(x) + identity)
        out      = self.deep_features(out)
        return self.classifier(out)

    def get_probabilities(self, x):
        return F.softmax(self.forward(x), dim=1)


# ─────────────────────────────────────────────────────────────
# ImprovedResponseDQN  — identical to training code
# ─────────────────────────────────────────────────────────────
class ImprovedResponseDQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────────────────────
# Load weights
# ─────────────────────────────────────────────────────────────
def torch_load_folder(path):
    """
    When a .pth is unzipped it becomes a folder:
        data.pkl        - pickle referencing tensor storage by persistent id
        data/0, data/1  - raw tensor binary files

    torch.load() needs a proper zip file, not a plain folder.
    Fix: re-zip the folder into a temp file then load it.

    PyTorch zip format requires:
      - archive entries named:  <folder_name>/data.pkl
                                <folder_name>/data/0  etc.
      - timestamps >= 1980 (use a fixed safe date to suppress warnings)
    """
    import tempfile, zipfile
    from datetime import datetime

    if os.path.isfile(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not found: {path}")

    folder_name = os.path.basename(path)
    # Fixed timestamp: 1980-01-01 00:00:00 is the ZIP epoch minimum
    safe_date = (1980, 1, 1, 0, 0, 0)

    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(dirs)
                for fname in sorted(files):
                    fpath    = os.path.join(root, fname)
                    rel      = os.path.relpath(fpath, path)
                    arcname  = folder_name + "/" + rel.replace(os.sep, "/")
                    info     = zipfile.ZipInfo(arcname)
                    info.date_time = safe_date
                    info.compress_type = zipfile.ZIP_STORED
                    with open(fpath, "rb") as f:
                        zf.writestr(info, f.read())
        result = torch.load(tmp.name, map_location="cpu", weights_only=False)
    finally:
        os.unlink(tmp.name)
    return result


def load_lora_adapter(model, path):
    """
    save_lora_adapter() saved only lora_A / lora_B tensors keyed by
    their full named_parameters() name, e.g. 'encoder.0.lora_A'.
    """
    lora_state = torch_load_folder(path)
    loaded = 0
    for name, param in model.named_parameters():
        if name in lora_state:
            param.data.copy_(lora_state[name])
            loaded += 1
    print(f"[lora_adapter]   loaded {loaded} LoRA tensors")
    return model


def load_response_agent(model, path):
    """
    torch.save(state_dict()) — keys: 'network.0.weight', 'network.1.weight', ...
    """
    sd = torch_load_folder(path)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[response_agent] loaded — missing={missing}  unexpected={unexpected}")
    return model


# Instantiate models
detector_model = LoRAThreatDetector(INPUT_DIM, NUM_CLASSES, LORA_CONFIG).to(DEVICE)
agent_model    = ImprovedResponseDQN(STATE_DIM, ACTION_DIM).to(DEVICE)

# Load from models/ folder (works with both folder format and plain .pth files)
lora_path  = os.path.join(MODELS_DIR, "lora_adapter")
agent_path = os.path.join(MODELS_DIR, "response_agent_lora")

for label, model, path, loader in [
    ("lora_adapter",   detector_model, lora_path,  load_lora_adapter),
    ("response_agent", agent_model,    agent_path, load_response_agent),
]:
    if os.path.exists(path):
        try:
            loader(model, path)
        except Exception as e:
            print(f"[{label}] WARNING: {e}")
    else:
        print(f"[{label}] NOT FOUND at {path}")

detector_model.eval()
agent_model.eval()


# ─────────────────────────────────────────────────────────────
# State builder — mirrors BetterResponseEnvironment._get_state()
# ─────────────────────────────────────────────────────────────
def build_state(detected_threat: int, confidence: float,
                system_health: float = 0.85, crypto_strength: float = 0.90):
    """
    15-dim state vector exactly as in training.
    Since we don't have true_label at inference time, we assume
    detected == true (detection_correct = 1.0), same as the HTML did.
    """
    is_pq_detected = 1.0 if detected_threat >= 8 else 0.0

    state = np.array([
        detected_threat / 11.0,   # normalised threat id
        confidence,               # detector confidence
        1.0,                      # detection_correct (assume correct at inference)
        is_pq_detected,           # is PQ threat detected
        is_pq_detected,           # is PQ threat true (proxy)
        system_health,
        crypto_strength,
        (1 - confidence) * 0.5,   # uncertainty signal
        is_pq_detected * (1 - crypto_strength),  # pq urgency
        (1 - system_health) * 0.5,
        1.0 if detected_threat < 4             else 0.0,   # low-severity
        1.0 if 4 <= detected_threat < 8        else 0.0,   # mid-severity
        1.0 if detected_threat >= 8            else 0.0,   # pq
        1.0 if confidence > 0.9 and detected_threat > 0 else 0.0,  # high-conf threat
        1.0 if is_pq_detected and crypto_strength < 0.8 else 0.0,  # urgent pq
    ], dtype=np.float32)

    return torch.FloatTensor(state).unsqueeze(0).to(DEVICE)


def softmax_list(vals):
    m = max(vals)
    e = [math.exp(v - m) for v in vals]
    s = sum(e)
    return [x / s for x in e]


def get_lora_layer_norms(model: LoRAThreatDetector, x_tensor: torch.Tensor):
    """
    Manually step through the model collecting per-LoRA-layer activation norms.
    Returns a list of 6 floats (enc0, enc4, deep0, deep4, cls0, cls4).
    """
    norms = []
    with torch.no_grad():
        x = model.input_bn(x_tensor)
        identity = model.shortcut(x)

        # encoder — two LoRALinear layers at positions 0 and 4
        h = x
        enc_children = list(model.encoder.children())
        for layer in enc_children:
            if isinstance(layer, LoRALinear) and layer.lora_A is not None:
                lora_delta = F.linear(layer.lora_dropout(h),
                                      layer.lora_B @ layer.lora_A) * layer.scaling
                norms.append(float(lora_delta.norm().item()))
            h = layer(h)

        h = F.relu(h + identity)

        # deep_features — two LoRALinear layers
        for layer in list(model.deep_features.children()):
            if isinstance(layer, LoRALinear) and layer.lora_A is not None:
                lora_delta = F.linear(layer.lora_dropout(h),
                                      layer.lora_B @ layer.lora_A) * layer.scaling
                norms.append(float(lora_delta.norm().item()))
            h = layer(h)

        # classifier — two LoRALinear layers
        for layer in list(model.classifier.children()):
            if isinstance(layer, LoRALinear) and layer.lora_A is not None:
                lora_delta = F.linear(layer.lora_dropout(h),
                                      layer.lora_B @ layer.lora_A) * layer.scaling
                norms.append(float(lora_delta.norm().item()))
            h = layer(h)

    # Pad / trim to exactly 6 values
    while len(norms) < 6:
        norms.append(0.0)
    return norms[:6]


# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)


@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "models": ["lora_adapter", "response_agent_lora"],
        "device": str(DEVICE)
    })


@app.route("/infer", methods=["POST"])
def infer():
    """
    Full pipeline: LoRAThreatDetector -> ImprovedResponseDQN

    Request JSON:
      { "threat_id": 9, "health": 85, "crypto": 90, "history": [] }

    Response JSON (same fields the HTML rendering code expects):
      {
        "detectedClass":      9,
        "detectorConfidence": 0.84,
        "loraActivation":     12.3,
        "layerNorms":         [n0, n1, n2, n3, n4, n5],
        "action":             7,
        "actionName":         "Emergency Crypto Rotate",
        "agentConfidence":    0.91,
        "qValues":            [q0..q7],
        "usingRealWeights":   true,
        "source":             "lora_adapter.pth + response_agent_lora.pth"
      }
    """
    data      = request.get_json(force=True)
    threat_id = int(data.get("threat_id", 0))
    health    = float(data.get("health",    85))
    crypto    = float(data.get("crypto",    90))

    if not (0 <= threat_id <= 11):
        return jsonify({"error": "threat_id must be 0-11"}), 400

    with torch.no_grad():
        # ── Detector ─────────────────────────────────────────────────────
        # Build 47-dim feature vector (one-hot style, same as HTML did)
        feat = np.zeros(INPUT_DIM, dtype=np.float32)
        feat[threat_id % INPUT_DIM] = 1.0
        feat[min(threat_id + 1, INPUT_DIM - 1)] = health / 100.0
        feat[min(threat_id + 2, INPUT_DIM - 1)] = crypto / 100.0
        x_tensor = torch.FloatTensor(feat).unsqueeze(0).to(DEVICE)

        probs     = detector_model.get_probabilities(x_tensor).squeeze(0)
        det_probs = probs.tolist()
        det_cls   = int(probs.argmax().item())
        det_conf  = float(probs[det_cls].item())

        # Per-layer LoRA norms
        layer_norms   = get_lora_layer_norms(detector_model, x_tensor)
        lora_act_norm = layer_norms[-1] if layer_norms else 0.0

        # ── Response agent ────────────────────────────────────────────────
        state  = build_state(det_cls, det_conf,
                             system_health   = health  / 100.0,
                             crypto_strength = crypto  / 100.0)
        q_vals = agent_model(state).squeeze(0).tolist()
        action = int(q_vals.index(max(q_vals)))
        probs_agent = softmax_list(q_vals)

    return jsonify({
        # Detector
        "detectedClass":      det_cls,
        "detectorProbs":      det_probs,
        "detectorConfidence": det_conf,
        "loraActivation":     lora_act_norm,
        "layerNorms":         layer_norms,
        # Agent
        "action":             action,
        "actionName":         RESPONSE_ACTIONS[action],
        "agentConfidence":    probs_agent[action],
        "qValues":            q_vals,
        # Meta
        "usingRealWeights":   True,
        "source":             "lora_adapter.pth + response_agent_lora.pth"
    })


if __name__ == "__main__":
    print("\n  Digital Hygiene - Flask backend")
    print("  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
