from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import numpy as np
import os
from api.model import BehavioralTransformerEncoder

app = FastAPI(title="Keystroke Authentication API")

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cpu")
model  = BehavioralTransformerEncoder(
    input_size=3, d_model=64, nhead=4,
    num_layers=3, dim_feedforward=256,
    dropout=0.1, num_classes=51
).to(device)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bte_best.pt")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ── Load normalization parameters ─────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEAT_MEAN  = np.load(os.path.join(DATA_DIR, "feature_mean.npy"))
FEAT_STD   = np.load(os.path.join(DATA_DIR, "feature_std.npy"))

# ── Known subjects ────────────────────────────────────────────────────────────
SUBJECTS = [
    "s002","s003","s004","s005","s007","s008","s010","s011","s012","s013",
    "s015","s016","s017","s018","s019","s020","s021","s022","s024","s025",
    "s026","s027","s028","s029","s030","s031","s032","s033","s034","s035",
    "s036","s037","s038","s039","s040","s041","s042","s043","s044","s046",
    "s047","s048","s049","s050","s051","s052","s053","s054","s055","s056","s057"
]

# ── Request / Response models ─────────────────────────────────────────────────
class KeystrokeSequence(BaseModel):
    hold_times:    list[float]
    dd_times:      list[float]
    ud_times:      list[float]

class AuthRequest(BaseModel):
    subject_id:  str
    sequence:    KeystrokeSequence

class AuthResponse(BaseModel):
    subject_id:   str
    decision:     str
    confidence:   float
    threshold:    float
    score:        float

# ── Helper: preprocess raw timing into normalized (11,3) tensor ───────────────
def preprocess(seq: KeystrokeSequence) -> torch.Tensor:
    h  = seq.hold_times[:10]  + [seq.hold_times[-1] if len(seq.hold_times) > 10 else 0.0]
    dd = seq.dd_times[:10]    + [0.0]
    ud = seq.ud_times[:10]    + [0.0]

    arr = np.array([[h[i], dd[i], ud[i]] for i in range(11)], dtype=np.float32)
    arr = np.clip(arr, None, 2.0)
    arr = (arr - FEAT_MEAN.squeeze()) / (FEAT_STD.squeeze() + 1e-8)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok", "model": "BTE", "subjects": len(SUBJECTS)}

@app.get("/subjects")
async def list_subjects():
    return {"subjects": SUBJECTS}

@app.post("/authenticate", response_model=AuthResponse)
async def authenticate(request: AuthRequest):
    if request.subject_id not in SUBJECTS:
        raise HTTPException(
            status_code=404,
            detail=f"Subject {request.subject_id} not found. "
                   f"Valid subjects: {SUBJECTS}"
        )

    subject_idx = SUBJECTS.index(request.subject_id)
    tensor      = preprocess(request.sequence).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        score  = probs[0, subject_idx].item()

    # Threshold calibrated to EER operating point
    THRESHOLD = 0.0054
    decision  = "ACCEPTED" if score >= THRESHOLD else "REJECTED"
    confidence = score if decision == "ACCEPTED" else 1.0 - score

    return AuthResponse(
        subject_id=request.subject_id,
        decision=decision,
        confidence=round(confidence * 100, 2),
        threshold=THRESHOLD,
        score=round(score, 6)
    )

# ── Mount static files ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")