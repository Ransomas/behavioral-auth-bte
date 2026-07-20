from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(title="Keystroke Authentication API")

# ── Load ONNX model ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bte_model.onnx")
session    = ort.InferenceSession(MODEL_PATH)

# ── Load normalization parameters ─────────────────────────────────────────────
API_DIR   = os.path.dirname(__file__)
FEAT_MEAN = np.load(os.path.join(API_DIR, "feature_mean.npy"))
FEAT_STD  = np.load(os.path.join(API_DIR, "feature_std.npy"))

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
    hold_times: list[float]
    dd_times:   list[float]
    ud_times:   list[float]

class AuthRequest(BaseModel):
    subject_id: str
    sequence:   KeystrokeSequence

class AuthResponse(BaseModel):
    subject_id:  str
    decision:    str
    confidence:  float
    threshold:   float
    score:       float

# ── Helper: preprocess raw timing into normalized (1, 11, 3) array ────────────
def preprocess(seq: KeystrokeSequence) -> np.ndarray:
    h  = list(seq.hold_times[:10]) + [seq.hold_times[-1] if len(seq.hold_times) > 10 else 0.0]
    dd = list(seq.dd_times[:10])   + [0.0]
    ud = list(seq.ud_times[:10])   + [0.0]

    arr = np.array([[h[i], dd[i], ud[i]] for i in range(11)], dtype=np.float32)
    arr = np.clip(arr, None, 2.0)
    arr = (arr - FEAT_MEAN.squeeze()) / (FEAT_STD.squeeze() + 1e-8)
    return arr.reshape(1, 11, 3)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok", "model": "BTE-ONNX", "subjects": len(SUBJECTS)}

@app.get("/subjects")
async def list_subjects():
    return {"subjects": SUBJECTS}

@app.post("/authenticate", response_model=AuthResponse)
async def authenticate(request: AuthRequest):
    if request.subject_id not in SUBJECTS:
        raise HTTPException(
            status_code=404,
            detail=f"Subject {request.subject_id} not found."
        )

    subject_idx = SUBJECTS.index(request.subject_id)
    arr         = preprocess(request.sequence)

    logits = session.run(["output"], {"input": arr})[0]

    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs      = exp_logits / exp_logits.sum()
    score      = float(probs[0, subject_idx])

    THRESHOLD  = 0.0054
    decision   = "ACCEPTED" if score >= THRESHOLD else "REJECTED"
    confidence = score if decision == "ACCEPTED" else 1.0 - score

    return AuthResponse(
        subject_id=request.subject_id,
        decision=decision,
        confidence=round(confidence * 100, 2),
        threshold=THRESHOLD,
        score=round(score, 6)
    )

# ── Mount static files ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(
    directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static"
)