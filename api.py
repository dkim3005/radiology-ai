"""
Radiology AI — chest X-ray analysis platform.
FastAPI on port 10001.

Three workflows over the NIH ChestX-ray14 corpus, one shared DenseNet:
  Triage         — DenseNet121 multi-label classification + FHIR context
  Explainability — Class Activation Maps over the classifier
  Search         — CLIP ViT-B/32 semantic image search
"""

from __future__ import annotations

import base64
import io
import os
import threading
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image as PILImage
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm                                       # noqa: E402

HEALTH = Path("/home/dk/Project/health_informatics")
DATA_ENTRY_CSV = HEALTH / "data" / "Data_Entry_2017.csv"
DATA_IMAGES_DIR = HEALTH / "data"
MODEL_CACHE_DIR = HEALTH / "model_cache"
WEIGHTS = "densenet121-res224-nih"
SAMPLE_SIZE = 60

IMAGE_DIRS = [str(DATA_IMAGES_DIR / "images_001" / "images"),
              str(DATA_IMAGES_DIR / "images_002" / "images")]
MAX_IMAGES = 40
CACHE_FILE = Path(__file__).parent / "embeddings_cache.npz"

MEDICAL_TAGS = ["chest x-ray", "cardiomegaly", "pleural effusion", "pneumonia",
                "emphysema", "nodule", "infiltration", "atelectasis",
                "no finding", "consolidation", "edema", "pneumothorax",
                "mass", "fibrosis"]


def _build_corpus_bg():
    try:
        _build_corpus()
    except Exception as e:                       # noqa: BLE001
        print(f"[clip] corpus build failed: {e}")


from contextlib import asynccontextmanager                       # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_build_corpus_bg, daemon=True).start()
    yield


app = FastAPI(title="Radiology AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

# ── Studies / patient data ────────────────────────────────────────────────────

_FIRST = ["Avery", "Jordan", "Morgan", "Riley", "Casey", "Taylor",
          "Cameron", "Quinn", "Hayden", "Rowan", "Reese", "Parker"]
_LAST = ["Kim", "Patel", "Nguyen", "Garcia", "Smith", "Johnson",
         "Brown", "Davis", "Miller", "Wilson", "Anderson", "Thomas"]


def _synth_name(i: int) -> str:
    return f"{_FIRST[i % len(_FIRST)]} {_LAST[(i * 3) % len(_LAST)]}"


@lru_cache(maxsize=1)
def _image_inventory() -> dict[str, str]:
    paths: dict[str, str] = {}
    for root, _, files in os.walk(DATA_IMAGES_DIR):
        if os.path.basename(root) != "images":
            continue
        for f in files:
            if f.lower().endswith(".png"):
                paths[f] = os.path.join(root, f)
    return paths


@lru_cache(maxsize=1)
def _load_studies() -> list[dict]:
    import pandas as pd
    inv = _image_inventory()
    df = pd.read_csv(DATA_ENTRY_CSV)
    df = df[df["Image Index"].isin(inv)].copy()
    df = (df.sort_values(["Patient ID", "Follow-up #", "Image Index"])
            .head(SAMPLE_SIZE).reset_index(drop=True))
    base = date(2026, 5, 1)
    rows = []
    for i, row in df.iterrows():
        rows.append({
            "id": f"FHIR-XR-{i + 1:04d}",
            "mrn": f"MRN-{700000 + i:06d}",
            "patient_name": _synth_name(i),
            "patient_age": int(row["Patient Age"]),
            "patient_gender": row["Patient Gender"],
            "view_position": row["View Position"],
            "study_date": (base - timedelta(days=i % 21)).isoformat(),
            "image_index": row["Image Index"],
            "finding_labels": row["Finding Labels"],
            "fhir_patient": f"Patient/FHIR-XR-{i + 1:04d}",
            "fhir_observation": f"Observation/FHIR-XR-{i + 1:04d}",
            "status": "Ready for AI review",
        })
    return rows


# ── Shared DenseNet ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_densenet():
    import torchxrayvision as xrv
    import torch.nn.functional as F
    import types

    m = xrv.models.DenseNet(weights=WEIGHTS, cache_dir=str(MODEL_CACHE_DIR))
    m.eval()

    # patch features2 to a non-inplace ReLU (safe for CAM + classification)
    def _features2_safe(self, x):
        if hasattr(self, "input_resolution"):
            x = xrv.utils.fix_resolution(x, self.input_resolution, self)
            xrv.utils.warn_normalization(x)
        feats = self.features(x)
        out = F.relu(feats, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(feats.size(0), -1)
        return out

    m.features2 = types.MethodType(_features2_safe, m)
    return m


def _infer(model, tensor):
    import torch
    with torch.inference_mode():
        feat_map = model.features(tensor)
        relu_feat = torch.relu(feat_map)
        output = model(tensor)
    return output, relu_feat


def _confidence(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.55:
        return "Moderate"
    return "Low"


# ── Triage endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "radiology-ai"}


@app.get("/api/studies")
def list_studies():
    return _load_studies()


@app.get("/api/studies/{study_id}")
def get_study(study_id: str):
    for s in _load_studies():
        if s["id"] == study_id:
            return s
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/images/{image_name}")
def serve_image(image_name: str):
    path = _image_inventory().get(image_name)
    if not path:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/studies/{study_id}/infer")
def infer_study(study_id: str):
    import torch
    import torchxrayvision as xrv

    study = next((s for s in _load_studies() if s["id"] == study_id), None)
    if not study:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = _image_inventory().get(study["image_index"])
    if not path:
        return JSONResponse({"error": "image not found"}, status_code=404)

    model = _load_densenet()
    image = xrv.utils.load_image(path)
    tensor = torch.from_numpy(image).unsqueeze(0).float()
    output, _ = _infer(model, tensor)
    preds = output[0].numpy()

    findings = []
    for label, score in zip(model.pathologies, preds):
        if label and not np.isnan(score):
            findings.append({"finding": label, "score": round(float(score), 4),
                              "confidence": _confidence(float(score))})
    findings.sort(key=lambda x: x["score"], reverse=True)
    primary = findings[0] if findings else {}
    supporting = [f for f in findings[1:] if f["score"] >= 0.55] or findings[1:4]
    return {"primary": primary, "supporting": supporting, "all": findings}


@app.get("/api/studies/{study_id}/fhir")
def fhir_template(study_id: str):
    study = next((s for s in _load_studies() if s["id"] == study_id), None)
    if not study:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"resources": [
        {"resource": "Patient", "fhir_id": study["fhir_patient"],
         "content": study["patient_name"]},
        {"resource": "Observation", "fhir_id": study["fhir_observation"],
         "content": "Chest X-ray imaging result"},
        {"resource": "Media", "fhir_id": f"DocumentReference/{study['id']}",
         "content": study["image_index"]},
        {"resource": "Reference Label", "fhir_id": "NIH ChestX-ray14",
         "content": study["finding_labels"]},
    ]}


# ── Explainability — Class Activation Map ─────────────────────────────────────

@app.get("/api/studies/{study_id}/explain")
def explain_study(study_id: str):
    import torch
    import torchxrayvision as xrv

    study = next((s for s in _load_studies() if s["id"] == study_id), None)
    if not study:
        raise HTTPException(status_code=404, detail="study not found")
    path = _image_inventory().get(study["image_index"])
    if not path:
        raise HTTPException(status_code=404, detail="image not found")

    model = _load_densenet()
    image_arr = xrv.utils.load_image(path)
    tensor = torch.from_numpy(image_arr).unsqueeze(0).float()
    output, relu_feat = _infer(model, tensor)
    scores = output[0].numpy()
    labels = model.pathologies

    preds = [{"finding": labels[i], "score": float(scores[i])}
             for i in range(len(labels)) if labels[i]]
    preds.sort(key=lambda x: x["score"], reverse=True)
    top5 = preds[:5]
    top_idx = list(labels).index(top5[0]["finding"])

    # CAM = classifier weights · feature map
    weights = model.classifier.weight[top_idx].detach()
    cam = torch.einsum("f,fhw->hw", weights, relu_feat.squeeze(0).detach())
    cam = torch.relu(cam)
    mn, mx = cam.min(), cam.max()
    cam_np = ((cam - mn) / (mx - mn + 1e-8)).numpy()
    cam_resized = np.array(
        PILImage.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224)))

    heatmap = cm.get_cmap("hot")(cam_resized / 255.0)[:, :, :3]
    norm = (image_arr[0] - image_arr[0].min()) / (
        image_arr[0].max() - image_arr[0].min() + 1e-8)
    rgb = np.stack([norm, norm, norm], axis=-1)
    rgb = np.array(PILImage.fromarray((rgb * 255).astype(np.uint8))
                   .resize((224, 224))) / 255.0
    overlay = np.clip(0.55 * rgb + 0.45 * heatmap, 0, 1)
    buf = io.BytesIO()
    PILImage.fromarray((overlay * 255).astype(np.uint8)).save(buf, format="PNG")

    return {
        "study_id": study_id,
        "top_class": top5[0]["finding"],
        "predictions": top5,
        "cam_b64": base64.b64encode(buf.getvalue()).decode(),
        "explanation": (
            f"The model focused on regions consistent with "
            f"{top5[0]['finding']}. The CAM heatmap highlights the areas that "
            f"most influenced this prediction (score {top5[0]['score']:.3f}). "
            f"Warmer colours indicate higher attention."),
    }


# ── CLIP semantic search ──────────────────────────────────────────────────────

_clip = None
_preprocess = None
_tokenizer = None
_corpus: list[dict] | None = None
_corpus_matrix: np.ndarray | None = None
_ready = threading.Event()


def _load_clip():
    global _clip, _preprocess, _tokenizer
    if _clip is not None:
        return
    import open_clip
    _clip, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _clip.eval()


def _collect_image_paths() -> list[Path]:
    paths: list[Path] = []
    for d in IMAGE_DIRS:
        p = Path(d)
        if p.exists():
            paths.extend(sorted(p.glob("*.png")))
    return sorted(paths, key=lambda x: x.name)[:MAX_IMAGES]


def _build_corpus():
    global _corpus, _corpus_matrix
    import torch
    paths = _collect_image_paths()
    if CACHE_FILE.exists():
        try:
            data = np.load(CACHE_FILE, allow_pickle=True)
            filenames = data["filenames"].tolist()
            path_map = {p.name: str(p) for p in paths}
            records = [{"filename": fn, "path": path_map[fn]}
                       for fn in filenames if fn in path_map]
            idx = [i for i, fn in enumerate(filenames) if fn in path_map]
            if records:
                _corpus = records
                _corpus_matrix = data["matrix"][idx].astype(np.float32)
                _ready.set()
                print(f"[clip] loaded {len(_corpus)} embeddings from cache")
                return
        except Exception as e:                   # noqa: BLE001
            print(f"[clip] cache load failed ({e}), rebuilding")

    _load_clip()
    embeddings, records = [], []
    for path in paths:
        try:
            img = PILImage.open(path).convert("RGB")
            tensor = _preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feat = _clip.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embeddings.append(feat.cpu().numpy()[0])
            records.append({"filename": path.name, "path": str(path)})
        except Exception as e:                   # noqa: BLE001
            print(f"[clip] skip {path.name}: {e}")
    _corpus = records
    _corpus_matrix = np.stack(embeddings).astype(np.float32)
    np.savez(CACHE_FILE, filenames=[r["filename"] for r in records],
             matrix=_corpus_matrix)
    _ready.set()
    print(f"[clip] indexed {len(_corpus)} images")


def _ensure_corpus():
    if not _ready.wait(timeout=120):
        raise HTTPException(status_code=503, detail="search index warming up")


def _embed_text(text: str) -> np.ndarray:
    import torch
    _load_clip()
    with torch.no_grad():
        feat = _clip.encode_text(_tokenizer([text]))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def _embed_image(img: PILImage.Image) -> np.ndarray:
    import torch
    _load_clip()
    tensor = _preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = _clip.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def _top_k(query: np.ndarray, k: int) -> list[dict]:
    q = query / (np.linalg.norm(query) + 1e-9)
    norms = np.linalg.norm(_corpus_matrix, axis=1, keepdims=True) + 1e-9
    sims = (_corpus_matrix / norms) @ q
    order = np.argsort(sims)[::-1][:k]
    return [{"filename": _corpus[i]["filename"], "score": float(sims[i]),
             "rank": r + 1} for r, i in enumerate(order)]


class TextSearch(BaseModel):
    query: str
    top_k: int = 8


@app.get("/api/search/status")
def search_status():
    return {"ready": _ready.is_set(),
            "indexed": len(_corpus) if _corpus else 0}


@app.post("/api/search/text")
def search_text(req: TextSearch):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    _ensure_corpus()
    k = max(1, min(req.top_k, len(_corpus)))
    return _top_k(_embed_text(req.query), k)


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 8):
    _ensure_corpus()
    data = await file.read()
    try:
        img = PILImage.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    k = max(1, min(top_k, len(_corpus)))
    return _top_k(_embed_image(img), k)


@app.get("/api/search/tags/{filename}")
def search_tags(filename: str):
    _ensure_corpus()
    vec = None
    for i, r in enumerate(_corpus):
        if r["filename"] == filename:
            vec = _corpus_matrix[i]
            break
    if vec is None:
        raise HTTPException(status_code=404, detail="image not in index")
    vn = vec / (np.linalg.norm(vec) + 1e-9)
    scored = []
    for tag in MEDICAL_TAGS:
        tv = _embed_text(tag)
        scored.append({"tag": tag,
                       "score": float(np.dot(vn, tv / (np.linalg.norm(tv) + 1e-9)))})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10001, reload=False)
