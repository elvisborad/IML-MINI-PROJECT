"""Small upload-and-predict web app for Render or local use."""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import os
import tempfile
import urllib.request

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
from torchvision import transforms

from .predict_video import load_checkpoint
from .video_processor import extract_frames


ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "index.html"
CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", str(ROOT / "results" / "models" / "best_model.pt")))
MODEL_URL = os.getenv("MODEL_URL", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
ALLOWED_EXTENSIONS = {".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".mkv", ".webm"}

MODEL = None
CHECKPOINT = None
TRANSFORM = None
MODEL_ERROR = "Model checkpoint is not configured. Set MODEL_URL to a trusted checkpoint download URL."
INFERENCE_LOCK = asyncio.Semaphore(1)


def resolve_checkpoint() -> Path:
    if CHECKPOINT_PATH.is_file():
        return CHECKPOINT_PATH
    if not MODEL_URL:
        raise FileNotFoundError(MODEL_ERROR)
    if not MODEL_URL.startswith("https://"):
        raise ValueError("MODEL_URL must use HTTPS and point to a trusted checkpoint.")

    target = Path(tempfile.gettempdir()) / "ai_video_detection_best_model.pt"
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AI-Video-Detector/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        limit = 500 * 1024 * 1024
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                target.unlink(missing_ok=True)
                raise ValueError("Checkpoint download exceeds the 500 MB limit.")
            output.write(chunk)
    return target


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, CHECKPOINT, TRANSFORM, MODEL_ERROR
    try:
        checkpoint_path = resolve_checkpoint()
        MODEL, CHECKPOINT = load_checkpoint(str(checkpoint_path), "cpu")
        size = CHECKPOINT["image_size"]
        norm = CHECKPOINT["normalization"]
        TRANSFORM = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ])
        MODEL_ERROR = ""
    except Exception as exc:
        MODEL_ERROR = str(exc) if isinstance(exc, (FileNotFoundError, ValueError)) else "Could not load the model checkpoint. Check Render logs and MODEL_URL."
    yield


app = FastAPI(title="AI Video Detection", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PAGE)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_error": MODEL_ERROR or None}


def analyze_video(video_path: Path):
    if MODEL is None or CHECKPOINT is None or TRANSFORM is None:
        raise RuntimeError(MODEL_ERROR or "Model is not available.")

    with tempfile.TemporaryDirectory(prefix="video_frames_") as frame_dir:
        paths, error = extract_frames(
            video_path,
            Path(frame_dir),
            interval_seconds=1.0,
            max_frames=12,
            size=CHECKPOINT["image_size"],
        )
        if error:
            raise ValueError(error)

        batch = []
        for frame_path in paths:
            with Image.open(frame_path) as image:
                batch.append(TRANSFORM(image.convert("RGB")))
        scores = []
        with torch.inference_mode():
            for start in range(0, len(batch), 2):
                logits = MODEL(torch.stack(batch[start:start + 2]))
                scores.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

    score = sum(scores) / len(scores)
    threshold = CHECKPOINT.get("decision_thresholds", {}).get("mean", 0.5)
    return {
        "prediction": "AI_GENERATED" if score >= threshold else "REAL",
        "ai_generated_score": score,
        "decision_threshold": threshold,
        "frames_analyzed": len(scores),
        "aggregation": "mean",
    }


@app.post("/api/predict")
async def predict_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a supported video file (MP4, MPEG, MOV, AVI, MKV, or WEBM).")
    if MODEL is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model is not available.")

    limit = MAX_UPLOAD_MB * 1024 * 1024
    temp_path = None
    byte_count = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="video_upload_", suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := await file.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > limit:
                    raise HTTPException(status_code=413, detail=f"Video exceeds the {MAX_UPLOAD_MB} MB upload limit.")
                temp_file.write(chunk)
        async with INFERENCE_LOCK:
            result = await run_in_threadpool(analyze_video, temp_path)
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed. Check that the video is readable and try another file.") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()
