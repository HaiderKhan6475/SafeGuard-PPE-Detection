from __future__ import annotations

import base64
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Allow the trusted Ultralytics model class in PyTorch 2.6+
torch.serialization.add_safe_globals([DetectionModel])

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_REPO = os.getenv("PPE_MODEL_REPO", "Hexmon/vyra-yolo-ppe-detection")
MODEL_FILENAME = os.getenv("PPE_MODEL_FILENAME", "best.pt")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "250"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "300"))
DEVICE = os.getenv("PPE_DEVICE", "0" if __import__("torch").cuda.is_available() else "cpu")

app = FastAPI(
    title="SafeGuard PPE Detection API",
    description="Image and video PPE detection using a pretrained YOLOv8 PPE model.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: YOLO | None = None

VIOLATION_KEYS = ("no-hardhat", "no-mask", "no-safety-vest", "no-gloves", "no-goggles")


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def class_name(class_id: int) -> str:
    assert model is not None
    names = model.names
    return str(names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, class_id))


def get_model() -> YOLO:
    if model is None:
        raise HTTPException(status_code=503, detail="PPE model is still loading. Try again shortly.")
    return model


@app.on_event("startup")
def load_model() -> None:
    global model
    try:
        weights = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, cache_dir=str(MODEL_DIR / "hf_cache"))
        loaded = YOLO(weights)
        names = loaded.names
        values = list(names.values()) if isinstance(names, dict) else list(names)
        normalized = [normalize_name(n) for n in values]
        if not any("hardhat" in n or "helmet" in n for n in normalized):
            raise RuntimeError(f"Downloaded model does not appear to contain PPE classes: {values}")
        model = loaded
        print(f"SafeGuard loaded pretrained PPE model: {MODEL_REPO}/{MODEL_FILENAME}")
        print(f"Classes: {values}")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the pretrained PPE model. Check internet access and model dependencies. "
            f"Details: {exc}"
        ) from exc


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "SafeGuard PPE Detection API", "status": "active", "version": "2.0.0"}


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = model is not None
    names = model.names if loaded else {}
    return {"status": "ready" if loaded else "loading", "model_loaded": loaded, "model_repo": MODEL_REPO,
            "device": DEVICE, "classes": list(names.values()) if isinstance(names, dict) else list(names)}


def read_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(__import__("io").BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


def run_detection(frame: np.ndarray, confidence: float, imgsz: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    detector = get_model()
    results = detector.predict(frame, conf=confidence, imgsz=imgsz, device=DEVICE, verbose=False)
    result = results[0]
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = class_name(class_id)
            normalized = normalize_name(name)
            confidence_score = float(box.conf.item())
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id": class_id,
                "class_name": name,
                "confidence": round(confidence_score, 4),
                "bbox": [x1, y1, x2, y2],
                "type": "violation" if any(key in normalized for key in VIOLATION_KEYS) else "detection",
            })
    annotated = result.plot()
    return annotated, detections


def summarize(detections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["class_name"] for item in detections)
    violations = [item for item in detections if item["type"] == "violation"]
    people = sum(1 for item in detections if normalize_name(item["class_name"]) == "person")
    return {
        "total_detections": len(detections),
        "people_detected": people,
        "violation_detections": len(violations),
        "class_counts": dict(counts),
        "violations": violations,
    }


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
    return data


@app.post("/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    imgsz: int = Query(640, ge=320, le=1280),
) -> Response:
    """Run PPE detection on one webcam frame and return the annotated JPEG."""
    data = await read_upload(file)
    frame_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode webcam frame.")
    annotated, _ = run_detection(frame, conf, imgsz)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated webcam frame.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    imgsz: int = Query(640, ge=320, le=1280),
) -> dict[str, Any]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a JPG, JPEG, or PNG image.")
    data = await read_upload(file)
    frame = read_image(data)
    annotated, detections = run_detection(frame, conf, imgsz)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")
    return {
        "filename": file.filename,
        "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "confidence_threshold": conf,
        "summary": summarize(detections),
        "detections": detections,
    }


@app.post("/detect/video")
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    imgsz: int = Query(640, ge=320, le=1280),
) -> FileResponse:
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(status_code=415, detail="Supported video formats: MP4, MOV, AVI, MKV, M4V.")
    data = await read_upload(file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_tmp:
        input_tmp.write(data)
        input_path = input_tmp.name
    output_path = str(Path(input_path).with_name(Path(input_path).stem + "_safeguard.mp4"))
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        Path(input_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read this video. Try a standard MP4 file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 120 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0
    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        Path(input_path).unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"Video is {duration:.1f}s; maximum allowed is {MAX_VIDEO_SECONDS}s.")
    if width < 2 or height < 2:
        cap.release()
        Path(input_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Video has invalid dimensions.")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        Path(input_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not create processed MP4 video.")

    frame_counter = 0
    aggregate: Counter[str] = Counter()
    violation_counter: Counter[str] = Counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, detections = run_detection(frame, conf, imgsz)
            writer.write(annotated)
            frame_counter += 1
            aggregate.update(item["class_name"] for item in detections)
            violation_counter.update(item["class_name"] for item in detections if item["type"] == "violation")
    except Exception:
        cap.release()
        writer.release()
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        writer.release()

    Path(input_path).unlink(missing_ok=True)
    if frame_counter == 0 or not Path(output_path).exists():
        Path(output_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No readable frames were found in this video.")

    summary = {
        "frames_processed": frame_counter,
        "video_duration_seconds": round(frame_counter / fps, 2),
        "fps": round(fps, 2),
        "class_detection_occurrences": dict(aggregate),
        "violation_occurrences": dict(violation_counter),
        "note": "Counts are detection occurrences across frames, not unique workers.",
    }
    encoded_summary = base64.urlsafe_b64encode(json.dumps(summary).encode()).decode()
    background_tasks.add_task(Path(output_path).unlink, missing_ok=True)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{Path(file.filename or 'video').stem}_safeguard.mp4",
        background=background_tasks,
        headers={"X-SafeGuard-Summary": encoded_summary},
    )
