# SafeGuard PPE Detection — upgraded image + video version

This version upgrades the original image-only FastAPI + Streamlit project to support:

- Image upload and annotated PPE detection (JPG/JPEG/PNG)
- Video upload and frame-by-frame annotated MP4 output (MP4/MOV/AVI/MKV/M4V)
- Live webcam PPE detection with on-screen annotated camera feed (browser camera permission required)
- Confidence and inference-size controls
- Detection summaries, class breakdowns, violation-class counts, and local inspection history
- Automatic download/cache of a pretrained PPE model on the first backend startup

## Important model correction

The original `backend/models/ppe_yolov8.pt` in the supplied project had COCO-style classes such as `person`, `car`, and `bicycle`, rather than PPE-specific classes. This upgraded version does not use that checkpoint.

It uses **Hexmon/vyra-yolo-ppe-detection** from Hugging Face (`best.pt`), a YOLOv8-medium PPE detector with 14 classes: `Fall-Detected`, `Gloves`, `Goggles`, `Hardhat`, `Ladder`, `Mask`, `NO-Gloves`, `NO-Goggles`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, and `Safety Vest`.

Model card: https://huggingface.co/Hexmon/vyra-yolo-ppe-detection
License listed by the model repository: CC BY 4.0. Give attribution to the model creator/repository when redistributing or publishing outputs that incorporate the model. Review the model card and license terms before commercial deployment.

**Accuracy note:** A pretrained detector is not guaranteed to be accurate on every camera, distance, lighting condition, PPE type, or site. The app reports model detections, not certified workplace compliance. In particular, video class counts are detection occurrences across frames, not unique workers.

## Requirements

- Python 3.10 or 3.11 recommended
- Internet access on first backend startup (to download model weights, about 52 MB)
- 8 GB RAM can run CPU inference, but video processing may be slow. An NVIDIA CUDA GPU is recommended for longer videos.

## Windows setup (recommended)

Open two terminals in the `SafeGuard_PPE_Upgraded` folder.

### Terminal 1 — Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

On first start, the backend downloads and loads the pretrained PPE model. Wait until the terminal prints `SafeGuard loaded pretrained PPE model`.

Backend health: http://127.0.0.1:8001/health  
Swagger API docs: http://127.0.0.1:8001/docs

### Terminal 2 — Frontend

```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (normally http://localhost:8501).

Alternatively, run `run_backend.bat` and `run_frontend.bat` in separate terminals.

## API endpoints

- `GET /health` — backend and model status
- `POST /detect/frame?conf=0.25&imgsz=416` — accepts one webcam frame and returns an annotated JPEG
- `POST /detect/image?conf=0.25&imgsz=640` — returns JSON with annotated image (base64), detection boxes, confidence, and summary
- `POST /detect/video?conf=0.25&imgsz=640` — returns an annotated MP4; summary is in the `X-SafeGuard-Summary` response header (URL-safe base64 encoded JSON)

## Settings

Environment variables (optional):

- `PPE_MODEL_REPO` (default `Hexmon/vyra-yolo-ppe-detection`)
- `PPE_MODEL_FILENAME` (default `best.pt`)
- `PPE_DEVICE` (default CUDA GPU 0 if available, otherwise CPU)
- `MAX_UPLOAD_MB` (default `250`)
- `MAX_VIDEO_SECONDS` (default `300`)
- `SAFEGUARD_API_URL` (default `http://127.0.0.1:8001`)

## Known limitations in this first upgrade

- Video is processed sequentially, frame by frame; long videos can take time on CPU.
- Video counts are frame-level detections, not persistent worker IDs. Tracking and per-worker compliance association are a separate next phase.
- Live webcam inference is throttled to about 4 frames per second to reduce CPU/API load. Actual speed depends on your CPU/GPU and camera resolution.
- Webcam access works on localhost in supported browsers. If you access Streamlit from another device or non-localhost address, browsers generally require HTTPS for camera permissions.
- Detection of missing equipment depends on the model's explicit `NO-*` classes and visible body regions. It must not be interpreted as a definitive safety audit.
- The current frontend remains Streamlit to preserve the existing app and make this upgrade straightforward. A React/Tailwind redesign can be done as a separate frontend migration after image/video workflows are validated.

## Folder structure

```text
SafeGuard_PPE_Upgraded/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── data/history/
├── .env.example
├── run_backend.bat
├── run_frontend.bat
└── README.md
```
