from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = ROOT / "data" / "history" / "history.jsonl"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_API = os.getenv("SAFEGUARD_API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="SafeGuard | PPE Intelligence", page_icon="🦺", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --sg-bg:#0b1220; --sg-panel:#111c2e; --sg-line:#24344b; --sg-text:#e8eef7; --sg-muted:#91a2b8; --sg-accent:#45d6a5; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(circle at 75% -10%, #18324a 0, #0b1220 38%); color:var(--sg-text); }
[data-testid="stSidebar"] { background:#0d1727; border-right:1px solid #24344b; }
[data-testid="stSidebar"] * { color:#dce7f5; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.hero { padding:1.25rem 1.5rem; border:1px solid #2b4059; border-radius:18px; background:linear-gradient(115deg,rgba(25,48,70,.96),rgba(16,29,46,.88)); margin-bottom:1.2rem; }
.hero-kicker { color:#45d6a5; font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-family:'Space Grotesk'; font-size:2.15rem; font-weight:700; color:#f4f8ff; margin:.3rem 0; }
.hero-sub { color:#a9b9cc; font-size:.95rem; }
.metric { background:linear-gradient(145deg,#14243a,#101b2c); border:1px solid #2a3b52; border-radius:15px; padding:1rem 1.1rem; min-height:112px; }
.metric-label { color:#91a2b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#f4f8ff; font-family:'Space Grotesk'; font-size:1.9rem; font-weight:700; margin-top:.25rem; }
.metric-note { color:#45d6a5; font-size:.78rem; margin-top:.1rem; }
.section-card { background:#111c2e; border:1px solid #263850; border-radius:16px; padding:1rem 1.15rem; }
.small-muted { color:#91a2b8; font-size:.85rem; }
.stButton > button { border-radius:10px; border:1px solid #45d6a5; background:#45d6a5; color:#07151a; font-weight:700; min-height:2.8rem; }
.stButton > button:hover { background:#6de5bb; border-color:#6de5bb; color:#07151a; }
[data-testid="stFileUploader"] { background:#111c2e; border:1px dashed #3b5572; border-radius:14px; padding:.4rem; }
[data-testid="stMetric"] { background:#111c2e; border:1px solid #263850; padding:1rem; border-radius:14px; }
hr { border-color:#263850; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def save_history(record: dict) -> None:
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    rows = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def metric_card(label: str, value: str | int, note: str = "") -> None:
    st.markdown(f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)


def api_health(api_url: str) -> dict:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "offline", "model_loaded": False}


class LivePPEProcessor(VideoProcessorBase):
    """Send throttled webcam frames to the local FastAPI detector."""
    def __init__(self, backend_url: str, confidence: float, image_size: int):
        self.backend_url = backend_url.rstrip("/")
        self.confidence = confidence
        self.image_size = image_size
        self.last_sent = 0.0
        self.interval = 0.25  # Target up to ~4 inference requests/second.

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        if now - self.last_sent < self.interval:
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        self.last_sent = now
        try:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                response = requests.post(
                    f"{self.backend_url}/detect/frame",
                    files={"file": ("webcam.jpg", encoded.tobytes(), "image/jpeg")},
                    params={"conf": self.confidence, "imgsz": self.image_size},
                    timeout=15,
                )
                if response.ok:
                    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        image = decoded
        except requests.RequestException:
            # Keep the camera preview alive if the backend briefly misses a frame.
            pass
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def show_summary(summary: dict) -> None:
    a, b, c = st.columns(3)
    with a: metric_card("Total detections", summary.get("total_detections", 0))
    with b: metric_card("People detected", summary.get("people_detected", 0))
    with c: metric_card("PPE violation detections", summary.get("violation_detections", 0), "Model-labelled violation classes")
    counts = summary.get("class_counts", {})
    if counts:
        st.markdown("#### Class breakdown")
        st.bar_chart(pd.DataFrame({"Detections": counts}).sort_values("Detections", ascending=False))
    violations = summary.get("violations", [])
    if violations:
        st.markdown("#### Violation detections")
        st.dataframe(pd.DataFrame(violations)[["class_name", "confidence", "bbox"]], width="stretch", hide_index=True)
    else:
        st.success("No model-labelled PPE violation was detected in this image. This is not a guarantee of workplace compliance.")


with st.sidebar:
    st.markdown("## 🦺 SafeGuard")
    st.caption("AI WORKPLACE SAFETY INTELLIGENCE")
    page = st.radio("WORKSPACE", ["Overview", "Image inspection", "Video inspection", "Detection history", "Model & settings"], label_visibility="collapsed")
    st.divider()
    api_url = st.text_input("Backend API URL", DEFAULT_API)
    health = api_health(api_url)
    if health.get("model_loaded"):
        st.success("AI engine online")
    elif health.get("status") == "loading":
        st.warning("AI engine loading")
    else:
        st.error("Backend offline")
    st.caption("Pretrained YOLOv8m PPE model · Local inference")

st.markdown("<div class='hero'><div class='hero-kicker'>Workplace safety · Computer vision</div><div class='hero-title'>SafeGuard PPE Intelligence</div><div class='hero-sub'>Inspect images and videos for visible protective equipment and model-labelled safety violations.</div></div>", unsafe_allow_html=True)

if page == "Overview":
    history = load_history()
    image_runs = sum(1 for x in history if x.get("kind") == "image")
    video_runs = sum(1 for x in history if x.get("kind") == "video")
    violation_total = sum(int(x.get("violation_count", 0)) for x in history)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Inspection runs", len(history), "Saved locally")
    with c2: metric_card("Image inspections", image_runs)
    with c3: metric_card("Video inspections", video_runs)
    with c4: metric_card("Violation detections", violation_total, "Across saved runs")
    st.markdown("### Start an inspection")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='section-card'><h3>🖼️ Image inspection</h3><p class='small-muted'>Upload a JPG, JPEG or PNG image and receive annotated detections, confidence scores and class counts.</p></div>", unsafe_allow_html=True)
        if st.button("Open image inspection", width="stretch"): st.session_state["page_jump"] = "Image inspection"
    with right:
        st.markdown("<div class='section-card'><h3>🎞️ Video inspection</h3><p class='small-muted'>Process MP4, MOV, AVI, MKV or M4V footage frame by frame and download the annotated result.</p></div>", unsafe_allow_html=True)
        if st.button("Open video inspection", width="stretch"): st.session_state["page_jump"] = "Video inspection"
    if st.session_state.get("page_jump"):
        st.info(f"Choose **{st.session_state['page_jump']}** from the left sidebar to continue.")
    st.markdown("### Recent activity")
    if history:
        recent = pd.DataFrame(history[-8:][::-1])
        keep = [c for c in ["timestamp", "kind", "filename", "detection_count", "violation_count"] if c in recent.columns]
        st.dataframe(recent[keep], width="stretch", hide_index=True)
    else:
        st.info("No inspections yet. Run your first image or video inspection to populate this dashboard.")

elif page == "Image inspection":
    st.markdown("### Image inspection")
    st.caption("Supported formats: JPG · JPEG · PNG")
    uploaded = st.file_uploader("Choose a workplace image", type=["jpg", "jpeg", "png"], key="image_upload")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, help="Higher values reduce weak predictions but may miss small or distant PPE.")
    imgsz = st.select_slider("Inference image size", options=[416, 512, 640, 768, 960], value=640)
    if uploaded:
        raw = uploaded.getvalue()
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Original image**")
            st.image(raw, width="stretch")
        with col2:
            st.markdown("**Detection output**")
            if st.button("Run image inspection", type="primary", width="stretch"):
                if not health.get("model_loaded"):
                    st.error("Backend/model is not ready. Start the FastAPI backend first.")
                else:
                    try:
                        with st.spinner("Running PPE detection..."):
                            response = requests.post(f"{api_url.rstrip('/')}/detect/image", files={"file": (uploaded.name, raw, uploaded.type)}, params={"conf": conf, "imgsz": imgsz}, timeout=180)
                        if response.ok:
                            result = response.json()
                            annotated_bytes = base64.b64decode(result["image_base64"])
                            st.image(annotated_bytes, width="stretch")
                            summary = result["summary"]
                            save_history({"kind": "image", "filename": uploaded.name, "detection_count": summary["total_detections"], "violation_count": summary["violation_detections"]})
                            st.session_state["last_image_summary"] = summary
                            st.session_state["last_image_name"] = uploaded.name
                            st.success("Image inspection completed.")
                        else:
                            st.error(f"API error {response.status_code}: {response.text}")
                    except requests.RequestException as exc:
                        st.error(f"Could not reach the backend: {exc}")
            if st.session_state.get("last_image_name") == uploaded.name and st.session_state.get("last_image_summary"):
                st.caption("Latest result is shown below.")
    if st.session_state.get("last_image_summary"):
        st.divider()
        show_summary(st.session_state["last_image_summary"])

elif page == "Video inspection":
    st.markdown("### Live camera detection")
    st.caption("Start your webcam to run live PPE detection. Allow camera access when your browser asks.")
    live_conf = st.slider("Live confidence threshold", 0.05, 0.95, 0.25, 0.05, key="live_conf")
    live_imgsz = st.select_slider("Live inference image size", options=[416, 512, 640, 768], value=416, key="live_imgsz")
    if not health.get("model_loaded"):
        st.error("Backend/model is not ready. Start the FastAPI backend first.")
    else:
        live_ctx = webrtc_streamer(
            key="safeguard-live-camera",
            video_processor_factory=lambda: LivePPEProcessor(api_url, live_conf, live_imgsz),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if live_ctx.state.playing:
            st.success("Live camera is running. PPE detections are drawn on the camera feed.")
        else:
            st.info("Press START to turn on the camera. Press STOP when the inspection is complete.")
    st.divider()
    st.markdown("### Uploaded video inspection")
    st.caption("You can still upload and process recorded footage: MP4 · MOV · AVI · MKV · M4V · Maximum duration: 5 minutes")
    uploaded = st.file_uploader("Choose a workplace video", type=["mp4", "mov", "avi", "mkv", "m4v"], key="video_upload")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="video_conf")
    imgsz = st.select_slider("Inference image size", options=[416, 512, 640, 768, 960], value=640, key="video_imgsz")
    if uploaded:
        video_bytes = uploaded.getvalue()
        st.video(video_bytes)
        if st.button("Process video", type="primary", width="stretch"):
            if not health.get("model_loaded"):
                st.error("Backend/model is not ready. Start the FastAPI backend first.")
            else:
                progress = st.progress(0, text="Uploading video and starting frame-by-frame detection...")
                try:
                    response = requests.post(f"{api_url.rstrip('/')}/detect/video", files={"file": (uploaded.name, video_bytes, uploaded.type or "video/mp4")}, params={"conf": conf, "imgsz": imgsz}, timeout=1800)
                    progress.progress(100, text="Video processing finished.")
                    if response.ok:
                        st.success("Video processing completed.")
                        st.markdown("#### Annotated video")
                        st.video(response.content)
                        st.download_button("Download processed video", data=response.content, file_name=f"{Path(uploaded.name).stem}_safeguard.mp4", mime="video/mp4", width="stretch")
                        raw_summary = response.headers.get("X-SafeGuard-Summary")
                        if raw_summary:
                            summary = json.loads(base64.urlsafe_b64decode(raw_summary.encode()).decode())
                            st.markdown("#### Video analysis summary")
                            a, b, c = st.columns(3)
                            with a: metric_card("Frames processed", summary.get("frames_processed", 0))
                            with b: metric_card("Video duration", f"{summary.get('video_duration_seconds', 0)}s")
                            with c: metric_card("FPS", summary.get("fps", "—"))
                            class_counts = summary.get("class_detection_occurrences", {})
                            if class_counts:
                                st.bar_chart(pd.DataFrame({"Frame-level occurrences": class_counts}).sort_values("Frame-level occurrences", ascending=False))
                            violations = summary.get("violation_occurrences", {})
                            if violations:
                                st.warning("Model-labelled violation occurrences were found across the video frames.")
                                st.dataframe(pd.DataFrame([{"class_name": k, "occurrences": v} for k, v in violations.items()]), width="stretch", hide_index=True)
                            else:
                                st.success("No model-labelled PPE violation was detected in processed frames. This is not a guarantee of workplace compliance.")
                            save_history({"kind": "video", "filename": uploaded.name, "detection_count": sum(class_counts.values()), "violation_count": sum(violations.values()), "frames_processed": summary.get("frames_processed", 0)})
                            st.caption(summary.get("note", ""))
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")
                except requests.Timeout:
                    st.error("Video processing timed out. Try a shorter or lower-resolution video.")
                except requests.RequestException as exc:
                    st.error(f"Could not reach the backend: {exc}")

elif page == "Detection history":
    st.markdown("### Detection history")
    history = load_history()
    if not history:
        st.info("No saved inspection history yet.")
    else:
        df = pd.DataFrame(history[::-1])
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Export history as CSV", df.to_csv(index=False).encode("utf-8"), "safeguard_detection_history.csv", "text/csv")
        if st.button("Clear local history", type="secondary"):
            HISTORY_FILE.write_text("", encoding="utf-8")
            st.rerun()

elif page == "Model & settings":
    st.markdown("### Model & settings")
    st.write("SafeGuard uses a pretrained YOLOv8-medium PPE detector downloaded from Hugging Face on the first backend startup.")
    st.code("Hexmon/vyra-yolo-ppe-detection · best.pt", language="text")
    st.markdown("**Supported model classes**")
    classes = health.get("classes", [])
    if classes:
        st.write(" · ".join(classes))
    else:
        st.warning("Model class names are unavailable while the backend is offline.")
    st.markdown("**Runtime configuration**")
    st.write({"API URL": api_url, "Model loaded": health.get("model_loaded", False), "Device": health.get("device", "unknown"), "Model repository": health.get("model_repo", "unknown")})
    st.info("The model reports explicit missing-PPE classes (for example NO-Hardhat). The app does not infer that an unseen item is missing just because a positive PPE box was not detected.")
