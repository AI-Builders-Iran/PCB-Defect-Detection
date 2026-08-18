# 🔬 PCB Defect Detection

A two-stage computer vision system for **detecting, classifying, and tracking assembly defects on PCB boards**, built on two YOLO models (board segmentation + defect detection) with multi-frame tracking support (ByteTrack). The project includes a **shared core pipeline**, a **REST API built with FastAPI**, and an **interactive dashboard built with Streamlit**, all runnable from a single **Docker** image.

<p>
  <img alt="python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688">
  <img alt="streamlit" src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B">
  <img alt="yolo" src="https://img.shields.io/badge/YOLO-Ultralytics-purple">
  <img alt="docker" src="https://img.shields.io/badge/Docker-Ready-2496ED">
</p>

---

## 📑 Table of Contents

- [Architecture & Pipeline Logic](#-architecture--pipeline-logic)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Preparing Model Weights](#-preparing-model-weights)
- [Running Locally (without Docker)](#-running-locally-without-docker)
- [Running with Docker](#-running-with-docker)
- [API Documentation](#-api-documentation)
  - [General Endpoints](#general-endpoints)
  - [Image Endpoints](#image-endpoints)
  - [Video Endpoints (Job-based)](#video-endpoints-job-based)
  - [Example Calls with curl](#example-calls-with-curl)
  - [Example Calls with Python](#example-calls-with-python)
- [Environment Variables](#-environment-variables)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Common Troubleshooting](#-common-troubleshooting)
- [Roadmap](#-roadmap)

---

## 🧠 Architecture & Pipeline Logic

The pipeline (`src/pipeline.py`) works in two stages:

```
        Input frame (image/video/webcam)
                    │
                    ▼
     ┌────────────────────────────┐
     │  Model 1: PCB-SEG (Segment) │  → locates and masks the PCB board in the frame
     └────────────────────────────┘
                    │  (crop the board region)
                    ▼
     ┌────────────────────────────┐
     │ Model 2: Defect Detection   │  → detects and classifies defects + ByteTrack
     └────────────────────────────┘
                    │
                    ▼
       Structured result (DetectionResult)
     → annotated frame + downloadable JSON
```

If no PCB board is found in the frame, the second model falls back to running on the whole frame so no defect is missed. The core class `PCBPipeline` supports three usage modes:

| Method | Purpose |
|---|---|
| `process_image(path)` | Process a single image, returns a `DetectionResult` |
| `process(progress_callback=...)` | Process a full video/webcam stream, saving the output video + JSON |
| `_draw_results(frame, result)` | Draws boxes/mask on a frame for display |

---

## 📂 Project Structure

```
PCB-Defect-Detection/
├── app/
│   ├── api/
│   │   ├── API_app.py          # FastAPI service (this project)
│   │   └── models/             # ⚠️ Model weights go here (git-ignored)
│   └── streamlit_app/
│       ├── app.py              # Interactive dashboard (FA/EN, dark/light)
│       └── models/             # Model weights for the dashboard
├── src/
│   └── pipeline.py             # Core pipeline (shared by the API and dashboard)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml          # Run the API + dashboard together
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

## ✅ Prerequisites

- Python **3.11+** (for local execution)
- or Docker **24+** and, optionally, Docker Compose **v2** (for containerized execution)
- Weights for both models: `best-pcb.onnx` (or `.pt`) and `best_detect2.onnx` (or `.pt`)
- (Optional) A CUDA-capable GPU for faster inference — the default is `cpu`

---

## 📦 Preparing Model Weights

Model weights are **not included in the repo or the Docker image** (which is why they're excluded via `.gitignore`/`.dockerignore` — they're typically large and often proprietary). Before running the project, copy the weight files to:

```
app/api/models/best-pcb.onnx
app/api/models/best_detect2.onnx

app/streamlit_app/models/best-pcb.onnx
app/streamlit_app/models/best_detect2.onnx
```

You can also change the paths/filenames via the `MODEL1_PATH` / `MODEL2_PATH` environment variables (for the API) or from the dashboard's settings panel (for Streamlit).

---

## 🖥 Running Locally (without Docker)

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the model weights into the paths above

# 4. Run FastAPI (from the project root — important, since it imports from src.pipeline)
uvicorn app.api.API_app:app --host 0.0.0.0 --port 8000 --reload

# 5. Run the Streamlit dashboard (in a separate terminal)
streamlit run app/streamlit_app/app.py
```

Once running:
- API: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`
- Streamlit dashboard: `http://localhost:8501`

---

## 🐳 Running with Docker

### Option 1: plain `docker build` / `docker run`

```bash
# Build the image
docker build -t pcb-defect-detection:latest .

# Run the FastAPI service (model weights are mounted from the host, not baked into the image)
docker run -d \
  --name pcb-api \
  -p 8000:8000 \
  -v "$(pwd)/app/api/models:/app/app/api/models:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  pcb-defect-detection:latest

# Run the Streamlit dashboard instead of FastAPI (same image, different CMD)
docker run -d \
  --name pcb-dashboard \
  -p 8501:8501 \
  -v "$(pwd)/app/streamlit_app/models:/app/app/streamlit_app/models:ro" \
  pcb-defect-detection:latest \
  streamlit run app/streamlit_app/app.py --server.address=0.0.0.0 --server.port=8501
```

Check service health:
```bash
curl http://localhost:8000/health
```

### Option 2: `docker compose` (recommended — both services at once)

```bash
docker compose up --build
```

This spins up two containers:

| Service | Address | Description |
|---|---|---|
| `api` | http://localhost:8000/docs | FastAPI service |
| `dashboard` | http://localhost:8501 | Streamlit dashboard |

Stop with:
```bash
docker compose down
```

---

## 📡 API Documentation

> Interactive, always up-to-date documentation (Swagger UI) is available at **`/docs`** — also **`/redoc`**.

### General Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Basic service info |
| `GET` | `/health` | Service health status and whether models loaded successfully |

### Image Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `POST` | `/predict/image` | Image file (`multipart/form-data`) | JSON result (defect count, bboxes, confidence, ...) |
| `POST` | `/predict/image/annotated` | Image file + `return_format=png\|jpg` | The annotated image (binary) + a summary header |

### Video Endpoints (Job-based)

Video processing can take a while, so it's designed as an **asynchronous job**: the request immediately returns a `job_id`, and the actual processing runs in the background (a thread pool).

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/video` | Upload a video + optional params (`conf_threshold`, `iou_threshold`, `imgsz`, `enable_tracking`) → `job_id` |
| `GET` | `/predict/video/{job_id}/status` | Processing status: `queued` → `processing` → `done`/`failed` + progress percentage |
| `GET` | `/predict/video/{job_id}/download/video` | Download the annotated video (only once `done`) |
| `GET` | `/predict/video/{job_id}/download/json` | Download the full JSON result (stats + per-frame defects) |
| `DELETE` | `/predict/video/{job_id}` | Delete a job and its output files |

### Example Calls with curl

```bash
# Process an image → JSON
curl -X POST http://localhost:8000/predict/image \
  -F "file=@sample_pcb.jpg"

# Process an image → get the annotated image back
curl -X POST "http://localhost:8000/predict/image/annotated?return_format=png" \
  -F "file=@sample_pcb.jpg" \
  -o annotated_result.png

# Start video processing
curl -X POST "http://localhost:8000/predict/video?conf_threshold=0.3" \
  -F "file=@line_stream.mp4"
# → {"job_id": "a1b2c3...", "status": "queued", ...}

# Poll status
curl http://localhost:8000/predict/video/a1b2c3.../status

# Download the final video (once status=done)
curl -o result.mp4 http://localhost:8000/predict/video/a1b2c3.../download/video
```

### Example Calls with Python

```python
import requests, time

BASE = "http://localhost:8000"

# --- Image ---
with open("sample_pcb.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict/image", files={"file": f})
print(r.json())

# --- Video ---
with open("line_stream.mp4", "rb") as f:
    r = requests.post(f"{BASE}/predict/video", files={"file": f})
job_id = r.json()["job_id"]

while True:
    status = requests.get(f"{BASE}/predict/video/{job_id}/status").json()
    print(status["status"], status.get("progress"))
    if status["status"] in ("done", "failed"):
        break
    time.sleep(2)

if status["status"] == "done":
    video = requests.get(f"{BASE}/predict/video/{job_id}/download/video")
    open("result.mp4", "wb").write(video.content)
```

---

## ⚙️ Environment Variables

All of these are optional and have sensible defaults (see the full list in `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODEL1_PATH` | `app/api/models/best-pcb.onnx` | Path to the board segmentation model weights |
| `MODEL2_PATH` | `app/api/models/best_detect2.onnx` | Path to the defect detection model weights |
| `CONF_THRESHOLD` | `0.25` | Default confidence threshold |
| `IOU_THRESHOLD` | `0.45` | Default IoU/NMS threshold |
| `IMG_SIZE` | `416` | Inference input resolution |
| `DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `OUTPUT_DIR` | `/tmp/pcb_api_outputs` | Where temporary job output (video/JSON) is stored |
| `MAX_UPLOAD_MB` | `200` | Maximum upload file size |
| `MAX_WORKERS` | `2` | Number of threads processing videos concurrently |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

---

## 🖼 Streamlit Dashboard

The dashboard (`app/streamlit_app/app.py`) is a full UI with Persian/English support (RTL/LTR), dark/light mode, single-image analysis, and full video processing with KPIs and downloadable outputs — it runs independently of FastAPI (calling `src.pipeline` directly) and can run alongside the API or on its own.

---

## 🩺 Common Troubleshooting

| Issue | Fix |
|---|---|
| `/health` returns `"status": "degraded"` | Check the `MODEL1_PATH`/`MODEL2_PATH` paths; in Docker, make sure the models folder is mounted |
| `ImportError: ultralytics not installed` | Run `pip install -r requirements.txt` from the project root |
| Output video won't play in the browser | `ffmpeg` must be installed in the system/container (already installed in the Dockerfile) |
| Inference is slow | Lower `IMG_SIZE`, or set `DEVICE=cuda:0` on a GPU-equipped machine |
| 413 error on upload | Increase `MAX_UPLOAD_MB` |

---

## 🗺 Roadmap

- [ ] Authentication (API key / JWT) for endpoints
- [ ] Job queue backed by Redis/Celery instead of an in-memory thread pool
- [ ] WebSocket support for live webcam streaming
- [ ] Batch inference for multiple images at once
- [ ] Prometheus/Grafana metrics integration
