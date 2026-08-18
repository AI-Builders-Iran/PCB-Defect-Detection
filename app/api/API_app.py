"""
PCB Defect Detection API
=========================

FastAPI service that exposes the PCB defect detection pipeline
(`src.pipeline.PCBPipeline`) as a REST API. It has two main endpoint groups:

    * /predict/image  -> process a single image/frame and get the result
                          (JSON or the annotated image)
    * /predict/video   -> process a full video asynchronously (job-based)
                          and retrieve its status/output

Run (locally, from the project root):
    uvicorn app.api.API_app:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs
"""

import os
import io
import json
import uuid
import shutil
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from src.pipeline import PCBPipeline, PipelineConfig, DetectionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pcb-api")

# ---------------------------------------------------------------------------
# Configuration via environment variables (overridable in Docker/Compose)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

MODEL1_PATH = os.getenv("MODEL1_PATH", str(MODELS_DIR / "best-pcb.onnx"))
MODEL2_PATH = os.getenv("MODEL2_PATH", str(MODELS_DIR / "best_detect2.onnx"))

DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
DEFAULT_IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))
DEFAULT_IMGSZ = int(os.getenv("IMG_SIZE", "416"))
DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")

# Directory for temporary outputs (annotated images/videos, result JSON files)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/pcb_api_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ---------------------------------------------------------------------------
# Thread pool for CPU-bound work (long video processing) so the main FastAPI
# event loop is never blocked.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "2")))

# ---------------------------------------------------------------------------
# In-memory state for video jobs (fine for a demo/MVP; for production, use
# Redis or a database instead).
# ---------------------------------------------------------------------------
_video_jobs: Dict[str, Dict] = {}


class PipelineHolder:
    pipeline: Optional[PCBPipeline] = None
    load_error: Optional[str] = None


holder = PipelineHolder()


def build_pipeline() -> PCBPipeline:
    config = PipelineConfig(
        model1_path=MODEL1_PATH,
        model2_path=MODEL2_PATH,
        conf_threshold=DEFAULT_CONF,
        iou_threshold=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        enable_tracking=True,
        save_video=False,
        save_json=False,
        show_preview=False,
        verbose=False,
    )
    return PCBPipeline(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: attempt to load the models ---
    try:
        logger.info("Loading models ...")
        logger.info(f"model1_path={MODEL1_PATH}")
        logger.info(f"model2_path={MODEL2_PATH}")
        holder.pipeline = build_pipeline()
        holder.load_error = None
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        holder.pipeline = None
        holder.load_error = str(e)
        logger.error(f"❌ Failed to load models: {e}")
        logger.error(
            "The service will still start, but /predict/* endpoints will "
            "return 503 until this is fixed. Check the model paths in the "
            "environment (MODEL1_PATH / MODEL2_PATH)."
        )
    yield
    # --- Shutdown ---
    _executor.shutdown(wait=False)
    logger.info("Service shutting down.")


app = FastAPI(
    title="PCB Defect Detection API",
    description=(
        "Computer vision service for detecting and tracking assembly/solder "
        "defects on PCB boards. Built on a two-stage YOLO pipeline (board "
        "segmentation + defect detection) with multi-frame tracking (ByteTrack)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS so the API can be consumed from the Streamlit dashboard / a separate frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model1_path: str
    model2_path: str
    device: str
    error: Optional[str] = None


class ImageResultResponse(BaseModel):
    frame: int
    has_pcb: bool
    pcb_id: int
    defect_count: int
    defects: list
    processing_time_ms: float


class VideoJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class VideoJobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | done | failed
    progress: Optional[float] = None
    stats: Optional[Dict] = None
    video_download_url: Optional[str] = None
    json_download_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_models_ready():
    if holder.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not loaded. Check the MODEL1_PATH / MODEL2_PATH "
                f"environment variables. Load error details: {holder.load_error}"
            ),
        )


def validate_upload(file: UploadFile, allowed_ext: set) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format: '{ext}'. Allowed formats: {sorted(allowed_ext)}",
        )
    return ext


async def save_upload_to_temp(file: UploadFile, ext: str) -> Path:
    size = 0
    tmp_path = Path(tempfile.mkstemp(suffix=ext)[1])
    with open(tmp_path, "wb") as out_f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                out_f.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than the allowed limit ({MAX_UPLOAD_MB}MB).",
                )
            out_f.write(chunk)
    return tmp_path


def run_video_job(job_id: str, input_path: str, conf: float, iou: float, imgsz: int, tracking: bool):
    """Runs inside the ThreadPoolExecutor; processes the full video and saves the outputs."""
    job = _video_jobs[job_id]
    job["status"] = "processing"
    job["updated_at"] = datetime.utcnow().isoformat()

    out_video_path = OUTPUT_DIR / f"{job_id}.mp4"
    out_json_path = OUTPUT_DIR / f"{job_id}.json"

    try:
        config = PipelineConfig(
            model1_path=MODEL1_PATH,
            model2_path=MODEL2_PATH,
            input_source=input_path,
            output_path=str(out_video_path),
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=imgsz,
            enable_tracking=tracking,
            save_video=True,
            save_json=True,
            show_preview=False,
            verbose=False,
        )
        pipeline = PCBPipeline(config)

        def on_progress(current, total):
            if total > 0:
                job["progress"] = round(current / total, 4)
                job["updated_at"] = datetime.utcnow().isoformat()

        results = pipeline.process(progress_callback=on_progress)

        # pipeline._save_json derives the JSON name from the .mp4 path; to keep it
        # aligned with job_id, copy the produced file if the path differs.
        produced_json = results.get("output_json")
        if produced_json and Path(produced_json).exists() and Path(produced_json) != out_json_path:
            shutil.copyfile(produced_json, out_json_path)

        job["status"] = "done"
        job["progress"] = 1.0
        job["stats"] = results["stats"]
        job["video_download_url"] = f"/predict/video/{job_id}/download/video"
        job["json_download_url"] = f"/predict/video/{job_id}/download/json"
        job["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[{job_id}] Video processing finished successfully.")

    except Exception as e:
        logger.exception(f"[{job_id}] Error while processing video")
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = datetime.utcnow().isoformat()

    finally:
        Path(input_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# General endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "service": "PCB Defect Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok" if holder.pipeline is not None else "degraded",
        models_loaded=holder.pipeline is not None,
        model1_path=MODEL1_PATH,
        model2_path=MODEL2_PATH,
        device=DEFAULT_DEVICE,
        error=holder.load_error,
    )


# ---------------------------------------------------------------------------
# Image endpoints
# ---------------------------------------------------------------------------
@app.post("/predict/image", response_model=ImageResultResponse, tags=["Image"])
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a PCB image and get the defect detection result as JSON
    (no annotated image; use /predict/image/annotated for that).
    """
    ensure_models_ready()
    ext = validate_upload(file, ALLOWED_IMAGE_EXT)
    tmp_path = await save_upload_to_temp(file, ext)

    try:
        result: DetectionResult = holder.pipeline.process_image(str(tmp_path))
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        logger.exception("Error while processing image")
        raise HTTPException(status_code=500, detail=f"Error while processing image: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/predict/image/annotated", tags=["Image"])
async def predict_image_annotated(
    file: UploadFile = File(...),
    return_format: str = Query("png", enum=["png", "jpg"]),
):
    """
    Same as /predict/image, but instead of JSON it returns the annotated
    image itself (with defect bounding boxes) as binary data. A short summary
    of the result is included in the `X-Detection-Result` response header.
    """
    ensure_models_ready()
    ext = validate_upload(file, ALLOWED_IMAGE_EXT)
    tmp_path = await save_upload_to_temp(file, ext)

    try:
        result: DetectionResult = holder.pipeline.process_image(str(tmp_path))
        frame = cv2.imread(str(tmp_path))
        if frame is None:
            raise HTTPException(status_code=422, detail="The image file could not be read.")

        annotated = holder.pipeline._draw_results(frame, result)

        encode_ext = ".png" if return_format == "png" else ".jpg"
        ok, buf = cv2.imencode(encode_ext, annotated)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode the output image.")

        media_type = "image/png" if return_format == "png" else "image/jpeg"
        headers = {
            "X-Detection-Result": json.dumps(
                {"has_pcb": result.has_pcb, "defect_count": len(result.defects)}
            )
        }
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type=media_type, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error while processing image (annotated)")
        raise HTTPException(status_code=500, detail=f"Error while processing image: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Video endpoints (Async / Job-based)
# Video processing can take a while, so a request immediately returns a
# job_id, and the client polls /predict/video/{job_id}/status for progress.
# ---------------------------------------------------------------------------
@app.post("/predict/video", response_model=VideoJobResponse, tags=["Video"])
async def predict_video(
    file: UploadFile = File(...),
    conf_threshold: float = Query(DEFAULT_CONF, ge=0.05, le=0.95),
    iou_threshold: float = Query(DEFAULT_IOU, ge=0.05, le=0.95),
    imgsz: int = Query(DEFAULT_IMGSZ),
    enable_tracking: bool = Query(True),
):
    """Starts processing a video in the background and immediately returns a job_id."""
    ensure_models_ready()
    ext = validate_upload(file, ALLOWED_VIDEO_EXT)
    tmp_path = await save_upload_to_temp(file, ext)

    job_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    _video_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "stats": None,
        "video_download_url": None,
        "json_download_url": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }

    _executor.submit(
        run_video_job, job_id, str(tmp_path), conf_threshold, iou_threshold, imgsz, enable_tracking
    )

    return VideoJobResponse(
        job_id=job_id,
        status="queued",
        message="Video processing started in the background. Check status at /predict/video/{job_id}/status.",
    )


@app.get("/predict/video/{job_id}/status", response_model=VideoJobStatus, tags=["Video"])
def video_job_status(job_id: str):
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found.")
    return VideoJobStatus(**job)


@app.get("/predict/video/{job_id}/download/video", tags=["Video"])
def download_video(job_id: str):
    job = _video_jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="The output video is not ready or job_id is invalid.")
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(path, media_type="video/mp4", filename=f"annotated_{job_id}.mp4")


@app.get("/predict/video/{job_id}/download/json", tags=["Video"])
def download_json(job_id: str):
    job = _video_jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="The JSON result is not ready or job_id is invalid.")
    path = OUTPUT_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="JSON file not found.")
    return FileResponse(path, media_type="application/json", filename=f"result_{job_id}.json")


@app.delete("/predict/video/{job_id}", tags=["Video"])
def delete_video_job(job_id: str):
    """Cleans up a job's output files to free up disk space."""
    job = _video_jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found.")
    for suffix in (".mp4", ".json"):
        p = OUTPUT_DIR / f"{job_id}{suffix}"
        p.unlink(missing_ok=True)
    return {"deleted": job_id}
