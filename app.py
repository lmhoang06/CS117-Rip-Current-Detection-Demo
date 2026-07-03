"""
FastAPI app: CBAM YOLO-seg stream.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tqdm import tqdm
from ultralytics import YOLO

from cs117_demo.cbam_runtime import reload_and_inject_cbam_runtime
from cs117_demo.overlay import draw_polygons_overlay, iter_result_polygons

# --- Constants (edit paths to your checkpoints; resolved relative to repo root) ---
_REPO_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "model_weight/yolo26n_cbam_p3p4.pt"
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = ""  # e.g. "0" or "cpu"; empty = Ultralytics default
ALPHA = 0.35
COLOR_BGR = (0, 255, 0)  # green
MAX_FRAMES = 0  # 0 = all frames

_DEMO_DIR = Path(__file__).resolve().parent
_OUTPUTS_DIR = _DEMO_DIR / "_outputs"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = _OUTPUTS_DIR / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

model: YOLO | None = None

jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_render_lock = threading.Lock()


def _predict_kwargs() -> dict:
    kw: dict = {"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "verbose": False}
    if str(DEVICE).strip():
        kw["device"] = str(DEVICE).strip()
    return kw


def _render_job(job_id: str, upload_path: Path, job_dir: Path, original_filename: str) -> None:
    global jobs
    cap = None
    w_out = None
    try:
        with _render_lock:
            with _jobs_lock:
                jobs[job_id]["status"] = "running"
            cap = cv2.VideoCapture(str(upload_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {upload_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                raise RuntimeError("Invalid video dimensions.")

            count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_cap = count_reported if count_reported > 0 else None
            if MAX_FRAMES > 0 and total_cap is not None:
                total_cap = min(total_cap, MAX_FRAMES)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_file = job_dir / "output.mp4"
            w_out = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
            if not w_out.isOpened():
                raise RuntimeError("Could not open VideoWriter for MP4 output.")

            pk = _predict_kwargs()
            m = model
            if m is None:
                raise RuntimeError("Model not loaded.")

            outer_total = total_cap
            with tqdm(total=outer_total, desc="render", unit="fr") as pbar:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if MAX_FRAMES > 0 and idx >= MAX_FRAMES:
                        break

                    r = m.predict(source=frame, **pk)
                    r0 = r[0] if r else None
                    vis = draw_polygons_overlay(
                        frame, iter_result_polygons(r0), alpha=ALPHA, color_bgr=COLOR_BGR
                    )
                    wr = w_out.write(vis)
                    if wr is False:
                        raise RuntimeError("Failed to write frame to output video.")

                    idx += 1
                    pbar.update(1)
                    with _jobs_lock:
                        jobs[job_id]["progress"] = {
                            "current": idx,
                            "total": outer_total if outer_total is not None else idx,
                        }

            if idx == 0:
                raise RuntimeError("No frames decoded from video.")

            # IMPORTANT: finalize MP4 before any post-processing/transcode.
            try:
                w_out.release()
            except Exception:
                pass
            w_out = None

            # Transcode to browser-friendly H.264 MP4 (Chrome often rejects mp4v streams).
            out_h264 = job_dir / "output_h264.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(out_file),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_h264),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {out_file.name}: {(p.stderr or '').strip()}")

            # Copy H.264 version to cache
            cache_name = f"{Path(original_filename).stem}_processed.mp4"
            # Sanitize filename
            cache_name = "".join(c for c in cache_name if c.isalnum() or c in "._-")
            cache_dest = _CACHE_DIR / cache_name
            try:
                shutil.copy2(out_h264, cache_dest)
            except Exception:
                pass

            with _jobs_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = {"current": idx, "total": idx}
                jobs[job_id]["urls"] = {
                    "url": f"/outputs/{job_id}/{out_h264.name if out_h264.exists() else out_file.name}"
                }
    except Exception as e:
        with _jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
    finally:
        if w_out is not None:
            w_out.release()
        if cap is not None:
            cap.release()


def _start_render_worker(job_id: str, upload_path: Path, job_dir: Path, original_filename: str) -> None:
    t = threading.Thread(target=_render_job, args=(job_id, upload_path, job_dir, original_filename), daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reload_and_inject_cbam_runtime()
    path_m = Path(MODEL_NAME)
    if not path_m.is_absolute():
        path_m = _REPO_ROOT / path_m
    model = YOLO(str(path_m))
    try:
        model.fuse()
    except Exception:
        pass
    yield


app = FastAPI(title="CS117 CBAM YOLO demo", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")


@app.get("/")
async def index_page():
    return FileResponse(_DEMO_DIR / "static" / "index.html")


@app.get("/api/cache")
async def api_cache_list():
    files = []
    if _CACHE_DIR.exists():
        for p in sorted(_CACHE_DIR.glob("*.mp4")):
            files.append({
                "name": p.name,
                "url": f"/outputs/cache/{p.name}"
            })
    return files


@app.post("/api/render")
async def api_render(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    job_id = uuid.uuid4().hex
    job_dir = _OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename).suffix or ".mp4"
    upload_path = job_dir / f"upload{suffix}"

    with _jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": None,
            "error": None,
            "urls": None,
        }

    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        with _jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Upload failed: {e}"
        raise HTTPException(status_code=500, detail=str(e)) from e

    _start_render_worker(job_id, upload_path, job_dir, video.filename)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str):
    with _jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")

    body: dict = {"status": job["status"]}
    if job.get("error"):
        body["error"] = job["error"]
    if job.get("progress") is not None:
        body["progress"] = job["progress"]
    if job.get("urls"):
        body["urls"] = job["urls"]
    return body
