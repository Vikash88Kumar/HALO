"""
JerseyIQ backend -- Flask app that serves the API the static index.html
frontend expects (GET /inputs, POST /upload, POST /process,
GET /progress/<filename>, GET /result/<filename>, GET /outputs/<file>).

Run with:
    python app.py

Then open static/index.html in a browser (double-click it, or serve it any
way you like) and point the "Backend" field at http://127.0.0.1:5000
(that's the default already baked into the page).
"""
import os
import threading
import traceback
import time

from flask import Flask, request, jsonify, send_from_directory

from models import load_jersey_cnn, load_ccnn_filter
from pipeline import VideoProcessor, DEVICE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in (INPUTS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Flask(__name__)

# --------------------------------------------------------------------------
# CORS (the frontend is typically opened as a static file / different origin
# from the Flask server, so we allow cross-origin requests explicitly)
# --------------------------------------------------------------------------
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return ("", 204)


# --------------------------------------------------------------------------
# Lazy model loading (so the server starts instantly and only pays the
# model-loading cost on first use / at startup in the background)
# --------------------------------------------------------------------------
_models_lock = threading.Lock()
_models = {"detector": None, "jersey": None, "ccnn": None, "processor": None, "error": None}


def get_processor():
    with _models_lock:
        if _models["processor"] is not None or _models["error"] is not None:
            if _models["error"] is not None:
                raise RuntimeError(_models["error"])
            return _models["processor"]

        try:
            from ultralytics import YOLO
        except ImportError as e:
            _models["error"] = (
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            )
            raise RuntimeError(_models["error"]) from e

        try:
            detector = YOLO(os.path.join(MODELS_DIR, "detection_best.pt"))
            jersey_model = load_jersey_cnn(os.path.join(MODELS_DIR, "jersey_ocr_best.pt"), device=DEVICE)
            ccnn_model = load_ccnn_filter(os.path.join(MODELS_DIR, "ccnn_best.pt"), device=DEVICE)
        except Exception as e:
            _models["error"] = f"Failed to load models: {e}"
            raise RuntimeError(_models["error"]) from e

        processor = VideoProcessor(detector, jersey_model, ccnn_model, device=DEVICE)
        _models.update(detector=detector, jersey=jersey_model, ccnn=ccnn_model, processor=processor)
        return processor


# --------------------------------------------------------------------------
# In-memory job/progress store
# --------------------------------------------------------------------------
_jobs_lock = threading.Lock()
_jobs = {}  # filename -> {"percent":int, "message":str, "done":bool, "error":str|None, "result":dict|None, "annotated_url":str|None}


def _set_progress(filename, percent=None, message=None, done=None, error=None):
    with _jobs_lock:
        job = _jobs.setdefault(filename, {"percent": 0, "message": "Queued", "done": False, "error": None})
        if percent is not None:
            job["percent"] = percent
        if message is not None:
            job["message"] = message
        if done is not None:
            job["done"] = done
        if error is not None:
            job["error"] = error


def _run_job(filename, input_path):
    output_name = f"annotated_{os.path.splitext(filename)[0]}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_name)
    try:
        processor = get_processor()
    except Exception as e:
        _set_progress(filename, done=True, error=str(e))
        return

    def cb(pct, msg):
        _set_progress(filename, percent=pct, message=msg)

    try:
        result = processor.process(input_path, output_path, progress_cb=cb)
        with _jobs_lock:
            _jobs[filename]["result"] = result
            _jobs[filename]["annotated_url"] = f"/outputs/{output_name}"
        _set_progress(filename, percent=100, message="Complete", done=True)
    except Exception as e:
        traceback.print_exc()
        _set_progress(filename, done=True, error=f"{type(e).__name__}: {e}")


def _start_job(filename, input_path):
    _set_progress(filename, percent=0, message="Starting...", done=False, error=None)
    with _jobs_lock:
        _jobs[filename]["result"] = None
        _jobs[filename]["annotated_url"] = None
    t = threading.Thread(target=_run_job, args=(filename, input_path), daemon=True)
    t.start()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/inputs", methods=["GET"])
def list_inputs():
    try:
        files = sorted(
            f for f in os.listdir(INPUTS_DIR)
            if os.path.splitext(f)[1].lower() in ALLOWED_EXT
        )
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No 'video' file in request."}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"success": False, "error": f"Unsupported file type '{ext}'."}), 400

    safe_name = f"{int(time.time())}_{os.path.basename(file.filename)}"
    save_path = os.path.join(UPLOADS_DIR, safe_name)
    file.save(save_path)

    _start_job(safe_name, save_path)
    return jsonify({"success": True, "filename": safe_name})


@app.route("/process", methods=["POST"])
def process_existing():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "Missing 'filename'."}), 400

    input_path = os.path.join(INPUTS_DIR, filename)
    if not os.path.isfile(input_path):
        return jsonify({"success": False, "error": f"'{filename}' not found in inputs/."}), 404

    _start_job(filename, input_path)
    return jsonify({"success": True, "filename": filename})


@app.route("/progress/<path:filename>", methods=["GET"])
def progress(filename):
    with _jobs_lock:
        job = _jobs.get(filename)
    if job is None:
        return jsonify({"percent": 0, "message": "No such job.", "done": False, "error": None})
    return jsonify({
        "percent": job["percent"],
        "message": job["message"],
        "done": job["done"],
        "error": job["error"],
    })


@app.route("/result/<path:filename>", methods=["GET"])
def result(filename):
    with _jobs_lock:
        job = _jobs.get(filename)
    if job is None or job.get("result") is None:
        return jsonify({"success": False, "error": "Result not ready yet."}), 404
    return jsonify({
        "success": True,
        "annotated_url": job["annotated_url"],
        "result": job["result"],
    })


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")

@app.route("/about", methods=["GET"])
def about():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "about.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": DEVICE})


if __name__ == "__main__":
    print(f"JerseyIQ backend starting on http://127.0.0.1:5000  (device={DEVICE})")
    print("Pre-loading models (this can take a while the first time)...")
    try:
        get_processor()
        print("Models loaded OK.")
    except Exception as e:
        print(f"WARNING: model preload failed, will retry on first request: {e}")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
