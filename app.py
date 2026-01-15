import os
import sys
import io
import base64
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file

# ------------------------------
# Configuration
# ------------------------------
MODEL_LOCAL_PATH = "/tmp/inswapper_128.onnx"
BUCKET_NAME = "face-swap-app"
MODEL_BLOB_PATH = "face_swap_app/models/inswapper_128.onnx"
MAX_IMAGE_SIZE = 1024

# Globals
face_app = None
swapper = None
models_loaded = False

# Flask app
app_flask = Flask(__name__)

# ------------------------------
# Helper functions
# ------------------------------

def download_model_if_needed():
    """Download ONNX model from GCS if not exists."""
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f"Model already exists at {MODEL_LOCAL_PATH}")
        return

    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed.")
        sys.exit(1)

    print("Downloading model from GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_PATH)
    blob.download_to_filename(MODEL_LOCAL_PATH)
    print("Model downloaded successfully.")

def lazy_imports():
    """Import ONNX Runtime and InsightFace only after model is ready."""
    try:
        import onnxruntime  # must be installed in Dockerfile
        import insightface
        from insightface.app import FaceAnalysis
        return onnxruntime, insightface, FaceAnalysis
    except ImportError as e:
        print(f"Critical import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def init_models():
    """Initialize face detection and swapper model."""
    global face_app, swapper, models_loaded

    if models_loaded:
        return True

    # Ensure model is downloaded first
    download_model_if_needed()

    # Lazy imports
    onnxruntime, insightface, FaceAnalysis = lazy_imports()

    try:
        # Initialize face analysis model
        if face_app is None:
            print("Loading face analysis model...")
            face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
            print("Face analysis model loaded.")

        # Initialize face swapper
        if swapper is None:
            print("Loading swapper model...")
            swapper = insightface.model_zoo.get_model(MODEL_LOCAL_PATH, providers=['CPUExecutionProvider'])
            print("Swapper model loaded.")

        models_loaded = True
        return True

    except Exception as e:
        print(f"Failed to initialize models: {e}")
        traceback.print_exc()
        sys.exit(1)  # Stop server if models cannot be loaded

# ------------------------------
# Image Utilities
# ------------------------------

def read_image_from_request(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def resize_image(img, max_size=MAX_IMAGE_SIZE):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

def extract_face_crop(img, face, padding=30, target_size=150):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    face_crop = img[y1:y2, x1:x2]
    return cv2.resize(face_crop, (target_size, target_size))

def create_faces_grid(faces_list, face_size=150):
    if not faces_list:
        blank = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        cv2.putText(blank, "No faces", (10, face_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return blank

    n = len(faces_list)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * face_size, cols * face_size, 3), dtype=np.uint8)

    for i, face_img in enumerate(faces_list):
        row = i // cols
        col = i % cols
        y = row * face_size
        x = col * face_size
        grid[y:y+face_size, x:x+face_size] = face_img
        cv2.putText(grid, str(i), (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return grid

# ------------------------------
# Routes
# ------------------------------

@app_flask.route("/", methods=["GET"])
def root():
    return "OK", 200

@app_flask.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": models_loaded}), 200

@app_flask.route("/detect_source", methods=["POST"])
def detect_source():
    if not init_models():
        return jsonify({"error": "Models failed to load"}), 503
    try:
        if "source" not in request.files:
            return jsonify({"error": "Missing 'source' file."}), 400

        src_img = read_image_from_request(request.files["source"])
        src_img = resize_image(src_img)
        src_faces = face_app.get(src_img)

        faces_data = []
        for i, face in enumerate(src_faces):
            crop = extract_face_crop(src_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                faces_data.append({
                    "index": i,
                    "image": base64.b64encode(buf.tobytes()).decode('utf-8')
                })
        return jsonify({"count": len(faces_data), "faces": faces_data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Preload models before server starts
# ------------------------------
if __name__ == "__main__":
    print("Starting server: ensuring all dependencies and model are loaded first...")
    init_models()
    print("All dependencies loaded. Starting Flask server.")
    app_flask.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
