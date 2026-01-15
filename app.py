import os
import io
import traceback
from flask import Flask, request, jsonify, send_file

# ------------------------
# Critical imports
# ------------------------
try:
    import numpy as np
    import cv2
    import base64
    import insightface
    from insightface.app import FaceAnalysis
    from google.cloud import storage
except ImportError as e:
    print(f"Critical dependency missing: {e}")
    raise e  # Fail fast, container won't start if imports fail

# ------------------------
# Flask app
# ------------------------
app_flask = Flask(__name__)

# ------------------------
# Globals
# ------------------------
face_app = None
swapper = None
models_loaded = False

MODEL_LOCAL_PATH = "/tmp/inswapper_128.onnx"
BUCKET_NAME = "face-swap-app"
MODEL_BLOB_PATH = "face_swap_app/models/inswapper_128.onnx"

# ------------------------
# Utilities
# ------------------------
def download_model_if_needed():
    if os.path.exists(MODEL_LOCAL_PATH):
        return True

    try:
        print("Downloading model from GCS...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB_PATH)
        blob.download_to_filename(MODEL_LOCAL_PATH)
        print("Model downloaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        traceback.print_exc()
        return False

def init_models():
    """Initialize face detection and swap models safely."""
    global face_app, swapper, models_loaded

    if models_loaded:
        return True

    try:
        # Ensure model file exists
        if not download_model_if_needed():
            raise RuntimeError("Model file could not be downloaded.")

        # Load FaceAnalysis model
        if face_app is None:
            print("Loading FaceAnalysis model...")
            face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
            print("FaceAnalysis loaded.")

        # Load face swapper model
        if swapper is None:
            print("Loading InSwapper model...")
            swapper = insightface.model_zoo.get_model(
                MODEL_LOCAL_PATH, providers=['CPUExecutionProvider']
            )
            print("InSwapper model loaded.")

        models_loaded = True
        return True
    except Exception as e:
        print(f"Model initialization failed: {e}")
        traceback.print_exc()
        models_loaded = False
        return False

# ------------------------
# Routes
# ------------------------
@app_flask.route("/health", methods=["GET"])
def health():
    """Check if models are loaded."""
    return jsonify({"status": "ok", "models_loaded": models_loaded}), 200

@app_flask.route("/", methods=["GET"])
def root():
    return "OK", 200

# ------------------------
# Start-up check
# ------------------------
# Fail fast if models cannot be loaded
if not init_models():
    print("Critical error: models could not be initialized. Exiting.")
    raise RuntimeError("Models initialization failed. Container cannot start.")

# ------------------------
# You can now safely add other endpoints that use face_app and swapper
# ------------------------

# Example:
@app_flask.route("/detect_faces", methods=["POST"])
def detect_faces():
    try:
        if not models_loaded:
            return jsonify({"error": "Models not loaded."}), 503

        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' file."}), 400

        file = request.files["image"]
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        faces = face_app.get(img)
        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            _, buf = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")
            results.append({"index": i, "face": b64_img})

        return jsonify({"faces_detected": len(faces), "faces": results})
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------
# Run Flask
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host="0.0.0.0", port=port)
