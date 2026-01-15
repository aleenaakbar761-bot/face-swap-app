import os
import sys
import io
import cv2
import base64
import numpy as np
import traceback
from flask import Flask, request, send_file, jsonify

# Globals
app_flask = Flask(__name__)
face_app = None
swapper = None
models_loaded = False

# Model paths
MODEL_LOCAL_PATH = "/tmp/inswapper_128.onnx"
BUCKET_NAME = "face-swap-app"
MODEL_BLOB_PATH = "face_swap_app/models/inswapper_128.onnx"

def download_model_if_needed():
    """Download model from GCS if not present."""
    from google.cloud import storage

    if os.path.exists(MODEL_LOCAL_PATH):
        return

    print("Downloading model from GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_PATH)
    blob.download_to_filename(MODEL_LOCAL_PATH)
    print("Model downloaded.")

def init_models():
    """Initialize all imports and AI models. Exit if fails."""
    global face_app, swapper, models_loaded, insightface

    try:
        print("Importing InsightFace...")
        import insightface
        from insightface.app import FaceAnalysis

        # Download the face swapper model first
        download_model_if_needed()

        # Load face detection/analysis model
        print("Loading face analysis model...")
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(320, 320))
        print("Face analysis model loaded.")

        # Load face swapper model
        print("Loading swapper model...")
        swapper = insightface.model_zoo.get_model(MODEL_LOCAL_PATH, providers=['CPUExecutionProvider'])
        print("Swapper model loaded.")

        models_loaded = True
        print("All models loaded successfully!")

    except Exception as e:
        print("Failed to initialize models or imports!")
        traceback.print_exc()
        sys.exit(1)  # Stop container immediately

# Helper functions
def read_image_from_request(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def resize_image(img, max_size=1024):
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
    face_crop = cv2.resize(face_crop, (target_size, target_size))
    return face_crop

def create_faces_grid(faces_list, face_size=150):
    if len(faces_list) == 0:
        blank = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        cv2.putText(blank, "No faces", (10, face_size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return blank

    n = len(faces_list)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    grid_h = rows * face_size
    grid_w = cols * face_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, face_img in enumerate(faces_list):
        row = i // cols
        col = i % cols
        y = row * face_size
        x = col * face_size
        grid[y:y+face_size, x:x+face_size] = face_img
        cv2.putText(grid, str(i), (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return grid

# Routes
@app_flask.route("/", methods=["GET"])
def root():
    return "OK", 200

@app_flask.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": models_loaded}), 200

# Detect faces in source image
@app_flask.route("/detect_source", methods=["POST"])
def detect_source():
    try:
        if not models_loaded:
            return jsonify({"error": "Models not loaded"}), 503
        if "source" not in request.files:
            return jsonify({"error": "Missing 'source' file"}), 400

        src_img = read_image_from_request(request.files["source"])
        if src_img is None:
            return jsonify({"error": "Invalid image data"}), 400

        src_img = resize_image(src_img)
        src_faces = face_app.get(src_img)

        faces_data = []
        for i, face in enumerate(src_faces):
            crop = extract_face_crop(src_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                faces_data.append({"index": i, "image": b64_str})

        return jsonify({"count": len(faces_data), "faces": faces_data})

    except Exception as e:
        print(f"Error in detect_source: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Swap faces endpoint
@app_flask.route("/swap", methods=["POST"])
def swap():
    try:
        if not models_loaded:
            return jsonify({"error": "Models not loaded"}), 503
        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files"}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])
        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data"}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        source_index = int(request.form.get("source_index", 0))
        target_index = request.form.get("target_index", None)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0 or len(tgt_faces) == 0:
            return jsonify({"error": "No faces detected"}), 400

        if source_index >= len(src_faces):
            source_index = 0
        src_face = src_faces[source_index]

        out_img = tgt_img.copy()
        if target_index is not None:
            target_index = int(target_index)
            if target_index < len(tgt_faces):
                out_img = swapper.get(out_img, tgt_faces[target_index], src_face, paste_back=True)
        else:
            for tgt_face in tgt_faces:
                out_img = swapper.get(out_img, tgt_face, src_face, paste_back=True)
                break

        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode output image"}), 500

        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg",
                         as_attachment=False, download_name="swap.jpg")

    except Exception as e:
        print(f"Error in swap: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Add more endpoints here (detect_target, swap_all, detect_faces, etc.) in same style

# Ensure initialization happens **before server starts**
init_models()

# Run Flask only if this is the main process
if __name__ == "__main__":
    print("Starting Flask server...")
    app_flask.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
