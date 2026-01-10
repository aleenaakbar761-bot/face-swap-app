import io
import os
import cv2
import base64
import numpy as np
from flask import Flask, request, send_file, jsonify
import insightface
from insightface.app import FaceAnalysis
import traceback
from google.cloud import storage

MODEL_LOCAL_PATH = "/tmp/inswapper_128.onnx"
BUCKET_NAME = "face-swap-app"
MODEL_BLOB_PATH = "face_swap_app/models/inswapper_128.onnx"



def download_model_if_needed():
    if os.path.exists(MODEL_LOCAL_PATH):
        return

    print("Downloading model from GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_PATH)
    blob.download_to_filename(MODEL_LOCAL_PATH)
    print("Model downloaded.")


def init_models():
    global face_app, swapper, models_loaded

    if models_loaded:
        return True

    try:
        # Load face detection / analysis model
        if face_app is None:
            print("Loading face analysis model...")
            face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
            print("Face analysis model loaded.")

        # Load face swap model
        if swapper is None:
            print("Loading swapper model...")

            download_model_if_needed()

            swapper = insightface.model_zoo.get_model(
                MODEL_LOCAL_PATH,
                providers=['CPUExecutionProvider']
            )

            print("Swapper model loaded.")

        models_loaded = True
        return True

    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False


app_flask = Flask(__name__)
face_app = None
swapper = None
models_loaded = False

# Preload models so container is ready immediately on start
init_models()


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
        cv2.putText(blank, "No faces", (10, face_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
        cv2.putText(grid, str(i), (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return grid

@app_flask.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": models_loaded}), 200



@app_flask.route("/detect_source", methods=["POST"])
def detect_source():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "source" not in request.files:
            return jsonify({"error": "Missing 'source' file."}), 400

        src_img = read_image_from_request(request.files["source"])
        if src_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        src_faces = face_app.get(src_img)

        faces_data = []
        for i, face in enumerate(src_faces):
            crop = extract_face_crop(src_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                faces_data.append({
                    "index": i,
                    "image": b64_str
                })

        return jsonify({
            "count": len(faces_data),
            "faces": faces_data
        })
    except Exception as e:
        print(f"Error in detect_source: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/detect_target", methods=["POST"])
def detect_target():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "target" not in request.files:
            return jsonify({"error": "Missing 'target' file."}), 400

        tgt_img = read_image_from_request(request.files["target"])
        if tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        tgt_img = resize_image(tgt_img)
        tgt_faces = face_app.get(tgt_img)

        face_crops = []
        for face in tgt_faces:
            crop = extract_face_crop(tgt_img, face)
            face_crops.append(crop)

        grid = create_faces_grid(face_crops)
        ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode image."}), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="target_faces.jpg"
        )
    except Exception as e:
        print(f"Error in detect_target: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/detect", methods=["POST"])
def detect_both():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        src_crops = [extract_face_crop(src_img, f) for f in src_faces]
        tgt_crops = [extract_face_crop(tgt_img, f) for f in tgt_faces]

        src_grid = create_faces_grid(src_crops)
        tgt_grid = create_faces_grid(tgt_crops)

        src_h, src_w = src_grid.shape[:2]
        tgt_h, tgt_w = tgt_grid.shape[:2]

        label_h = 30
        max_w = max(src_w, tgt_w)
        
        src_grid_padded = np.zeros((src_h, max_w, 3), dtype=np.uint8)
        src_grid_padded[:, :src_w] = src_grid
        
        tgt_grid_padded = np.zeros((tgt_h, max_w, 3), dtype=np.uint8)
        tgt_grid_padded[:, :tgt_w] = tgt_grid

        src_label = np.zeros((label_h, max_w, 3), dtype=np.uint8)
        cv2.putText(src_label, f"SOURCE FACES ({len(src_faces)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        tgt_label = np.zeros((label_h, max_w, 3), dtype=np.uint8)
        cv2.putText(tgt_label, f"TARGET FACES ({len(tgt_faces)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        combined = np.vstack([src_label, src_grid_padded, tgt_label, tgt_grid_padded])

        ok, buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode image."}), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="detected_faces.jpg"
        )
    except Exception as e:
        print(f"Error in detect: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/swap", methods=["POST"])
def swap():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models. Please try again."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        source_index = int(request.form.get("source_index", 0))
        target_index = request.form.get("target_index", None)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0:
            return jsonify({"error": "No face detected in source image."}), 400
        if len(tgt_faces) == 0:
            return jsonify({"error": "No face detected in target image."}), 400

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
            return jsonify({"error": "Failed to encode output image."}), 500
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="swap.jpg"
        )
    except Exception as e:
        print(f"Error in swap: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/swap_all", methods=["POST"])
def swap_all():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models. Please try again."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        source_index = int(request.form.get("source_index", 0))

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0:
            return jsonify({"error": "No face detected in source image."}), 400
        if len(tgt_faces) == 0:
            return jsonify({"error": "No face detected in target image."}), 400

        if source_index >= len(src_faces):
            source_index = 0
        src_face = src_faces[source_index]

        out_img = tgt_img.copy()
        for tgt_face in tgt_faces:
            out_img = swapper.get(out_img, tgt_face, src_face, paste_back=True)

        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode output image."}), 500
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="swap.jpg"
        )
    except Exception as e:
        print(f"Error in swap_all: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/detect_faces", methods=["POST"])
def detect_faces():
    """
    Detects faces in both source and target images.
    Returns all faces as base64 JPEGs with type identifier (source/target).
    """
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        all_faces = []

        for i, face in enumerate(src_faces):
            crop = extract_face_crop(src_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                all_faces.append({
                    "type": "source",
                    "index": i,
                    "image": b64_str
                })

        for i, face in enumerate(tgt_faces):
            crop = extract_face_crop(tgt_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                all_faces.append({
                    "type": "target",
                    "index": i,
                    "image": b64_str
                })

        return jsonify({
            "source_count": len(src_faces),
            "target_count": len(tgt_faces),
            "faces": all_faces
        })
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/swap_selected", methods=["POST"])
def swap_selected():
    """
    Swaps selected faces between source and target images.
    Expects:
      - source: source image file
      - target: target image file
      - swaps: JSON string of swap pairs, e.g. [{"source_index": 0, "target_index": 1}]
    Returns the output image with all specified swaps applied.
    """
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models. Please try again."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        swaps_json = request.form.get("swaps", "[]")
        try:
            import json
            swaps = json.loads(swaps_json)
        except:
            return jsonify({"error": "Invalid 'swaps' format. Expected JSON array."}), 400

        if not isinstance(swaps, list) or len(swaps) == 0:
            return jsonify({"error": "No swap pairs provided. Expected array of {source_index, target_index}."}), 400

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0:
            return jsonify({"error": "No face detected in source image."}), 400
        if len(tgt_faces) == 0:
            return jsonify({"error": "No face detected in target image."}), 400

        out_img = tgt_img.copy()

        for swap_pair in swaps:
            source_idx = int(swap_pair.get("source_index", 0))
            target_idx = int(swap_pair.get("target_index", 0))

            if source_idx >= len(src_faces):
                source_idx = 0
            if target_idx >= len(tgt_faces):
                continue

            src_face = src_faces[source_idx]
            tgt_face = tgt_faces[target_idx]
            out_img = swapper.get(out_img, tgt_face, src_face, paste_back=True)

        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode output image."}), 500
        
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="swap_selected.jpg"
        )
    except Exception as e:
        print(f"Error in swap_selected: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

