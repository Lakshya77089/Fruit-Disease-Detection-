# ...existing code...
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import io
import base64
import os
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def _safe_load(*paths):
    """
    Try a list of candidate paths and return the first successfully loaded YOLO model.
    If none found or load fails, return None.
    """
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            logging.info(f"Loading model from: {p}")
            try:
                return YOLO(p)
            except Exception as e:
                logging.error(f"Failed to load model from {p}: {e}")
    logging.warning(f"No model file found among: {paths}")
    return None


# Try common local repo paths first, then fallbacks used by some notebooks/environments
fruit_detection_model = _safe_load("weights_3/best.pt", "weights/best.pt")
banana_disease_detection_model = _safe_load("train2/weights/best.pt", "/config/workspace/train2/weights/best.pt")
mango_disease_detection_model = _safe_load("train/weights/best.pt", "/config/workspace/train/weights/best.pt")
pomogranate_disease_detection_model = _safe_load("train4/weights/best.pt", "/config/workspace/train4/weights/best.pt")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    # Expose model availability to the homepage for better UX
    return render_template('index.html',
                           fruit_loaded=(fruit_detection_model is not None),
                           banana_loaded=(banana_disease_detection_model is not None),
                           mango_loaded=(mango_disease_detection_model is not None),
                           pomogranate_loaded=(pomogranate_disease_detection_model is not None))


@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Receive image data from the client
    data_url = request.json.get('image_data', '')
    if not data_url:
        return jsonify([])

    # Remove the data URL prefix if present
    if ',' in data_url:
        image_data = data_url.split(',', 1)[1]
    else:
        image_data = data_url

    # Decode base64 image data
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception:
        return jsonify([])

    # Convert image bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if fruit_detection_model is None:
        logging.warning("Fruit detection model not loaded.")
        return jsonify([])

    # Perform object detection using YOLO
    try:
        results = fruit_detection_model(image)
    except Exception as e:
        logging.error(f"Detection error: {e}")
        return jsonify([])

    # Extract detection results
    detected_objects = []
    for res in results:
        names = getattr(res, "names", {})
        # boxes may be a Boxes object; try to access .xywh, .cls, .conf
        try:
            boxes = res.boxes
            cls_vals = getattr(boxes, "cls", None)
            conf_vals = getattr(boxes, "conf", None)
            # convert tensors to python lists if needed
            if hasattr(cls_vals, "cpu"):
                cls_list = cls_vals.cpu().numpy().tolist()
            elif cls_vals is not None:
                cls_list = cls_vals.tolist()
            else:
                cls_list = []

            if hasattr(conf_vals, "cpu"):
                conf_list = conf_vals.cpu().numpy().tolist()
            elif conf_vals is not None:
                conf_list = conf_vals.tolist()
            else:
                conf_list = []

            # xywh or xyxy: prefer xywh if available
            xywh = getattr(boxes, "xywh", None)
            if xywh is not None and hasattr(xywh, "cpu"):
                bbox_list = xywh.cpu().numpy().tolist()
            elif xywh is not None:
                bbox_list = xywh.tolist()
            else:
                # fallback to xyxy -> convert to xywh
                xyxy = getattr(boxes, "xyxy", None)
                if xyxy is not None and hasattr(xyxy, "cpu"):
                    bbox_xyxy = xyxy.cpu().numpy().tolist()
                elif xyxy is not None:
                    bbox_xyxy = xyxy.tolist()
                else:
                    bbox_xyxy = []

                bbox_list = []
                for bb in bbox_xyxy:
                    if len(bb) >= 4:
                        x1, y1, x2, y2 = bb[:4]
                        w = x2 - x1
                        h = y2 - y1
                        bbox_list.append([x1, y1, w, h])

            for box, cls, conf in zip(bbox_list, cls_list, conf_list):
                try:
                    cls_idx = int(cls)
                    label = names[cls_idx] if isinstance(names, dict) else names[cls_idx]
                except Exception:
                    label = str(cls)
                detected_objects.append({'class': label, 'bbox': box, 'confidence': float(conf)})
        except Exception:
            # Fallback: try res.probs (classification results)
            probs = getattr(res, "probs", None)
            if probs is not None:
                try:
                    top1 = getattr(probs, "top1", None)
                    top1conf = getattr(probs, "top1conf", None)
                    idx = int(top1) if top1 is not None else None
                    label = res.names[idx] if idx is not None else "unknown"
                    conf_val = float(top1conf.cpu().numpy()) if hasattr(top1conf, "cpu") else float(top1conf) if top1conf is not None else 0.0
                    detected_objects.append({'class': label, 'bbox': None, 'confidence': conf_val})
                except Exception:
                    pass

    return jsonify(detected_objects)


@app.route('/disease_detection')
def disease_detection():
    return render_template('disease_detection.html')


@app.route('/banana_detection', methods=['GET', 'POST'])
def banana_detection():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(banana_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('banana_detection.html')


@app.route('/mango_detection', methods=['GET', 'POST'])
def mango_detection():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(mango_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('mango_detection.html')


@app.route('/pomogranate_detection', methods=['GET', 'POST'])
def pomogranate_detection():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            class_names = detect_disease(pomogranate_disease_detection_model, img)
            img_str = image_to_base64(img)
            return render_template('uploaded_image.html', img_str=img_str, class_names=class_names)
    return render_template('pomogranate_detection.html')


def detect_disease(model, image):
    """
    Run classification/disease detection on a PIL.Image or numpy image.
    Returns a list of formatted strings like "label (score)" or ["model_not_loaded"].
    """
    if model is None:
        return ["model_not_loaded"]

    # Convert PIL Image to numpy RGB (ultralytics accepts numpy/PIL, but standardize)
    if isinstance(image, Image.Image):
        img_np = np.array(image)  # RGB
    else:
        img_np = image

    try:
        results = model(img_np)
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return ["inference_error"]

    class_names = []
    for res in results:
        # 1) Try classification-style attributes (probs.top1 / top1conf)
        probs = getattr(res, "probs", None)
        if probs is not None:
            top1 = getattr(probs, "top1", None)
            top1conf = getattr(probs, "top1conf", None)
            try:
                idx = int(top1) if top1 is not None else None
                label = res.names[idx] if idx is not None else "unknown"
                score = float(top1conf.cpu().numpy()) if hasattr(top1conf, "cpu") else float(top1conf) if top1conf is not None else None
                class_names.append(f"{label} ({score:.3f})" if score is not None else label)
                continue
            except Exception:
                pass

        # 2) Try detection-style boxes with classes and confidences
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            cls_vals = getattr(boxes, "cls", None)
            conf_vals = getattr(boxes, "conf", None)
            try:
                if hasattr(cls_vals, "cpu"):
                    cls_list = cls_vals.cpu().numpy().tolist()
                elif cls_vals is not None:
                    cls_list = cls_vals.tolist()
                else:
                    cls_list = []

                if hasattr(conf_vals, "cpu"):
                    conf_list = conf_vals.cpu().numpy().tolist()
                elif conf_vals is not None:
                    conf_list = conf_vals.tolist()
                else:
                    conf_list = []

                for cls_idx, conf in zip(cls_list, conf_list):
                    try:
                        idx = int(cls_idx)
                        label = res.names[idx] if isinstance(res.names, (list, dict)) else res.names[idx]
                    except Exception:
                        label = str(cls_idx)
                    class_names.append(f"{label} ({float(conf):.3f})")
                if class_names:
                    continue
            except Exception:
                pass

        # fallback
        class_names.append("unknown")

    return class_names


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return render_template('uploaded_image.html', filename=filename)


def generate_frames():
    # Try multiple camera indices to be more robust on different machines
    camera = None
    for idx in range(0, 4):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            camera = cap
            logging.info(f"Opened camera index {idx} for server stream")
            break
        else:
            try:
                cap.release()
            except Exception:
                pass

    if camera is None:
        logging.warning("No camera available for server-side streaming. Sending placeholder frames.")
        # create a simple placeholder image
        placeholder = (np.ones((480, 640, 3), dtype=np.uint8) * 30)
        cv2.putText(placeholder, 'Camera unavailable', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.5)
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logging.warning('Failed to read frame from camera')
                time.sleep(0.1)
                continue

            image = frame.copy()
            if fruit_detection_model is not None:
                try:
                    fruit_results = fruit_detection_model(frame)
                    for result in fruit_results:
                        try:
                            im_array = result.plot()
                            im = Image.fromarray(im_array[..., ::-1])
                            image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                        except Exception:
                            pass
                except Exception as e:
                    logging.error(f"Realtime detection error: {e}")

            ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # small sleep to avoid maxing CPU
            time.sleep(0.03)
    finally:
        try:
            camera.release()
        except Exception:
            pass


if __name__ == '__main__':
    # Use 0.0.0.0 so other devices on the LAN can reach the dev server if needed.
    app.run(host="0.0.0.0", debug=True)