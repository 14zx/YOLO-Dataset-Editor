"""
YOLO Inference Server — loads model once into GPU, serves via HTTP.
Runs on port 3777.
"""
import sys, os, json, io, warnings, logging, zipfile, pickle, random, traceback, subprocess
from pathlib import Path
import numpy as np
import cv2

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["YOLO_VERBOSE"] = "false"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from flask_cors import CORS
import tempfile

app = Flask(__name__)
CORS(app)

# Global model reference
MODEL = None
MODEL_NAMES = {}
MODEL_PATH = None


def _attach_mask_polygons(preds, result):
    """
    Safely attach segmentation polygons to existing preds.
    Compatible with different Ultralytics versions where masks.xyn can fail.
    """
    masks = getattr(result, "masks", None)
    if masks is None:
        return preds

    polys = None
    try:
        # Preferred: already normalized polygons.
        polys = masks.xyn
    except Exception:
        polys = None

    # Fallback: absolute polygons -> normalize manually.
    if polys is None:
        try:
            xy = masks.xy
            h, w = result.orig_shape[:2]
            polys = []
            for pts in xy:
                if pts is None or len(pts) < 3:
                    polys.append(None)
                    continue
                polys.append([(float(p[0]) / max(1, w), float(p[1]) / max(1, h)) for p in pts])
        except Exception:
            polys = None

    if polys is None:
        return preds

    for i, pts in enumerate(polys):
        if i >= len(preds):
            break
        if pts is None or len(pts) < 3:
            continue
        preds[i]["shape"] = "polygon"
        preds[i]["points"] = [{"x": float(p[0]), "y": float(p[1])} for p in pts]
    return preds


def _bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def extract_classes_from_pt(pt_path):
    """Extract class names from .pt file using custom unpickler (no torch needed)."""
    names = {}
    try:
        class Dummy:
            def __init__(self, *a, **kw): pass
            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
            def __reduce_ex__(self, p):
                return (type(self), ())

        class TorchUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                return type(name, (Dummy,), {})
            def persistent_load(self, pid):
                return None

        with open(pt_path, 'rb') as f:
            z = zipfile.ZipFile(f)
            pkl_files = [n for n in z.namelist() if n.endswith('.pkl')]
            if pkl_files:
                obj = TorchUnpickler(io.BytesIO(z.read(pkl_files[0]))).load()
                if isinstance(obj, dict):
                    for key in ['model', 'ema']:
                        if key in obj:
                            m = obj[key]
                            if hasattr(m, 'names') and isinstance(m.names, dict) and m.names:
                                names = m.names
                                break
    except Exception as e:
        print(f"  [PY] Pickle extract error: {e}")
    return names


def load_model(pt_path):
    """Load YOLO model into GPU (or CPU if no GPU)."""
    global MODEL, MODEL_NAMES, MODEL_PATH
    from ultralytics import YOLO
    import torch

    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"  [PY] Loading model: {pt_path}")
    print(f"  [PY] Device: {'CUDA GPU' if device == '0' else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  [PY] GPU: {torch.cuda.get_device_name(0)}")

    MODEL = YOLO(pt_path)
    MODEL_PATH = pt_path
    # Warm up — run once to initialize GPU context
    MODEL.predict(
        source=os.path.join(tempfile.gettempdir(), "__warmup.png")
        if False else None,
        device=device, verbose=False
    ) if False else None

    # Store device preference
    MODEL._device = device

    # Get names
    if hasattr(MODEL, 'names') and MODEL.names:
        MODEL_NAMES = dict(MODEL.names)
    else:
        MODEL_NAMES = extract_classes_from_pt(pt_path)

    print(f"  [PY] Model loaded! Classes: {len(MODEL_NAMES)}")
    if MODEL_NAMES:
        first = list(MODEL_NAMES.values())[:5]
        print(f"  [PY] Names: {', '.join(str(n) for n in first)}{'...' if len(MODEL_NAMES) > 5 else ''}")

    return MODEL_NAMES


def _predict_with_task_fallback(source, conf, device):
    """
    Run predict and recover from Ultralytics task-mismatch crashes like:
    AttributeError: 'str' object has no attribute 'shape'
    by retrying with explicit task reload.
    """
    global MODEL, MODEL_PATH
    try:
        return MODEL.predict(source=source, conf=conf, device=device, verbose=False)
    except Exception as e:
        msg = str(e)
        if "object has no attribute 'shape'" not in msg or not MODEL_PATH:
            raise

        print("  [PY] Predictor task mismatch detected, trying explicit task reload...")
        from ultralytics import YOLO

        last_err = e
        # Try detect first, then segment.
        for forced_task in ("detect", "segment"):
            try:
                test_model = YOLO(MODEL_PATH)
                # Hard-force task to avoid checkpoint overrides selecting wrong predictor.
                test_model.task = forced_task
                test_model.overrides["task"] = forced_task
                test_model.predictor = None
                if hasattr(test_model, "model") and test_model.model is not None:
                    try:
                        test_model.model.task = forced_task
                    except Exception:
                        pass
                test_model._device = device
                out = test_model.predict(source=source, conf=conf, device=device, verbose=False)
                MODEL = test_model
                print(f"  [PY] Recovered with task='{forced_task}'")
                return out
            except Exception as e2:
                last_err = e2
                print(f"  [PY]   task='{forced_task}' failed: {e2}")
        raise last_err


@app.route('/api/extract-classes', methods=['POST'])
def api_extract_classes():
    if 'pt_file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    f = request.files['pt_file']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    f.save(tmp.name)
    tmp.close()

    print(f"  [PY] Extracting classes from: {f.filename}")

    # First extract names via pickle (fast)
    names = extract_classes_from_pt(tmp.name)

    # Then load the full model for inference
    try:
        loaded_names = load_model(tmp.name)
        if loaded_names and not names:
            names = loaded_names
        elif not loaded_names and names:
            MODEL_NAMES.update(names)
    except Exception as e:
        print(f"  [PY] Model load error: {e}")
        # Still return the names from pickle
        if not names:
            os.unlink(tmp.name)
            return jsonify({"error": str(e), "names": {}, "nc": 0})

    # Keep the temp file for inference (don't delete)
    result = {str(k): str(v) for k, v in names.items()}
    return jsonify({"names": result, "nc": len(result)})


@app.route('/api/inference', methods=['POST'])
def api_inference():
    if MODEL is None:
        return jsonify({"error": "No model loaded. Upload a .pt file first.", "predictions": [], "count": 0})

    if 'image' not in request.files:
        return jsonify({"error": "No image", "predictions": [], "count": 0})

    f = request.files['image']
    conf = float(request.form.get('conf', 0.25))
    mode = str(request.form.get('mode', 'auto')).strip().lower()  # auto | detect | segment

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    f.save(tmp.name)
    tmp.close()

    print(f"  [PY] Inference: {f.filename} (conf={conf})")

    try:
        device = getattr(MODEL, '_device', '0')
        results = _predict_with_task_fallback(source=tmp.name, conf=conf, device=device)
        predictions = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cid = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xywhn = box.xywhn[0].tolist()
                    predictions.append({
                        "class_id": cid,
                        "bbox": xywhn,
                        "conf": round(confidence, 3),
                        "shape": "rectangle",
                    })
            predictions = _attach_mask_polygons(predictions, result)
        if mode == "segment":
            seg_count = sum(1 for p in predictions if p.get("shape") == "polygon")
            if seg_count == 0:
                os.unlink(tmp.name)
                return jsonify({
                    "error": "Segment mode requested, but model did not return masks. Use a valid segment checkpoint.",
                    "predictions": [],
                    "count": 0
                }), 400
        print(f"  [PY] Found {len(predictions)} objects")
        os.unlink(tmp.name)
        return jsonify({"predictions": predictions, "count": len(predictions)})
    except Exception as e:
        print(f"  [PY] Inference error: {e}")
        print(traceback.format_exc())
        os.unlink(tmp.name)
        return jsonify({"error": str(e), "predictions": [], "count": 0})


@app.route('/api/inference-batch', methods=['POST'])
def api_inference_batch():
    if MODEL is None:
        return jsonify({"error": "No model loaded", "results": {}})

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No images", "results": {}})

    conf = float(request.form.get('conf', 0.25))
    mode = str(request.form.get('mode', 'auto')).strip().lower()  # auto | detect | segment
    device = getattr(MODEL, '_device', '0')

    print(f"  [PY] Batch: {len(files)} images (conf={conf}, device={device})")

    # Save all images to temp files
    tmp_paths = []
    filenames = []
    for f in files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        f.save(tmp.name)
        tmp.close()
        tmp_paths.append(tmp.name)
        filenames.append(f.filename)

    results_dict = {}

    try:
        # Run inference on ALL images at once (batch) for GPU efficiency
        batch_results = _predict_with_task_fallback(source=tmp_paths, conf=conf, device=device)

        for i, result in enumerate(batch_results):
            preds = []
            if result.boxes is not None:
                for box in result.boxes:
                    cid = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xywhn = box.xywhn[0].tolist()
                    preds.append({
                        "class_id": cid,
                        "bbox": xywhn,
                        "conf": round(confidence, 3),
                        "shape": "rectangle",
                    })
            preds = _attach_mask_polygons(preds, result)
            if mode == "segment":
                if not any(p.get("shape") == "polygon" for p in preds):
                    results_dict[filenames[i]] = []
                    print(f"  [PY]   {filenames[i]} -> segment masks unavailable")
                    continue
            results_dict[filenames[i]] = preds
            print(f"  [PY]   {filenames[i]} -> {len(preds)} objects")
    except Exception as e:
        print(f"  [PY] Batch error: {e}")
        print(traceback.format_exc())
        # Fallback: process one by one
        for i, tmp_path in enumerate(tmp_paths):
            try:
                res = _predict_with_task_fallback(source=tmp_path, conf=conf, device=device)
                preds = []
                for result in res:
                    if result.boxes is not None:
                        for box in result.boxes:
                            preds.append({
                                "class_id": int(box.cls[0]),
                                "bbox": box.xywhn[0].tolist(),
                                "conf": round(float(box.conf[0]), 3),
                                "shape": "rectangle",
                            })
                    preds = _attach_mask_polygons(preds, result)
                results_dict[filenames[i]] = preds
                print(f"  [PY]   {filenames[i]} -> {len(preds)} objects")
            except Exception as e2:
                results_dict[filenames[i]] = []
                print(f"  [PY]   {filenames[i]} -> error: {e2}")
                print(traceback.format_exc())

    # Cleanup
    for p in tmp_paths:
        try: os.unlink(p)
        except: pass

    return jsonify({"results": results_dict})


@app.route('/api/inference-segment-cli-batch', methods=['POST'])
def api_inference_segment_cli_batch():
    """
    Segment inference via Ultralytics CLI subprocess.
    Used as robust fallback when in-process segment predictor is unstable.
    """
    if MODEL_PATH is None:
        return jsonify({"error": "No model loaded. Upload a .pt file first.", "results": {}}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No images", "results": {}}), 400

    conf = float(request.form.get('conf', 0.25))
    imgsz = int(float(request.form.get('imgsz', 640)))

    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as out_dir:
        # Save input files
        names = []
        for f in files:
            name = os.path.basename(f.filename)
            names.append(name)
            f.save(os.path.join(src_dir, name))

        # Run CLI in subprocess
        cmd = [
            sys.executable,
            "-m",
            "ultralytics",
            "yolo",
            "segment",
            "predict",
            f"model={MODEL_PATH}",
            f"source={src_dir}",
            f"conf={conf}",
            f"imgsz={imgsz}",
            "save=False",
            "save_txt=True",
            "save_conf=False",
            f"project={out_dir}",
            "name=pred",
            "exist_ok=True",
            "verbose=False",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return jsonify({
                    "error": "CLI segment predict failed",
                    "stderr": proc.stderr[-2000:],
                    "stdout": proc.stdout[-2000:],
                    "results": {}
                }), 500
        except Exception as e:
            return jsonify({"error": f"Failed to run CLI: {e}", "results": {}}), 500

        labels_dir = os.path.join(out_dir, "pred", "labels")
        results_dict = {}
        for name in names:
            stem = os.path.splitext(name)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            preds = []
            if os.path.exists(label_path):
                try:
                    with open(label_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 7:
                                continue
                            cls = int(float(parts[0]))
                            coords = [float(v) for v in parts[1:]]
                            if len(coords) % 2 != 0:
                                continue
                            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                            if len(pts) < 3:
                                continue
                            preds.append({
                                "class_id": cls,
                                "shape": "polygon",
                                "points": [{"x": x, "y": y} for x, y in pts],
                                "bbox": _bbox_from_points(pts),
                            })
                except Exception:
                    preds = []
            results_dict[name] = preds

        return jsonify({"results": results_dict})


@app.route('/api/health', methods=['GET'])
def health():
    import torch
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "classes": len(MODEL_NAMES),
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })


@app.route('/api/export-yolo-dataset', methods=['POST'])
def api_export_yolo_dataset():
    """
    Build and return YOLO dataset zip:
      dataset/
        images/train, images/val
        labels/train, labels/val
        data.yaml
    """
    payload_raw = request.form.get("payload", "")
    if not payload_raw:
        return jsonify({"error": "No payload"}), 400

    try:
        payload = json.loads(payload_raw)
    except Exception:
        return jsonify({"error": "Invalid payload JSON"}), 400

    classes = payload.get("classes", [])
    task = str(payload.get("task", "detect")).strip().lower()
    items = payload.get("items", [])
    val_split = float(payload.get("val_split", 0.2))
    seed = int(payload.get("seed", 42))
    tiling = payload.get("tiling", {}) or {}
    augment = payload.get("augment", {}) or {}
    use_tiling = bool(tiling.get("enabled", False))
    tile_size = int(tiling.get("tile_size", 640) or 640)
    overlap = float(tiling.get("overlap", 0.2) or 0.2)
    flip_h = bool(augment.get("flip_h", False))
    flip_v = bool(augment.get("flip_v", False))
    yaml_path = payload.get("yaml_path")
    min_object_fraction = float(payload.get("min_object_fraction", 0.30))
    if not isinstance(classes, list) or not isinstance(items, list):
        return jsonify({"error": "Invalid payload fields"}), 400
    if task not in {"detect", "segment"}:
        return jsonify({"error": "Unsupported task"}), 400

    files = request.files.getlist("images")
    file_map = {f.filename: f for f in files}

    pairs = []
    for it in items:
        name = it.get("name")
        label = it.get("label_segment", "") if task == "segment" else it.get("label_detect", "")
        if name in file_map:
            pairs.append((name, label))

    if not pairs:
        return jsonify({"error": "No matched image-label pairs"}), 400

    val_split = max(0.0, min(0.9, val_split))
    tile_size = max(128, min(4096, tile_size))
    overlap = max(0.0, min(0.7, overlap))
    min_object_fraction = max(0.01, min(0.95, min_object_fraction))

    def parse_label_txt(txt):
        anns = []
        for line in (txt or "").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = [float(v) for v in parts[1:5]]
                anns.append((cls, cx, cy, w, h))
            except Exception:
                continue
        return anns

    def parse_segment_label_txt(txt):
        anns = []
        for line in (txt or "").splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                cls = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
                if len(coords) % 2 != 0:
                    continue
                pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                anns.append((cls, pts))
            except Exception:
                continue
        return anns

    def encode_png(img):
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None

    def anns_to_txt(anns):
        lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in anns]
        return "\n".join(lines)

    def flip_annotations(anns, do_h=False, do_v=False):
        out = []
        for cls, cx, cy, w, h in anns:
            nx = 1.0 - cx if do_h else cx
            ny = 1.0 - cy if do_v else cy
            out.append((cls, nx, ny, w, h))
        return out

    def flip_segment_annotations(anns, do_h=False, do_v=False):
        out = []
        for cls, pts in anns:
            npts = []
            for x, y in pts:
                nx = 1.0 - x if do_h else x
                ny = 1.0 - y if do_v else y
                npts.append((nx, ny))
            out.append((cls, npts))
        return out

    def segment_anns_to_txt(anns):
        lines = []
        for cls, pts in anns:
            if len(pts) < 3:
                continue
            body = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
            lines.append(f"{cls} {body}")
        return "\n".join(lines)

    def poly_area(pts):
        if len(pts) < 3:
            return 0.0
        s = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    def clip_polygon_to_rect(pts, x_min, y_min, x_max, y_max):
        # Sutherland–Hodgman clipping against axis-aligned rectangle
        def clip_edge(subject, inside_fn, intersect_fn):
            if not subject:
                return []
            out = []
            prev = subject[-1]
            prev_in = inside_fn(prev)
            for cur in subject:
                cur_in = inside_fn(cur)
                if cur_in:
                    if not prev_in:
                        out.append(intersect_fn(prev, cur))
                    out.append(cur)
                elif prev_in:
                    out.append(intersect_fn(prev, cur))
                prev, prev_in = cur, cur_in
            return out

        def intersect_vertical(p1, p2, x):
            x1, y1 = p1
            x2, y2 = p2
            if abs(x2 - x1) < 1e-9:
                return (x, y1)
            t = (x - x1) / (x2 - x1)
            return (x, y1 + t * (y2 - y1))

        def intersect_horizontal(p1, p2, y):
            x1, y1 = p1
            x2, y2 = p2
            if abs(y2 - y1) < 1e-9:
                return (x1, y)
            t = (y - y1) / (y2 - y1)
            return (x1 + t * (x2 - x1), y)

        out = [(float(x), float(y)) for x, y in pts]
        out = clip_edge(out, lambda p: p[0] >= x_min, lambda a, b: intersect_vertical(a, b, x_min))
        out = clip_edge(out, lambda p: p[0] <= x_max, lambda a, b: intersect_vertical(a, b, x_max))
        out = clip_edge(out, lambda p: p[1] >= y_min, lambda a, b: intersect_horizontal(a, b, y_min))
        out = clip_edge(out, lambda p: p[1] <= y_max, lambda a, b: intersect_horizontal(a, b, y_max))
        # dedupe adjacent almost-identical points
        cleaned = []
        for p in out:
            if not cleaned or abs(cleaned[-1][0] - p[0]) > 1e-6 or abs(cleaned[-1][1] - p[1]) > 1e-6:
                cleaned.append(p)
        if len(cleaned) >= 2 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6:
            cleaned = cleaned[:-1]
        return cleaned

    def tile_sample(img, anns):
        ih, iw = img.shape[:2]
        step = int(tile_size * (1.0 - overlap))
        step = max(32, step)
        tiles = []
        if iw <= tile_size and ih <= tile_size:
            tiles.append((img, anns, 0, 0))
            return tiles

        xs = list(range(0, max(1, iw - tile_size + 1), step))
        ys = list(range(0, max(1, ih - tile_size + 1), step))
        if xs[-1] != max(0, iw - tile_size):
            xs.append(max(0, iw - tile_size))
        if ys[-1] != max(0, ih - tile_size):
            ys.append(max(0, ih - tile_size))

        for y0 in ys:
            for x0 in xs:
                x1 = min(iw, x0 + tile_size)
                y1 = min(ih, y0 + tile_size)
                tile_img = img[y0:y1, x0:x1]
                tw = x1 - x0
                th = y1 - y0
                tile_anns = []
                for cls, cx, cy, w, h in anns:
                    bx1 = (cx - w / 2.0) * iw
                    by1 = (cy - h / 2.0) * ih
                    bx2 = (cx + w / 2.0) * iw
                    by2 = (cy + h / 2.0) * ih
                    ix1 = max(bx1, x0)
                    iy1 = max(by1, y0)
                    ix2 = min(bx2, x1)
                    iy2 = min(by2, y1)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    orig_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
                    if inter_area / orig_area < min_object_fraction:
                        continue
                    tcx = ((ix1 + ix2) / 2.0 - x0) / max(1, tw)
                    tcy = ((iy1 + iy2) / 2.0 - y0) / max(1, th)
                    twb = (ix2 - ix1) / max(1, tw)
                    thb = (iy2 - iy1) / max(1, th)
                    tile_anns.append((cls, tcx, tcy, twb, thb))
                if tile_anns:
                    tiles.append((tile_img, tile_anns, x0, y0))
        return tiles

    random.seed(seed)
    random.shuffle(pairs)
    val_count = int(len(pairs) * val_split)
    if len(pairs) > 1:
        val_count = max(1, val_count)
    val_names = set(name for name, _ in pairs[:val_count])

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, label in pairs:
            split = "val" if name in val_names else "train"
            img_file = file_map[name]
            img_file.stream.seek(0)
            raw = img_file.read()
            np_arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if task == "segment":
                base_anns = parse_segment_label_txt(label)
            else:
                base_anns = parse_label_txt(label)
            base_name = os.path.splitext(name)[0]

            variants = [("orig", img, base_anns)]
            if flip_h:
                variants.append(
                    ("fh", cv2.flip(img, 1), flip_segment_annotations(base_anns, do_h=True) if task == "segment" else flip_annotations(base_anns, do_h=True))
                )
            if flip_v:
                variants.append(
                    ("fv", cv2.flip(img, 0), flip_segment_annotations(base_anns, do_v=True) if task == "segment" else flip_annotations(base_anns, do_v=True))
                )

            for vtag, vimg, vanns in variants:
                if use_tiling and task == "detect":
                    tiles = tile_sample(vimg, vanns)
                    for idx, (timg, tanns, tx, ty) in enumerate(tiles):
                        img_bytes = encode_png(timg)
                        if not img_bytes:
                            continue
                        fname = f"{base_name}_{vtag}_tile_{idx:04d}.png"
                        zf.writestr(f"dataset/images/{split}/{fname}", img_bytes)
                        zf.writestr(
                            f"dataset/labels/{split}/{os.path.splitext(fname)[0]}.txt",
                            anns_to_txt(tanns),
                        )
                elif use_tiling and task == "segment":
                    ih, iw = vimg.shape[:2]
                    step = int(tile_size * (1.0 - overlap))
                    step = max(32, step)
                    xs = list(range(0, max(1, iw - tile_size + 1), step))
                    ys = list(range(0, max(1, ih - tile_size + 1), step))
                    if xs[-1] != max(0, iw - tile_size):
                        xs.append(max(0, iw - tile_size))
                    if ys[-1] != max(0, ih - tile_size):
                        ys.append(max(0, ih - tile_size))

                    tile_idx = 0
                    for y0 in ys:
                        for x0 in xs:
                            x1 = min(iw, x0 + tile_size)
                            y1 = min(ih, y0 + tile_size)
                            timg = vimg[y0:y1, x0:x1]
                            tw = max(1, x1 - x0)
                            th = max(1, y1 - y0)
                            tile_anns = []
                            for cls, pts in vanns:
                                abs_pts = [(px * iw, py * ih) for px, py in pts]
                                orig_area = poly_area(abs_pts)
                                if orig_area <= 1e-6:
                                    continue
                                clipped = clip_polygon_to_rect(abs_pts, x0, y0, x1, y1)
                                if len(clipped) < 3:
                                    continue
                                clipped_area = poly_area(clipped)
                                if clipped_area / orig_area < min_object_fraction:
                                    continue
                                norm = [((px - x0) / tw, (py - y0) / th) for px, py in clipped]
                                tile_anns.append((cls, norm))
                            if not tile_anns:
                                continue
                            img_bytes = encode_png(timg)
                            if not img_bytes:
                                continue
                            fname = f"{base_name}_{vtag}_tile_{tile_idx:04d}.png"
                            tile_idx += 1
                            zf.writestr(f"dataset/images/{split}/{fname}", img_bytes)
                            zf.writestr(
                                f"dataset/labels/{split}/{os.path.splitext(fname)[0]}.txt",
                                segment_anns_to_txt(tile_anns),
                            )
                else:
                    img_bytes = encode_png(vimg)
                    if not img_bytes:
                        continue
                    fname = f"{base_name}_{vtag}.png" if vtag != "orig" else f"{base_name}.png"
                    zf.writestr(f"dataset/images/{split}/{fname}", img_bytes)
                    zf.writestr(
                        f"dataset/labels/{split}/{os.path.splitext(fname)[0]}.txt",
                        segment_anns_to_txt(vanns) if task == "segment" else anns_to_txt(vanns),
                    )

        # Optional explicit path for older Ultralytics/Windows setups.
        path_line = f"path: {yaml_path}\n" if isinstance(yaml_path, str) and yaml_path.strip() else "path: .\n"
        yaml_text = (
            path_line +
            "train: images/train\n"
            "val: images/val\n"
            f"nc: {len(classes)}\n"
            f"editor_task: {task}\n"
            f"editor_imgsz_hint: {tile_size if use_tiling else 640}\n"
            f"editor_min_object_fraction: {min_object_fraction:.2f}\n"
            "names:\n" +
            "".join([f"  {i}: {str(c)}\n" for i, c in enumerate(classes)])
        )
        zf.writestr("dataset/data.yaml", yaml_text)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="yolo_dataset_export.zip",
    )


def _resolve_static_root():
    """Bundled Vite `dist/` next to exe (_MEIPASS) or `web/dist` in dev."""
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        root = Path(__file__).resolve().parent
    for cand in (root / "dist", root / "web" / "dist"):
        if cand.is_dir():
            return cand
    return None


STATIC_ROOT = _resolve_static_root()


if STATIC_ROOT is not None:

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def _serve_yolo_editor_frontend(path: str):
        if path.startswith("api"):
            abort(404)
        base = STATIC_ROOT.resolve()
        if path:
            target = (STATIC_ROOT / path).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                abort(404)
            if target.is_file():
                return send_from_directory(str(STATIC_ROOT), path.replace("\\", "/"))
        return send_from_directory(str(STATIC_ROOT), "index.html")


def run_server(host="0.0.0.0", port=3777):
    print(f"  [PY] Starting YOLO inference server on port {port}...")
    if STATIC_ROOT is not None:
        print(f"  [PY] UI: http://127.0.0.1:{port}/")
    else:
        print(f"  [PY] API only (no dist/). For UI run: cd web && npm run dev")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    run_server()
