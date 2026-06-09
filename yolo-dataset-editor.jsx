import { useState, useRef, useCallback, useEffect } from "react";

const DEMO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush"
];

const CLASS_COLORS = [
  "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE", "#30B0C7",
  "#007AFF", "#5856D6", "#AF52DE", "#FF2D55", "#A2845E", "#8E8E93",
  "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
  "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F1948A", "#82E0AA",
];
const AUTOSAVE_KEY = "yolo-editor-autosave-v1";

function getColor(classId) {
  return CLASS_COLORS[classId % CLASS_COLORS.length];
}

// Convert YOLO format (cx, cy, w, h) to pixel coords
function yoloToPixel(yolo, imgW, imgH) {
  const [cx, cy, w, h] = yolo;
  return {
    x: (cx - w / 2) * imgW,
    y: (cy - h / 2) * imgH,
    w: w * imgW,
    h: h * imgH,
  };
}

// Convert pixel coords back to YOLO format
function pixelToYolo(px, imgW, imgH) {
  const cx = (px.x + px.w / 2) / imgW;
  const cy = (px.y + px.h / 2) / imgH;
  const w = px.w / imgW;
  const h = px.h / imgH;
  return [cx, cy, w, h];
}

function bboxIouYolo(a, b) {
  const aX1 = a[0] - a[2] / 2;
  const aY1 = a[1] - a[3] / 2;
  const aX2 = a[0] + a[2] / 2;
  const aY2 = a[1] + a[3] / 2;
  const bX1 = b[0] - b[2] / 2;
  const bY1 = b[1] - b[3] / 2;
  const bX2 = b[0] + b[2] / 2;
  const bY2 = b[1] + b[3] / 2;
  const ix1 = Math.max(aX1, bX1);
  const iy1 = Math.max(aY1, bY1);
  const ix2 = Math.min(aX2, bX2);
  const iy2 = Math.min(aY2, bY2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const areaA = Math.max(0, aX2 - aX1) * Math.max(0, aY2 - aY1);
  const areaB = Math.max(0, bX2 - bX1) * Math.max(0, bY2 - bY1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function parseYoloLabel(text) {
  return text
    .trim()
    .split("\n")
    .filter((l) => l.trim())
    .map((line) => {
      const parts = line.trim().split(/\s+/);
      return {
        classId: parseInt(parts[0]),
        bbox: parts.slice(1, 5).map(Number),
        shape: "rectangle",
        id: Math.random().toString(36).slice(2, 10),
      };
    });
}

function toYoloText(annotations) {
  return annotations
    .map((a) => `${a.classId} ${a.bbox.map((v) => v.toFixed(6)).join(" ")}`)
    .join("\n");
}

export default function YoloDatasetEditor() {
  const [images, setImages] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [annotations, setAnnotations] = useState({});
  const [classes, setClasses] = useState(DEMO_CLASSES);
  const [newClassName, setNewClassName] = useState("");
  const [selectedBox, setSelectedBox] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [drawEnd, setDrawEnd] = useState(null);
  const [activeClassId, setActiveClassId] = useState(0);
  const [tool, setTool] = useState("select"); // select | draw | draw_circle | draw_poly | replace_class | delete
  const [polygonPoints, setPolygonPoints] = useState([]);
  const [polygonHover, setPolygonHover] = useState(null);
  const [dragInfo, setDragInfo] = useState(null);
  const [resizeInfo, setResizeInfo] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [showClasses, setShowClasses] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [editingClass, setEditingClass] = useState(null);
  const [editingClassName, setEditingClassName] = useState("");
  const [remapFrom, setRemapFrom] = useState("");
  const [remapTo, setRemapTo] = useState("");
  const [ptFile, setPtFile] = useState(null);
  const [ptStatus, setPtStatus] = useState(""); // "" | "loading" | "done" | "error"
  const [inferring, setInferring] = useState(false); // single image
  const [batchInferring, setBatchInferring] = useState(false);
  const [segmentCliInferring, setSegmentCliInferring] = useState(false);
  const [batchProgress, setBatchProgress] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [aiMode, setAiMode] = useState("auto"); // auto | detect | segment
  const [duplicatesSkipped, setDuplicatesSkipped] = useState(0);
  const [exportImageMode, setExportImageMode] = useState("labeled"); // all | labeled | current
  const [exportTask, setExportTask] = useState("detect"); // detect | segment
  const [exportClassFilter, setExportClassFilter] = useState([]);
  const [exportSkipEmptyAfterFilter, setExportSkipEmptyAfterFilter] = useState(true);
  const [exportValSplit, setExportValSplit] = useState(0.2);
  const [exportTileSize, setExportTileSize] = useState(640);
  const [exportTileOverlap, setExportTileOverlap] = useState(0.2);
  const [exportMinObjectFraction, setExportMinObjectFraction] = useState(0.3);
  const [exportUseTiling, setExportUseTiling] = useState(false);
  const [exportFlipH, setExportFlipH] = useState(true);
  const [exportFlipV, setExportFlipV] = useState(false);
  const [exportStatus, setExportStatus] = useState("");
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportEstimate, setExportEstimate] = useState("");
  const [importConflicts, setImportConflicts] = useState([]);
  const [autosaveState, setAutosaveState] = useState("idle"); // idle | restored | saved | error
  const [view, setView] = useState("editor"); // editor | script | export
  const imageHashesRef = useRef(new Set()); // stores hashes of loaded images
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const stateInputRef = useRef(null);

  const currentImage = images[currentIdx];
  const currentAnnotations = currentImage
    ? annotations[currentImage.name] || []
    : [];

  const detectImportConflicts = useCallback((imageName, anns) => {
    const messages = [];
    for (let i = 0; i < anns.length; i++) {
      for (let j = i + 1; j < anns.length; j++) {
        if (anns[i].classId === anns[j].classId) continue;
        const iou = bboxIouYolo(anns[i].bbox, anns[j].bbox);
        if (iou >= 0.7) {
          messages.push(
            `${imageName}: overlap ${iou.toFixed(2)} между классами ${anns[i].classId} и ${anns[j].classId}`
          );
        }
      }
    }
    return messages;
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(AUTOSAVE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.classes)) setClasses(parsed.classes);
      if (parsed.annotations && typeof parsed.annotations === "object") {
        setAnnotations(parsed.annotations);
      }
      if (Array.isArray(parsed.exportClassFilter)) {
        setExportClassFilter(parsed.exportClassFilter);
      }
      if (typeof parsed.exportTask === "string") setExportTask(parsed.exportTask);
      if (typeof parsed.exportImageMode === "string") {
        setExportImageMode(parsed.exportImageMode);
      }
      if (typeof parsed.exportSkipEmptyAfterFilter === "boolean") {
        setExportSkipEmptyAfterFilter(parsed.exportSkipEmptyAfterFilter);
      }
      if (typeof parsed.exportTileSize === "number") setExportTileSize(parsed.exportTileSize);
      if (typeof parsed.exportTileOverlap === "number") setExportTileOverlap(parsed.exportTileOverlap);
      if (typeof parsed.exportMinObjectFraction === "number") setExportMinObjectFraction(parsed.exportMinObjectFraction);
      if (typeof parsed.exportUseTiling === "boolean") setExportUseTiling(parsed.exportUseTiling);
      if (typeof parsed.exportFlipH === "boolean") setExportFlipH(parsed.exportFlipH);
      if (typeof parsed.exportFlipV === "boolean") setExportFlipV(parsed.exportFlipV);
      setAutosaveState("restored");
    } catch {
      setAutosaveState("error");
    }
  }, []);

  useEffect(() => {
    try {
      const payload = {
        version: 1,
        classes,
        annotations,
        exportClassFilter,
        exportTask,
        exportImageMode,
        exportSkipEmptyAfterFilter,
        exportTileSize,
        exportTileOverlap,
        exportMinObjectFraction,
        exportUseTiling,
        exportFlipH,
        exportFlipV,
      };
      localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(payload));
      setAutosaveState("saved");
    } catch {
      setAutosaveState("error");
    }
  }, [
    classes,
    annotations,
    exportClassFilter,
    exportTask,
    exportImageMode,
    exportSkipEmptyAfterFilter,
    exportTileSize,
    exportTileOverlap,
    exportMinObjectFraction,
    exportUseTiling,
    exportFlipH,
    exportFlipV,
  ]);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    const imgFiles = files.filter((f) => f.type.startsWith("image/"));
    const txtFiles = files.filter((f) => f.name.endsWith(".txt"));
    const yamlFiles = files.filter(
      (f) => f.name.endsWith(".yaml") || f.name.endsWith(".yml")
    );
    const ptFiles = files.filter((f) => f.name.endsWith(".pt"));

    if (ptFiles.length > 0) {
      setPtFile(ptFiles[0]);
      setPtStatus("loading");
      const formData = new FormData();
      formData.append("pt_file", ptFiles[0]);
      fetch("/api/extract-classes", { method: "POST", body: formData })
        .then((r) => r.json())
        .then((data) => {
          console.log("PT extraction result:", data);
          if (data.names && !data.error) {
            let names;
            if (typeof data.names === "object" && !Array.isArray(data.names)) {
              // Sort by numeric key: {"0": "person", "1": "car"} -> ["person", "car"]
              const keys = Object.keys(data.names).sort((a, b) => Number(a) - Number(b));
              names = keys.map((k) => data.names[k]);
            } else {
              names = data.names;
            }
            if (names.length > 0) {
              setClasses(names);
              setPtStatus("done");
            } else {
              setPtStatus("error");
            }
          } else {
            console.log("PT error:", data.error);
            setPtStatus("error");
          }
        })
        .catch((e) => { console.error("PT fetch error:", e); setPtStatus("error"); });
    }

    // Parse YAML for class names
    yamlFiles.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target.result;
        const namesMatch = text.match(/names:\s*\n([\s\S]*?)(?:\n\w|\n*$)/);
        if (namesMatch) {
          const names = namesMatch[1]
            .split("\n")
            .map((l) => l.replace(/^\s*-\s*/, "").replace(/['"\s]/g, ""))
            .filter(Boolean);
          if (names.length > 0) setClasses(names);
        }
      };
      reader.readAsText(file);
    });

    // ─── Fast hash: SHA-256 of first 64KB + file size (enough for dedup) ───
    const hashFileFast = (file) => {
      return new Promise((resolve) => {
        const chunk = file.slice(0, 65536); // first 64KB only
        const reader = new FileReader();
        reader.onload = async (ev) => {
          try {
            const buffer = ev.target.result;
            const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
            const hashArray = Array.from(new Uint8Array(hashBuffer));
            const hex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
            resolve(hex + "_" + file.size);
          } catch {
            resolve(null);
          }
        };
        reader.onerror = () => resolve(null);
        reader.readAsArrayBuffer(chunk);
      });
    };

    // ─── Process images with dedup (parallel batches) ───
    const processImages = async () => {
      // Build existing names set for O(1) lookup
      const existingNames = new Set(images.map((img) => img.name));

      // 1. Fast filter by name
      const nameFiltered = imgFiles.filter((f) => !existingNames.has(f.name));
      const skippedByName = imgFiles.length - nameFiltered.length;

      // 2. Hash all remaining in parallel
      const hashPromises = nameFiltered.map((f) => hashFileFast(f).then((h) => ({ file: f, hash: h })));
      const hashed = await Promise.all(hashPromises);

      // 3. Dedup by hash
      const newImages = [];
      const newAnnotations = {};
      let skippedByHash = 0;

      for (const { file: imgFile, hash } of hashed) {
        if (hash && imageHashesRef.current.has(hash)) {
          skippedByHash++;
          continue;
        }
        if (hash) imageHashesRef.current.add(hash);

        const url = URL.createObjectURL(imgFile);
        newImages.push({ name: imgFile.name, url, file: imgFile });
      }

      // 4. Load labels in parallel
      const labelPromises = newImages.map(({ name }) => {
        const baseName = name.replace(/\.[^.]+$/, "");
        const labelFile = txtFiles.find((f) => f.name === baseName + ".txt");
        if (!labelFile) return Promise.resolve(null);
        return new Promise((res) => {
          const reader = new FileReader();
          reader.onload = (ev) => res({ name, text: ev.target.result });
          reader.readAsText(labelFile);
        });
      });

      const labels = await Promise.all(labelPromises);
      const conflicts = [];
      for (const label of labels) {
        if (label) {
          const parsed = parseYoloLabel(label.text);
          newAnnotations[label.name] = parsed;
          conflicts.push(...detectImportConflicts(label.name, parsed));
        }
      }
      if (conflicts.length > 0) {
        setImportConflicts((prev) => [...conflicts.slice(0, 200), ...prev].slice(0, 500));
      }

      const totalSkipped = skippedByName + skippedByHash;
      if (totalSkipped > 0) {
        setDuplicatesSkipped((prev) => prev + totalSkipped);
      }

      if (newImages.length > 0) {
        setImages((prev) => [...prev, ...newImages]);
        setAnnotations((prev) => {
          const next = { ...prev };
          // Keep previously restored annotations if they already exist.
          newImages.forEach(({ name }) => {
            if (!next[name]) next[name] = [];
          });
          Object.entries(newAnnotations).forEach(([name, anns]) => {
            next[name] = anns;
          });
          return next;
        });
        if (images.length === 0) setCurrentIdx(0);
      }
    };

    processImages();
  };

  const handleImgLoad = () => {
    if (imgRef.current) {
      const w = imgRef.current.naturalWidth;
      const h = imgRef.current.naturalHeight;
      setImgSize({ w, h });
      // Auto-fit zoom to container
      if (containerRef.current) {
        const cw = containerRef.current.clientWidth - 40;
        const ch = containerRef.current.clientHeight - 40;
        const fitZoom = Math.min(cw / w, ch / h, 1);
        setZoom(Math.round(fitZoom * 100) / 100);
        setPan({ x: 0, y: 0 });
      }
    }
  };

  // Scroll wheel zoom — zooms toward cursor position
  const handleWheel = useCallback(
    (e) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setZoom((prev) => {
        const next = Math.round(Math.max(0.1, Math.min(5, prev + delta)) * 100) / 100;
        // Adjust pan to zoom toward cursor
        const rect = canvasRef.current?.getBoundingClientRect();
        if (rect) {
          const cx = e.clientX - rect.left;
          const cy = e.clientY - rect.top;
          const scale = next / prev;
          setPan((p) => ({
            x: cx - scale * (cx - p.x),
            y: cy - scale * (cy - p.y),
          }));
        }
        return next;
      });
    },
    []
  );

  // Middle mouse or space+drag to pan
  const handleCanvasMouseDown = (e) => {
    e.preventDefault();
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      // Middle click or Alt+click = pan
      setPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      return;
    }
    handleMouseDown(e);
  };

  const handleCanvasMouseMove = (e) => {
    if (panning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
      return;
    }
    handleMouseMove(e);
  };

  const handleCanvasMouseUp = (e) => {
    if (panning) {
      setPanning(false);
      return;
    }
    handleMouseUp(e);
  };

  // Reset pan/zoom on image change
  useEffect(() => {
    setPan({ x: 0, y: 0 });
  }, [currentIdx]);

  const getMousePos = useCallback(
    (e) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      return {
        // rect already includes translate(pan) and scale(zoom), so no extra pan subtraction here
        x: (e.clientX - rect.left) / zoom,
        y: (e.clientY - rect.top) / zoom,
      };
    },
    [zoom]
  );

  const displayW = imgSize.w || 800;
  const displayH = imgSize.h || 600;
  const isDrawingTool = tool === "draw" || tool === "draw_circle" || tool === "draw_poly";
  const closeDistancePx = 10 / Math.max(zoom, 0.1);

  const pointsToPixelBBox = useCallback((points) => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return {
      x: Math.min(...xs),
      y: Math.min(...ys),
      w: Math.max(...xs) - Math.min(...xs),
      h: Math.max(...ys) - Math.min(...ys),
    };
  }, []);

  const polygonToYoloPoints = useCallback(
    (points) =>
      points.map((p) => ({
        x: Math.max(0, Math.min(1, p.x / displayW)),
        y: Math.max(0, Math.min(1, p.y / displayH)),
      })),
    [displayH, displayW]
  );

  const yoloToPixelPoints = useCallback(
    (points = []) =>
      points.map((p) => ({
        x: p.x * displayW,
        y: p.y * displayH,
      })),
    [displayH, displayW]
  );

  const isPointInPolygon = useCallback((point, polygon) => {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x;
      const yi = polygon[i].y;
      const xj = polygon[j].x;
      const yj = polygon[j].y;
      const intersects = (yi > point.y) !== (yj > point.y)
        && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
      if (intersects) inside = !inside;
    }
    return inside;
  }, []);

  const isAnnotationHit = useCallback(
    (ann, pos) => {
      const px = yoloToPixel(ann.bbox, displayW, displayH);
      if (ann.shape === "circle") {
        return (
          Math.hypot(pos.x - (px.x + px.w / 2), pos.y - (px.y + px.h / 2)) <=
          Math.min(px.w, px.h) / 2
        );
      }
      if (ann.shape === "polygon" && ann.points) {
        return isPointInPolygon(pos, yoloToPixelPoints(ann.points));
      }
      return pos.x >= px.x && pos.x <= px.x + px.w && pos.y >= px.y && pos.y <= px.y + px.h;
    },
    [displayH, displayW, isPointInPolygon, yoloToPixelPoints]
  );

  const addAnnotationFromPixelBox = useCallback(
    (px, extra = {}) => {
      if (!currentImage) return;
      const x1 = Math.max(0, Math.min(displayW, px.x));
      const y1 = Math.max(0, Math.min(displayH, px.y));
      const x2 = Math.max(0, Math.min(displayW, px.x + px.w));
      const y2 = Math.max(0, Math.min(displayH, px.y + px.h));
      const clamped = {
        x: Math.min(x1, x2),
        y: Math.min(y1, y2),
        w: Math.abs(x2 - x1),
        h: Math.abs(y2 - y1),
      };
      if (clamped.w <= 5 || clamped.h <= 5) return;

      const bbox = pixelToYolo(clamped, displayW, displayH);
      const newAnn = {
        classId: activeClassId,
        bbox,
        shape: "rectangle",
        id: Math.random().toString(36).slice(2, 10),
        ...extra,
      };
      setAnnotations((prev) => ({
        ...prev,
        [currentImage.name]: [...(prev[currentImage.name] || []), newAnn],
      }));
      setSelectedBox(newAnn.id);
    },
    [activeClassId, currentImage, displayH, displayW]
  );

  const addPolygonAnnotation = useCallback(
    (points) => {
      if (points.length < 3) return;
      const bboxPx = pointsToPixelBBox(points);
      addAnnotationFromPixelBox(bboxPx, {
        shape: "polygon",
        points: polygonToYoloPoints(points),
      });
    },
    [addAnnotationFromPixelBox, pointsToPixelBBox, polygonToYoloPoints]
  );

  const handleMouseDown = (e) => {
    const pos = getMousePos(e);

    if (tool === "draw" || tool === "draw_circle") {
      setDrawing(true);
      setDrawStart(pos);
      setDrawEnd(pos);
      setSelectedBox(null);
      return;
    }

    if (tool === "draw_poly") {
      if (e.button === 2) {
        e.preventDefault();
        setPolygonPoints((prev) => prev.slice(0, -1));
        return;
      }
      if (e.button !== 0) return;
      setSelectedBox(null);
      if (polygonPoints.length >= 3) {
        const first = polygonPoints[0];
        const isCloseToFirst = Math.hypot(pos.x - first.x, pos.y - first.y) <= closeDistancePx;
        if (isCloseToFirst) {
          finishPolygon();
          return;
        }
      }
      setPolygonPoints((prev) => [...prev, pos]);
      return;
    }

    if (tool === "replace_class") {
      for (const ann of [...currentAnnotations].reverse()) {
        if (isAnnotationHit(ann, pos)) {
          changeAnnotationClass(ann.id, activeClassId);
          setSelectedBox(ann.id);
          return;
        }
      }
      return;
    }

    if (tool === "select") {
      // Check resize handles first
      for (const ann of currentAnnotations) {
        if (ann.shape === "polygon") continue;
        const px = yoloToPixel(ann.bbox, displayW, displayH);
        const handles = [
          { cursor: "nw", x: px.x, y: px.y },
          { cursor: "ne", x: px.x + px.w, y: px.y },
          { cursor: "sw", x: px.x, y: px.y + px.h },
          { cursor: "se", x: px.x + px.w, y: px.y + px.h },
        ];
        for (const h of handles) {
          if (Math.abs(pos.x - h.x) < 8 && Math.abs(pos.y - h.y) < 8) {
            setResizeInfo({
              id: ann.id,
              handle: h.cursor,
              startPos: pos,
              origBox: { ...px },
            });
            setSelectedBox(ann.id);
            return;
          }
        }
      }

      // Check box click
      for (const ann of [...currentAnnotations].reverse()) {
        const px = yoloToPixel(ann.bbox, displayW, displayH);
        if (isAnnotationHit(ann, pos)) {
          setSelectedBox(ann.id);
          setDragInfo({
            id: ann.id,
            shape: ann.shape || "rectangle",
            startPos: pos,
            origBox: { ...px },
            origPoints: ann.points ? yoloToPixelPoints(ann.points) : null,
          });

          if (tool === "delete" || e.shiftKey) {
            deleteAnnotation(ann.id);
          }
          return;
        }
      }
      setSelectedBox(null);
    }

    if (tool === "delete") {
      for (const ann of [...currentAnnotations].reverse()) {
        if (isAnnotationHit(ann, pos)) {
          deleteAnnotation(ann.id);
          return;
        }
      }
    }
  };

  const handleMouseMove = (e) => {
    const pos = getMousePos(e);

    if (drawing && drawStart) {
      setDrawEnd(pos);
      return;
    }

    if (tool === "draw_poly") {
      setPolygonHover(pos);
      return;
    }

    if (dragInfo) {
      const dx = pos.x - dragInfo.startPos.x;
      const dy = pos.y - dragInfo.startPos.y;
      if (dragInfo.shape === "polygon" && dragInfo.origPoints) {
        const moved = dragInfo.origPoints.map((p) => ({ x: p.x + dx, y: p.y + dy }));
        const bbox = pointsToPixelBBox(moved);
        const offsetX = bbox.x < 0 ? -bbox.x : bbox.x + bbox.w > displayW ? displayW - (bbox.x + bbox.w) : 0;
        const offsetY = bbox.y < 0 ? -bbox.y : bbox.y + bbox.h > displayH ? displayH - (bbox.y + bbox.h) : 0;
        const clamped = moved.map((p) => ({ x: p.x + offsetX, y: p.y + offsetY }));
        const clampedBBox = pointsToPixelBBox(clamped);
        setAnnotations((prev) => ({
          ...prev,
          [currentImage.name]: prev[currentImage.name].map((a) =>
            a.id === dragInfo.id
              ? {
                  ...a,
                  bbox: pixelToYolo(clampedBBox, displayW, displayH),
                  points: polygonToYoloPoints(clamped),
                }
              : a
          ),
        }));
        return;
      }
      const newPx = {
        x: Math.max(0, Math.min(displayW - dragInfo.origBox.w, dragInfo.origBox.x + dx)),
        y: Math.max(0, Math.min(displayH - dragInfo.origBox.h, dragInfo.origBox.y + dy)),
        w: dragInfo.origBox.w,
        h: dragInfo.origBox.h,
      };
      updateAnnotationPixel(dragInfo.id, newPx);
      return;
    }

    if (resizeInfo) {
      const dx = pos.x - resizeInfo.startPos.x;
      const dy = pos.y - resizeInfo.startPos.y;
      const orig = resizeInfo.origBox;
      let newPx = { ...orig };

      if (resizeInfo.handle.includes("e")) newPx.w = Math.max(10, orig.w + dx);
      if (resizeInfo.handle.includes("w")) {
        newPx.x = orig.x + dx;
        newPx.w = Math.max(10, orig.w - dx);
      }
      if (resizeInfo.handle.includes("s")) newPx.h = Math.max(10, orig.h + dy);
      if (resizeInfo.handle.includes("n")) {
        newPx.y = orig.y + dy;
        newPx.h = Math.max(10, orig.h - dy);
      }
      updateAnnotationPixel(resizeInfo.id, newPx);
      return;
    }
  };

  const handleMouseUp = () => {
    if (drawing && drawStart && drawEnd) {
      if (tool === "draw") {
        const x = Math.min(drawStart.x, drawEnd.x);
        const y = Math.min(drawStart.y, drawEnd.y);
        const w = Math.abs(drawEnd.x - drawStart.x);
        const h = Math.abs(drawEnd.y - drawStart.y);
        addAnnotationFromPixelBox({ x, y, w, h }, { shape: "rectangle" });
      } else if (tool === "draw_circle") {
        const dx = drawEnd.x - drawStart.x;
        const dy = drawEnd.y - drawStart.y;
        const radius = Math.hypot(dx, dy);
        addAnnotationFromPixelBox({
          x: drawStart.x - radius,
          y: drawStart.y - radius,
          w: radius * 2,
          h: radius * 2,
        }, { shape: "circle" });
      }
    }
    setDrawing(false);
    setDrawStart(null);
    setDrawEnd(null);
    setDragInfo(null);
    setResizeInfo(null);
  };

  const finishPolygon = useCallback(() => {
    if (polygonPoints.length < 3) return;
    addPolygonAnnotation(polygonPoints);
    setPolygonPoints([]);
    setPolygonHover(null);
  }, [addPolygonAnnotation, polygonPoints]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (tool !== "draw_poly") return;
      if (e.key === "Escape") {
        setPolygonPoints([]);
        setPolygonHover(null);
      }
      if (e.key === "Enter") {
        e.preventDefault();
        finishPolygon();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [finishPolygon, tool]);

  const updateAnnotationPixel = (id, px) => {
    const bbox = pixelToYolo(px, displayW, displayH);
    setAnnotations((prev) => ({
      ...prev,
      [currentImage.name]: prev[currentImage.name].map((a) =>
        a.id === id ? { ...a, bbox } : a
      ),
    }));
  };

  const deleteAnnotation = (id) => {
    setAnnotations((prev) => ({
      ...prev,
      [currentImage.name]: prev[currentImage.name].filter((a) => a.id !== id),
    }));
    if (selectedBox === id) setSelectedBox(null);
  };

  const changeAnnotationClass = (id, newClassId) => {
    setAnnotations((prev) => ({
      ...prev,
      [currentImage.name]: prev[currentImage.name].map((a) =>
        a.id === id ? { ...a, classId: newClassId } : a
      ),
    }));
  };

  const handleRemap = () => {
    const from = parseInt(remapFrom);
    const to = parseInt(remapTo);
    if (isNaN(from) || isNaN(to)) return;
    setAnnotations((prev) => {
      const next = { ...prev };
      Object.keys(next).forEach((key) => {
        next[key] = next[key].map((a) =>
          a.classId === from ? { ...a, classId: to } : a
        );
      });
      return next;
    });
    setRemapFrom("");
    setRemapTo("");
  };

  const handleExport = () => {
    if (images.length === 0) return;
    const allowedClasses = new Set(exportClassFilter);
    let imageNames = images.map((img) => img.name);
    if (exportImageMode === "current" && currentImage) {
      imageNames = [currentImage.name];
    } else if (exportImageMode === "labeled") {
      imageNames = imageNames.filter((name) => (annotations[name] || []).length > 0);
    }

    imageNames.forEach((imgName) => {
      let anns = annotations[imgName] || [];
      if (allowedClasses.size > 0) {
        anns = anns.filter((a) => allowedClasses.has(a.classId));
      }
      if (exportSkipEmptyAfterFilter && anns.length === 0) return;

      const baseName = imgName.replace(/\.[^.]+$/, "");
      const text = toYoloText(anns);
      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = baseName + ".txt";
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  const annToSegmentLine = useCallback(
    (ann) => {
      const cls = ann.classId;
      if (ann.shape === "polygon" && Array.isArray(ann.points) && ann.points.length >= 3) {
        return `${cls} ${ann.points
          .map((p) => `${Math.max(0, Math.min(1, p.x)).toFixed(6)} ${Math.max(0, Math.min(1, p.y)).toFixed(6)}`)
          .join(" ")}`;
      }

      const px = yoloToPixel(ann.bbox, displayW, displayH);
      if (ann.shape === "circle") {
        const cx = px.x + px.w / 2;
        const cy = px.y + px.h / 2;
        const rx = px.w / 2;
        const ry = px.h / 2;
        const pts = [];
        const n = 20;
        for (let i = 0; i < n; i++) {
          const a = (2 * Math.PI * i) / n;
          const x = (cx + Math.cos(a) * rx) / displayW;
          const y = (cy + Math.sin(a) * ry) / displayH;
          pts.push(`${Math.max(0, Math.min(1, x)).toFixed(6)} ${Math.max(0, Math.min(1, y)).toFixed(6)}`);
        }
        return `${cls} ${pts.join(" ")}`;
      }

      // rectangle fallback as 4-point polygon
      const x1 = ann.bbox[0] - ann.bbox[2] / 2;
      const y1 = ann.bbox[1] - ann.bbox[3] / 2;
      const x2 = ann.bbox[0] + ann.bbox[2] / 2;
      const y2 = ann.bbox[1] + ann.bbox[3] / 2;
      return `${cls} ${x1.toFixed(6)} ${y1.toFixed(6)} ${x2.toFixed(6)} ${y1.toFixed(6)} ${x2.toFixed(6)} ${y2.toFixed(6)} ${x1.toFixed(6)} ${y2.toFixed(6)}`;
    },
    [displayH, displayW]
  );

  const getFilteredExportItems = useCallback(() => {
    const allowedClasses = new Set(exportClassFilter);
    let imageNames = images.map((img) => img.name);
    if (exportImageMode === "current" && currentImage) {
      imageNames = [currentImage.name];
    } else if (exportImageMode === "labeled") {
      imageNames = imageNames.filter((name) => (annotations[name] || []).length > 0);
    }

    return imageNames
      .map((imgName) => {
        let anns = annotations[imgName] || [];
        if (allowedClasses.size > 0) {
          anns = anns.filter((a) => allowedClasses.has(a.classId));
        }
        if (exportSkipEmptyAfterFilter && anns.length === 0) return null;
        const detectText = toYoloText(anns);
        const segmentText = anns.map((a) => annToSegmentLine(a)).join("\n");
        return {
          name: imgName,
          label_detect: detectText,
          label_segment: segmentText,
        };
      })
      .filter(Boolean);
  }, [
    annToSegmentLine,
    annotations,
    currentImage,
    exportClassFilter,
    exportImageMode,
    exportSkipEmptyAfterFilter,
    images,
  ]);

  const handleExportDatasetZip = async () => {
    if (exporting) return;
    setExportStatus("");
    setExportProgress(0);
    const items = getFilteredExportItems();
    if (items.length === 0) {
      setExportStatus("Нечего экспортировать: нет изображений после фильтров.");
      return;
    }
    const variantMult = 1 + (exportFlipH ? 1 : 0) + (exportFlipV ? 1 : 0);
    const approx = exportUseTiling
      ? `Оценка: минимум ~${items.length * variantMult} изображений после аугментации (тайлинг увеличит больше).`
      : `Оценка: ~${items.length * variantMult} изображений в архиве.`;
    setExportEstimate(approx);
    const fileByName = new Map(images.map((img) => [img.name, img.file]).filter((x) => x[1]));
    const formData = new FormData();
    let uploaded = 0;
    items.forEach((it) => {
      const file = fileByName.get(it.name);
      if (file) {
        formData.append("images", file, it.name);
        uploaded += 1;
      }
    });
    if (uploaded === 0) {
      setExportStatus("Не найдены исходные файлы изображений в памяти. Загрузите изображения снова (можно тем же набором), затем повторите экспорт.");
      return;
    }
    setExporting(true);
    setExportProgress(5);
    formData.append(
      "payload",
      JSON.stringify({
        classes,
        task: exportTask,
        val_split: exportValSplit,
        items,
        tiling: {
          enabled: exportUseTiling,
          tile_size: exportTileSize,
          overlap: exportTileOverlap,
        },
        augment: {
          flip_h: exportFlipH,
          flip_v: exportFlipV,
        },
        min_object_fraction: exportMinObjectFraction,
      })
    );
    await new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/export-yolo-dataset");
      xhr.responseType = "blob";
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const p = 5 + Math.round((e.loaded / e.total) * 45);
        setExportProgress(Math.min(55, p));
        setExportStatus("Загрузка данных на сервер...");
      };
      xhr.onprogress = (e) => {
        if (!e.lengthComputable) {
          setExportProgress((prev) => Math.min(95, prev + 1));
          setExportStatus("Генерация архива...");
          return;
        }
        const p = 55 + Math.round((e.loaded / e.total) * 40);
        setExportProgress(Math.min(95, p));
        setExportStatus("Скачивание архива...");
      };
      xhr.onload = async () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          const blob = xhr.response;
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "yolo_dataset_export.zip";
          a.click();
          URL.revokeObjectURL(url);
          setExportProgress(100);
          setExportStatus(`Готово: архив ${exportTask} датасета скачан.`);
        } else {
          let message = `Ошибка экспорта (${xhr.status})`;
          try {
            const text = await xhr.response.text();
            const err = JSON.parse(text);
            if (err?.error) message = err.error;
          } catch {}
          setExportStatus(message);
          setExportProgress(0);
        }
        setExporting(false);
        resolve();
      };
      xhr.onerror = () => {
        setExportStatus("Сетевая ошибка при экспорте.");
        setExportProgress(0);
        setExporting(false);
        resolve();
      };
      xhr.send(formData);
    });
  };

  const toggleExportClass = (classId) => {
    setExportClassFilter((prev) =>
      prev.includes(classId)
        ? prev.filter((id) => id !== classId)
        : [...prev, classId]
    );
  };

  const removeImage = (name) => {
    setImages((prev) => {
      const idx = prev.findIndex((img) => img.name === name);
      if (idx === -1) return prev;
      if (prev[idx].url) URL.revokeObjectURL(prev[idx].url);
      const next = prev.filter((img) => img.name !== name);
      setCurrentIdx((cur) => {
        if (next.length === 0) return 0;
        if (cur > idx) return cur - 1;
        if (cur >= next.length) return next.length - 1;
        return cur;
      });
      return next;
    });
    setAnnotations((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
    setSelectedBox(null);
  };

  const removeClass = (classId) => {
    if (classes.length <= 1) return;
    setClasses((prev) => prev.filter((_, i) => i !== classId));
    setAnnotations((prev) => {
      const next = { ...prev };
      Object.keys(next).forEach((key) => {
        next[key] = next[key]
          .filter((a) => a.classId !== classId)
          .map((a) => (a.classId > classId ? { ...a, classId: a.classId - 1 } : a));
      });
      return next;
    });
    setExportClassFilter((prev) =>
      prev
        .filter((id) => id !== classId)
        .map((id) => (id > classId ? id - 1 : id))
    );
    setSelectedBox(null);
    setActiveClassId((prev) => {
      if (prev === classId) return 0;
      if (prev > classId) return prev - 1;
      return prev;
    });
  };

  const handleSaveState = () => {
    const payload = {
      version: 1,
      createdAt: new Date().toISOString(),
      classes,
      annotations,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "yolo-editor-state.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoadState = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(String(ev.target.result || "{}"));
        if (!Array.isArray(data.classes) || typeof data.annotations !== "object") {
          return;
        }
        const imageNames = new Set(images.map((img) => img.name));
        const filteredAnnotations = {};
        Object.entries(data.annotations).forEach(([name, anns]) => {
          if (!imageNames.has(name) || !Array.isArray(anns)) return;
          filteredAnnotations[name] = anns.map((a) => ({
            ...a,
            id: a.id || Math.random().toString(36).slice(2, 10),
            shape: a.shape || "rectangle",
          }));
        });
        setClasses(data.classes);
        setAnnotations((prev) => ({ ...prev, ...filteredAnnotations }));
        setSelectedBox(null);
      } catch {
        // ignore malformed files
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const handleDownloadDatasetBuilder = () => {
    const namesPy = classes.map((c) => `'${String(c).replace(/'/g, "\\'")}'`).join(", ");
    const script = `#!/usr/bin/env python3
"""
Build YOLO dataset from images + exported labels.

Usage:
  python build_yolo_dataset.py --images ./raw_images --labels ./exported_labels --out ./dataset --split 0.2
"""
import argparse
import random
import shutil
from pathlib import Path
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder with source images")
    parser.add_argument("--labels", required=True, help="Folder with YOLO txt labels")
    parser.add_argument("--out", default="./dataset", help="Output dataset root")
    parser.add_argument("--split", type=float, default=0.2, help="Validation fraction (0..1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out = Path(args.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    labeled = []
    for img in image_files:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            labeled.append((img, lbl))

    if not labeled:
        raise SystemExit("No labeled pairs found (image + .txt)")

    random.seed(args.seed)
    random.shuffle(labeled)
    val_count = max(1, int(len(labeled) * args.split))
    val_set = set(x[0].name for x in labeled[:val_count])

    for img, lbl in labeled:
        split = "val" if img.name in val_set else "train"
        shutil.copy2(img, out / "images" / split / img.name)
        shutil.copy2(lbl, out / "labels" / split / lbl.name)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": ${classes.length},
        "names": [${namesPy}],
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"Done: {len(labeled)} labeled images")
    print(f"Train: {len(labeled) - val_count}, Val: {val_count}")
    print(f"data.yaml: {out / 'data.yaml'}")

if __name__ == "__main__":
    main()
`;
    const blob = new Blob([script], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "build_yolo_dataset.py";
    a.click();
    URL.revokeObjectURL(url);
  };

  const addClass = () => {
    if (newClassName.trim() && !classes.includes(newClassName.trim())) {
      setClasses((prev) => [...prev, newClassName.trim()]);
      setNewClassName("");
    }
  };

  const startEditClass = (classId) => {
    setEditingClass(classId);
    setEditingClassName(classes[classId] || "");
  };

  const saveEditClass = () => {
    if (editingClass === null) return;
    const name = editingClassName.trim();
    if (!name) {
      setEditingClass(null);
      setEditingClassName("");
      return;
    }
    setClasses((prev) =>
      prev.map((cls, idx) => (idx === editingClass ? name : cls))
    );
    setEditingClass(null);
    setEditingClassName("");
  };

  const cancelEditClass = () => {
    setEditingClass(null);
    setEditingClassName("");
  };

  // ─── Auto-annotate current image ───
  const runInference = async () => {
    if (!currentImage || !ptFile) return;
    setInferring(true);
    try {
      const formData = new FormData();
      formData.append("image", currentImage.file);
      formData.append("conf", String(confidence));
      formData.append("mode", aiMode);
      const res = await fetch("/api/inference", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        console.error("Inference error:", data?.error || res.statusText);
        return;
      }
      if (data.predictions && data.predictions.length > 0) {
        const newAnns = data.predictions.map((p) => ({
          classId: p.class_id,
          bbox: p.bbox,
          shape: p.shape === "polygon" ? "polygon" : "rectangle",
          points: p.shape === "polygon" && Array.isArray(p.points) ? p.points : undefined,
          id: Math.random().toString(36).slice(2, 10),
        }));
        setAnnotations((prev) => ({
          ...prev,
          [currentImage.name]: [...(prev[currentImage.name] || []), ...newAnns],
        }));
      }
    } catch (e) {
      console.error("Inference error:", e);
    }
    setInferring(false);
  };

  // ─── Replace annotations on current image ───
  const runInferenceReplace = async () => {
    if (!currentImage || !ptFile) return;
    setInferring(true);
    try {
      const formData = new FormData();
      formData.append("image", currentImage.file);
      formData.append("conf", String(confidence));
      formData.append("mode", aiMode);
      const res = await fetch("/api/inference", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        console.error("Inference error:", data?.error || res.statusText);
        return;
      }
      if (data.predictions) {
        const newAnns = data.predictions.map((p) => ({
          classId: p.class_id,
          bbox: p.bbox,
          shape: p.shape === "polygon" ? "polygon" : "rectangle",
          points: p.shape === "polygon" && Array.isArray(p.points) ? p.points : undefined,
          id: Math.random().toString(36).slice(2, 10),
        }));
        setAnnotations((prev) => ({
          ...prev,
          [currentImage.name]: newAnns,
        }));
      }
    } catch (e) {
      console.error("Inference error:", e);
    }
    setInferring(false);
  };

  // ─── Batch auto-annotate ALL images ───
  const runBatchInference = async () => {
    if (!ptFile || images.length === 0) return;
    setBatchInferring(true);
    setBatchProgress("0/" + images.length);

    // Process in chunks of 5
    const chunkSize = 5;
    for (let i = 0; i < images.length; i += chunkSize) {
      const chunk = images.slice(i, i + chunkSize);
      const formData = new FormData();
      chunk.forEach((img) => formData.append("images", img.file));
      formData.append("conf", String(confidence));
      formData.append("mode", aiMode);

      try {
        const res = await fetch("/api/inference-batch", { method: "POST", body: formData });
        const data = await res.json();
        if (data.results) {
          setAnnotations((prev) => {
            const next = { ...prev };
            Object.entries(data.results).forEach(([filename, preds]) => {
              const newAnns = preds.map((p) => ({
                classId: p.class_id,
                bbox: p.bbox,
                shape: p.shape === "polygon" ? "polygon" : "rectangle",
                points: p.shape === "polygon" && Array.isArray(p.points) ? p.points : undefined,
                id: Math.random().toString(36).slice(2, 10),
              }));
              // Merge with existing
              next[filename] = [...(next[filename] || []), ...newAnns];
            });
            return next;
          });
        }
      } catch (e) {
        console.error("Batch error:", e);
      }
      setBatchProgress(Math.min(i + chunkSize, images.length) + "/" + images.length);
    }
    setBatchInferring(false);
    setBatchProgress("");
  };

  // ─── Batch segment via CLI fallback ───
  const runBatchSegmentCli = async () => {
    if (!ptFile || images.length === 0) return;
    setSegmentCliInferring(true);
    setBatchProgress("0/" + images.length);

    const chunkSize = 5;
    for (let i = 0; i < images.length; i += chunkSize) {
      const chunk = images.slice(i, i + chunkSize);
      const formData = new FormData();
      chunk.forEach((img) => formData.append("images", img.file));
      formData.append("conf", String(confidence));
      formData.append("imgsz", "640");

      try {
        const res = await fetch("/api/inference-segment-cli-batch", { method: "POST", body: formData });
        const data = await res.json();
        if (data.results) {
          setAnnotations((prev) => {
            const next = { ...prev };
            Object.entries(data.results).forEach(([filename, preds]) => {
              const newAnns = preds.map((p) => ({
                classId: p.class_id,
                bbox: p.bbox,
                shape: "polygon",
                points: p.points || [],
                id: Math.random().toString(36).slice(2, 10),
              }));
              next[filename] = [...(next[filename] || []), ...newAnns];
            });
            return next;
          });
        }
      } catch (e) {
        console.error("Segment CLI batch error:", e);
      }
      setBatchProgress(Math.min(i + chunkSize, images.length) + "/" + images.length);
    }
    setSegmentCliInferring(false);
    setBatchProgress("");
  };

  const selectedAnn = currentAnnotations.find((a) => a.id === selectedBox);

  const pythonScript = `#!/usr/bin/env python3
"""
YOLO Dataset Enhancement Script
Использует .pt файл модели YOLO для автоматической разметки
и доработки датасета с возможностью изменения классов.

Использование:
  python yolo_enhance.py --model best.pt --images ./images --labels ./labels --output ./output

Зависимости:
  pip install ultralytics opencv-python numpy pyyaml
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import yaml
except ImportError:
    print("Установите зависимости: pip install ultralytics opencv-python numpy pyyaml")
    sys.exit(1)


def load_existing_labels(label_path):
    """Загрузка существующих аннотаций YOLO"""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    annotations.append({
                        'class_id': class_id,
                        'bbox': bbox,
                        'source': 'manual'
                    })
    return annotations


def save_labels(label_path, annotations):
    """Сохранение аннотаций в формате YOLO"""
    with open(label_path, 'w') as f:
        for ann in annotations:
            bbox_str = ' '.join(f'{v:.6f}' for v in ann['bbox'])
            f.write(f"{ann['class_id']} {bbox_str}\\n")


def iou(box1, box2):
    """Вычисление IoU между двумя YOLO bbox"""
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def remap_classes(annotations, class_map):
    """Переназначение классов по маппингу {old_id: new_id}"""
    for ann in annotations:
        old_id = ann['class_id']
        if old_id in class_map:
            ann['class_id'] = class_map[old_id]
    return annotations


def filter_classes(annotations, keep_classes):
    """Фильтрация: оставить только указанные классы"""
    return [a for a in annotations if a['class_id'] in keep_classes]


def merge_annotations(existing, predicted, iou_threshold=0.5, confidence_threshold=0.25):
    """
    Слияние существующих и предсказанных аннотаций.
    Если предсказание перекрывается с существующей разметкой (IoU > порог),
    оставляем существующую. Иначе добавляем предсказание.
    """
    merged = list(existing)

    for pred in predicted:
        is_duplicate = False
        for ext in existing:
            if iou(pred['bbox'], ext['bbox']) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(pred)

    return merged


def run_inference(model, image_path, conf_threshold=0.25):
    """Запуск инференса модели на изображении"""
    results = model(image_path, conf=conf_threshold, verbose=False)
    predictions = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Конвертация из xyxy в xywh (нормализованные)
                xywhn = box.xywhn[0].tolist()
                predictions.append({
                    'class_id': class_id,
                    'bbox': xywhn,
                    'confidence': conf,
                    'source': 'model'
                })

    return predictions


def enhance_dataset(
    model_path,
    images_dir,
    labels_dir,
    output_dir,
    conf_threshold=0.25,
    iou_threshold=0.5,
    class_map=None,
    keep_classes=None,
    mode='merge'  # merge | replace | predict_only | remap_only
):
    """
    Основная функция доработки датасета.

    Режимы:
    - merge: слияние существующей и новой разметки
    - replace: полная замена разметки предсказаниями модели
    - predict_only: разметка только неразмеченных изображений
    - remap_only: только переназначение классов без инференса
    """
    # Загрузка модели
    if mode != 'remap_only':
        print(f"Загрузка модели: {model_path}")
        model = YOLO(model_path)
    else:
        model = None

    # Создание выходных директорий
    out_images = os.path.join(output_dir, 'images')
    out_labels = os.path.join(output_dir, 'labels')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # Получение списка изображений
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in img_extensions
    ])

    stats = {
        'total': len(image_files),
        'processed': 0,
        'new_boxes': 0,
        'kept_boxes': 0,
        'remapped': 0,
        'filtered_out': 0,
    }

    print(f"Найдено {len(image_files)} изображений")
    print(f"Режим: {mode}")
    if class_map:
        print(f"Маппинг классов: {class_map}")
    if keep_classes:
        print(f"Оставить классы: {keep_classes}")
    print("-" * 60)

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        base_name = Path(img_file).stem
        label_path = os.path.join(labels_dir, base_name + '.txt') if labels_dir else None
        out_label_path = os.path.join(out_labels, base_name + '.txt')
        out_img_path = os.path.join(out_images, img_file)

        # Копируем изображение
        shutil.copy2(img_path, out_img_path)

        # Загружаем существующие аннотации
        existing = load_existing_labels(label_path) if label_path else []

        if mode == 'remap_only':
            final = existing
        elif mode == 'replace':
            predictions = run_inference(model, img_path, conf_threshold)
            final = predictions
            stats['new_boxes'] += len(predictions)
        elif mode == 'predict_only':
            if len(existing) == 0:
                predictions = run_inference(model, img_path, conf_threshold)
                final = predictions
                stats['new_boxes'] += len(predictions)
            else:
                final = existing
                stats['kept_boxes'] += len(existing)
        elif mode == 'merge':
            predictions = run_inference(model, img_path, conf_threshold)
            before_count = len(existing)
            final = merge_annotations(existing, predictions, iou_threshold, conf_threshold)
            stats['new_boxes'] += len(final) - before_count
            stats['kept_boxes'] += before_count
        else:
            final = existing

        # Применение маппинга классов
        if class_map:
            before_remap = [a['class_id'] for a in final]
            final = remap_classes(final, class_map)
            after_remap = [a['class_id'] for a in final]
            stats['remapped'] += sum(1 for b, a in zip(before_remap, after_remap) if b != a)

        # Фильтрация классов
        if keep_classes is not None:
            before_filter = len(final)
            final = filter_classes(final, keep_classes)
            stats['filtered_out'] += before_filter - len(final)

        # Сохранение
        save_labels(out_label_path, final)
        stats['processed'] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  [{i+1}/{len(image_files)}] Обработано")

    # Генерация data.yaml
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_data = {
        'path': os.path.abspath(output_dir),
        'train': 'images',
        'val': 'images',
        'nc': max(
            max((a['class_id'] for anns in
                 [load_existing_labels(os.path.join(out_labels, f))
                  for f in os.listdir(out_labels)]
                 for a in anns), default=0) + 1,
            1
        ),
        'names': {}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print("\\n" + "=" * 60)
    print("ИТОГИ:")
    print(f"  Обработано изображений: {stats['processed']}/{stats['total']}")
    print(f"  Сохранено существующих bbox: {stats['kept_boxes']}")
    print(f"  Добавлено новых bbox: {stats['new_boxes']}")
    print(f"  Переназначено классов: {stats['remapped']}")
    print(f"  Отфильтровано bbox: {stats['filtered_out']}")
    print(f"  Выходная директория: {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print("=" * 60)


def parse_class_map(s):
    """Парсинг строки маппинга классов: '0:1,2:3,5:0'"""
    if not s:
        return None
    mapping = {}
    for pair in s.split(','):
        old, new = pair.split(':')
        mapping[int(old.strip())] = int(new.strip())
    return mapping


def parse_keep_classes(s):
    """Парсинг строки классов для фильтрации: '0,1,2,5'"""
    if not s:
        return None
    return set(int(x.strip()) for x in s.split(','))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLO Dataset Enhancement — доработка датасета через .pt модель'
    )
    parser.add_argument('--model', type=str, default='best.pt',
                        help='Путь к .pt файлу модели YOLO')
    parser.add_argument('--images', type=str, required=True,
                        help='Директория с изображениями')
    parser.add_argument('--labels', type=str, default=None,
                        help='Директория с существующими лейблами (опционально)')
    parser.add_argument('--output', type=str, default='./enhanced_dataset',
                        help='Выходная директория')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Порог confidence (по умолчанию 0.25)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Порог IoU для слияния (по умолчанию 0.5)')
    parser.add_argument('--mode', type=str, default='merge',
                        choices=['merge', 'replace', 'predict_only', 'remap_only'],
                        help='Режим работы')
    parser.add_argument('--remap', type=str, default=None,
                        help='Маппинг классов, напр: "0:1,2:3,5:0"')
    parser.add_argument('--keep-classes', type=str, default=None,
                        help='Оставить только указанные классы, напр: "0,1,2"')

    args = parser.parse_args()

    enhance_dataset(
        model_path=args.model,
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        class_map=parse_class_map(args.remap),
        keep_classes=parse_keep_classes(args.keep_classes),
        mode=args.mode,
    )
`;

  return (
    <div
      style={{
        width: "100%",
        minHeight: "100vh",
        background: "#0a0a0b",
        color: "#e4e4e7",
        fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "16px 24px",
          borderBottom: "1px solid #27272a",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "linear-gradient(180deg, #111113 0%, #0a0a0b 100%)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 32,
              height: 32,
              background: "linear-gradient(135deg, #f97316, #ef4444)",
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              fontWeight: 800,
            }}
          >
            Y
          </div>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: "0.05em" }}>
            YOLO DATASET EDITOR
          </span>
          {ptFile && (
            <span
              style={{
                fontSize: 11,
                background: ptStatus === "loading" ? "#f9731620" : ptStatus === "error" ? "#ef444420" : "#22c55e20",
                color: ptStatus === "loading" ? "#f97316" : ptStatus === "error" ? "#ef4444" : "#22c55e",
                padding: "3px 8px",
                borderRadius: 4,
                border: `1px solid ${ptStatus === "loading" ? "#f9731640" : ptStatus === "error" ? "#ef444440" : "#22c55e40"}`,
              }}
            >
              {ptStatus === "loading" ? "Loading .pt..." : ptStatus === "error" ? ".pt: no Python/ultralytics" : `.pt: ${ptFile.name} (${classes.length} classes)`}
            </span>
          )}
          {duplicatesSkipped > 0 && (
            <span
              style={{
                fontSize: 11,
                background: "#ef444420",
                color: "#ef4444",
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid #ef444440",
              }}
            >
              {duplicatesSkipped} duplicates skipped
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {["editor", "script", "export"].map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                padding: "6px 14px",
                borderRadius: 6,
                border: "none",
                background: view === v ? "#f97316" : "#27272a",
                color: view === v ? "#000" : "#a1a1aa",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "inherit",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              {v === "editor" ? "Редактор" : v === "script" ? "Python" : "Экспорт"}
            </button>
          ))}
        </div>
      </div>

      {view === "script" && (
        <div style={{ padding: 24, maxWidth: 900, margin: "0 auto" }}>
          <h2
            style={{
              fontSize: 18,
              fontWeight: 700,
              marginBottom: 16,
              color: "#f97316",
            }}
          >
            Python-скрипт для доработки датасета
          </h2>
          <p style={{ fontSize: 13, color: "#a1a1aa", marginBottom: 16, lineHeight: 1.6 }}>
            Этот скрипт использует .pt файл модели YOLO для автоматической разметки
            и доработки датасета. Поддерживает 4 режима: merge (слияние), replace (замена),
            predict_only (только новые), remap_only (переназначение классов).
          </p>
          <div
            style={{
              background: "#111113",
              border: "1px solid #27272a",
              borderRadius: 8,
              padding: 16,
              fontSize: 12,
              lineHeight: 1.6,
              overflowX: "auto",
              maxHeight: "70vh",
              overflowY: "auto",
            }}
          >
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{pythonScript}</pre>
          </div>
          <div
            style={{
              marginTop: 16,
              padding: 16,
              background: "#18181b",
              borderRadius: 8,
              border: "1px solid #27272a",
            }}
          >
            <p style={{ fontSize: 13, fontWeight: 600, color: "#f97316", marginBottom: 8 }}>
              Примеры использования:
            </p>
            <code style={{ fontSize: 12, color: "#a1a1aa", display: "block", marginBottom: 6 }}>
              # Слияние с существующей разметкой
            </code>
            <code style={{ fontSize: 12, color: "#e4e4e7", display: "block", marginBottom: 12 }}>
              python yolo_enhance.py --model best.pt --images ./data/images --labels ./data/labels --mode merge
            </code>
            <code style={{ fontSize: 12, color: "#a1a1aa", display: "block", marginBottom: 6 }}>
              # Переназначение классов: 0→1, 2→0
            </code>
            <code style={{ fontSize: 12, color: "#e4e4e7", display: "block", marginBottom: 12 }}>
              python yolo_enhance.py --model best.pt --images ./data/images --mode remap_only --remap "0:1,2:0"
            </code>
            <code style={{ fontSize: 12, color: "#a1a1aa", display: "block", marginBottom: 6 }}>
              # Оставить только классы 0 и 1
            </code>
            <code style={{ fontSize: 12, color: "#e4e4e7", display: "block" }}>
              python yolo_enhance.py --model best.pt --images ./data/images --labels ./data/labels --keep-classes "0,1"
            </code>
          </div>
        </div>
      )}

      {view === "export" && (
        <div style={{ padding: 24, maxWidth: 700, margin: "0 auto" }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16, color: "#f97316" }}>
            Экспорт данных
          </h2>
          <div
            style={{
              fontSize: 12,
              color:
                autosaveState === "error"
                  ? "#ef4444"
                  : autosaveState === "restored"
                  ? "#22c55e"
                  : "#71717a",
              marginBottom: 10,
            }}
          >
            Автосохранение:{" "}
            {autosaveState === "error"
              ? "ошибка"
              : autosaveState === "restored"
              ? "восстановлено после перезагрузки"
              : autosaveState === "saved"
              ? "сохранено"
              : "ожидание"}
          </div>

          <div
            style={{
              background: "#18181b",
              border: "1px solid #27272a",
              borderRadius: 8,
              padding: 16,
              marginBottom: 16,
              display: "flex",
              gap: 8,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <button
              onClick={handleSaveState}
              style={{
                padding: "8px 12px",
                borderRadius: 6,
                border: "1px solid #3f3f46",
                background: "#27272a",
                color: "#e4e4e7",
                fontSize: 12,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Сохранить состояние (.json)
            </button>
            <button
              onClick={() => stateInputRef.current?.click()}
              style={{
                padding: "8px 12px",
                borderRadius: 6,
                border: "1px solid #3f3f46",
                background: "#27272a",
                color: "#e4e4e7",
                fontSize: 12,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Загрузить состояние
            </button>
            <button
              onClick={handleDownloadDatasetBuilder}
              style={{
                padding: "8px 12px",
                borderRadius: 6,
                border: "1px solid #3f3f46",
                background: "#27272a",
                color: "#e4e4e7",
                fontSize: 12,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Скачать скрипт сборки датасета
            </button>
            <button
              onClick={() => {
                localStorage.removeItem(AUTOSAVE_KEY);
                setAutosaveState("idle");
              }}
              style={{
                padding: "8px 12px",
                borderRadius: 6,
                border: "1px solid #3f3f46",
                background: "transparent",
                color: "#a1a1aa",
                fontSize: 12,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Сбросить автосохранение
            </button>
            <input
              ref={stateInputRef}
              type="file"
              accept=".json"
              onChange={handleLoadState}
              style={{ display: "none" }}
            />
          </div>

          <div
            style={{
              background: "#18181b",
              border: "1px solid #27272a",
              borderRadius: 8,
              padding: 20,
              marginBottom: 16,
            }}
          >
            <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
              Статистика датасета
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ padding: 12, background: "#0a0a0b", borderRadius: 6 }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: "#f97316" }}>{images.length}</div>
                <div style={{ fontSize: 11, color: "#71717a" }}>Изображений</div>
              </div>
              <div style={{ padding: 12, background: "#0a0a0b", borderRadius: 6 }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: "#22c55e" }}>
                  {Object.values(annotations).reduce((s, a) => s + a.length, 0)}
                </div>
                <div style={{ fontSize: 11, color: "#71717a" }}>Аннотаций</div>
              </div>
              <div style={{ padding: 12, background: "#0a0a0b", borderRadius: 6 }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: "#3b82f6" }}>{classes.length}</div>
                <div style={{ fontSize: 11, color: "#71717a" }}>Классов</div>
              </div>
              <div style={{ padding: 12, background: "#0a0a0b", borderRadius: 6 }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: "#a855f7" }}>
                  {images.filter((img) => (annotations[img.name] || []).length === 0).length}
                </div>
                <div style={{ fontSize: 11, color: "#71717a" }}>Без разметки</div>
              </div>
            </div>
          </div>

          {/* Global class remap */}
          <div
            style={{
              background: "#18181b",
              border: "1px solid #27272a",
              borderRadius: 8,
              padding: 20,
              marginBottom: 16,
            }}
          >
            <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
              Глобальная замена классов
            </p>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="number"
                placeholder="Из ID"
                value={remapFrom}
                onChange={(e) => setRemapFrom(e.target.value)}
                style={{
                  width: 80,
                  padding: "8px 12px",
                  borderRadius: 6,
                  border: "1px solid #3f3f46",
                  background: "#0a0a0b",
                  color: "#e4e4e7",
                  fontSize: 13,
                  fontFamily: "inherit",
                }}
              />
              <span style={{ color: "#71717a", fontSize: 18 }}>→</span>
              <input
                type="number"
                placeholder="В ID"
                value={remapTo}
                onChange={(e) => setRemapTo(e.target.value)}
                style={{
                  width: 80,
                  padding: "8px 12px",
                  borderRadius: 6,
                  border: "1px solid #3f3f46",
                  background: "#0a0a0b",
                  color: "#e4e4e7",
                  fontSize: 13,
                  fontFamily: "inherit",
                }}
              />
              <button
                onClick={handleRemap}
                style={{
                  padding: "8px 16px",
                  borderRadius: 6,
                  border: "none",
                  background: "#f97316",
                  color: "#000",
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                Заменить во всех
              </button>
            </div>
          </div>

          <div
            style={{
              background: "#18181b",
              border: "1px solid #27272a",
              borderRadius: 8,
              padding: 20,
              marginBottom: 16,
            }}
          >
            <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
              Параметры экспорта
            </p>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              {[
                { id: "detect", label: "YOLO Detect (bbox)" },
                { id: "segment", label: "YOLO Segment (polygon)" },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setExportTask(m.id)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    border: "1px solid",
                    borderColor: exportTask === m.id ? "#22c55e" : "#3f3f46",
                    background: exportTask === m.id ? "#22c55e20" : "transparent",
                    color: exportTask === m.id ? "#22c55e" : "#a1a1aa",
                    fontSize: 12,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              {[
                { id: "labeled", label: "Только размеченные" },
                { id: "current", label: "Текущее изображение" },
                { id: "all", label: "Все изображения" },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setExportImageMode(m.id)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    border: "1px solid",
                    borderColor: exportImageMode === m.id ? "#f97316" : "#3f3f46",
                    background: exportImageMode === m.id ? "#f9731620" : "transparent",
                    color: exportImageMode === m.id ? "#f97316" : "#a1a1aa",
                    fontSize: 12,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
              <input
                type="checkbox"
                checked={exportSkipEmptyAfterFilter}
                onChange={(e) => setExportSkipEmptyAfterFilter(e.target.checked)}
              />
              <span style={{ fontSize: 12, color: "#a1a1aa" }}>
                Не экспортировать пустые .txt после фильтра
              </span>
            </label>
            <div style={{ fontSize: 12, color: "#71717a", marginBottom: 6 }}>
              Фильтр по классам (если ничего не выбрано — экспорт всех):
            </div>
            <div style={{ marginBottom: 10 }}>
              <span style={{ fontSize: 12, color: "#a1a1aa", marginRight: 8 }}>
                Доля валидации: {(exportValSplit * 100).toFixed(0)}%
              </span>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.05"
                value={exportValSplit}
                onChange={(e) => setExportValSplit(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "#f97316" }}
              />
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <input
                type="checkbox"
                checked={exportUseTiling}
                onChange={(e) => setExportUseTiling(e.target.checked)}
              />
              <span style={{ fontSize: 12, color: "#a1a1aa" }}>
                Дробить изображения на тайлы (полезно для мелких деталей)
              </span>
            </label>
            {exportUseTiling && (
              <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: "#71717a", marginBottom: 4 }}>Размер тайла</div>
                  <input
                    type="number"
                    min="256"
                    max="2048"
                    step="32"
                    value={exportTileSize}
                    onChange={(e) => setExportTileSize(Math.max(128, parseInt(e.target.value || "640", 10)))}
                    style={{
                      width: "100%",
                      padding: "6px 8px",
                      borderRadius: 4,
                      border: "1px solid #3f3f46",
                      background: "#0a0a0b",
                      color: "#e4e4e7",
                      fontSize: 12,
                    }}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: "#71717a", marginBottom: 4 }}>Overlap</div>
                  <input
                    type="number"
                    min="0"
                    max="0.7"
                    step="0.05"
                    value={exportTileOverlap}
                    onChange={(e) => setExportTileOverlap(Math.max(0, Math.min(0.7, parseFloat(e.target.value || "0"))))}
                    style={{
                      width: "100%",
                      padding: "6px 8px",
                      borderRadius: 4,
                      border: "1px solid #3f3f46",
                      background: "#0a0a0b",
                      color: "#e4e4e7",
                      fontSize: 12,
                    }}
                  />
                </div>
              </div>
            )}
            {exportUseTiling && (
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11, color: "#71717a", marginBottom: 4 }}>
                  Мин. доля объекта в тайле: {(exportMinObjectFraction * 100).toFixed(0)}%
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.8"
                  step="0.05"
                  value={exportMinObjectFraction}
                  onChange={(e) => setExportMinObjectFraction(parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "#22c55e" }}
                />
              </div>
            )}
            <div style={{ display: "flex", gap: 14, marginBottom: 10 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#a1a1aa" }}>
                <input
                  type="checkbox"
                  checked={exportFlipH}
                  onChange={(e) => setExportFlipH(e.target.checked)}
                />
                Mirror H
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#a1a1aa" }}>
                <input
                  type="checkbox"
                  checked={exportFlipV}
                  onChange={(e) => setExportFlipV(e.target.checked)}
                />
                Mirror V
              </label>
            </div>
            <div style={{ maxHeight: 120, overflowY: "auto", display: "grid", gap: 4 }}>
              {classes.map((cls, i) => (
                <label key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={exportClassFilter.includes(i)}
                    onChange={() => toggleExportClass(i)}
                  />
                  <span style={{ fontSize: 12, color: "#a1a1aa" }}>
                    {i}: {cls}
                  </span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={handleExport}
            disabled={images.length === 0}
            style={{
              width: "100%",
              padding: "14px 24px",
              borderRadius: 8,
              border: "none",
              background: images.length > 0
                ? "linear-gradient(135deg, #f97316, #ef4444)"
                : "#27272a",
              color: images.length > 0 ? "#fff" : "#71717a",
              fontSize: 15,
              fontWeight: 700,
              cursor: images.length > 0 ? "pointer" : "not-allowed",
              fontFamily: "inherit",
              letterSpacing: "0.03em",
            }}
          >
            Скачать все лейблы (.txt)
          </button>
          <button
            onClick={handleExportDatasetZip}
            disabled={images.length === 0 || exporting}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "12px 24px",
              borderRadius: 8,
              border: "1px solid #22c55e40",
              background: images.length > 0 ? "#22c55e20" : "#27272a",
              color: images.length > 0 ? "#22c55e" : "#71717a",
              fontSize: 14,
              fontWeight: 700,
              cursor: images.length > 0 && !exporting ? "pointer" : "not-allowed",
              fontFamily: "inherit",
            }}
          >
            {exporting ? "Генерация архива..." : "Скачать YOLO датасет (.zip)"}
          </button>
          {exporting && (
            <div style={{ marginTop: 8 }}>
              <div style={{ height: 8, background: "#27272a", borderRadius: 999, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${Math.max(2, exportProgress)}%`,
                    height: "100%",
                    background: "linear-gradient(90deg, #22c55e, #16a34a)",
                    transition: "width 0.2s ease",
                  }}
                />
              </div>
              <div style={{ fontSize: 11, color: "#71717a", marginTop: 4 }}>
                Прогресс: {exportProgress}%
              </div>
            </div>
          )}
          {exportEstimate && (
            <div style={{ marginTop: 6, fontSize: 11, color: "#71717a" }}>
              {exportEstimate}
            </div>
          )}
          {exportStatus && (
            <div
              style={{
                marginTop: 8,
                fontSize: 12,
                color: exportStatus.startsWith("Готово") ? "#22c55e" : "#f97316",
              }}
            >
              {exportStatus}
            </div>
          )}
        </div>
      )}

      {view === "editor" && (
        <div style={{ display: "flex", height: "calc(100vh - 65px)" }}>
          {/* Sidebar — file list */}
          <div
            style={{
              width: 220,
              borderRight: "1px solid #27272a",
              overflowY: "auto",
              background: "#111113",
              flexShrink: 0,
            }}
          >
            <div style={{ padding: 12 }}>
              <label
                style={{
                  display: "block",
                  padding: "10px 14px",
                  borderRadius: 6,
                  border: "1px dashed #3f3f46",
                  textAlign: "center",
                  cursor: "pointer",
                  fontSize: 12,
                  color: "#a1a1aa",
                  transition: "all 0.2s",
                }}
              >
                📂 Загрузить файлы
                <br />
                <span style={{ fontSize: 10, color: "#52525b" }}>
                  img + txt + yaml + pt
                </span>
                <input
                  type="file"
                  multiple
                  accept="image/*,.txt,.yaml,.yml,.pt"
                  onChange={handleImageUpload}
                  style={{ display: "none" }}
                />
              </label>
              {importConflicts.length > 0 && (
                <div
                  style={{
                    marginTop: 8,
                    background: "#ef444415",
                    border: "1px solid #ef444440",
                    borderRadius: 6,
                    padding: 8,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 10, color: "#ef4444" }}>
                      Конфликты импорта ({importConflicts.length})
                    </span>
                    <button
                      onClick={() => setImportConflicts([])}
                      style={{
                        border: "none",
                        background: "transparent",
                        color: "#a1a1aa",
                        cursor: "pointer",
                        fontSize: 10,
                      }}
                    >
                      очистить
                    </button>
                  </div>
                  <div style={{ maxHeight: 100, overflowY: "auto", marginTop: 6 }}>
                    {importConflicts.slice(0, 20).map((msg, i) => (
                      <div key={i} style={{ fontSize: 10, color: "#fca5a5", marginBottom: 2 }}>
                        {msg}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <div style={{ padding: "0 8px" }}>
              {images.map((img, idx) => {
                const annCount = (annotations[img.name] || []).length;
                return (
                  <div
                    key={img.name}
                    onClick={() => {
                      setCurrentIdx(idx);
                      setSelectedBox(null);
                    }}
                    style={{
                      padding: "8px 10px",
                      borderRadius: 6,
                      cursor: "pointer",
                      background: idx === currentIdx ? "#f9731620" : "transparent",
                      border:
                        idx === currentIdx
                          ? "1px solid #f9731640"
                          : "1px solid transparent",
                      marginBottom: 2,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
                      <span
                        style={{
                          fontSize: 11,
                          color: idx === currentIdx ? "#f97316" : "#a1a1aa",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          maxWidth: 120,
                        }}
                      >
                        {img.name}
                      </span>
                      <span
                        style={{
                          fontSize: 10,
                          background: annCount > 0 ? "#22c55e20" : "#52525b20",
                          color: annCount > 0 ? "#22c55e" : "#52525b",
                          padding: "1px 6px",
                          borderRadius: 4,
                        }}
                      >
                        {annCount}
                      </span>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeImage(img.name);
                      }}
                      style={{
                        border: "none",
                        background: "transparent",
                        color: "#ef4444",
                        cursor: "pointer",
                        fontSize: 12,
                        padding: "0 4px",
                      }}
                      title="Удалить изображение"
                    >
                      ✕
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Main canvas area */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            {/* Toolbar */}
            <div
              style={{
                padding: "8px 16px",
                borderBottom: "1px solid #27272a",
                display: "flex",
                alignItems: "center",
                gap: 8,
                background: "#111113",
                flexWrap: "wrap",
              }}
            >
              {[
                { id: "select", label: "Выбор", icon: "↖" },
                { id: "draw", label: "Рисовать", icon: "□" },
                { id: "draw_circle", label: "Круг", icon: "◯" },
                { id: "draw_poly", label: "По точкам", icon: "⨯" },
                { id: "replace_class", label: "Замена класса", icon: "⇄" },
                { id: "delete", label: "Удалить", icon: "✕" },
              ].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTool(t.id)}
                  style={{
                    padding: "5px 12px",
                    borderRadius: 5,
                    border: "1px solid",
                    borderColor: tool === t.id ? "#f97316" : "#3f3f46",
                    background: tool === t.id ? "#f9731620" : "transparent",
                    color: tool === t.id ? "#f97316" : "#a1a1aa",
                    fontSize: 12,
                    cursor: "pointer",
                    fontFamily: "inherit",
                    display: "flex",
                    gap: 4,
                    alignItems: "center",
                  }}
                >
                  <span>{t.icon}</span> {t.label}
                </button>
              ))}

              <div style={{ width: 1, height: 24, background: "#27272a", margin: "0 4px" }} />

              <button
                onClick={() => setShowLabels(!showLabels)}
                style={{
                  padding: "5px 10px",
                  borderRadius: 5,
                  border: "1px solid #3f3f46",
                  background: showLabels ? "#3b82f620" : "transparent",
                  color: showLabels ? "#3b82f6" : "#71717a",
                  fontSize: 11,
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                Labels
              </button>

              <div style={{ width: 1, height: 24, background: "#27272a", margin: "0 4px" }} />

              <span style={{ fontSize: 11, color: "#71717a" }}>Zoom:</span>
              <button
                onClick={() => {
                  if (containerRef.current && imgSize.w) {
                    const cw = containerRef.current.clientWidth - 40;
                    const ch = containerRef.current.clientHeight - 40;
                    const fitZoom = Math.min(cw / imgSize.w, ch / imgSize.h, 1);
                    setZoom(Math.round(fitZoom * 100) / 100);
                    setPan({ x: 0, y: 0 });
                  }
                }}
                style={{
                  padding: "3px 8px", borderRadius: 4, border: "none",
                  background: "#27272a", color: "#a1a1aa", fontSize: 11,
                  cursor: "pointer", fontFamily: "inherit",
                }}
              >
                Fit
              </button>
              {[0.5, 1, 1.5, 2, 3].map((z) => (
                <button
                  key={z}
                  onClick={() => { setZoom(z); setPan({ x: 0, y: 0 }); }}
                  style={{
                    padding: "3px 8px", borderRadius: 4, border: "none",
                    background: Math.abs(zoom - z) < 0.01 ? "#f97316" : "#27272a",
                    color: Math.abs(zoom - z) < 0.01 ? "#000" : "#71717a",
                    fontSize: 11, cursor: "pointer", fontFamily: "inherit",
                    fontWeight: Math.abs(zoom - z) < 0.01 ? 700 : 400,
                  }}
                >
                  {z}x
                </button>
              ))}
              <span style={{ fontSize: 10, color: "#52525b", minWidth: 35, textAlign: "center" }}>
                {Math.round(zoom * 100)}%
              </span>

              {ptStatus === "done" && (
                <>
                  <div style={{ width: 1, height: 24, background: "#27272a", margin: "0 4px" }} />

                  <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ fontSize: 10, color: "#71717a" }}>conf:</span>
                    <input
                      type="range"
                      min="0.05"
                      max="0.95"
                      step="0.05"
                      value={confidence}
                      onChange={(e) => setConfidence(parseFloat(e.target.value))}
                      style={{ width: 60, accentColor: "#f97316" }}
                    />
                    <span style={{ fontSize: 10, color: "#a1a1aa", width: 28 }}>{confidence.toFixed(2)}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ fontSize: 10, color: "#71717a" }}>AI:</span>
                    {[
                      { id: "auto", label: "auto" },
                      { id: "detect", label: "bbox" },
                      { id: "segment", label: "segment" },
                    ].map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setAiMode(m.id)}
                        style={{
                          padding: "2px 6px",
                          borderRadius: 4,
                          border: "1px solid #3f3f46",
                          background: aiMode === m.id ? "#22c55e20" : "transparent",
                          color: aiMode === m.id ? "#22c55e" : "#a1a1aa",
                          fontSize: 10,
                          cursor: "pointer",
                          fontFamily: "inherit",
                        }}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>

                  <button
                    onClick={runInference}
                    disabled={inferring || !currentImage}
                    style={{
                      padding: "5px 10px",
                      borderRadius: 5,
                      border: "1px solid #22c55e40",
                      background: "#22c55e20",
                      color: inferring ? "#71717a" : "#22c55e",
                      fontSize: 11,
                      cursor: inferring ? "wait" : "pointer",
                      fontFamily: "inherit",
                      fontWeight: 600,
                    }}
                  >
                    {inferring ? "..." : "AI detect"}
                  </button>

                  <button
                    onClick={runInferenceReplace}
                    disabled={inferring || !currentImage}
                    style={{
                      padding: "5px 10px",
                      borderRadius: 5,
                      border: "1px solid #3b82f640",
                      background: "#3b82f620",
                      color: inferring ? "#71717a" : "#3b82f6",
                      fontSize: 11,
                      cursor: inferring ? "wait" : "pointer",
                      fontFamily: "inherit",
                    }}
                  >
                    AI replace
                  </button>

                  <button
                    onClick={runBatchInference}
                    disabled={batchInferring || images.length === 0}
                    style={{
                      padding: "5px 10px",
                      borderRadius: 5,
                      border: "1px solid #a855f740",
                      background: "#a855f720",
                      color: batchInferring ? "#71717a" : "#a855f7",
                      fontSize: 11,
                      cursor: batchInferring ? "wait" : "pointer",
                      fontFamily: "inherit",
                      fontWeight: 600,
                    }}
                  >
                    {batchInferring ? `Batch ${batchProgress}` : `AI all (${images.length})`}
                  </button>
                  <button
                    onClick={runBatchSegmentCli}
                    disabled={segmentCliInferring || images.length === 0}
                    style={{
                      padding: "5px 10px",
                      borderRadius: 5,
                      border: "1px solid #f59e0b40",
                      background: "#f59e0b20",
                      color: segmentCliInferring ? "#71717a" : "#f59e0b",
                      fontSize: 11,
                      cursor: segmentCliInferring ? "wait" : "pointer",
                      fontFamily: "inherit",
                      fontWeight: 600,
                    }}
                  >
                    {segmentCliInferring ? `Seg ${batchProgress}` : `AI seg CLI (${images.length})`}
                  </button>
                  <span style={{ fontSize: 10, color: "#71717a" }}>
                    AI ставит bbox (ось-ориент.), для наклона используйте "По точкам"
                  </span>
                </>
              )}

              <div style={{ flex: 1 }} />

              {images.length > 0 && (
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <button
                    onClick={() => {
                      setCurrentIdx(Math.max(0, currentIdx - 1));
                      setSelectedBox(null);
                    }}
                    disabled={currentIdx === 0}
                    style={{
                      padding: "4px 10px",
                      borderRadius: 5,
                      border: "1px solid #3f3f46",
                      background: "transparent",
                      color: currentIdx === 0 ? "#3f3f46" : "#e4e4e7",
                      fontSize: 13,
                      cursor: currentIdx === 0 ? "default" : "pointer",
                      fontFamily: "inherit",
                    }}
                  >
                    ←
                  </button>
                  <span style={{ fontSize: 12, color: "#71717a" }}>
                    {currentIdx + 1}/{images.length}
                  </span>
                  <button
                    onClick={() => {
                      setCurrentIdx(Math.min(images.length - 1, currentIdx + 1));
                      setSelectedBox(null);
                    }}
                    disabled={currentIdx >= images.length - 1}
                    style={{
                      padding: "4px 10px",
                      borderRadius: 5,
                      border: "1px solid #3f3f46",
                      background: "transparent",
                      color: currentIdx >= images.length - 1 ? "#3f3f46" : "#e4e4e7",
                      fontSize: 13,
                      cursor: currentIdx >= images.length - 1 ? "default" : "pointer",
                      fontFamily: "inherit",
                    }}
                  >
                    →
                  </button>
                </div>
              )}
            </div>

            {/* Canvas + Right panel */}
            <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
              {/* Canvas */}
              <div
                ref={containerRef}
                style={{
                  flex: 1,
                  overflow: "hidden",
                  background: "#09090b",
                  position: "relative",
                }}
                onWheel={handleWheel}
              >
                {currentImage ? (
                  <div
                    ref={canvasRef}
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      width: displayW,
                      height: displayH,
                      transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                      transformOrigin: "0 0",
                      cursor: panning
                        ? "grabbing"
                        : isDrawingTool
                        ? "crosshair"
                        : tool === "replace_class"
                        ? "copy"
                        : tool === "delete"
                        ? "pointer"
                        : "default",
                    }}
                    onDragStart={(e) => e.preventDefault()}
                    onMouseDown={handleCanvasMouseDown}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={handleCanvasMouseUp}
                    onMouseLeave={handleCanvasMouseUp}
                    onDoubleClick={() => {
                      if (tool === "draw_poly") finishPolygon();
                    }}
                    onContextMenu={(e) => {
                      if (tool === "draw_poly") e.preventDefault();
                    }}
                  >
                    <img
                      ref={imgRef}
                      src={currentImage.url}
                      onLoad={handleImgLoad}
                      style={{
                        width: "100%",
                        height: "100%",
                        display: "block",
                        pointerEvents: "none",
                        userSelect: "none",
                      }}
                      draggable={false}
                      onDragStart={(e) => e.preventDefault()}
                    />

                    {/* Annotations */}
                    {currentAnnotations.map((ann) => {
                      const px = yoloToPixel(ann.bbox, displayW, displayH);
                      const color = getColor(ann.classId);
                      const isSelected = selectedBox === ann.id;
                      const polygonPxPoints = ann.shape === "polygon" && ann.points
                        ? yoloToPixelPoints(ann.points)
                        : [];
                      return (
                        <div key={ann.id}>
                          {ann.shape === "polygon" && polygonPxPoints.length >= 3 ? (
                            <svg
                              style={{
                                position: "absolute",
                                inset: 0,
                                width: "100%",
                                height: "100%",
                                pointerEvents: "none",
                              }}
                            >
                              <polygon
                                points={polygonPxPoints.map((p) => `${p.x},${p.y}`).join(" ")}
                                fill={isSelected ? `${color}30` : `${color}18`}
                                stroke={color}
                                strokeWidth={isSelected ? 3 : 2}
                              />
                            </svg>
                          ) : (
                            <div
                              style={{
                                position: "absolute",
                                left: px.x,
                                top: px.y,
                                width: px.w,
                                height: px.h,
                                border: `${isSelected ? 3 : 2}px solid ${color}`,
                                background: isSelected
                                  ? `${color}20`
                                  : `${color}10`,
                                pointerEvents: "none",
                                boxSizing: "border-box",
                                borderRadius: ann.shape === "circle" ? "50%" : 0,
                              }}
                            />
                          )}
                          {showLabels && (
                            <div
                              style={{
                                position: "absolute",
                                left: px.x,
                                top: px.y - 18,
                                background: color,
                                color: "#000",
                                fontSize: 10,
                                fontWeight: 700,
                                padding: "1px 6px",
                                borderRadius: "3px 3px 0 0",
                                whiteSpace: "nowrap",
                                pointerEvents: "none",
                              }}
                            >
                              {ann.classId}: {classes[ann.classId] || `class_${ann.classId}`}
                            </div>
                          )}
                          {/* Resize handles */}
                          {isSelected && ann.shape !== "polygon" &&
                            [
                              { x: px.x, y: px.y },
                              { x: px.x + px.w, y: px.y },
                              { x: px.x, y: px.y + px.h },
                              { x: px.x + px.w, y: px.y + px.h },
                            ].map((h, i) => (
                              <div
                                key={i}
                                style={{
                                  position: "absolute",
                                  left: h.x - 4,
                                  top: h.y - 4,
                                  width: 8,
                                  height: 8,
                                  background: "#fff",
                                  border: `2px solid ${color}`,
                                  borderRadius: 2,
                                  pointerEvents: "none",
                                }}
                              />
                            ))}
                        </div>
                      );
                    })}

                    {/* Drawing preview: rectangle */}
                    {tool === "draw" && drawing && drawStart && drawEnd && (
                      <div
                        style={{
                          position: "absolute",
                          left: Math.min(drawStart.x, drawEnd.x),
                          top: Math.min(drawStart.y, drawEnd.y),
                          width: Math.abs(drawEnd.x - drawStart.x),
                          height: Math.abs(drawEnd.y - drawStart.y),
                          border: `2px dashed ${getColor(activeClassId)}`,
                          background: `${getColor(activeClassId)}15`,
                          pointerEvents: "none",
                        }}
                      />
                    )}
                    {/* Drawing preview: circle */}
                    {tool === "draw_circle" && drawing && drawStart && drawEnd && (() => {
                      const radius = Math.hypot(drawEnd.x - drawStart.x, drawEnd.y - drawStart.y);
                      return (
                        <div
                          style={{
                            position: "absolute",
                            left: drawStart.x - radius,
                            top: drawStart.y - radius,
                            width: radius * 2,
                            height: radius * 2,
                            border: `2px dashed ${getColor(activeClassId)}`,
                            background: `${getColor(activeClassId)}15`,
                            borderRadius: "50%",
                            pointerEvents: "none",
                          }}
                        />
                      );
                    })()}
                    {/* Drawing preview: polygon by points */}
                    {tool === "draw_poly" && polygonPoints.length > 0 && (
                      <svg
                        style={{
                          position: "absolute",
                          inset: 0,
                          width: "100%",
                          height: "100%",
                          pointerEvents: "none",
                        }}
                      >
                        <polyline
                          points={polygonPoints
                            .map((p) => `${p.x},${p.y}`)
                            .concat(polygonHover ? `${polygonHover.x},${polygonHover.y}` : [])
                            .join(" ")}
                          fill="none"
                          stroke={getColor(activeClassId)}
                          strokeWidth="2"
                          strokeDasharray="6 4"
                        />
                        {polygonPoints.map((p, i) => (
                          <circle
                            key={i}
                            cx={p.x}
                            cy={p.y}
                            r={i === 0 ? 5 : 3}
                            fill={i === 0 ? "#ffffff" : getColor(activeClassId)}
                            stroke={getColor(activeClassId)}
                            strokeWidth={i === 0 ? 2 : 0}
                          />
                        ))}
                        {polygonPoints.length >= 3 && polygonHover && (
                          <circle
                            cx={polygonPoints[0].x}
                            cy={polygonPoints[0].y}
                            r={Math.hypot(polygonHover.x - polygonPoints[0].x, polygonHover.y - polygonPoints[0].y) <= closeDistancePx ? 8 : 6}
                            fill="none"
                            stroke={getColor(activeClassId)}
                            strokeWidth="2"
                            strokeDasharray="4 4"
                          />
                        )}
                      </svg>
                    )}
                  </div>
                ) : (
                  <div
                    style={{
                      textAlign: "center",
                      color: "#52525b",
                      padding: 40,
                    }}
                  >
                    <div style={{ fontSize: 48, marginBottom: 16 }}>📷</div>
                    <p style={{ fontSize: 14, marginBottom: 8 }}>Загрузите изображения и лейблы</p>
                    <p style={{ fontSize: 12, color: "#3f3f46" }}>
                      Поддерживаются: jpg, png + txt (YOLO формат) + yaml + .pt
                    </p>
                  </div>
                )}
              </div>

              {/* Right panel */}
              <div
                style={{
                  width: 260,
                  borderLeft: "1px solid #27272a",
                  overflowY: "auto",
                  background: "#111113",
                  flexShrink: 0,
                }}
              >
                {/* Active class for drawing */}
                <div style={{ padding: 12, borderBottom: "1px solid #27272a" }}>
                  <div
                    style={{
                      fontSize: 11,
                      color: "#71717a",
                      marginBottom: 8,
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                    }}
                  >
                    Активный класс
                  </div>
                  <select
                    value={activeClassId}
                    onChange={(e) => setActiveClassId(parseInt(e.target.value))}
                    style={{
                      width: "100%",
                      padding: "6px 10px",
                      borderRadius: 5,
                      border: "1px solid #3f3f46",
                      background: "#0a0a0b",
                      color: "#e4e4e7",
                      fontSize: 12,
                      fontFamily: "inherit",
                    }}
                  >
                    {classes.map((cls, i) => (
                      <option key={i} value={i}>
                        {i}: {cls}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Selected box info */}
                {selectedAnn && (
                  <div style={{ padding: 12, borderBottom: "1px solid #27272a" }}>
                    <div
                      style={{
                        fontSize: 11,
                        color: "#71717a",
                        marginBottom: 8,
                        textTransform: "uppercase",
                        letterSpacing: "0.1em",
                      }}
                    >
                      Выбранный объект
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <span style={{ fontSize: 11, color: "#a1a1aa" }}>Класс:</span>
                      <select
                        value={selectedAnn.classId}
                        onChange={(e) =>
                          changeAnnotationClass(selectedAnn.id, parseInt(e.target.value))
                        }
                        style={{
                          width: "100%",
                          marginTop: 4,
                          padding: "5px 8px",
                          borderRadius: 4,
                          border: "1px solid #3f3f46",
                          background: "#0a0a0b",
                          color: "#e4e4e7",
                          fontSize: 12,
                          fontFamily: "inherit",
                        }}
                      >
                        {classes.map((cls, i) => (
                          <option key={i} value={i}>
                            {i}: {cls}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div style={{ fontSize: 11, color: "#52525b", lineHeight: 1.5 }}>
                      cx: {selectedAnn.bbox[0].toFixed(4)}
                      {" | "}cy: {selectedAnn.bbox[1].toFixed(4)}
                      <br />
                      w: {selectedAnn.bbox[2].toFixed(4)}
                      {" | "}h: {selectedAnn.bbox[3].toFixed(4)}
                    </div>
                    <button
                      onClick={() => deleteAnnotation(selectedAnn.id)}
                      style={{
                        marginTop: 8,
                        width: "100%",
                        padding: "6px",
                        borderRadius: 5,
                        border: "1px solid #ef444440",
                        background: "#ef444415",
                        color: "#ef4444",
                        fontSize: 12,
                        cursor: "pointer",
                        fontFamily: "inherit",
                      }}
                    >
                      Удалить bbox
                    </button>
                  </div>
                )}

                {/* Annotations list */}
                <div style={{ padding: 12, borderBottom: "1px solid #27272a" }}>
                  <div
                    style={{
                      fontSize: 11,
                      color: "#71717a",
                      marginBottom: 8,
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                    }}
                  >
                    Аннотации ({currentAnnotations.length})
                  </div>
                  <div style={{ maxHeight: 200, overflowY: "auto" }}>
                    {currentAnnotations.map((ann) => (
                      <div
                        key={ann.id}
                        onClick={() => setSelectedBox(ann.id)}
                        style={{
                          padding: "5px 8px",
                          borderRadius: 4,
                          cursor: "pointer",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          background:
                            selectedBox === ann.id ? "#f9731615" : "transparent",
                          marginBottom: 1,
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div
                            style={{
                              width: 10,
                              height: 10,
                              borderRadius: 2,
                              background: getColor(ann.classId),
                            }}
                          />
                          <span style={{ fontSize: 11, color: "#a1a1aa" }}>
                            {classes[ann.classId] || `class_${ann.classId}`}
                          </span>
                        </div>
                        <span style={{ fontSize: 10, color: "#52525b" }}>
                          {ann.classId}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Classes */}
                <div style={{ padding: 12 }}>
                  <div
                    style={{
                      fontSize: 11,
                      color: "#71717a",
                      marginBottom: 8,
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    Классы ({classes.length})
                    <button
                      onClick={() => setShowClasses(!showClasses)}
                      style={{
                        background: "none",
                        border: "none",
                        color: "#71717a",
                        cursor: "pointer",
                        fontSize: 10,
                      }}
                    >
                      {showClasses ? "▾" : "▸"}
                    </button>
                  </div>
                  {showClasses && (
                    <>
                      <div style={{ maxHeight: 200, overflowY: "auto", marginBottom: 8 }}>
                        {classes.map((cls, i) => (
                          <div
                            key={i}
                            onClick={() => setActiveClassId(i)}
                            style={{
                              padding: "4px 8px",
                              borderRadius: 4,
                              cursor: "pointer",
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              background:
                                activeClassId === i ? "#f9731615" : "transparent",
                              marginBottom: 1,
                            }}
                          >
                            <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0, flex: 1 }}>
                              <div
                                style={{
                                  width: 8,
                                  height: 8,
                                  borderRadius: 2,
                                  background: getColor(i),
                                  flexShrink: 0,
                                }}
                              />
                              {editingClass === i ? (
                                <input
                                  autoFocus
                                  value={editingClassName}
                                  onChange={(e) => setEditingClassName(e.target.value)}
                                  onClick={(e) => e.stopPropagation()}
                                  onBlur={saveEditClass}
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") saveEditClass();
                                    if (e.key === "Escape") cancelEditClass();
                                  }}
                                  style={{
                                    flex: 1,
                                    padding: "3px 6px",
                                    borderRadius: 4,
                                    border: "1px solid #f97316",
                                    background: "#0a0a0b",
                                    color: "#e4e4e7",
                                    fontSize: 11,
                                    fontFamily: "inherit",
                                  }}
                                />
                              ) : (
                                <span
                                  style={{
                                    fontSize: 11,
                                    color: activeClassId === i ? "#f97316" : "#a1a1aa",
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                  }}
                                >
                                  {i}: {cls}
                                </span>
                              )}
                            </div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                startEditClass(i);
                              }}
                              style={{
                                border: "none",
                                background: "transparent",
                                color: "#a1a1aa",
                                cursor: "pointer",
                                fontSize: 12,
                                padding: "0 4px",
                              }}
                              title="Переименовать класс"
                            >
                              ✎
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                removeClass(i);
                              }}
                              style={{
                                border: "none",
                                background: "transparent",
                                color: "#ef4444",
                                cursor: "pointer",
                                fontSize: 12,
                                padding: "0 4px",
                              }}
                              title="Удалить класс"
                            >
                              ✕
                            </button>
                          </div>
                        ))}
                      </div>
                      <div style={{ display: "flex", gap: 4 }}>
                        <input
                          value={newClassName}
                          onChange={(e) => setNewClassName(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && addClass()}
                          placeholder="Новый класс"
                          style={{
                            flex: 1,
                            padding: "5px 8px",
                            borderRadius: 4,
                            border: "1px solid #3f3f46",
                            background: "#0a0a0b",
                            color: "#e4e4e7",
                            fontSize: 11,
                            fontFamily: "inherit",
                          }}
                        />
                        <button
                          onClick={addClass}
                          style={{
                            padding: "5px 10px",
                            borderRadius: 4,
                            border: "none",
                            background: "#f97316",
                            color: "#000",
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: "pointer",
                            fontFamily: "inherit",
                          }}
                        >
                          +
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
