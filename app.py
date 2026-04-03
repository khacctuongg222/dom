import os
import sys
import json
import re
import base64
import logging
from io import BytesIO
from typing import List, Tuple, Dict, Any, Generator

import gradio as gr
from PIL import Image, ImageOps
from google import genai
from transformers import AutoProcessor, AutoModelForImageTextToText

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
GLM_PATH       = "zai-org/GLM-OCR"
MODEL_PATH     = "faster_rcnn_resnet50_bom.pth"

MAX_INFER_SIZE = 2048

# Palette BGR
_PALETTE_BGR = [
    (30,  80, 230),
    (220, 100, 30),
    (30,  180, 60),
    (180, 230, 30),
    (230, 30, 180),
    (128, 0, 128),
    (100, 100, 100),
]

# ─────────────────────────────────────────────
# GLOBAL MODELS
# ─────────────────────────────────────────────

_gemini        = None
_glm_processor = None
_glm_model     = None
_detector      = None
_device        = None
_class_names: List[str] = []
_class_colors: Dict[str, Tuple[int, int, int]] = {}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_color_map(names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    return {
        name: _PALETTE_BGR[i % len(_PALETTE_BGR)]
        for i, name in enumerate(names)
    }

def _resize_for_inference(img_rgb: np.ndarray, max_size: int = MAX_INFER_SIZE) -> Tuple[np.ndarray, float]:
    """
    Resize ảnh sao cho cạnh dài nhất <= max_size.
    Trả về (img_resized, scale).
    """
    h, w = img_rgb.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale == 1.0:
        return img_rgb.copy(), 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def _crop_from_original(
    img_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    padding_ratio: float = 0.02,
) -> np.ndarray:
    """Crop vùng box từ ảnh gốc full-resolution (RGB)."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box

    pad_x = int((x2 - x1) * padding_ratio + w * 0.005)
    pad_y = int((y2 - y1) * padding_ratio + h * 0.005)

    x1c = max(0, x1 - pad_x)
    y1c = max(0, y1 - pad_y)
    x2c = min(w, x2 + pad_x)
    y2c = min(h, y2 + pad_y)

    return img_rgb[y1c:y2c, x1c:x2c].copy()

def pil_to_base64(pil_img: Image.Image, quality: int = 90) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def draw_boxes(
    img_rgb: np.ndarray,
    boxes_orig: np.ndarray,
    class_indices: List[int],
    scores: List[float],
) -> np.ndarray:
    """Vẽ bounding boxes lên ảnh RGB, trả về ảnh RGB."""
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, cls_idx, score in zip(boxes_orig, class_indices, scores):
        x1, y1, x2, y2 = map(int, box)

        if 0 <= cls_idx < len(_class_names):
            cls_name  = _class_names[cls_idx]
            color_bgr = _class_colors.get(cls_name, (128, 128, 128))
        else:
            cls_name  = "Unknown"
            color_bgr = (128, 128, 128)

        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)

        cv2.rectangle(out_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(out_bgr, label, (x1 + 2, y1 - 4),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

def has_enough_vram(required_gb: float = 4.0) -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        free, _ = torch.cuda.mem_get_info()
        return (free / (1024 ** 3)) >= required_gb
    except Exception:
        return False


def load_faster_rcnn(
    model_path: str, device: torch.device
) -> Tuple[nn.Module, List[str]]:
    """Load Faster R-CNN từ checkpoint, tự động xác định số class."""
    log.info("Loading Faster R-CNN (MobileNetV3 backbone)...")

    default_names = ["PartDrawing", "Note", "Table"]

    if not os.path.exists(model_path):
        log.warning(f"Checkpoint không tồn tại: {model_path}. Chạy với random weights.")
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(default_names) + 1)
        model.to(device).eval()
        return model, default_names

    log.info(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cls_weight = state_dict.get("roi_heads.box_predictor.cls_score.weight")
    if cls_weight is None:
        raise RuntimeError("Checkpoint thiếu key 'roi_heads.box_predictor.cls_score.weight'")

    actual_num_classes = cls_weight.shape[0]
    num_obj_classes    = actual_num_classes - 1
    log.info(f"Checkpoint num_classes (incl. background): {actual_num_classes}")

    if "class_names" in checkpoint and isinstance(checkpoint["class_names"], list):
        class_names = checkpoint["class_names"]
        log.info(f"Class names từ checkpoint: {class_names}")
    else:
        fallback = ["PartDrawing", "Note", "Table", "Other"]
        if num_obj_classes <= len(fallback):
            class_names = fallback[:num_obj_classes]
        else:
            class_names = [f"Class_{i + 1}" for i in range(num_obj_classes)]
        log.info(f"Inferred class_names ({num_obj_classes} objects): {class_names}")

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, actual_num_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log.warning(f"strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        log.info("Checkpoint loaded hoàn toàn khớp.")

    model.to(device).eval()
    return model, class_names


def load_all_models() -> None:
    global _gemini, _glm_processor, _glm_model, _detector, _device
    global _class_names, _class_colors

    _device = torch.device("cuda") if has_enough_vram(4) else torch.device("cpu")
    log.info(f"Device: {_device}")

    # Gemini
    try:
        api_key = GEMINI_API_KEY
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY chưa được đặt.")
        _gemini = genai.Client(api_key=api_key)
        log.info("Gemini client loaded.")
    except Exception as e:
        log.error(f"Không thể khởi tạo Gemini client: {e}")
        _gemini = None

    # GLM OCR
    try:
        log.info(f"Loading GLM OCR from {GLM_PATH}...")
        _glm_processor = AutoProcessor.from_pretrained(GLM_PATH, trust_remote_code=True)

        try:
            if torch.cuda.is_available():
                _glm_model = AutoModelForImageTextToText.from_pretrained(
                    GLM_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).eval()
                log.info("GLM-OCR loaded on GPU (float16)")
            else:
                raise RuntimeError("No CUDA available")
        except Exception as e:
            log.warning(f"GPU load failed ({e}), fallback to CPU")
            _glm_model = AutoModelForImageTextToText.from_pretrained(
                GLM_PATH,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
            log.info("GLM-OCR loaded on CPU")

        log.info("GLM OCR loaded.")
    except Exception as e:
        log.error(f"Không thể load GLM OCR: {e}")
        _glm_processor = None
        _glm_model = None

    # Faster R-CNN
    try:
        _detector, _class_names = load_faster_rcnn(MODEL_PATH, _device)
        _class_colors = _build_color_map(_class_names)
        log.info(f"Detector loaded. Classes: {_class_names}")
    except Exception as e:
        log.error(f"Không thể load detector: {e}")
        _detector     = None
        _class_names  = ["PartDrawing", "Note", "Table"]
        _class_colors = _build_color_map(_class_names)

    log.info("Model loading complete.")

# ─────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────

def run_detection(
    img_rgb: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, List[int], List[float]]:
    """Chạy Faster R-CNN. Trả về boxes, class_indices (0-based), scores."""
    if _detector is None:
        raise RuntimeError("Detector chưa được load.")

    img_tensor = T.ToTensor()(Image.fromarray(img_rgb)).to(_device)

    with torch.no_grad():
        predictions = _detector([img_tensor])

    pred   = predictions[0]
    boxes  = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    mask = scores > threshold
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), [], []

    boxes  = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # labels từ model là 1-based (0 = background)
    class_indices = (labels - 1).tolist()
    return boxes, class_indices, scores.tolist()

# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

def perform_glm_ocr(crop_pil: Image.Image, task_type: str, max_new_tokens: int = 1500) -> str:
    """GLM OCR — nhận crop PIL (RGB)."""
    if _glm_processor is None or _glm_model is None:
        return "[GLM OCR chưa được load]"

    if crop_pil.mode not in ("RGB",):
        crop_pil = crop_pil.convert("RGB")

    prompt = "Table Recognition:" if task_type == "Table" else "Text Recognition:"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": crop_pil},
            {"type": "text",  "text": prompt},
        ],
    }]

    device = next(_glm_model.parameters()).device
    inputs = _glm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = _glm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    raw  = _glm_processor.decode(output_ids[0], skip_special_tokens=True)
    text = re.sub(r"<\|.*?\|>", "", raw).strip()
    
    if task_type == "Table":
        text_lower = text.lower()
        if "<table" in text_lower:
            start     = text_lower.find("<table")
            end_table = text_lower.rfind("</table>")
            if end_table != -1:
                end = end_table + len("</table>")
            else:
                end_tr = text_lower.rfind("</tr>")
                end    = (end_tr + len("</tr>")) if end_tr != -1 else len(text)
            text = text[start:end]
            
    # ĐỐI TÁC LẬP TRÌNH ĐÃ SỬA Ở ĐÂY: Thêm lệnh return để trả về kết quả
    return text


def refine_with_gemini(raw: str, task_type: str) -> str:
    """Dùng Gemini để sửa lỗi OCR; nếu lỗi thì fallback xóa tiêu đề prompt."""
    
    # Kiểm tra an toàn: nếu raw là None do lỗi trước đó, trả về chuỗi rỗng
    if not raw:
        return ""

    def fallback_clean(t: str) -> str:
        return re.sub(r"^(Text|Table) Recognition:\s*", "", t.strip()).strip()

    if _gemini is None:
        return fallback_clean(raw)

    if task_type == "Table":
        prompt = f"""
Bạn là chuyên gia sửa lỗi OCR cho bản vẽ kỹ thuật cơ khí.
Nhiệm vụ:
1. Nhận mã HTML từ kết quả OCR.
2. Sửa lỗi chính tả, giữ nguyên cấu trúc HTML (colspan, rowspan).
3. Loại bỏ hoàn toàn cụm từ "Table Recognition:" nếu có.
4. CHỈ trả về mã HTML sạch bên trong cặp thẻ <table>, không giải thích.

Dữ liệu OCR:
{raw}
"""
    else:
        prompt = f"""
Bạn là chuyên gia sửa lỗi OCR cho bản vẽ kỹ thuật cơ khí.
Nhiệm vụ:
1. Nhận văn bản OCR.
2. Sửa lỗi chính tả, giữ nguyên thuật ngữ chuyên ngành.
3. Loại bỏ cụm từ "Text Recognition:" nếu có.
4. CHỈ trả về văn bản đã sửa, không giải thích.

Dữ liệu OCR:
{raw}
"""

    try:
        resp = _gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = resp.text.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    except Exception as e:
        log.warning(f"Gemini refinement failed: {e}")
        return fallback_clean(raw)


def ocr_one_region(
    obj_id: int, cls_name: str, crop_rgb: np.ndarray, max_new_tokens: int = 1500
) -> Tuple[int, str, str, str]:
    """
    Xử lý OCR cho 1 vùng.
    Trả về (obj_id, cls_name, ocr_content, base64_jpeg).
    """
    if cls_name in ("PartDrawing", "OTHER"):
        crop_pil = Image.fromarray(crop_rgb)
        return obj_id, cls_name, "", pil_to_base64(crop_pil)

    crop_pil = Image.fromarray(crop_rgb)
    b64      = pil_to_base64(crop_pil)

    try:
        raw     = perform_glm_ocr(crop_pil, cls_name, max_new_tokens)
        refined = refine_with_gemini(raw, cls_name)
    except Exception as e:
        log.error(f"OCR error for object {obj_id}: {e}")
        refined = f"Lỗi OCR: {e}"

    return obj_id, cls_name, refined, b64

# ─────────────────────────────────────────────
# HTML RESULT
# ─────────────────────────────────────────────

def build_result_html(objects: List[Dict[str, Any]]) -> str:
    if not objects:
        return "<p style='color:gray;padding:1rem'>Chưa phát hiện đối tượng nào hoặc đang chờ xử lý.</p>"

    badge_style: Dict[str, Tuple[str, str]] = {}
    for cls, (b, g, r) in _class_colors.items():
        rgb  = f"rgb({r},{g},{b})"
        rgba = f"rgba({r},{g},{b},0.12)"
        badge_style[cls] = (
            f"background:{rgba};color:{rgb}",
            f"border:1.5px solid {rgb}",
        )

    counts: Dict[str, int] = {}
    for o in objects:
        counts[o["class"]] = counts.get(o["class"], 0) + 1

    global_table_css = """
.det-table-content table { border-collapse:collapse; min-width:100%; }
.det-table-content th,
.det-table-content td  { border:1px solid #d1d5db; padding:5px 10px; text-align:left; }
.det-table-content th  { background:#f3f4f6; font-weight:500; }
.det-table-content tr:nth-child(even) { background:#f9fafb; }
"""

    def make_card(obj: Dict[str, Any]) -> str:
        cls   = obj["class"]
        oid   = obj["id"]
        conf  = obj["confidence"]
        b64   = obj.get("crop_b64", "")
        bg_s, bd_s = badge_style.get(cls, ("background:#f3f4f6;color:#374151", ""))

        if b64:
            crop_html = f"""
<div style="flex-shrink:0;width:180px;max-width:30%;min-height:80px;
            border-radius:8px;overflow:hidden;border:1px solid #e5e7eb;
            background:#f9fafb;display:flex;align-items:center;justify-content:center;">
    <img src="data:image/jpeg;base64,{b64}"
         style="max-width:100%;max-height:150px;object-fit:contain;display:block;">
</div>"""
        else:
            crop_html = f"""
<div style="flex-shrink:0;width:180px;max-width:30%;height:100px;border-radius:8px;
            border:1px dashed #d1d5db;background:#f9fafb;display:flex;
            align-items:center;justify-content:center;font-size:12px;color:#9ca3af">
    {cls} #{oid}
</div>"""

        content = obj.get("ocr_content", "")
        if content:
            if cls == "Table":
                right_html = f"""
<div style="overflow-x:auto;width:100%;">
    <div class="det-table-content">{content}</div>
</div>"""
            else:
                right_html = f"""
<p style="font-size:14px;line-height:1.5;margin:0;word-break:break-word;white-space:normal;">
    {content}
</p>"""
        else:
            bbox = obj.get("bbox", {})
            right_html = f"""
<p style="font-size:13px;color:#6b7280;margin:0;word-break:break-word;">
    Vị trí: ({bbox.get('x1',0):.0f}, {bbox.get('y1',0):.0f})
    → ({bbox.get('x2',0):.0f}, {bbox.get('y2',0):.0f})
</p>"""

        return f"""
<div class="det-card" data-class="{cls}"
     style="display:flex;flex-direction:row;gap:14px;align-items:flex-start;
       border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;
       padding:14px;margin-bottom:10px;width:100%;box-sizing:border-box;
       flex-shrink:0;overflow:hidden;">
    {crop_html}
    <div style="flex:1;min-width:0;overflow:hidden;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;">
            <span style="font-size:12px;font-weight:500;padding:3px 10px;
                         border-radius:6px;{bg_s}">{cls}</span>
            <span style="font-size:12px;color:#6b7280">#{oid}&nbsp;·&nbsp;{conf:.0%}</span>
        </div>
        {right_html}
    </div>
</div>"""

    all_cards = "".join(make_card(o) for o in objects)
    total     = len(objects)

    summary_parts = []
    for cls, cnt in counts.items():
        if cnt == 0:
            continue
        bg_s, bd_s = badge_style.get(cls, ("background:#f3f4f6;color:#374151", ""))
        summary_parts.append(
            f"<span style='font-size:13px;padding:4px 12px;border-radius:6px;"
            f"{bg_s};{bd_s};font-weight:500'>{cls}&nbsp;{cnt}</span>"
        )

    tab_btns = ['<button class="det-tab active" onclick="detFilter(\'all\',this)">Tất cả</button>']
    for cls in counts:
        if counts[cls] > 0:
            tab_btns.append(
                f'<button class="det-tab" onclick="detFilter(\'{cls}\',this)">{cls}</button>'
            )

    return f"""
<style>
{global_table_css}
.det-tab {{
    background:#f3f4f6;border:1px solid #d1d5db;border-radius:6px;
    padding:5px 14px;font-size:13px;cursor:pointer;color:#374151;
    transition:background .15s;
}}
.det-tab:hover {{ background:#e5e7eb; }}
.det-tab.active {{ background:#1e293b;color:#f8fafc;border-color:#1e293b; }}
.det-card {{ box-sizing:border-box; }}
.det-card img {{ max-width:100%;height:auto; }}
</style>
<div style="margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <span style="font-size:14px;font-weight:500;color:#111827;">{total} đối tượng đã xử lý</span>
    {''.join(summary_parts)}
</div>
<div style="display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;">{''.join(tab_btns)}</div>
<div id="det-cards-wrap" style="display:flex;flex-direction:column;gap:0;">{all_cards}</div>
<script>
function detFilter(cls, btn) {{
    document.querySelectorAll('.det-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.det-card').forEach(card => {{
        card.style.display = (cls === 'all' || card.dataset.class === cls) ? 'flex' : 'none';
    }});
}}
</script>"""

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main_pipeline(
    pil_img: Image.Image,
    confidence_threshold: float = 0.9,
    max_new_tokens: int = 1500,
) -> Generator[Tuple[Image.Image, str, str], None, None]:
    """
    Generator trả về (ảnh đã vẽ box, json, html) sau mỗi bước.
    """
    if pil_img is None:
        yield None, "{}", "<p style='color:gray'>Chưa có ảnh đầu vào.</p>"
        return

    if _detector is None:
        yield None, "{}", "<p style='color:red'>❌ Detector chưa được load. Kiểm tra logs khởi động.</p>"
        return

    # 1. Chuẩn hoá ảnh
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img_rgb_orig = np.array(pil_img)
    h_orig, w_orig = img_rgb_orig.shape[:2]
    log.info(f"Input image: {w_orig}x{h_orig}")

    # 2. Resize cho detection
    img_resized, scale = _resize_for_inference(img_rgb_orig, MAX_INFER_SIZE)
    log.info(f"Resized to {img_resized.shape[1]}x{img_resized.shape[0]}, scale={scale:.3f}")

    # 3. Detection
    try:
        boxes_resized, class_indices, scores = run_detection(img_resized, confidence_threshold)
        log.info(f"Detected {len(scores)} objects (threshold={confidence_threshold})")
    except Exception as e:
        log.error(f"Detection error: {e}")
        yield None, "{}", f"<p style='color:red'>❌ Lỗi detection: {e}</p>"
        return

    if boxes_resized.shape[0] > 0:
        boxes_orig = boxes_resized / scale
    else:
        boxes_orig = boxes_resized  # shape (0,4), không cần chia

    # Vẽ boxes
    viz_rgb = draw_boxes(img_rgb_orig, boxes_orig, class_indices, scores)
    viz_pil = Image.fromarray(viz_rgb)

    # Chuẩn bị tasks
    tasks: List[Tuple[int, str, np.ndarray]] = []
    result_objects_meta: List[Dict[str, Any]] = []

    for box_orig, cls_idx, score in zip(boxes_orig, class_indices, scores):
        if not (0 <= cls_idx < len(_class_names)):
            continue
        cls_name = _class_names[cls_idx]
        x1, y1, x2, y2 = map(int, box_orig)

        crop_rgb = _crop_from_original(img_rgb_orig, (x1, y1, x2, y2), padding_ratio=0.05)
        if crop_rgb.size == 0:
            continue

        obj_id = len(tasks) + 1
        tasks.append((obj_id, cls_name, crop_rgb))
        result_objects_meta.append({
            "id":         obj_id,
            "class":      cls_name,
            "confidence": round(float(score), 4),
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    total_objects = len(tasks)
    initial_html  = (
        f"<p style='color:#2563eb;font-weight:bold;font-size:16px;'>"
        f"✓ Đã tìm thấy {total_objects} đối tượng. Đang bắt đầu xử lý OCR...</p>"
    )
    yield viz_pil, "{}", initial_html

    # 4. OCR tuần tự
    final_objects: List[Dict[str, Any]] = []
    json_objects: List[Dict[str, Any]]  = []

    for idx, (obj_id, cls_name, crop_rgb) in enumerate(tasks):
        meta = result_objects_meta[idx]

        try:
            _, _, content, b64 = ocr_one_region(obj_id, cls_name, crop_rgb, max_new_tokens)
        except Exception as e:
            log.error(f"OCR failed for object {obj_id}: {e}")
            b64     = pil_to_base64(Image.fromarray(crop_rgb))
            content = f"Lỗi OCR: {e}"

        current_obj = {**meta, "ocr_content": content, "crop_b64": b64}
        final_objects.append(current_obj)
        json_objects.append({
            "id":          obj_id,
            "class":       meta["class"],
            "confidence":  meta["confidence"],
            "bbox":        meta["bbox"],
            "ocr_content": content,
        })

        current_json = json.dumps(
            {"image": "uploaded.jpg", "objects": json_objects},
            indent=2, ensure_ascii=False,
        )
        progress_html = (
            f"<div style='margin-bottom:15px;padding:12px;background:#eff6ff;"
            f"border-left:4px solid #3b82f6;border-radius:4px;font-weight:500;color:#1e40af;'>"
            f"⏳ Tiến độ OCR: {idx + 1}/{total_objects}</div>"
        )
        yield viz_pil, current_json, progress_html + build_result_html(final_objects)

    # Hoàn tất
    final_json = json.dumps(
        {"image": "uploaded.jpg", "objects": json_objects},
        indent=2, ensure_ascii=False,
    )
    done_html = (
        f"<div style='margin-bottom:15px;padding:12px;background:#ecfdf5;"
        f"border-left:4px solid #10b981;border-radius:4px;font-weight:500;color:#065f46;'>"
        f"✅ Hoàn tất phân tích {total_objects} đối tượng!</div>"
    )
    yield viz_pil, final_json, done_html + build_result_html(final_objects)

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Engineering Drawing Analyzer") as demo:
        gr.Markdown("# 🔩 Engineering Drawing Analyzer")
        gr.Markdown(
            f"**Backbone:** Faster R-CNN MobileNetV3 · "
            f"**Classes:** {', '.join(_class_names)} · "
            f"**Device:** {_device}"
        )

        with gr.Row():
            img_input  = gr.Image(type="pil", label="Ảnh đầu vào")
            viz_output = gr.Image(label="Kết quả detection")

        confidence_slider = gr.Slider(
            minimum=0.1, maximum=0.9, value=0.9, step=0.05,
            label="Ngưỡng confidence",
        )
        token_slider = gr.Slider(
            minimum=256, maximum=10240, value=1500, step=128,
            label="Max tokens OCR",
        )

        btn = gr.Button("🚀 Analyze", variant="primary")

        result_output = gr.HTML(label="Kết quả chi tiết")

        with gr.Accordion("📄 JSON output", open=False):
            json_output = gr.Code(language="json")


        btn.click(
            fn=main_pipeline,
            inputs=[img_input, confidence_slider, token_slider],
            outputs=[viz_output, json_output, result_output],
        )

    return demo

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting — loading all models...")
    load_all_models()
    log.info(f"Active classes: {_class_names}")
    log.info(f"Color map:      {_class_colors}")

    demo = build_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)