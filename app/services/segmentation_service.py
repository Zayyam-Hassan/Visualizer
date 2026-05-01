"""
Semantic segmentation service using HuggingFace Transformers.

Default model: nvidia/segformer-b2-finetuned-ade-512-512 (ADE20K, 150 classes)
Model is configurable via MODEL_NAME env var — any AutoModelForSemanticSegmentation
compatible checkpoint that outputs per-pixel class logits will work.

ADE20K surface label mapping
------------------------------
The ADE20K dataset has 150 classes.  The indices below are the *model output*
class IDs (matching the id2label dict in the model config).  If you swap the
model for one trained on a different dataset, update SURFACE_LABEL_ALIASES so
that the string keys map to the correct integer class IDs for that dataset.

Known ADE20K class IDs relevant to interior surfaces:
  0  → wall
  3  → floor, flooring
  5  → ceiling
  6  → road, route       (outdoor; included as floor fallback)
 53  → rug, carpet, mat  (can be added as a floor sub-class)

To adjust: edit SURFACE_LABEL_ALIASES at the bottom of this file.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from app.config import get_settings
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Label aliases
# ---------------------------------------------------------------------------
# Maps the user-facing target_surface string to one or more ADE20K class IDs.
# Order matters: earlier IDs are preferred when multiple classes match.
# Adjust for your model / dataset if you change MODEL_NAME.
SURFACE_LABEL_ALIASES: Dict[str, List[str]] = {
    # rug/carpet/mat stay in the coverage mask so the new texture replaces them.
    # Do not remove them here — lighting exclusion is handled separately via
    # LIGHTING_EXCLUDE_LABEL_ALIASES (same model pass, no luminance heuristics).
    "floor": ["floor", "flooring", "rug", "carpet", "mat", "ground"],
    "wall": ["wall", "partition"],
    "ceiling": ["ceiling"],
}

# Classes matched here become 255 in *lighting_exclude_mask*: pixels omitted from
# the low-pass lighting reference (still textured like the rest of the surface).
# Tune per dataset when MODEL_NAME changes (same id2label rules as above).
LIGHTING_EXCLUDE_LABEL_ALIASES: Dict[str, List[str]] = {
    "floor": ["rug", "carpet", "mat"],
    "wall": [],
    "ceiling": [],
}

SUPPORTED_SURFACES = list(SURFACE_LABEL_ALIASES.keys())

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_processor: Optional[AutoImageProcessor] = None
_model: Optional[AutoModelForSemanticSegmentation] = None
_id2label: Optional[Dict[int, str]] = None
_device: Optional[torch.device] = None
_model_loaded: bool = False
_sam2_predictor = None
_sam2_ready: bool = False


def _resolve_device() -> torch.device:
    settings = get_settings()
    raw = settings.device.lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def load_model() -> None:
    """Load the segmentation model and processor into the global cache."""
    global _processor, _model, _id2label, _device, _model_loaded

    if _model_loaded:
        return

    settings = get_settings()
    model_name = settings.model_name
    _device = _resolve_device()

    log.info("Loading segmentation model '%s' on device '%s' …", model_name, _device)
    t0 = time.perf_counter()

    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()

    _id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
    _model_loaded = True

    elapsed = time.perf_counter() - t0
    log.info("Model loaded in %.2f s.  Classes: %d", elapsed, len(_id2label))
    log.debug("id2label sample: %s", dict(list(_id2label.items())[:10]))


def is_model_loaded() -> bool:
    return _model_loaded


def get_device_info() -> str:
    if _device is None:
        return "not initialized"
    if _device.type == "cuda":
        idx = _device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return "cpu"


# ---------------------------------------------------------------------------
# Label → class-ID resolution
# ---------------------------------------------------------------------------

def _resolve_class_ids(target_surface: str) -> List[int]:
    """
    Return a list of model class IDs that correspond to *target_surface*.

    Resolution order:
    1. Exact match on id2label values.
    2. Substring match using SURFACE_LABEL_ALIASES.
    3. Empty list → caller should raise an error.
    """
    if _id2label is None:
        raise RuntimeError("Model not loaded — call load_model() first.")

    aliases = SURFACE_LABEL_ALIASES.get(target_surface.lower(), [target_surface.lower()])

    matched: List[int] = []
    for class_id, label in _id2label.items():
        label_lower = label.lower()
        for alias in aliases:
            if alias in label_lower or label_lower in alias:
                matched.append(class_id)
                break

    if not matched:
        log.warning(
            "No class IDs found for surface '%s'. id2label excerpt: %s",
            target_surface,
            dict(list(_id2label.items())[:20]),
        )

    log.debug("Surface '%s' → class IDs %s", target_surface, matched)
    return matched


def _resolve_class_ids_from_alias_strings(aliases: List[str]) -> List[int]:
    """Match *aliases* against id2label (substring), same rules as surface resolution."""
    if _id2label is None:
        raise RuntimeError("Model not loaded — call load_model() first.")
    if not aliases:
        return []
    matched: List[int] = []
    for class_id, label in _id2label.items():
        label_lower = label.lower()
        for alias in aliases:
            al = alias.lower()
            if al in label_lower or label_lower in al:
                matched.append(class_id)
                break
    log.debug("Lighting-exclusion aliases %s → class IDs %s", aliases, matched)
    return matched


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _preprocess_for_inference(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    """Optionally downscale *image* for inference, returning (processed_img, original_size)."""
    original_size = image.size  # (W, H)
    settings = get_settings()
    max_res = settings.max_inference_resolution

    if max_res > 0:
        w, h = image.size
        if max(w, h) > max_res:
            scale = max_res / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            log.debug("Downscaled for inference: %s → %s", original_size, image.size)

    return image, original_size


def run_segmentation(
    image: Image.Image,
    target_surface: str = "floor",
    use_sam2_refine: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[float]]:
    """
    Run semantic segmentation on *image* and return a binary mask for *target_surface*.

    Returns
    -------
    mask : np.ndarray  shape (H, W) uint8, values 0 or 255 — full surface coverage
    lighting_exclude_mask : np.ndarray  uint8, 255 where pixels should not contribute
        to the lighting reference (e.g. rug/carpet/mat on floor). Same shape as *mask*.
        Zeros when no exclusion aliases apply or none matched in id2label.
    class_ids_used : list[int]   the model class IDs merged into *mask*
    confidence : float | None    placeholder; SegFormer does not expose per-class confidence
    """
    if not _model_loaded:
        raise RuntimeError("Segmentation model is not loaded.")

    target_surface = target_surface.lower()
    if target_surface not in SUPPORTED_SURFACES:
        raise ValueError(
            f"Unsupported target surface '{target_surface}'. "
            f"Supported: {SUPPORTED_SURFACES}"
        )

    class_ids = _resolve_class_ids(target_surface)
    if not class_ids:
        raise ValueError(
            f"No model classes found for surface '{target_surface}'. "
            f"Check SURFACE_LABEL_ALIASES in segmentation_service.py."
        )

    # --- Preprocessing -------------------------------------------------------
    infer_image, original_size = _preprocess_for_inference(image.convert("RGB"))

    inputs = _processor(images=infer_image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # --- Inference -----------------------------------------------------------
    with torch.no_grad():
        outputs = _model(**inputs)

    # logits: (1, num_classes, H', W')
    logits = outputs.logits  # type: ignore[union-attr]
    predicted = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)  # (H', W')

    # --- Build binary mask ---------------------------------------------------
    binary = np.zeros_like(predicted, dtype=np.uint8)
    for cid in class_ids:
        binary[predicted == cid] = 255

    exclude_aliases = LIGHTING_EXCLUDE_LABEL_ALIASES.get(target_surface, [])
    exclude_ids = _resolve_class_ids_from_alias_strings(exclude_aliases)
    exclude_binary = np.zeros_like(predicted, dtype=np.uint8)
    for cid in exclude_ids:
        exclude_binary[predicted == cid] = 255

    # Resize mask back to original image dimensions (W, H) → cv resize wants (W, H)
    orig_w, orig_h = original_size
    if binary.shape != (orig_h, orig_w):
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        exclude_binary = cv2.resize(exclude_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if use_sam2_refine:
        binary = _refine_mask_with_sam2(image=image, coarse_mask=binary, target_surface=target_surface)

    log.debug(
        "Segmentation done — surface '%s', class_ids %s, mask coverage %.1f%%, "
        "lighting_exclude coverage %.2f%%",
        target_surface,
        class_ids,
        100.0 * binary.mean() / 255.0,
        100.0 * exclude_binary.mean() / 255.0,
    )

    return binary, exclude_binary, class_ids, None  # confidence not available from logits directly


def _refine_mask_with_sam2(
    image: Image.Image,
    coarse_mask: np.ndarray,
    target_surface: str,
) -> np.ndarray:
    """
    Refine a coarse surface mask using SAM-2 with a box prompt from the coarse mask.

    If SAM-2 is unavailable or not configured, this returns *coarse_mask* unchanged.
    """
    if not coarse_mask.any():
        return coarse_mask

    predictor = _get_sam2_predictor()
    if predictor is None:
        log.warning(
            "SAM-2 refinement requested for '%s' but SAM-2 is unavailable; using coarse mask.",
            target_surface,
        )
        return coarse_mask

    ys, xs = np.where(coarse_mask > 0)
    if len(xs) == 0:
        return coarse_mask
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    try:
        predictor.set_image(np.asarray(image.convert("RGB")))
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        if masks is None or len(masks) == 0:
            return coarse_mask
        refined = (masks[0].astype(np.uint8) * 255).astype(np.uint8)
        # Keep refinement conservative to avoid bleeding onto non-target areas.
        refined = cv2.bitwise_and(refined, coarse_mask)
        if refined.any():
            log.debug("SAM-2 refinement applied for '%s' (score=%s).", target_surface, scores[0] if len(scores) else "n/a")
            return refined
    except Exception:
        log.warning("SAM-2 refinement failed; falling back to coarse mask.", exc_info=True)

    return coarse_mask


def _get_sam2_predictor():
    """Lazily initialize SAM-2 predictor. Returns None when unavailable."""
    global _sam2_predictor, _sam2_ready
    if _sam2_ready:
        return _sam2_predictor

    settings = get_settings()
    try:
        # Optional dependency. Keep imports local so core segmentation works without SAM-2.
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception:
        return None

    try:
        device = _resolve_device()
        # Preferred path: use Hugging Face model ID, per official SAM-2 API.
        if settings.sam2_model_id:
            _sam2_predictor = SAM2ImagePredictor.from_pretrained(
                settings.sam2_model_id,
                device=device.type,
            )
        else:
            # Fallback path: local SAM-2 config + checkpoint.
            if not settings.sam2_checkpoint_path or not settings.sam2_config_path:
                return None
            from sam2.build_sam import build_sam2  # type: ignore

            sam2_model = build_sam2(
                config_file=settings.sam2_config_path,
                ckpt_path=settings.sam2_checkpoint_path,
                device=device.type,
            )
            _sam2_predictor = SAM2ImagePredictor(sam2_model)
        _sam2_ready = True
        log.info("SAM-2 predictor initialized on %s.", device)
    except Exception:
        log.warning("Failed to initialize SAM-2 predictor.", exc_info=True)
        _sam2_predictor = None

    return _sam2_predictor


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def warmup() -> None:
    """Run a tiny inference pass to JIT-compile kernels and warm the cache."""
    if not _model_loaded:
        return
    log.info("Running model warm-up …")
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    try:
        run_segmentation(dummy, target_surface="floor")
        log.info("Warm-up complete.")
    except Exception:
        log.warning("Warm-up inference failed (non-fatal).", exc_info=True)
