"""OpenCV / NumPy helper utilities."""

from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGBA/RGB/L image to an OpenCV BGR uint8 array."""
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def pil_to_rgb(image: Image.Image) -> np.ndarray:
    """Convert PIL image to a NumPy RGB uint8 array."""
    return np.array(image.convert("RGB"), dtype=np.uint8)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def rgb_to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb.astype(np.uint8))


def ndarray_to_base64_png(img: np.ndarray) -> str:
    """Encode a NumPy array (RGB or grayscale uint8) as a data-URI PNG string."""
    if img.ndim == 2:
        encode_img = img
    else:
        encode_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".png", encode_img)
    if not success:
        raise RuntimeError("cv2.imencode failed to encode image as PNG")

    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def resize_longest_side(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize *image* so its longest side is ≤ *max_side*; return (resized, scale)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0

    scale = max_side / longest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def safe_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
