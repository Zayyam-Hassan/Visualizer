"""Image I/O service: reading uploads, format conversion, saving outputs."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.utils.cv_utils import ndarray_to_base64_png
from app.utils.file_utils import validate_image_upload, safe_output_filename, output_file_path
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


async def read_upload_to_pil(file: UploadFile) -> Image.Image:
    """
    Read *file* bytes, validate size/type, and return a PIL Image.

    Raises HTTPException on validation failure or corrupt image data.
    """
    validate_image_upload(file)

    settings = get_settings()
    data = await file.read()

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large after reading. Maximum allowed: {settings.max_upload_mb} MB.",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decode so corrupt files raise here
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Cannot identify image file. Ensure it is a valid image.")
    except Exception as exc:
        log.exception("Unexpected error reading image upload")
        raise HTTPException(status_code=422, detail=f"Failed to read image: {exc}") from exc


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def numpy_rgb_to_base64(img: np.ndarray) -> str:
    return ndarray_to_base64_png(img)


def save_numpy_rgb_to_file(img: np.ndarray, prefix: str = "render") -> Tuple[Path, str]:
    """
    Save *img* (RGB uint8) to the outputs directory.

    Returns (absolute_path, relative_url_path).
    """
    filename = safe_output_filename(prefix=prefix, ext=".png")
    path = output_file_path(filename)

    pil = Image.fromarray(img.astype(np.uint8))
    pil.save(str(path), format="PNG")

    log.info("Saved output image → %s", path)
    return path, f"/outputs/{filename}"


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Return (width, height) of *image*."""
    return image.size  # PIL returns (width, height)
