"""File-handling utilities: validation, safe naming, path helpers."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException

from app.config import get_settings
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def validate_image_upload(file: UploadFile) -> None:
    """Raise HTTPException if *file* fails type or size validation."""
    settings = get_settings()

    # Extension check
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Content-type check (browsers set this; do not rely on it alone)
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        log.warning("Unexpected content-type '%s' for file '%s'", file.content_type, file.filename)

    # Size check — file.size may be None for chunked uploads; we re-check after reading
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed: {settings.max_upload_mb} MB.",
        )


def safe_output_filename(prefix: str = "render", ext: str = ".png") -> str:
    """Generate a collision-safe filename for saved output images."""
    return f"{prefix}_{uuid.uuid4().hex}{ext}"


def output_file_path(filename: str) -> Path:
    """Return the absolute path for a file in the configured output directory."""
    settings = get_settings()
    return settings.resolved_output_dir / filename
