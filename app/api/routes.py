"""
API route handlers for all /api/v1/* endpoints.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import RenderFileResponse, RenderResponse, SegmentResponse
from app.services import image_service, mask_service, render_service, segmentation_service
from app.services.render_service import RenderOptions
from app.utils.logging_utils import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["visualizer"])

SUPPORTED_SURFACES = segmentation_service.SUPPORTED_SURFACES


# ---------------------------------------------------------------------------
# /api/v1/segment
# ---------------------------------------------------------------------------

@router.post("/segment", response_model=SegmentResponse)
async def segment_surface(
    room_image: UploadFile = File(..., description="Room photo"),
    target_surface: str = Form("floor", description="floor | wall | ceiling"),
    use_sam2_refine: bool = Form(get_settings().use_sam2_refine_default),
) -> SegmentResponse:
    """
    Detect and return the segmentation mask for the requested surface.
    """
    _validate_surface(target_surface)

    room_pil = await image_service.read_upload_to_pil(room_image)
    width, height = image_service.get_image_dimensions(room_pil)

    try:
        raw_mask, _, _, confidence = segmentation_service.run_segmentation(
            room_pil,
            target_surface,
            use_sam2_refine=use_sam2_refine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    clean, _, polygon, _ = mask_service.process_mask(raw_mask)

    if not clean.any():
        raise HTTPException(
            status_code=404,
            detail=f"No '{target_surface}' surface detected. Try a different image.",
        )

    mask_b64 = image_service.numpy_rgb_to_base64(clean)

    return SegmentResponse(
        target_surface=target_surface,
        mask_image_base64=mask_b64,
        polygon=polygon,
        confidence=confidence,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# /api/v1/render   (base64 response)
# ---------------------------------------------------------------------------

@router.post("/render", response_model=RenderResponse)
async def render_surface(
    room_image: UploadFile = File(..., description="Room photo"),
    texture_image: UploadFile = File(..., description="Material / texture tile"),
    target_surface: str = Form("floor"),
    tile_scale: float = Form(1.0, ge=0.1, le=10.0),
    rotation_degrees: float = Form(0.0, ge=-360.0, le=360.0),
    blend_strength: float = Form(0.85, ge=0.0, le=1.0),
    preserve_lighting: bool = Form(True),
    use_sam2_refine: bool = Form(get_settings().use_sam2_refine_default),
    fill_enclosed_holes: bool = Form(False),
) -> RenderResponse:
    """
    Apply a material texture to the detected surface and return the result.
    """
    _validate_surface(target_surface)

    room_pil    = await image_service.read_upload_to_pil(room_image)
    texture_pil = await image_service.read_upload_to_pil(texture_image)
    width, height = image_service.get_image_dimensions(room_pil)

    options = RenderOptions(
        target_surface=target_surface,
        tile_scale=tile_scale,
        rotation_degrees=rotation_degrees,
        blend_strength=blend_strength,
        preserve_lighting_flag=preserve_lighting,
        use_sam2_refine=use_sam2_refine,
        fill_enclosed_holes=fill_enclosed_holes,
    )

    try:
        rendered_rgb, mask_rgb, polygon, elapsed_ms = render_service.render_visualization(
            room_pil, texture_pil, options
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Render pipeline failed")
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

    return RenderResponse(
        target_surface=target_surface,
        rendered_image_base64=image_service.numpy_rgb_to_base64(rendered_rgb),
        mask_image_base64=image_service.numpy_rgb_to_base64(mask_rgb),
        polygon=polygon,
        width=width,
        height=height,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# /api/v1/render/file   (save to disk + return path)
# ---------------------------------------------------------------------------

@router.post("/render/file", response_model=RenderFileResponse)
async def render_surface_to_file(
    room_image: UploadFile = File(...),
    texture_image: UploadFile = File(...),
    target_surface: str = Form("floor"),
    tile_scale: float = Form(1.0, ge=0.1, le=10.0),
    rotation_degrees: float = Form(0.0, ge=-360.0, le=360.0),
    blend_strength: float = Form(0.85, ge=0.0, le=1.0),
    preserve_lighting: bool = Form(True),
    use_sam2_refine: bool = Form(get_settings().use_sam2_refine_default),
    fill_enclosed_holes: bool = Form(False),
) -> RenderFileResponse:
    """
    Same as /render but also saves the output image to disk.
    """
    _validate_surface(target_surface)

    room_pil    = await image_service.read_upload_to_pil(room_image)
    texture_pil = await image_service.read_upload_to_pil(texture_image)
    width, height = image_service.get_image_dimensions(room_pil)

    options = RenderOptions(
        target_surface=target_surface,
        tile_scale=tile_scale,
        rotation_degrees=rotation_degrees,
        blend_strength=blend_strength,
        preserve_lighting_flag=preserve_lighting,
        use_sam2_refine=use_sam2_refine,
        fill_enclosed_holes=fill_enclosed_holes,
    )

    try:
        rendered_rgb, mask_rgb, polygon, elapsed_ms = render_service.render_visualization(
            room_pil, texture_pil, options
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Render pipeline failed")
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

    file_path, file_url = image_service.save_numpy_rgb_to_file(rendered_rgb, prefix="render")

    return RenderFileResponse(
        target_surface=target_surface,
        rendered_image_base64=image_service.numpy_rgb_to_base64(rendered_rgb),
        mask_image_base64=image_service.numpy_rgb_to_base64(mask_rgb),
        polygon=polygon,
        width=width,
        height=height,
        processing_time_ms=round(elapsed_ms, 2),
        file_path=str(file_path),
        file_url=file_url,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_surface(surface: str) -> None:
    if surface.lower() not in SUPPORTED_SURFACES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported target_surface '{surface}'. "
                f"Supported values: {SUPPORTED_SURFACES}"
            ),
        )
