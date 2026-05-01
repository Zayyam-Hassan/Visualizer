"""
Render service: lighting preservation, blending, and final compositing.

Pipeline
--------
1. preserve_lighting  — extract luminance from original and apply to texture
2. blend_texture      — alpha-composite texture over original using mask
3. render_visualization — orchestrates the full pipeline end-to-end
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.services import segmentation_service, mask_service, texture_service, image_service
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def preserve_lighting(
    original: np.ndarray,
    warped_texture: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Transfer luminance from *original* onto *warped_texture* inside *mask* so
    that shadows, highlights, and perspective shading remain visible.
    """
    orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    tex_lab  = cv2.cvtColor(warped_texture, cv2.COLOR_RGB2LAB).astype(np.float32)

    orig_L = orig_lab[:, :, 0]
    tex_L  = tex_lab[:, :, 0]

    mean_L = np.mean(orig_L[mask > 0]) if mask.any() else 128.0
    mean_L = max(mean_L, 1.0)

    factor = (orig_L / mean_L).clip(0.2, 2.0)
    lit_L = (tex_L * factor).clip(0, 255)

    alpha = (mask / 255.0).astype(np.float32)
    blended_L = orig_L * (1 - alpha) + lit_L * alpha

    result_lab = tex_lab.copy()
    result_lab[:, :, 0] = blended_L
    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def blend_texture(
    original: np.ndarray,
    lit_texture: np.ndarray,
    feather_mask: np.ndarray,
    blend_strength: float = 0.85,
) -> np.ndarray:
    """
    Alpha-composite *lit_texture* over *original* using the soft *feather_mask*.

    Parameters
    ----------
    original        : RGB uint8
    lit_texture     : RGB uint8  (lighting-corrected texture)
    feather_mask    : float32 [0, 1]  (soft alpha, same H×W)
    blend_strength  : [0, 1] — 1.0 = fully replace with texture

    Returns
    -------
    composite : RGB uint8
    """
    alpha = (feather_mask * blend_strength).astype(np.float32)
    alpha3 = alpha[:, :, np.newaxis]  # broadcast over RGB

    orig_f = original.astype(np.float32)
    tex_f  = lit_texture.astype(np.float32)

    composite = orig_f * (1.0 - alpha3) + tex_f * alpha3
    return np.clip(composite, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class RenderOptions:
    __slots__ = (
        "target_surface",
        "tile_scale",
        "rotation_degrees",
        "blend_strength",
        "preserve_lighting_flag",
        "use_sam2_refine",
        "fill_enclosed_holes",
    )

    def __init__(
        self,
        target_surface: str = "floor",
        tile_scale: float = 1.0,
        rotation_degrees: float = 0.0,
        blend_strength: float = 0.85,
        preserve_lighting_flag: bool = True,
        use_sam2_refine: bool = False,
        fill_enclosed_holes: bool = False,
    ) -> None:
        self.target_surface = target_surface
        self.tile_scale = tile_scale
        self.rotation_degrees = rotation_degrees
        self.blend_strength = blend_strength
        self.preserve_lighting_flag = preserve_lighting_flag
        self.use_sam2_refine = use_sam2_refine
        self.fill_enclosed_holes = fill_enclosed_holes


def render_visualization(
    room_pil: Image.Image,
    texture_pil: Image.Image,
    options: RenderOptions,
) -> Tuple[np.ndarray, np.ndarray, list, float]:
    """
    Full render pipeline.

    Returns
    -------
    rendered_rgb   : np.ndarray  RGB uint8 final image
    mask_rgb       : np.ndarray  RGB uint8 mask (for display)
    polygon        : list[[x,y], …]
    elapsed_ms     : float  total processing time in milliseconds
    """
    t_start = time.perf_counter()

    # ---- Stage 1: Load images -----------------------------------------------
    t0 = time.perf_counter()
    room_rgb   = image_service.pil_to_numpy_rgb(room_pil)
    tex_rgb    = image_service.pil_to_numpy_rgb(texture_pil)
    h, w = room_rgb.shape[:2]
    log.info("[stage:load] %dx%d room, %.1f ms", w, h, (time.perf_counter() - t0) * 1000)

    # ---- Stage 2: Segmentation ----------------------------------------------
    t0 = time.perf_counter()
    raw_mask, _, _, _ = segmentation_service.run_segmentation(
        room_pil,
        options.target_surface,
        use_sam2_refine=options.use_sam2_refine,
    )
    log.info("[stage:segment] %.1f ms", (time.perf_counter() - t0) * 1000)

    # ---- Stage 3: Mask cleanup ----------------------------------------------
    t0 = time.perf_counter()
    clean_mask, feather, polygon, quad = mask_service.process_mask(
        raw_mask,
        fill_enclosed_holes=options.fill_enclosed_holes,
    )

    if not clean_mask.any():
        raise ValueError(
            f"No '{options.target_surface}' surface detected in the image. "
            "Try a different image or target_surface value."
        )
    log.info("[stage:mask] polygon=%d pts, quad=%s, %.1f ms",
             len(polygon), quad, (time.perf_counter() - t0) * 1000)

    # ---- Stage 4: Texture warp ----------------------------------------------
    t0 = time.perf_counter()
    warped_texture = texture_service.prepare_warped_texture(
        texture_image=tex_rgb,
        quad_points=quad,
        room_size=(w, h),
        tile_scale=options.tile_scale,
        rotation_degrees=options.rotation_degrees,
    )
    log.info("[stage:warp] %.1f ms", (time.perf_counter() - t0) * 1000)

    # ---- Stage 5: Lighting & blending ---------------------------------------
    t0 = time.perf_counter()
    if options.preserve_lighting_flag:
        lit_tex = preserve_lighting(room_rgb, warped_texture, clean_mask)
    else:
        lit_tex = warped_texture

    rendered = blend_texture(room_rgb, lit_tex, feather, options.blend_strength)
    log.info("[stage:blend] %.1f ms", (time.perf_counter() - t0) * 1000)

    # ---- Stage 6: Encode ----------------------------------------------------
    t0 = time.perf_counter()
    mask_display = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB)
    log.info("[stage:encode] %.1f ms", (time.perf_counter() - t0) * 1000)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    log.info("[pipeline:total] %.1f ms", elapsed_ms)

    return rendered, mask_display, polygon, elapsed_ms
