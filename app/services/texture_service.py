"""
Texture tiling, rotation, and perspective warping service.

Core design principle
---------------------
The quad extracted from the floor mask is used ONLY to compute a perspective
homography (how the floor plane recedes into the distance).  The warped
texture is painted across the ENTIRE image canvas — not clipped to the quad.

Clipping is the mask's responsibility.  By separating these concerns:

  • Complex floor shapes work correctly (L-shaped floors, multiple patches).
  • Furniture sitting on the floor is not affected: the segmentation mask has
    zero values under furniture, so those pixels get zero texture blend.
  • The texture correctly continues "behind" furniture legs at the right
    perspective — if a table is lifted the floor looks correct underneath.

Pipeline
--------
1. create_repeated_texture  — tile the small texture to fill a large canvas
2. rotate_texture           — optional rotation before warping
3. warp_texture_perspective — apply floor-plane homography across full image
4. prepare_warped_texture   — convenience wrapper for the full pipeline
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from app.utils.logging_utils import get_logger

log = get_logger(__name__)

Point = List[int]
Quad = List[Point]  # exactly 4 points in TL, TR, BR, BL order


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def create_repeated_texture(
    texture: np.ndarray,
    output_width: int,
    output_height: int,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Tile *texture* (RGB uint8) to fill a canvas of (*output_height* × *output_width*).

    scale < 1  → smaller tiles (denser repetition, e.g. small mosaic tiles)
    scale > 1  → larger tiles (e.g. wide-plank wood)
    """
    th, tw = texture.shape[:2]
    tile_w = max(1, int(tw * scale))
    tile_h = max(1, int(th * scale))
    scaled_tile = cv2.resize(texture, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

    cols = int(np.ceil(output_width / tile_w)) + 1
    rows = int(np.ceil(output_height / tile_h)) + 1

    canvas = np.tile(scaled_tile, (rows, cols, 1))  # type: ignore[call-overload]
    canvas = canvas[:output_height, :output_width, :]

    log.debug(
        "Tiled: tile=%dx%d  canvas=%dx%d  reps=%dx%d",
        tile_w, tile_h, output_width, output_height, cols, rows,
    )
    return canvas.astype(np.uint8)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotate_texture(texture: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotate *texture* counter-clockwise by *degrees*, expanding the canvas so
    no pixel is cropped.
    """
    if degrees == 0.0:
        return texture

    h, w = texture.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, degrees, scale=1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(
        texture, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    log.debug("Rotated %.1f°: %dx%d → %dx%d", degrees, w, h, new_w, new_h)
    return rotated


# ---------------------------------------------------------------------------
# Perspective warping
# ---------------------------------------------------------------------------

def warp_texture_perspective(
    texture_canvas: np.ndarray,
    quad_points: Quad,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """
    Warp *texture_canvas* across the FULL output image using the perspective
    homography derived from *quad_points*.

    Why full-image, not quad-clipped
    ---------------------------------
    The quad is the convex hull of the floor and is used only to compute
    *how* the floor plane recedes in perspective (i.e., the homography matrix).
    The actual boundary of the floor — including concavities carved out by
    furniture — is defined by the segmentation mask applied later in
    render_service.blend_texture.

    Using BORDER_WRAP means the texture tile repeats seamlessly beyond the
    quad corners, so every pixel in the room image receives a valid texture
    sample with the correct perspective, and the mask decides which pixels
    are actually visible.

    Parameters
    ----------
    texture_canvas : RGB uint8  (pre-tiled canvas, same aspect as output)
    quad_points    : 4 points [TL, TR, BR, BL] on the floor plane
    output_size    : (width, height) of the room image

    Returns
    -------
    warped : RGB uint8, full image size, texture covers every pixel
    """
    if len(quad_points) < 4:
        raise ValueError(f"Perspective warp requires 4 quad points; got {len(quad_points)}")

    out_w, out_h = output_size
    tc_h, tc_w = texture_canvas.shape[:2]

    # Map texture corners → quad corners to establish the floor-plane homography
    src_pts = np.array(
        [[0, 0], [tc_w, 0], [tc_w, tc_h], [0, tc_h]],
        dtype=np.float32,
    )
    dst_pts = np.array(quad_points[:4], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed — degenerate quad points.")

    # BORDER_WRAP: texture tiles seamlessly outside the quad region
    # This gives every pixel a valid, perspective-correct texture sample.
    # The segmentation mask (applied in render_service) is the actual clip boundary.
    warped = cv2.warpPerspective(
        texture_canvas,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    log.debug("Perspective warp done: output %dx%d", out_w, out_h)
    return warped.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_warped_texture(
    texture_image: np.ndarray,
    quad_points: Quad,
    room_size: Tuple[int, int],
    tile_scale: float = 1.0,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    """
    Tile → rotate → perspective-warp the texture ready for mask-based compositing.

    Parameters
    ----------
    texture_image    : RGB uint8 texture tile (any size)
    quad_points      : 4-point floor convex-hull quad (homography source only)
    room_size        : (width, height) of the room image
    tile_scale       : tile size multiplier (0.1–10.0)
    rotation_degrees : counter-clockwise rotation applied before warping

    Returns
    -------
    warped : RGB uint8, same size as the room image
             Every pixel has a valid perspective-correct texture value.
             The segmentation mask in render_service clips what is visible.
    """
    out_w, out_h = room_size

    tiled = create_repeated_texture(texture_image, out_w, out_h, scale=tile_scale)

    if rotation_degrees != 0.0:
        tiled = rotate_texture(tiled, rotation_degrees)
        # Re-tile to fill any canvas gaps introduced by the rotation expand
        tiled = create_repeated_texture(tiled, out_w, out_h, scale=1.0)

    warped = warp_texture_perspective(tiled, quad_points, output_size=(out_w, out_h))
    return warped
