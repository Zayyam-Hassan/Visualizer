"""
Mask post-processing service.

Design contract
---------------
• The segmentation mask is the single source of truth for what is "floor".
  It naturally excludes furniture, walls, and anything the model did not label
  as the target surface — including pixels hidden under a sofa or table.

• We must NOT fill furniture-sized holes with morphological operations.
  A kernel large enough to bridge a sofa leg would bleed texture under it.
  Morphological ops here are only for removing pixel-level sensor noise.

• Multiple disconnected floor patches are valid — a sofa in the centre of a
  room creates two separate visible floor regions, both of which should receive
  the texture.  We therefore keep ALL components above a minimum size, not
  just the largest.

• The 4-point perspective quad is extracted from the CONVEX HULL of all floor
  pixels and is used only to estimate the floor-plane homography for texture
  warping.  It is never used as a clipping boundary — the mask does that.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from app.utils.logging_utils import get_logger

log = get_logger(__name__)

Point = List[int]
Polygon = List[Point]


# ---------------------------------------------------------------------------
# Morphological cleanup  (noise only — NOT furniture holes)
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove pixel-level noise without disturbing furniture-shaped holes.

    We intentionally use a small kernel (5×5) so we only close sub-pixel
    sensor noise gaps.  A large kernel would fill in sofa legs, table feet,
    and other furniture footprints, causing the texture to bleed under them.

    Parameters
    ----------
    mask : np.ndarray  shape (H, W) uint8, values 0 / 255

    Returns
    -------
    cleaned : np.ndarray  same shape/dtype
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Open: remove isolated noise dots outside the floor
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Close: fill sub-pixel cracks inside the floor only
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


# ---------------------------------------------------------------------------
# Component selection  (keep all significant patches)
# ---------------------------------------------------------------------------

def get_significant_components(
    mask: np.ndarray,
    min_area_fraction: float = 0.005,
) -> np.ndarray:
    """
    Keep every connected component whose area exceeds *min_area_fraction* of
    the total image area.

    This preserves multiple floor patches created by furniture in the centre
    of the room (e.g., sofa splits the floor into left and right regions).
    Genuine noise blobs are discarded because they are tiny.

    Parameters
    ----------
    mask              : uint8 binary mask
    min_area_fraction : minimum component area as a fraction of H×W
                        (default 0.5 % — drops noise, keeps real patches)

    Returns
    -------
    result : np.ndarray  uint8 mask containing all significant components
    """
    h, w = mask.shape[:2]
    total_pixels = h * w
    min_area = max(100, int(total_pixels * min_area_fraction))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        log.warning("No connected components found in mask.")
        return np.zeros_like(mask)

    result = np.zeros_like(mask)
    kept = 0
    for label in range(1, num_labels):  # 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == label] = 255
            kept += 1
            log.debug("Kept component %d: area=%d px", label, area)
        else:
            log.debug("Dropped component %d: area=%d px (< %d)", label, area, min_area)

    log.debug("Component selection: %d / %d kept", kept, num_labels - 1)
    return result


# ---------------------------------------------------------------------------
# Edge feathering
# ---------------------------------------------------------------------------

def feather_mask(mask: np.ndarray, blur_size: int = 21) -> np.ndarray:
    """
    Gaussian-blur the mask edges to create a soft alpha feather.

    The result is a float32 array in [0, 1].  Values near 0 are outside the
    floor; values near 1 are deep inside.  The gradient at the edge creates
    a smooth, realistic transition between the original floor and the texture.
    """
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    return np.clip(blurred / 255.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Polygon for API response  (full shape, preserves furniture cutouts)
# ---------------------------------------------------------------------------

def extract_polygon(mask: np.ndarray, max_points: int = 64) -> Polygon:
    """
    Extract a simplified outer contour of the floor mask.

    Returns a polygon with ≤ *max_points* vertices.  We use a generous
    default (64) so the polygon faithfully represents complex floor shapes
    including concavities from furniture.

    This polygon is returned in the API response so the frontend can overlay
    it on the room image; it is NOT used for texture warping.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning("No contours found; falling back to bounding box.")
        return _bounding_box_polygon(mask)

    # When multiple components exist, trace the outer boundary of all of them
    # by working on the combined mask contour
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, closed=True)

    epsilon = 0.005 * perimeter  # start fine
    approx = cv2.approxPolyDP(largest, epsilon, closed=True)

    while len(approx) > max_points and epsilon < perimeter:
        epsilon *= 1.4
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)

    polygon: Polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
    log.debug("Extracted polygon: %d vertices.", len(polygon))
    return polygon


# ---------------------------------------------------------------------------
# Perspective quad  (for homography estimation ONLY)
# ---------------------------------------------------------------------------

def get_perspective_quad(mask: np.ndarray) -> Polygon:
    """
    Estimate a 4-point perspective quad from the CONVEX HULL of all floor pixels.

    Purpose
    -------
    This quad is fed to cv2.findHomography so the texture tile is warped with
    the correct floor-plane perspective.  It is NOT used to clip the texture —
    the full segmentation mask handles clipping, so furniture holes and
    irregular floor shapes are always respected.

    Using the convex hull means the quad spans the true extent of the visible
    floor plane, giving a good perspective estimate even when furniture splits
    the mask into disconnected patches.
    """
    # Build convex hull over all non-zero pixels
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        log.warning("Empty mask; cannot estimate perspective quad.")
        return _bounding_box_polygon(mask)

    points = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(points)  # shape (N, 1, 2)

    # Simplify hull to 4 points
    perimeter = cv2.arcLength(hull, closed=True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, closed=True)

    for _ in range(40):
        if len(approx) <= 4:
            break
        epsilon *= 1.25
        approx = cv2.approxPolyDP(hull, epsilon, closed=True)

    if len(approx) >= 4:
        pts = [[int(p[0][0]), int(p[0][1])] for p in approx[:4]]
        return _sort_quad(pts)

    # Ultimate fallback: bounding box of all floor pixels
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return _sort_quad(pts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bounding_box_polygon(mask: np.ndarray) -> Polygon:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return [[0, 0], [w, 0], [w, h], [0, h]]
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _sort_quad(pts: List[List[int]]) -> Polygon:
    """Sort 4 points into (top-left, top-right, bottom-right, bottom-left) order."""
    pts_np = np.array(pts, dtype=np.float32)
    s = pts_np.sum(axis=1)
    d = np.diff(pts_np, axis=1).flatten()
    tl = pts_np[np.argmin(s)]
    br = pts_np[np.argmax(s)]
    tr = pts_np[np.argmin(d)]
    bl = pts_np[np.argmax(d)]
    return [
        [int(tl[0]), int(tl[1])],
        [int(tr[0]), int(tr[1])],
        [int(br[0]), int(br[1])],
        [int(bl[0]), int(bl[1])],
    ]


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def process_mask(
    raw_mask: np.ndarray,
    feather_size: int = 21,
    fill_enclosed_holes: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Polygon, Polygon]:
    """
    Full mask post-processing pipeline.

    Returns
    -------
    clean   : uint8 binary mask  (all significant floor patches)
    feather : float32 [0,1]      soft alpha for blending
    polygon : Polygon            simplified floor outline for API / UI overlay
    quad    : Polygon            4-point convex hull quad for perspective homography only
    """
    clean = clean_mask(raw_mask)
    clean = get_significant_components(clean)
    if fill_enclosed_holes:
        clean = _fill_enclosed_holes(clean)
    soft = feather_mask(clean, blur_size=feather_size)
    polygon = extract_polygon(clean)
    quad = get_perspective_quad(clean)
    return clean, soft, polygon, quad


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill interior holes that are fully enclosed by the floor mask.

    This allows swapping floor-like regions under objects (e.g., table gaps)
    while keeping external background untouched.
    """
    if not mask.any():
        return mask

    # Flood-fill background connected to image borders, then invert to get holes.
    flood = mask.copy()
    h, w = flood.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood) & cv2.bitwise_not(mask)
    return cv2.bitwise_or(mask, holes)
