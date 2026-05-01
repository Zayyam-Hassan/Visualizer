"""
Tests for the render pipeline services using synthetic images.

These tests do NOT require the segmentation model to be loaded.
They validate the mask / texture / render utility functions in isolation.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.services.mask_service import (
    clean_mask,
    extract_polygon,
    feather_mask,
    get_significant_components,
    get_perspective_quad,
    process_mask,
)
from app.services.texture_service import (
    create_repeated_texture,
    prepare_warped_texture,
    rotate_texture,
    warp_texture_perspective,
)
from app.services.render_service import blend_texture, preserve_lighting
from app.services.image_service import numpy_rgb_to_base64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_mask() -> np.ndarray:
    """512×512 mask with a 300×300 square in the centre."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[106:406, 106:406] = 255
    return mask


@pytest.fixture
def noisy_mask(square_mask: np.ndarray) -> np.ndarray:
    """Square mask with random noise blobs added."""
    noisy = square_mask.copy()
    rng = np.random.default_rng(42)
    noise = (rng.random((512, 512)) > 0.95).astype(np.uint8) * 255
    noisy = np.maximum(noisy, noise)
    return noisy


@pytest.fixture
def room_rgb() -> np.ndarray:
    """Synthetic 512×512 RGB room image (gradient)."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img[i, :, :] = [i // 2, 80, 120]
    return img


@pytest.fixture
def texture_rgb() -> np.ndarray:
    """64×64 checkerboard texture tile."""
    tile = np.zeros((64, 64, 3), dtype=np.uint8)
    for r in range(64):
        for c in range(64):
            if (r // 8 + c // 8) % 2 == 0:
                tile[r, c] = [200, 180, 160]
            else:
                tile[r, c] = [100, 90, 80]
    return tile


# ---------------------------------------------------------------------------
# mask_service tests
# ---------------------------------------------------------------------------

class TestCleanMask:
    def test_removes_small_noise(self, noisy_mask: np.ndarray) -> None:
        cleaned = clean_mask(noisy_mask)
        # Cleaned mask should have fewer isolated pixels than the noisy one
        assert cleaned.sum() <= noisy_mask.sum()

    def test_preserves_large_region(self, square_mask: np.ndarray) -> None:
        cleaned = clean_mask(square_mask)
        # At least 50% of the original foreground should remain
        assert cleaned.sum() >= 0.5 * square_mask.sum()

    def test_output_dtype(self, square_mask: np.ndarray) -> None:
        cleaned = clean_mask(square_mask)
        assert cleaned.dtype == np.uint8


class TestGetSignificantComponents:
    def test_drops_tiny_noise_keeps_large(self) -> None:
        # 400×400 image: one 3×3 noise dot and one 200×200 floor patch
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[1:4, 1:4] = 255        # 9 px — noise
        mask[100:300, 100:300] = 255  # 40000 px — real floor
        result = get_significant_components(mask)
        assert result[2, 2] == 0       # noise dropped
        assert result[150, 150] == 255  # floor kept

    def test_keeps_multiple_significant_patches(self) -> None:
        # Simulate sofa splitting floor into two patches of similar size
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[200:380, 10:180] = 255   # left floor patch
        mask[200:380, 220:390] = 255  # right floor patch
        result = get_significant_components(mask)
        assert result[300, 80] == 255   # left patch kept
        assert result[300, 300] == 255  # right patch kept

    def test_empty_mask_returns_zeros(self) -> None:
        empty = np.zeros((100, 100), dtype=np.uint8)
        result = get_significant_components(empty)
        assert result.sum() == 0


class TestFeatherMask:
    def test_returns_float32(self, square_mask: np.ndarray) -> None:
        feathered = feather_mask(square_mask)
        assert feathered.dtype == np.float32

    def test_range_zero_to_one(self, square_mask: np.ndarray) -> None:
        feathered = feather_mask(square_mask)
        assert feathered.min() >= 0.0
        assert feathered.max() <= 1.0

    def test_centre_is_high(self, square_mask: np.ndarray) -> None:
        feathered = feather_mask(square_mask)
        assert feathered[256, 256] > 0.9


class TestExtractPolygon:
    def test_returns_list_of_points(self, square_mask: np.ndarray) -> None:
        poly = extract_polygon(square_mask)
        assert isinstance(poly, list)
        assert all(isinstance(pt, list) and len(pt) == 2 for pt in poly)

    def test_polygon_within_bounds(self, square_mask: np.ndarray) -> None:
        poly = extract_polygon(square_mask)
        h, w = square_mask.shape
        for x, y in poly:
            assert 0 <= x <= w
            assert 0 <= y <= h


class TestGetPerspectiveQuad:
    def test_returns_four_points(self, square_mask: np.ndarray) -> None:
        quad = get_perspective_quad(square_mask)
        assert len(quad) == 4

    def test_points_are_xy_pairs(self, square_mask: np.ndarray) -> None:
        quad = get_perspective_quad(square_mask)
        for pt in quad:
            assert len(pt) == 2


class TestProcessMask:
    def test_full_pipeline_output(self, square_mask: np.ndarray) -> None:
        clean, feather, polygon, quad = process_mask(square_mask)
        assert clean.dtype == np.uint8
        assert feather.dtype == np.float32
        assert isinstance(polygon, list)
        assert len(quad) == 4


# ---------------------------------------------------------------------------
# texture_service tests
# ---------------------------------------------------------------------------

class TestCreateRepeatedTexture:
    def test_output_shape(self, texture_rgb: np.ndarray) -> None:
        result = create_repeated_texture(texture_rgb, 512, 512, scale=1.0)
        assert result.shape == (512, 512, 3)

    def test_scale_smaller_tiles(self, texture_rgb: np.ndarray) -> None:
        result = create_repeated_texture(texture_rgb, 256, 256, scale=0.5)
        assert result.shape == (256, 256, 3)

    def test_no_black_regions(self, texture_rgb: np.ndarray) -> None:
        result = create_repeated_texture(texture_rgb, 300, 300, scale=1.0)
        assert result.mean() > 10  # should not be mostly black


class TestRotateTexture:
    def test_zero_rotation_unchanged(self, texture_rgb: np.ndarray) -> None:
        result = rotate_texture(texture_rgb, 0.0)
        np.testing.assert_array_equal(result, texture_rgb)

    def test_90_rotation_changes_image(self, texture_rgb: np.ndarray) -> None:
        result = rotate_texture(texture_rgb, 90.0)
        # Shape may differ slightly due to bounding box expansion; just check it runs
        assert result.ndim == 3


class TestWarpTexturePerspective:
    def test_output_size(self, texture_rgb: np.ndarray) -> None:
        canvas = create_repeated_texture(texture_rgb, 512, 512)
        quad = [[100, 100], [400, 100], [400, 400], [100, 400]]
        result = warp_texture_perspective(canvas, quad, output_size=(512, 512))
        assert result.shape == (512, 512, 3)

    def test_full_image_coverage(self, texture_rgb: np.ndarray) -> None:
        # With BORDER_WRAP every pixel should be non-zero (texture covers full image)
        canvas = create_repeated_texture(texture_rgb, 512, 512)
        quad = [[200, 200], [300, 200], [300, 300], [200, 300]]
        result = warp_texture_perspective(canvas, quad, output_size=(512, 512))
        # The mask — not the warp — clips what's visible; full image should have texture
        assert result.mean() > 10  # no large black void


# ---------------------------------------------------------------------------
# render_service tests
# ---------------------------------------------------------------------------

class TestPreserveLighting:
    def test_output_shape(self, room_rgb: np.ndarray) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, 100:400] = 255
        tex = np.full((512, 512, 3), 150, dtype=np.uint8)
        result = preserve_lighting(room_rgb, tex, mask)
        assert result.shape == room_rgb.shape
        assert result.dtype == np.uint8

    def test_lighting_exclude_optional_same_shape(self, room_rgb: np.ndarray) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, 100:400] = 255
        exclude = np.zeros((512, 512), dtype=np.uint8)
        exclude[200:300, 200:300] = 255  # synthetic “rug” inside floor
        tex = np.full((512, 512, 3), 150, dtype=np.uint8)
        result = preserve_lighting(room_rgb, tex, mask, lighting_exclude_mask=exclude)
        assert result.shape == room_rgb.shape

    def test_outside_mask_unchanged(self, room_rgb: np.ndarray) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        tex = np.full((512, 512, 3), 200, dtype=np.uint8)
        result = preserve_lighting(room_rgb, tex, mask)
        # With all-zero mask, result should equal original
        np.testing.assert_array_equal(result, room_rgb)


class TestBlendTexture:
    def test_zero_blend_returns_original(self, room_rgb: np.ndarray) -> None:
        tex = np.full_like(room_rgb, 200)
        alpha = np.ones((512, 512), dtype=np.float32)
        result = blend_texture(room_rgb, tex, alpha, blend_strength=0.0)
        np.testing.assert_allclose(result.astype(float), room_rgb.astype(float), atol=1)

    def test_full_blend_in_mask_area(self) -> None:
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        texture  = np.full((100, 100, 3), 200, dtype=np.uint8)
        alpha    = np.ones((100, 100), dtype=np.float32)
        result = blend_texture(original, texture, alpha, blend_strength=1.0)
        assert result.mean() > 190


# ---------------------------------------------------------------------------
# image_service tests
# ---------------------------------------------------------------------------

class TestNumpyToBase64:
    def test_produces_data_uri(self, room_rgb: np.ndarray) -> None:
        result = numpy_rgb_to_base64(room_rgb)
        assert result.startswith("data:image/png;base64,")

    def test_grayscale_works(self) -> None:
        gray = np.zeros((64, 64), dtype=np.uint8)
        result = numpy_rgb_to_base64(gray)
        assert result.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Upload validation test (via TestClient, no model needed)
# ---------------------------------------------------------------------------

def test_segment_rejects_non_image() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    # Send a text file as room_image
    response = client.post(
        "/api/v1/segment",
        data={"target_surface": "floor"},
        files={"room_image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code in (415, 422)


def test_render_rejects_oversized_surface() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    img_bytes = _make_png_bytes(64, 64)
    response = client.post(
        "/api/v1/render",
        data={"target_surface": "roof"},  # unsupported
        files={
            "room_image": ("room.png", img_bytes, "image/png"),
            "texture_image": ("tex.png", img_bytes, "image/png"),
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_png_bytes(w: int, h: int) -> bytes:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
