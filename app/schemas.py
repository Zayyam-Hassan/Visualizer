"""Pydantic request / response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

Point = List[int]  # [x, y]
Polygon = List[Point]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


# ---------------------------------------------------------------------------
# /api/v1/segment
# ---------------------------------------------------------------------------

class SegmentResponse(BaseModel):
    success: bool = True
    target_surface: str
    mask_image_base64: str = Field(description="data:image/png;base64,... encoded mask")
    polygon: Polygon
    confidence: Optional[float] = None
    width: int
    height: int


# ---------------------------------------------------------------------------
# /api/v1/render  (base64 response)
# ---------------------------------------------------------------------------

class RenderResponse(BaseModel):
    success: bool = True
    target_surface: str
    rendered_image_base64: str = Field(
        description="data:image/png;base64,... encoded final image"
    )
    mask_image_base64: str = Field(description="data:image/png;base64,... encoded mask")
    polygon: Polygon
    width: int
    height: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# /api/v1/render/file  (saved-file response)
# ---------------------------------------------------------------------------

class RenderFileResponse(RenderResponse):
    file_path: str
    file_url: str


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = "ok"
    app_name: str
    app_env: str
    model_loaded: bool
    model_name: str
    device: str
    timestamp: str


# ---------------------------------------------------------------------------
# /  (root)
# ---------------------------------------------------------------------------

class RootResponse(BaseModel):
    service: str
    status: str = "running"
    docs: str = "/docs"
