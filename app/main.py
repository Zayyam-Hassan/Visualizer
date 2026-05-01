"""
FastAPI application entry-point.

Responsibilities
----------------
• Create the FastAPI app instance.
• Register CORS middleware.
• Mount the /outputs static directory.
• Register routers.
• Handle startup (model loading + warm-up) and shutdown events.
• Expose root / and /health endpoints.
"""

from __future__ import annotations

import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.schemas import HealthResponse, RootResponse
from app.services import segmentation_service
from app.utils.logging_utils import get_logger

log = get_logger("ai_visualizer.main")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    log.info("=== %s starting up (env=%s) ===", settings.app_name, settings.app_env)

    # Ensure output directory exists
    settings.resolved_output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation model
    try:
        segmentation_service.load_model()
        segmentation_service.warmup()
    except Exception:
        log.exception(
            "Failed to load segmentation model '%s'. "
            "The service will start, but /segment and /render will return 503.",
            settings.model_name,
        )

    log.info("=== Startup complete ===")
    yield
    log.info("=== %s shutting down ===", settings.app_name)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-powered 2D room visualizer — segment surfaces, apply material "
            "textures with perspective warping, and preserve realistic lighting."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS ----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Static files (serve rendered outputs) --------------------------------
    outputs_dir = settings.resolved_output_dir
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    # --- Routers -------------------------------------------------------------
    app.include_router(api_router)

    # --- Global error handler ------------------------------------------------
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "status_code": 500,
            },
        )

    # --- Root ----------------------------------------------------------------
    @app.get("/", response_model=RootResponse, tags=["meta"])
    async def root() -> RootResponse:
        return RootResponse(service=settings.app_name)

    # --- Health --------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        return HealthResponse(
            app_name=settings.app_name,
            app_env=settings.app_env,
            model_loaded=segmentation_service.is_model_loaded(),
            model_name=settings.model_name,
            device=segmentation_service.get_device_info(),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn)
# ---------------------------------------------------------------------------
app = create_app()
