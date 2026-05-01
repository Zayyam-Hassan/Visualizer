"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Suppress false-positive warnings for fields that start with "model_"
        protected_namespaces=(),
    )

    # -- App ------------------------------------------------------------------
    app_name: str = "AI Room Visualizer"
    app_env: str = "development"
    debug: bool = False

    # -- Model ----------------------------------------------------------------
    model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    # auto  → use CUDA if available, else CPU
    # cpu   → force CPU
    # cuda  → force CUDA (raises if not available)
    device: str = "auto"
    use_sam2_refine_default: bool = False
    sam2_model_id: str = "facebook/sam2-hiera-large"
    sam2_config_path: str = ""
    sam2_checkpoint_path: str = ""

    # -- Upload ---------------------------------------------------------------
    max_upload_mb: int = 20
    max_inference_resolution: int = 1024  # longest side; 0 = no limit

    # -- Storage --------------------------------------------------------------
    output_dir: Path = Path("outputs")

    # -- CORS -----------------------------------------------------------------
    # Stored as a plain string so pydantic-settings does NOT try to JSON-decode
    # it from .env (List[str] fields trigger json.loads internally).
    # Accepts either a comma-separated list or a JSON array.
    # Use the `cors_origins` property to get the parsed list.
    cors_origins_raw: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:5173"
    )

    # -------------------------------------------------------------------------

    @property
    def cors_origins(self) -> List[str]:
        """Parse cors_origins_raw into a list of origins."""
        v = self.cors_origins_raw.strip()
        if v.startswith("["):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(o).strip() for o in parsed if str(o).strip()]
            except json.JSONDecodeError:
                pass
        return [origin.strip() for origin in v.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_output_dir(self) -> Path:
        """Absolute path to the output directory."""
        p = self.output_dir
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
