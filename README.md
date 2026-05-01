# AI Room Visualizer — Backend

A production-ready FastAPI backend that segments room surfaces (floor, wall, ceiling) using a HuggingFace semantic segmentation model and renders material textures onto them with perspective warping and realistic lighting preservation.

---

## Table of Contents

1. [Quick Start (Local)](#quick-start-local)
2. [Docker](#docker)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Frontend Integration](#frontend-integration)
6. [Replacing the Segmentation Model](#replacing-the-segmentation-model)
7. [Architecture](#architecture)
8. [Limitations](#limitations)
9. [Model Licensing Warning](#model-licensing-warning)

---

## Quick Start (Local)

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: install SAM-2 for mask refinement
# Windows PowerShell:
$env:SAM2_BUILD_CUDA="0"; pip install -r requirements-sam2.txt
# Linux/macOS:
# SAM2_BUILD_CUDA=0 pip install -r requirements-sam2.txt

# Copy and edit environment variables
cp .env.example .env

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

> **Note:** On first run the model weights (~180 MB for SegFormer-B2) are downloaded automatically from HuggingFace Hub and cached in `~/.cache/huggingface/`.

---

## Docker

```bash
# Build and start
docker compose up --build

# Rebuild from scratch
docker compose up --build --force-recreate
```

The `outputs/` directory is mounted as a volume so rendered images persist between container restarts.

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `AI Room Visualizer` | Service display name |
| `APP_ENV` | `development` | `development` / `production` |
| `DEBUG` | `true` | Enable debug logging |
| `MODEL_NAME` | `nvidia/segformer-b2-finetuned-ade-512-512` | HuggingFace model ID |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `USE_SAM2_REFINE_DEFAULT` | `false` | Enable SAM-2 refinement by default |
| `SAM2_MODEL_ID` | `facebook/sam2-hiera-large` | HuggingFace SAM-2 model ID used by `SAM2ImagePredictor.from_pretrained` |
| `SAM2_CONFIG_PATH` | empty | Optional local SAM-2 config path (fallback mode) |
| `SAM2_CHECKPOINT_PATH` | empty | Optional local SAM-2 checkpoint path (fallback mode) |
| `MAX_UPLOAD_MB` | `20` | Maximum upload file size in MB |
| `MAX_INFERENCE_RESOLUTION` | `1024` | Longest side for inference (0 = no limit) |
| `OUTPUT_DIR` | `outputs` | Directory for saved render files |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated allowed origins |

---

## API Reference

### `GET /`

Returns service name and status.

```json
{ "service": "AI Room Visualizer", "status": "running", "docs": "/docs" }
```

---

### `GET /health`

```json
{
  "status": "ok",
  "app_name": "AI Room Visualizer",
  "app_env": "development",
  "model_loaded": true,
  "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
  "device": "cpu",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

---

### `POST /api/v1/segment`

Detect and return the segmentation mask for a surface.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `room_image` | file | required | Room photo (JPEG/PNG/WebP) |
| `target_surface` | string | `floor` | `floor` / `wall` / `ceiling` |
| `use_sam2_refine` | bool | `.env default` | Refine SegFormer mask with SAM-2 |

**curl example:**
```bash
curl -X POST http://localhost:8000/api/v1/segment \
  -F "room_image=@/path/to/room.jpg" \
  -F "target_surface=floor" \
  -F "use_sam2_refine=true"
```

**Response:**
```json
{
  "success": true,
  "target_surface": "floor",
  "mask_image_base64": "data:image/png;base64,...",
  "polygon": [[100, 400], [900, 400], [950, 700], [50, 700]],
  "confidence": null,
  "width": 1024,
  "height": 768
}
```

---

### `POST /api/v1/render`

Apply a texture to the detected surface and return the result as base64.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `room_image` | file | required | Room photo |
| `texture_image` | file | required | Material/texture tile |
| `target_surface` | string | `floor` | `floor` / `wall` / `ceiling` |
| `tile_scale` | float | `1.0` | Tile size scale (0.1–10.0) |
| `rotation_degrees` | float | `0` | Texture rotation (−360–360) |
| `blend_strength` | float | `0.85` | Blend opacity (0–1) |
| `preserve_lighting` | bool | `true` | Preserve shadow/highlight |
| `use_sam2_refine` | bool | `.env default` | Refine SegFormer mask with SAM-2 |

**curl example:**
```bash
curl -X POST http://localhost:8000/api/v1/render \
  -F "room_image=@room.jpg" \
  -F "texture_image=@hardwood.jpg" \
  -F "target_surface=floor" \
  -F "tile_scale=1.5" \
  -F "rotation_degrees=0" \
  -F "blend_strength=0.85" \
  -F "preserve_lighting=true" \
  -F "use_sam2_refine=true"
```

**Response:**
```json
{
  "success": true,
  "target_surface": "floor",
  "rendered_image_base64": "data:image/png;base64,...",
  "mask_image_base64": "data:image/png;base64,...",
  "polygon": [[x, y], ...],
  "width": 1024,
  "height": 768,
  "processing_time_ms": 1234.5
}
```

---

### `POST /api/v1/render/file`

Same as `/api/v1/render` but also saves the output to `outputs/` and returns `file_path` and `file_url`.

Additional response fields:
```json
{
  "file_path": "/app/outputs/render_abc123.png",
  "file_url": "/outputs/render_abc123.png"
}
```

Saved files are served statically at `http://localhost:8000/outputs/<filename>`.

---

## Frontend Integration

```javascript
const formData = new FormData();
formData.append("room_image", roomFile);        // File object
formData.append("texture_image", textureFile);  // File object
formData.append("target_surface", "floor");
formData.append("tile_scale", "1.0");
formData.append("rotation_degrees", "0");
formData.append("blend_strength", "0.85");
formData.append("preserve_lighting", "true");

const res = await fetch("http://localhost:8000/api/v1/render", {
  method: "POST",
  body: formData,
});

const data = await res.json();

if (data.success) {
  const img = document.createElement("img");
  img.src = data.rendered_image_base64;  // data:image/png;base64,...
  document.body.appendChild(img);
}
```

---

## Replacing the Segmentation Model

The model is fully configurable via the `MODEL_NAME` environment variable.  Any HuggingFace model that:

1. Is loadable with `AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)`
2. Accepts images via `AutoImageProcessor`
3. Outputs per-pixel class logits `(batch, num_classes, H, W)`

will work out of the box.

After changing the model you **must** update `SURFACE_LABEL_ALIASES` in `app/services/segmentation_service.py` to map `floor`, `wall`, and `ceiling` to the correct class name substrings for the new dataset.

**Example: Cityscapes model**
```python
SURFACE_LABEL_ALIASES = {
    "floor": ["road", "sidewalk", "terrain"],
    "wall":  ["building", "wall", "fence"],
    "ceiling": [],  # not available in Cityscapes
}
```

---

## Architecture

```
POST /api/v1/render
         │
         ▼
  image_service            ← read + validate uploads
         │
         ▼
  segmentation_service     ← HuggingFace model inference → raw binary mask
         │
         ▼
  mask_service             ← clean, largest component, feather, polygon/quad
         │
         ▼
  texture_service          ← tile, rotate, perspective warp into quad
         │
         ▼
  render_service           ← lighting preservation (LAB), alpha blend
         │
         ▼
  image_service            ← encode to base64 / save to disk
```

---

## Limitations

- **Segmentation quality** depends entirely on the model. Complex lighting, reflective floors, or unusual angles may produce poor masks.
- **4-point perspective quad** fitting can fail on non-rectangular surfaces (e.g., curved walls). The system falls back to the full contour polygon in that case.
- **Performance:** On CPU, a single render can take 2–10 seconds depending on image resolution. Use `MAX_INFERENCE_RESOLUTION=768` or lower to speed up inference.
- **CUDA:** Requires a compatible GPU and CUDA toolkit to be installed.

---

## Model Licensing Warning

> **IMPORTANT — Read before commercial deployment**
>
> The default model `nvidia/segformer-b2-finetuned-ade-512-512` is trained on the **ADE20K** dataset and the model weights are released under the **NVIDIA Source Code License**.  This license **may not permit unrestricted commercial use**.
>
> Before deploying this service commercially:
> 1. Review the license at [HuggingFace — nvidia/segformer-b2-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512).
> 2. Contact NVIDIA for commercial licensing if required.
> 3. Consider replacing the model with one that has a permissive license (e.g., MIT, Apache 2.0) for commercial use.
>
> This software is provided as-is with no warranty. The authors are not responsible for any licensing violations by users.

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Tests that exercise the render pipeline use synthetic images and do **not** require the segmentation model to be downloaded.
