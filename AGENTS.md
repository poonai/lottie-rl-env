# AGENTS.md

## Commands

- **Install/sync deps:** `uv sync`
- **Run server:** `uv run uvicorn server.app:app --reload --port 8000`
- **Run server (entry point):** `uv run --project . server`
- **Extract frames from Lottie JSON:** `uv run python lottie_cli.py --lottie_json_path <file.json>` (run from repo root; output goes to `server/lottie_frames/<stem>/`)
- **Tests:** `uv run pytest` (3 tests in `tests/`)
- **Run a single test:** `uv run pytest tests/test_environment.py::TestLifecycle::test_full_lifecycle`
- **Lint/typecheck:** No linter or typecheck config.

## Package structure

The repo root **is** the `lottie_env` Python package. This is configured via `pyproject.toml` `[tool.setuptools] package-dir`:

```
lottie_env         →  ./            (root)
lottie_env.server  →  ./server/
```

So `from lottie_env.models import ...` imports from `./models.py`. Imports within `server/` use relative imports (`from ..models import ...`) with a fallback to absolute imports for direct-script execution.

## Key conventions

- **Package manager:** `uv` only. Never use `pip install` directly.
- **openenv-core framework:** Do NOT build the FastAPI app manually. Call `create_app(EnvironmentClass, ActionClass, ObservationClass)` and add custom routes to the returned app.
- **Observation model** uses `extra="forbid"` — every field must be an explicit Pydantic field. No arbitrary keys.
- **State model** uses `extra="allow"` — arbitrary fields can be added.

## PngImage custom type (`models.py`)

Observation frame fields use `PngImage` — a custom annotated type:

```python
PngImage = Annotated[
    Image.Image | None,
    BeforeValidator(_to_image),
    PlainSerializer(_image_to_b64, return_type=str),
]
```

- **At runtime** fields are `Image.Image | None`. Construct observations by passing PIL Images directly or `None` for empty.
- **On JSON serialization** (`model_dump_json()` / FastAPI response), Pydantic auto-encodes to base64 PNG strings. `None` becomes `""`.
- `BeforeValidator` also accepts raw `bytes` (e.g. file contents) and converts them to PIL Images.
- Do NOT manually base64-encode images before passing to `LottieObservation`.

## Data directories (inside `server/`)

Both `lottie_frames/` and `submissions/` live inside `server/`. Paths are resolved relative to `__file__`:

```python
FRAMES_DIR = Path(__file__).resolve().parent / "lottie_frames"
SUBMISSIONS_DIR = Path(__file__).resolve().parent / "submissions"
```

- **`server/lottie_frames/`** — each subdirectory is one task (e.g., `bouncing_ball/`). A task folder must contain `frame_start.png`, `frame_middle.png`, `frame_end.png`. On `reset()`, a random valid task folder is picked.
- **`server/submissions/`** — submitted frames saved for debugging, organized as `<episode_id>/step_<n>/<frame>.png`.
- `.gitignore` excludes `*.png` and `server/submissions/*`. Frames are not committed — regenerate with `lottie_cli.py`.

## rlottie-python

Import as `from rlottie_python import LottieAnimation` (not `import rlottie`). Key methods:
- `LottieAnimation(path=...)` — load from file
- `LottieAnimation(data=...)` — load from JSON string
- `lottie_animation_get_totalframe()` — total frame count
- `save_frame(path, frame_num=)` — save a frame as PNG
- `render_pillow_frame(frame_num=)` — render to PIL Image
- **Always call** `lottie_animation_destroy()` when done to free the C object.

## Reward logic

- `step()` validates `action.lottie_json` against the Lottie JSON Schema (loaded via `lottie-specs`).
- Invalid/non-Lottie JSON → `reward=-1.0`.
- Valid JSON → reward is the mean SSIM between submitted and reference frames (0.0–1.0).
- Submitted frames are resized to match reference dimensions if needed.

## Docker

Build from repo root: `docker build -t lottie_env-env:latest -f server/Dockerfile .`

The Dockerfile copies the entire project to `/app/env` and runs `uvicorn server.app:app` from there. Since `FRAMES_DIR` and `SUBMISSIONS_DIR` resolve relative to `__file__` (which becomes `/app/env/server/lottie_env_environment.py`), they correctly point to `/app/env/server/lottie_frames/` and `/app/env/server/submissions/` inside the container.
