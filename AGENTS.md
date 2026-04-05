# AGENTS.md

## Commands

- **Install/sync deps:** `uv sync`
- **Run server:** `uv run uvicorn server.app:app --reload --port 8000`
- **Run server (entry point):** `uv run --project . server`
- **Extract frames from Lottie JSON:** `uv run python lottie_cli.py --lottie_json_path <file.json>`
- **Tests:** No tests exist yet. `pytest` is configured as a dev dependency (`uv run pytest`).
- **Lint/typecheck:** No linter or typecheck config found.

## Package structure

The repo root **is** the `lottie_env` Python package. This is configured via `pyproject.toml` `[tool.setuptools] package-dir`:

```
lottie_env  →  ./            (root)
lottie_env.server  →  ./server/
```

So `from lottie_env.models import ...` imports from `./models.py`. Imports within the package use relative imports (`from ..models import ...`) with a fallback to absolute imports for direct-script execution.

## Key conventions

- **Package manager:** `uv` only. Never use `pip install` directly.
- **openenv-core framework:** Do NOT build the FastAPI app manually. Call `create_app(EnvironmentClass, ActionClass, ObservationClass)` and add custom routes to the returned app.
- **Observation model** uses `extra="forbid"` — every field must be an explicit Pydantic field. No arbitrary keys.
- **State model** uses `extra="allow"` — arbitrary fields can be added.
- **Frame URLs** follow the pattern `/frames/{task_folder_name}/frame_{start|middle|end}` — no episode ID.
- **Frame endpoint** (`/frames/{task_name}/{frame_name}`) validates only the frame name against the set `{frame_start, frame_middle, frame_end}`. It does NOT validate episode IDs.

## rlottie-python

Import as `from rlottie_python import LottieAnimation` (not `import rlottie`). Key methods:
- `LottieAnimation(path=...)` — load from file
- `lottie_animation_get_totalframe()` — total frame count
- `save_frame(path, frame_num=)` — save a frame as PNG
- `render_pillow_frame(frame_num=)` — render to PIL Image

## lottie_frames/ directory

Each subdirectory is one task (e.g., `bouncing_ball/`). A task folder must contain all three files: `frame_start.png`, `frame_middle.png`, `frame_end.png`. On `reset()`, the environment picks a random valid task folder.

`.gitignore` excludes `*.png`, so frames are not committed. Regenerate with `lottie_cli.py`.

## Known issue

`client.py:45` references `action.message` but `LottieAction` now has field `lottie_json` (not `message`). This will cause an `AttributeError` on `step()`.

## Validation

`step()` validates `action.lottie_json` against the Lottie JSON Schema (loaded via `lottie-specs`). Invalid or non-Lottie JSON gets `reward=-1.0`. Empty string skips validation (`reward=0.0`). Positive reward for valid JSON is TBD.
