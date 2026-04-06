---
title: Lottie Env Environment Server
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - lottie
  - animation
---

# Lottie Env

An RL environment where agents receive 3 reference keyframes and must generate Lottie JSON animations. Reward is based on MSE similarity between reference and rendered frames.

## Task Flow

```
POST /reset → Observation: 3 reference frames → POST /step with Lottie JSON → Reward [0,1] + submitted frames
```

## Environment Design

The environment is designed to test an agent's ability to:

1. **Understand visual patterns** from sparse keyframe observations
2. **Generate valid Lottie JSON** that conforms to the Lottie specification
3. **Reconstruct animations** that match reference behavior

### Reference Keyframes

On each `reset()`, the agent receives 3 PNG frames:
- `frame_start.png` - Beginning of the animation
- `frame_middle.png` - Midpoint of the animation
- `frame_end.png` - End of the animation

These frames are extracted from a ground-truth Lottie animation and serve as the target the agent must approximate.

### Agent Action

The agent submits a complete Lottie JSON document via `LottieAction.lottie_json`. The JSON is validated against the official Lottie specification schema before rendering.

### Reward Calculation

| Condition | Reward |
|-----------|--------|
| Valid Lottie JSON | `1.0 - min(MSE/65025, 1.0)` |
| Invalid JSON / schema fail | `-1.0` |

The reward uses mean squared error (MSE) between reference and submitted frames:
- Submitted frames are rendered at matching timestamps
- MSE is averaged across all 3 frame positions
- Maximum possible reward is 1.0 (perfect reconstruction)
- MSE/65025 normalizes by max possible error (255² per pixel)

## Example: Bouncing Ball

The environment includes a `bouncing_ball` task as a canonical example:

**Reference Animation:**
- A ball moving in a parabolic arc
- Squash-and-stretch on impact
- 60 total frames at 30 FPS

**Reference Keyframes (what the agent sees):**
- Frame 0: Ball at top-left
- Frame 30: Ball at peak height, lowest width
- Frame 59: Ball at bottom-right

**Agent's Goal:**
Generate Lottie JSON that reproduces this motion, including:
- Position keyframes with ease-in/out timing
- Scale keyframes for squash effect
- Proper layer structure

## Quick Start

```python
from lottie_env import LottieAction, LottieEnv

try:
    # Create environment from Docker image
    env = LottieEnv.from_docker_image("lottie_env-env:latest")

    # Reset to get reference frames
    result = env.reset()
    print(f"Task: {result.observation.metadata['task_name']}")
    print(f"Start frame shape: {result.observation.start_frame.size}")

    # Agent processes frames and generates Lottie JSON
    lottie_json = your_model.generate_lottie(
        result.observation.start_frame,
        result.observation.middle_frame,
        result.observation.end_frame
    )

    # Submit action
    result = env.step(LottieAction(lottie_json=lottie_json))
    print(f"Reward: {result.reward}")

    # View submitted frames
    result.observation.submitted_start_frame.show()
    result.observation.submitted_middle_frame.show()
    result.observation.submitted_end_frame.show()

finally:
    env.close()
```

## Schemas

### Action

| Field | Type | Description |
|-------|------|-------------|
| `lottie_json` | `string` | Complete Lottie JSON document |

### Observation

| Field | Type | Description |
|-------|------|-------------|
| `start_frame` | `PngImage` | Reference frame at t=0 |
| `middle_frame` | `PngImage` | Reference frame at t=0.5 |
| `end_frame` | `PngImage` | Reference frame at t=1.0 |
| `submitted_start_frame` | `PngImage` | Agent's rendered frame at t=0 |
| `submitted_middle_frame` | `PngImage` | Agent's rendered frame at t=0.5 |
| `submitted_end_frame` | `PngImage` | Agent's rendered frame at t=1.0 |
| `reward` | `float` | Similarity score in [0, 1] or -1.0 |
| `done` | `bool` | Always false (continuous task) |
| `metadata` | `dict` | Task info: `task_name`, `width`, `height` |

### Image Type

`PngImage` fields are PIL `Image.Image` objects at runtime. When serialized via API, they become base64-encoded PNG strings.

## Building the Docker Image

```bash
# From project root
docker build -t lottie_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory
openenv push

# With options
openenv push --namespace my-org --private
```

Deployed spaces include:
- **Web Interface** at `/web` - Interactive UI with examples
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint

## Development

### Install Dependencies

```bash
uv sync
```

### Run Server Locally

```bash
uv run uvicorn server.app:app --reload --port 8000
```

### Extract Frames from Lottie JSON

To add new tasks, extract keyframes from any Lottie file:

```bash
uv run python lottie_cli.py --lottie_json_path path/to/animation.json
```

Output goes to `server/lottie_frames/<task_name>/` with `frame_start.png`, `frame_middle.png`, `frame_end.png`.

### Run Tests

```bash
uv run pytest
```

### Environment Validation

```bash
uv run openenv validate
```

## Project Structure

```
lottie_env/
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── models.py              # LottieAction, LottieObservation, PngImage type
├── client.py              # LottieEnv client (Docker + HTTP/WebSocket)
├── lottie_cli.py          # Extract frames from Lottie JSON
├── openenv.yaml           # OpenEnv manifest
├── server/
│   ├── app.py             # FastAPI application
│   ├── lottie_env_environment.py  # Core environment logic
│   ├── lottie_frames/     # Task reference frames (git-ignored)
│   ├── submissions/       # Agent submissions (git-ignored)
│   ├── index.html         # Web interface
│   ├── assets/            # Demo animations and frames
│   └── Dockerfile         # Container image definition
└── tests/                 # Unit tests
```

## Technical Details

### Frame Rendering

The environment uses `rlottie-python` to render Lottie animations:
- `LottieAnimation(data=json_string)` - Load from JSON
- `render_pillow_frame(frame_num)` - Render to PIL Image
- Frames are extracted at 0%, 50%, and 100% of total duration
- Images are resized to match reference dimensions for MSE calculation

### Schema Validation

Lottie JSON is validated against the official specification via `lottie-specs` package:
- Ensures structural correctness
- Catches malformed JSON before rendering
- Prevents runtime errors from invalid animations

### Reward Computation

```python
from skimage.metrics import structural_similarity as ssim

# MSE is computed per-channel, then averaged
mse = np.mean((ref_img - sub_img) ** 2)
reward = 1.0 - min(mse / 65025.0, 1.0)  # Normalize by max possible error
```

## Advanced Usage

### Connecting to Existing Server

```python
from lottie_env import LottieEnv

env = LottieEnv(base_url="http://localhost:8000")
result = env.reset()
# ... use normally
env.close()  # Does NOT stop the server
```

### Context Manager

```python
from lottie_env import LottieEnv

with LottieEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    result = env.step(LottieAction(lottie_json=lottie_json))
# Auto-closes connection
```

### WebSocket Sessions

The client uses WebSockets for low-latency, persistent sessions. For concurrent sessions, modify `server/app.py`:

```python
app = create_app(
    LottieEnvironment,
    LottieAction,
    LottieObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

## License

BSD-style license. See LICENSE file for details.
