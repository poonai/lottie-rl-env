# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Lottie Env Environment.

This module creates an HTTP server that exposes the LottieEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import LottieAction, LottieObservation
    from .lottie_env_environment import LottieEnvironment
except (ImportError, ModuleNotFoundError):
    from models import LottieAction, LottieObservation
    from server.lottie_env_environment import LottieEnvironment


_SERVER_DIR = Path(__file__).resolve().parent

app = create_app(
    LottieEnvironment,
    LottieAction,
    LottieObservation,
    env_name="lottie_env",
    max_concurrent_envs=1,
)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index():
    return (_SERVER_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/assets", StaticFiles(directory=str(_SERVER_DIR / "assets")), name="assets")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m lottie_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn lottie_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
