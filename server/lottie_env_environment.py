"""
Lottie Env Environment Implementation.

On reset(), picks a random task folder from lottie_frames/ and returns
frame URLs for the start, middle, and end frames.
"""

import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import LottieAction, LottieObservation
except ImportError:
    from models import LottieAction, LottieObservation

FRAMES_DIR = Path("lottie_frames")
FRAME_NAMES = ["frame_start.png", "frame_middle.png", "frame_end.png"]


class LottieEnvironment(Environment):
    """
    Environment that serves Lottie animation frames.

    On reset(), a random task folder is selected from lottie_frames/
    and the observation contains URLs to the start, middle, and end frames.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: str = ""

    def reset(self) -> LottieObservation:
        if not FRAMES_DIR.exists():
            raise RuntimeError(f"Frames directory not found: {FRAMES_DIR}")

        task_folders = [
            p.name
            for p in FRAMES_DIR.iterdir()
            if p.is_dir() and all((p / f).exists() for f in FRAME_NAMES)
        ]

        if not task_folders:
            raise RuntimeError(f"No valid task folders found in {FRAMES_DIR}")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = random.choice(task_folders)

        return LottieObservation(
            start_frame=f"/frames/{self._current_task}/frame_start",
            middle_frame=f"/frames/{self._current_task}/frame_middle",
            end_frame=f"/frames/{self._current_task}/frame_end",
            done=False,
            reward=0.0,
        )

    def step(self, action: LottieAction) -> LottieObservation:  # type: ignore[override]
        self._state.step_count += 1

        return LottieObservation(
            start_frame=f"/frames/{self._current_task}/frame_start",
            middle_frame=f"/frames/{self._current_task}/frame_middle",
            end_frame=f"/frames/{self._current_task}/frame_end",
            done=False,
            reward=0.0,
            metadata={"step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        return self._state
