"""
Lottie Env Environment Implementation.

On reset(), picks a random task folder from lottie_frames/ and returns
PIL Image frames (auto-serialized to base64 by Pydantic).
On step(), validates the submitted Lottie JSON schema, extracts frames,
and compares them against reference frames using MSE.
"""

import json
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
from jsonschema import ValidationError
from lottie_specs import load_specs
from rlottie_python import LottieAnimation
from skimage.metrics import mean_squared_error as mse
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from PIL import Image

try:
    from ..models import LottieAction, LottieObservation
except ImportError:
    from models import LottieAction, LottieObservation

FRAMES_DIR = Path(__file__).resolve().parent / "lottie_frames"
SUBMISSIONS_DIR = Path(__file__).resolve().parent / "submissions"
FRAME_NAMES = ["frame_start.png", "frame_middle.png", "frame_end.png"]
FRAME_LABELS = ["frame_start", "frame_middle", "frame_end"]
_LOTTIE_SCHEMA = load_specs()


class LottieEnvironment(Environment):
    """
    Environment that serves Lottie animation frames.

    On reset(), a random task folder is selected from lottie_frames/
    and the observation contains PIL Image frames.
    On step(), the submitted Lottie JSON is validated and its rendered
    frames are compared to the reference frames using MSE.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: str = ""

    def reset(self) -> LottieObservation:
        if not FRAMES_DIR.exists():
            raise RuntimeError(f"Frames directory not found: {FRAMES_DIR}")

        task_folders = []
        for diff_dir in FRAMES_DIR.iterdir():
            if diff_dir.is_dir():
                for task_dir in diff_dir.iterdir():
                    if task_dir.is_dir() and all(
                        (task_dir / f).exists() for f in FRAME_NAMES
                    ):
                        task_folders.append(task_dir.relative_to(FRAMES_DIR))

        if not task_folders:
            raise RuntimeError(f"No valid task folders found in {FRAMES_DIR}")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = str(random.choice(task_folders))

        return self._construct_observation(reward=0.0)

    def _validate_lottie(self, lottie_json: str) -> bool:
        if not lottie_json:
            return False
        try:
            data = json.loads(lottie_json)
        except (json.JSONDecodeError, TypeError):
            return False
        try:
            from jsonschema import validate

            validate(instance=data, schema=_LOTTIE_SCHEMA)
        except ValidationError:
            return False
        return True

    def _extract_frames(self, lottie_json: str) -> list[Image.Image] | None:
        try:
            anim = LottieAnimation(data=lottie_json)
            total = anim.lottie_animation_get_totalframe()
            if total < 3:
                indices = list(range(total))
            else:
                indices = [0, total // 2, total - 1]

            frames = [anim.render_pillow_frame(frame_num=i) for i in indices]
            anim.lottie_animation_destroy()
            return frames
        except Exception:
            return None

    def _compare_frames(self, submitted_frames: list[Image.Image]) -> float:
        task_dir = FRAMES_DIR / self._current_task
        ref_paths = [
            task_dir / "frame_start.png",
            task_dir / "frame_middle.png",
            task_dir / "frame_end.png",
        ]

        if len(submitted_frames) != 3 or len(ref_paths) != 3:
            return 0.0

        mse_values = []
        for sub_img, ref_path in zip(submitted_frames, ref_paths):
            ref_img = Image.open(ref_path).convert("RGB")
            sub_img = sub_img.convert("RGB")

            if sub_img.size != ref_img.size:
                sub_img = sub_img.resize(ref_img.size, Image.LANCZOS)

            ref_arr = np.array(ref_img)
            sub_arr = np.array(sub_img)

            mse_val = mse(ref_arr, sub_arr)
            mse_values.append(mse_val)

        # Transform MSE to reward in [0, 1] range
        # MSE = 0 → reward = 1.0 (perfect match)
        # Maximum MSE for RGB (0-255) is 255^2 = 65025
        max_mse = 65025.0
        mean_mse = float(np.mean(mse_values))
        reward = 1.0 - min(mean_mse / max_mse, 1.0)

        return reward

    def _save_submitted_frames(
        self, frames: list[Image.Image], episode_id: str, step_count: int
    ) -> None:
        out_dir = SUBMISSIONS_DIR / episode_id / f"step_{step_count}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for img, label in zip(frames, FRAME_LABELS):
            img.save(str(out_dir / f"{label}.png"))

    def step(self, action: LottieAction) -> LottieObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = -1.0

        valid = self._validate_lottie(action.lottie_json)
        if not valid:
            return self._construct_observation(reward=reward)

        submitted_frames = self._extract_frames(action.lottie_json)
        if submitted_frames is None:
            return self._construct_observation(reward=-1.0)

        self._save_submitted_frames(
            submitted_frames, self._state.episode_id, self._state.step_count
        )

        reward = self._compare_frames(submitted_frames)

        return self._construct_observation(
            reward=reward, submitted_frames=submitted_frames
        )

    def _construct_observation(
        self,
        reward: float,
        submitted_frames: list[Image.Image] | None = None,
    ) -> LottieObservation:
        task_dir = FRAMES_DIR / self._current_task

        def _load_img(path: Path) -> Image.Image | None:
            return Image.open(path).copy() if path.exists() else None

        return LottieObservation(
            start_frame=_load_img(task_dir / "frame_start.png"),
            middle_frame=_load_img(task_dir / "frame_middle.png"),
            end_frame=_load_img(task_dir / "frame_end.png"),
            submitted_start_frame=submitted_frames[0] if submitted_frames else None,
            submitted_middle_frame=submitted_frames[1] if submitted_frames else None,
            submitted_end_frame=submitted_frames[2] if submitted_frames else None,
            done=False,
            reward=reward,
            metadata={"step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        return self._state
