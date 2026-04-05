import base64
import json
from pathlib import Path

import pytest

from lottie_env.models import LottieAction
from lottie_env.server.lottie_env_environment import LottieEnvironment
from PIL import Image


class TestLifecycle:
    def test_full_lifecycle(
        self, bouncing_ball_json: str, sample_invalid_jsons: list[str], tmp_path: Path
    ):
        import lottie_env.server.lottie_env_environment as mod

        original_sub = mod.SUBMISSIONS_DIR
        mod.SUBMISSIONS_DIR = tmp_path
        try:
            env = LottieEnvironment()

            for bad_json in sample_invalid_jsons:
                assert env._validate_lottie(bad_json) is False
            assert env._validate_lottie(bouncing_ball_json) is True

            # --- reset returns PIL Images ---
            obs = env.reset()
            task = env._current_task
            assert task != ""
            assert isinstance(obs.start_frame, Image.Image)
            assert isinstance(obs.middle_frame, Image.Image)
            assert isinstance(obs.end_frame, Image.Image)
            assert env.state.step_count == 0
            ep1 = env.state.episode_id

            frames = env._extract_frames(bouncing_ball_json)
            assert frames is not None
            assert len(frames) == 3
            for f in frames:
                assert isinstance(f, Image.Image)

            score = env._compare_frames(frames)
            assert score >= 0.95

            assert not env._extract_frames("garbage")

            # --- step with valid JSON: PIL Images + files on disk ---
            obs = env.step(LottieAction(lottie_json=bouncing_ball_json))
            assert obs.reward >= 0.95
            assert env.state.step_count == 1
            assert isinstance(obs.submitted_start_frame, Image.Image)
            assert isinstance(obs.submitted_middle_frame, Image.Image)
            assert isinstance(obs.submitted_end_frame, Image.Image)
            step_dir = tmp_path / ep1 / "step_1"
            assert (step_dir / "frame_start.png").exists()
            assert (step_dir / "frame_middle.png").exists()
            assert (step_dir / "frame_end.png").exists()

            # --- step with invalid JSON: None frames ---
            obs = env.step(LottieAction(lottie_json="bad"))
            assert obs.reward == -1.0
            assert env.state.step_count == 2
            assert obs.submitted_start_frame is None
            assert obs.submitted_middle_frame is None
            assert obs.submitted_end_frame is None

            obs = env.reset()
            assert env.state.episode_id != ep1
            assert env.state.step_count == 0
            assert env._current_task != ""

            resized = [f.resize((10, 10)) for f in frames]
            resized_score = env._compare_frames(resized)
            assert 0.0 <= resized_score <= 1.0

            # --- model_dump_json serializes Images to base64 strings ---
            parsed = json.loads(obs.model_dump_json())
            raw = base64.b64decode(parsed["start_frame"])
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        finally:
            mod.SUBMISSIONS_DIR = original_sub
