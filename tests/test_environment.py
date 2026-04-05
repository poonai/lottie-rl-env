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

            # --- validation rejects bad inputs ---
            for bad_json in sample_invalid_jsons:
                assert env._validate_lottie(bad_json) is False
            assert env._validate_lottie(bouncing_ball_json) is True

            # --- reset picks a task and returns frame URLs ---
            obs = env.reset()
            task = env._current_task
            assert task != ""
            assert obs.start_frame == f"/frames/{task}/frame_start"
            assert obs.middle_frame == f"/frames/{task}/frame_middle"
            assert obs.end_frame == f"/frames/{task}/frame_end"
            assert env.state.step_count == 0
            ep1 = env.state.episode_id

            # --- extract frames from valid Lottie JSON ---
            frames = env._extract_frames(bouncing_ball_json)
            assert frames is not None
            assert len(frames) == 3
            for f in frames:
                assert isinstance(f, Image.Image)

            # --- identical frames score near 1.0 ---
            score = env._compare_frames(frames)
            assert score >= 0.95

            # --- garbage extraction returns empty ---
            assert not env._extract_frames("garbage")

            # --- step with valid JSON: high reward + submitted frame URLs + files on disk ---
            obs = env.step(LottieAction(lottie_json=bouncing_ball_json))
            assert obs.reward >= 0.95
            assert env.state.step_count == 1
            assert f"/submissions/{ep1}/step_1/" in obs.submitted_start_frame
            assert "/frame_start" in obs.submitted_start_frame
            assert "/frame_middle" in obs.submitted_middle_frame
            assert "/frame_end" in obs.submitted_end_frame
            step_dir = tmp_path / ep1 / "step_1"
            assert (step_dir / "frame_start.png").exists()
            assert (step_dir / "frame_middle.png").exists()
            assert (step_dir / "frame_end.png").exists()

            # --- step with invalid JSON: negative reward, no submitted URLs ---
            obs = env.step(LottieAction(lottie_json="bad"))
            assert obs.reward == -1.0
            assert env.state.step_count == 2
            assert obs.submitted_start_frame == ""
            assert obs.submitted_middle_frame == ""
            assert obs.submitted_end_frame == ""

            # --- second reset generates new episode_id and resets step count ---
            obs = env.reset()
            assert env.state.episode_id != ep1
            assert env.state.step_count == 0
            assert env._current_task != ""

            # --- resized frames still compare (with lower score) ---
            resized = [f.resize((10, 10)) for f in frames]
            resized_score = env._compare_frames(resized)
            assert 0.0 <= resized_score <= 1.0
        finally:
            mod.SUBMISSIONS_DIR = original_sub
