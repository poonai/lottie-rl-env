import json
from pathlib import Path

import pytest

from lottie_env.server.lottie_env_environment import LottieEnvironment

ROOT = Path(__file__).resolve().parent.parent
LOTTIE_JSON_PATH = ROOT / "bouncing_ball.json"
FRAMES_DIR = ROOT / "server" / "lottie_frames"


@pytest.fixture
def bouncing_ball_json() -> str:
    return LOTTIE_JSON_PATH.read_text()


@pytest.fixture
def env() -> LottieEnvironment:
    return LottieEnvironment()


@pytest.fixture
def env_with_task(env: LottieEnvironment) -> LottieEnvironment:
    env.reset()
    return env


@pytest.fixture
def sample_invalid_jsons() -> list[str]:
    return [
        "",
        "not json at all",
        '{"foo": "bar"}',
        json.dumps({"v": "5.1.1", "fr": 30, "ip": 0, "op": 60}),
        "<xml>not lottie</xml>",
    ]
