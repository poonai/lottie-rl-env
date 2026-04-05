# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lottie Env Environment Client."""

import base64
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LottieAction, LottieObservation

_FRAME_KEYS = [
    "start_frame",
    "middle_frame",
    "end_frame",
    "submitted_start_frame",
    "submitted_middle_frame",
    "submitted_end_frame",
]


def _decode_frame(data: dict, key: str):
    val = data.get(key, "")
    if isinstance(val, str) and val:
        return base64.b64decode(val)
    return None


class LottieEnv(EnvClient[LottieAction, LottieObservation, State]):
    """
    Client for the Lottie Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with LottieEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.start_frame)
        ...     print(result.observation.middle_frame)
        ...     print(result.observation.end_frame)
    """

    def _step_payload(self, action: LottieAction) -> Dict:
        return {
            "lottie_json": action.lottie_json,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LottieObservation]:
        obs_data = payload.get("observation", {})
        frame_kwargs = {k: _decode_frame(obs_data, k) for k in _FRAME_KEYS}
        observation = LottieObservation(
            **frame_kwargs,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
