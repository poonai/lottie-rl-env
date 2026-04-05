# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lottie Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LottieAction, LottieObservation


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
        """
        Convert LottieAction to JSON payload for step message.

        Args:
            action: LottieAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LottieObservation]:
        """
        Parse server response into StepResult[LottieObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with LottieObservation
        """
        obs_data = payload.get("observation", {})
        observation = LottieObservation(
            start_frame=obs_data.get("start_frame", ""),
            middle_frame=obs_data.get("middle_frame", ""),
            end_frame=obs_data.get("end_frame", ""),
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
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
