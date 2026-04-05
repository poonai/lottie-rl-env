# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Lottie Env Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class LottieAction(Action):
    """Action for the Lottie Env environment."""

    lottie_json: str = Field(default="", description="lottie json")


class LottieObservation(Observation):
    """Observation from the Lottie Env environment - frame URLs."""

    start_frame: str = Field(default="", description="URL to start frame")
    middle_frame: str = Field(default="", description="URL to middle frame")
    end_frame: str = Field(default="", description="URL to end frame")
    submitted_start_frame: str = Field(
        default="", description="URL to submitted start frame"
    )
    submitted_middle_frame: str = Field(
        default="", description="URL to submitted middle frame"
    )
    submitted_end_frame: str = Field(
        default="", description="URL to submitted end frame"
    )
