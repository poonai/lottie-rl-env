# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Lottie Env Environment.
"""

import base64
import io
from typing import Annotated

from openenv.core.env_server.types import Action, Observation
from pydantic import BeforeValidator, Field, PlainSerializer
from PIL import Image


def _to_image(v):
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, bytes):
        if not v:
            return None
        return Image.open(io.BytesIO(v)).copy()
    return v


def _image_to_b64(img):
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


PngImage = Annotated[
    Image.Image | None,
    BeforeValidator(_to_image),
    PlainSerializer(_image_to_b64, return_type=str),
]


class LottieAction(Action):
    """Action for the Lottie Env environment."""

    lottie_json: str = Field(default="", description="lottie json")


class LottieObservation(Observation):
    """Observation from the Lottie Env environment - PIL Image frames serialized as base64."""

    start_frame: PngImage = Field(default=None, description="Start frame")
    middle_frame: PngImage = Field(default=None, description="Middle frame")
    end_frame: PngImage = Field(default=None, description="End frame")
    submitted_start_frame: PngImage = Field(
        default=None, description="Submitted start frame"
    )
    submitted_middle_frame: PngImage = Field(
        default=None, description="Submitted middle frame"
    )
    submitted_end_frame: PngImage = Field(
        default=None, description="Submitted end frame"
    )
