"""
Inference Script for Lottie Env
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name for the environment.

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import base64
import io
import json
import os
import re
import textwrap
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

from lottie_env import LottieAction, LottieEnv
from server import LottieEnvironment

load_dotenv(Path(__file__).resolve().parent / ".env")

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "schoolboy/lottie_env-env:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-5.1-codex-max")
TASK_NAME = os.getenv("LOTTIE_TASK", "lottie_animation")
BENCHMARK = os.getenv("LOTTIE_BENCHMARK", "lottie_env")
MAX_STEPS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 8192
SUCCESS_SCORE_THRESHOLD = 0.8

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Lottie animation designer.
    You will be shown 3 reference frames (start, middle, end) of an animation.
    Your task is to generate valid Lottie JSON that reproduces this animation as closely as possible.

    OUTPUT RULES:
    - Output ONLY valid Lottie JSON. No explanations, no markdown fences, no commentary.
    - The JSON must validate against the Lottie schema.
    - Focus on matching shapes, colors, positions, and motion from the reference frames.

    Lottie JSON top-level structure:
    {
      "v": "5.7.4",
      "fr": 30,
      "ip": 0,
      "op": 60,
      "w": <width>,
      "h": <height>,
      "nm": "Animation",
      "ddd": 0,
      "assets": [],
      "layers": [ <layer objects> ]
    }

    Key layer properties:
    - ty: layer type (4=shape, 1=solid, 0=precomp)
    - ks: transform { o: opacity, r: rotation, p: position, a: anchor, s: scale }
    - shapes: shape items (el=ellipse, rc=rect, fl=fill, st=stroke)
    - Static prop:  {"a": 0, "k": <value>}
    - Animated prop: {"a": 1, "k": [<keyframe>, ...]}
    - Keyframe: {"t": <frame>, "s": [<start_val>], "e": [<end_val>], "i":{<ease>}, "o":{<ease>}}
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_summary = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_summary} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def extract_lottie_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


async def get_lottie_json(
    client: AsyncOpenAI,
    ref_frames: List[Optional[Image.Image]],
    step: int,
    last_reward: float,
    submitted_frames: Optional[List[Optional[Image.Image]]],
    history: List[str],
) -> str:
    content: list = []

    text_parts = [
        "Here are the 3 reference frames (start, middle, end) of the animation to reproduce:"
    ]
    for i, label in enumerate(["START", "MIDDLE", "END"]):
        img = ref_frames[i] if i < len(ref_frames) else None
        if img is not None:
            text_parts.append(f"[{label} frame]:")
            content.append({"type": "text", "text": f"[{label} frame]:"})
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}
            )
        else:
            content.append(
                {"type": "text", "text": f"[{label} frame]: (not available)"}
            )

    if step > 1 and submitted_frames:
        content.append(
            {
                "type": "text",
                "text": f"\nYour previous attempt (step {step - 1}) received reward: {last_reward:.2f}/1.00. Here are your submitted frames:",
            }
        )
        for i, label in enumerate(
            ["SUBMITTED START", "SUBMITTED MIDDLE", "SUBMITTED END"]
        ):
            img = submitted_frames[i] if i < len(submitted_frames) else None
            if img is not None:
                content.append({"type": "text", "text": f"[{label}]:"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img)},
                    }
                )

    if history:
        history_block = "\n".join(history[-6:])
        content.append({"type": "text", "text": f"\nAttempt history:\n{history_block}"})

    content.append(
        {
            "type": "text",
            "text": f"\nGenerate the complete Lottie JSON (attempt {step}/{MAX_STEPS}). Output ONLY the JSON:",
        }
    )

    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return extract_lottie_json(raw) if raw else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "{}"


async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await LottieEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        ref_frames = [obs.start_frame, obs.middle_frame, obs.end_frame]
        submitted_frames: Optional[List[Optional[Image.Image]]] = None
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            lottie_json = await get_lottie_json(
                client,
                ref_frames,
                step,
                last_reward,
                submitted_frames,
                history,
            )

            result = await env.step(LottieAction(lottie_json=lottie_json))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            if reward < 0:
                error = "Invalid Lottie JSON or render failure"

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            submitted_frames = [
                obs.submitted_start_frame,
                obs.submitted_middle_frame,
                obs.submitted_end_frame,
            ]

            log_step(
                step=step, action=lottie_json, reward=reward, done=done, error=error
            )

            history.append(f"Step {step}: reward={reward:.2f}")

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = max(score, 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
