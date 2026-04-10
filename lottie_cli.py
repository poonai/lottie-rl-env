from pathlib import Path

import fire
import imagehash
import numpy as np
from PIL import Image
from rlottie_python import LottieAnimation


def get_img_hash(images: list[Image.Image]) -> list[imagehash.ImageHash]:
    hashes = []
    for img in images:
        h = imagehash.whash(img, mode="db4")
        hashes.append(h)
    return hashes


def get_hamming_distance(hashes: list[imagehash.ImageHash]) -> list[int]:
    reference = hashes[0]
    distances = []
    for img_hash in hashes:
        distances.append(reference - img_hash)
    return distances


def find_nearest(array: np.ndarray, value: float) -> int:
    idx = (np.abs(array - value)).argmin()
    return idx


def extract_key_frames(hashes: list[imagehash.ImageHash]) -> tuple[int, int, int]:
    distances = np.array(get_hamming_distance(hashes))
    max_index = np.argmax(distances)
    mean = np.mean(distances[: max_index + 1])
    mean_index = find_nearest(distances[: max_index + 1], mean)
    return 0, mean_index, max_index


def extract_key_frames_recursively(hashes: list[imagehash.ImageHash]) -> list[int]:
    start_index, mean_index, max_index = extract_key_frames(hashes=hashes)
    key_frames = list(set([start_index, mean_index, max_index]))
    if len(hashes) - max_index < 3 or mean_index == 0:
        return key_frames

    next_key_frames = extract_key_frames_recursively(hashes=hashes[max_index + 1 :])
    next_key_frames = [frame_index + max_index for frame_index in next_key_frames]
    return key_frames + next_key_frames


def extract_frames(
    lottie_json_path: str, difficulty: str = "easy", all: bool = False
) -> None:
    lottie_path = Path(lottie_json_path)
    if not lottie_path.exists():
        raise FileNotFoundError(f"Lottie file not found: {lottie_path}")

    if difficulty not in ["easy", "medium", "hard"]:
        raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")

    name = lottie_path.stem
    out_dir = Path("server/lottie_frames") / difficulty / name
    out_dir.mkdir(parents=True, exist_ok=True)

    anim = LottieAnimation(path=str(lottie_path))
    total = anim.lottie_animation_get_totalframe()

    images = [anim.render_pillow_frame(frame_num=i) for i in range(total)]
    hashes = get_img_hash(images)
    key_frames = extract_key_frames_recursively(hashes=hashes)
    key_frames = sorted(list(set(key_frames)))

    if all:
        frames = key_frames
        labels = [f"key_frame_{i}" for i in range(len(frames))]
    else:
        frames = key_frames
        if len(key_frames) > 3:
            frames = key_frames[:3]
        frames.append(key_frames[-1])
        labels = [f"key_frame_{i}" for i in range(len(frames))]

    for frame_num, label in zip(frames, labels):
        out_path = out_dir / f"frame_{label}.png"
        anim.save_frame(str(out_path), frame_num=frame_num)
        print(f"Saved {out_path} (frame {frame_num}/{total - 1})")

    anim.lottie_animation_destroy()


if __name__ == "__main__":
    fire.Fire(extract_frames)
