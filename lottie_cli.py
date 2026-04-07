from pathlib import Path

import fire
from rlottie_python import LottieAnimation


def extract_frames(lottie_json_path: str, difficulty: str = "easy") -> None:
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

    if total < 3:
        frames = list(range(total))
    else:
        frames = [0, total // 2, total - 1]

    labels = ["start", "middle", "end"]

    for frame_num, label in zip(frames, labels):
        out_path = out_dir / f"frame_{label}.png"
        anim.save_frame(str(out_path), frame_num=frame_num)
        print(f"Saved {out_path} (frame {frame_num}/{total - 1})")

    anim.lottie_animation_destroy()


if __name__ == "__main__":
    fire.Fire(extract_frames)
