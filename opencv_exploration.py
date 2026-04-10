import marimo

__generated_with = "0.22.5"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    lottie_frames_path = Path(
        "/home/saint/lottie_env/server/lottie_frames/hard/robot thinking"
    )
    import marimo as mo

    return Path, lottie_frames_path, mo


@app.cell
def _(lottie_frames_path, mo):
    from PIL import Image
    import imagehash

    # Load all PNG images from the folder
    image_paths = sorted(lottie_frames_path.glob("*.png"))
    images = []
    hashes = []

    for img_path in image_paths:
        img = Image.open(img_path)
        images.append(img)
        h = imagehash.phash(img)
        hashes.append(h)

    # First frame hash as reference
    first_hash = hashes[0]

    # Compute and display hamming distances
    text = ""
    distances = []
    for idx, img_data in enumerate(zip(image_paths, hashes)):
        img_path, img_hash = img_data
        hamming_dist = first_hash - img_hash
        distances.append(hamming_dist)
        mo.image(src=img_path)
        text += f"""<img src="{img_path}" width="300" /> {hamming_dist} index {idx}\n"""
    mo.md(text)
    return Image, idx, image_paths, imagehash, images


@app.cell
def _(Image, Path, image_paths, imagehash, images, mo):
    import numpy as np

    def get_img_hash(img_paths: list[Path]):
        hashes = []

        for img_path in img_paths:
            img = Image.open(img_path)
            images.append(img)
            h = imagehash.whash(img)
            hashes.append(h)

        return hashes

    def get_hamming_distance(hashes: list[imagehash.ImageHash]):
        reference = hashes[0]
        distances = []
        for img_hash in hashes:
            distances.append(reference - img_hash)
        return distances

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def extract_key_frames(hashes: list[imagehash.ImageHash]):
        distances = np.array(get_hamming_distance(hashes))
        max_index = np.argmax(distances)
        print("max index", max_index)
        print("max batch", distances[: max_index + 1])
        mean = np.mean(distances[: max_index + 1])
        mean_index = find_nearest(distances[: max_index + 1], mean)
        return 0, mean_index, max_index

    def extract_key_frames_recursively(hashes: list[imagehash.ImageHash]):
        start_index, mean_index, max_index = extract_key_frames(hashes=hashes)
        key_frames = list(set([start_index, mean_index, max_index]))
        if len(hashes) - max_index < 3 or mean_index == 0:
            return key_frames

        next_key_frames = extract_key_frames_recursively(hashes=hashes[max_index + 1 :])
        next_key_frames = [frame_index + max_index for frame_index in next_key_frames]
        return key_frames + next_key_frames

    hashes2 = get_img_hash(img_paths=image_paths)
    key_frames = extract_key_frames_recursively(hashes=hashes2)
    key_frames = sorted(list(set(key_frames)))
    print("key frames", key_frames)
    text2 = ""

    for key_frame in key_frames:
        text2 += f"""<img src="{image_paths[int(key_frame)]}" width="300" />"""
    mo.md(text2)
    return (key_frames,)


@app.cell
def _(Image, Path, idx, image_paths, imagehash, images, key_frames):
    def get_avg_img_hash(img_paths: list[Path]):
        hashes = []

        for img_path in img_paths:
            img = Image.open(img_path)
            images.append(img)
            h = imagehash.whash(img, mode='db4')
            hashes.append(h)

        return hashes
    hashes3 = get_avg_img_hash(img_paths=image_paths)
    key_frame_hashes = [hashes3[idx] for key_frame in key_frames]

    for i in range(len(key_frame_hashes) - 1):
        diff = key_frame_hashes[i + 1] - key_frame_hashes[i]
        print(f"{i}-{i + 1}: {diff}")
    return


if __name__ == "__main__":
    app.run()
