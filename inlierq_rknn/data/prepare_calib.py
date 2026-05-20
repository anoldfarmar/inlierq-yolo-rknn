import argparse
import random
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(data_yaml):
    try:
        import yaml

        with open(data_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        data = {}
        with open(data_yaml, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip().strip("'\"")
        return data


def resolve_split_dir(data_yaml, split):
    data_yaml = Path(data_yaml).resolve()
    data = load_data_yaml(data_yaml)
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_value = data.get(split)
    if not split_value:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")

    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if image_dir.exists():
        return image_dir.resolve()

    fallback = root / "images" / split
    if fallback.exists():
        print(f"Warning: {image_dir} does not exist, using {fallback}")
        return fallback.resolve()

    return image_dir.resolve()


def collect_images(image_dir):
    image_dir = Path(image_dir).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMG_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        default=None,
        help="image directory for calibration; overrides --data",
    )
    parser.add_argument("--data", default="/home/paipaiqi01/workspace/data/spacer-merged_processed/data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="outputs/calib/spacer_train64.txt")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve() if args.image_dir else resolve_split_dir(args.data, args.split)
    images = collect_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No calibration images found under: {image_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    selected = images[: min(args.count, len(images))]

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")

    print(f"Image dir: {image_dir}")
    print(f"Found images: {len(images)}")
    print(f"Selected images: {len(selected)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
