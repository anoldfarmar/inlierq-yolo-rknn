import argparse
import os
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
        current_key = None
        with open(data_yaml, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].rstrip()
                if not line:
                    continue
                if line.startswith("-") and current_key:
                    data.setdefault(current_key, []).append(line[1:].strip().strip("'\""))
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                current_key = key
                if value:
                    data[key] = value.strip("'\"")
                else:
                    data[key] = []
        return data


def resolve_data_paths(data_yaml, splits):
    data_yaml = Path(data_yaml).resolve()
    data = load_data_yaml(data_yaml)

    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    paths = []
    for split in splits:
        split_value = data.get(split)
        if split_value is None:
            continue

        split_items = split_value if isinstance(split_value, list) else [split_value]
        for item in split_items:
            item = Path(item)
            if not item.is_absolute():
                item = root / item
            paths.append(item.resolve())

    return paths


def collect_images(paths):
    images = []
    for path in paths:
        if path.is_file():
            if path.suffix.lower() == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        image_path = line.strip()
                        if image_path:
                            images.append(Path(image_path).resolve())
            elif path.suffix.lower() in IMG_EXTS:
                images.append(path.resolve())
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for name in files:
                    image_path = Path(root) / name
                    if image_path.suffix.lower() in IMG_EXTS:
                        images.append(image_path.resolve())

    return sorted(set(images))


def write_calib_dataset(data_yaml, output_txt, splits, count, seed):
    paths = resolve_data_paths(data_yaml, splits)
    images = collect_images(paths)
    if not images:
        raise FileNotFoundError(f"No calibration images found from {data_yaml}, splits={splits}")

    rng = random.Random(seed)
    rng.shuffle(images)
    if count > 0:
        images = images[: min(count, len(images))]

    output_txt = Path(output_txt).resolve()
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for image_path in images:
            f.write(str(image_path) + "\n")

    return output_txt, len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="input ONNX model path")
    parser.add_argument("--output", required=True, help="output INT8 RKNN model path")
    parser.add_argument("--data", required=True, help="YOLO data.yaml path for calibration images")
    parser.add_argument("--dataset", default=None, help="optional RKNN calibration dataset txt")
    parser.add_argument("--splits", nargs="+", default=["train"], help="calibration splits in data.yaml")
    parser.add_argument("--calib-count", type=int, default=300, help="number of calibration images, <=0 means all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--mean", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--std", nargs=3, type=float, default=[255, 255, 255])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--output-optimize",
        action="store_true",
        help="enable RKNN output optimization; disabled by default to avoid int8 quantization on mixed box/score outputs",
    )
    parser.add_argument("--custom-string", default=None, help="optional RKNN custom_string, disabled by default")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from rknn.api import RKNN

    onnx_path = Path(args.onnx).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_txt = Path(args.dataset).resolve()
    else:
        dataset_txt = output_path.with_suffix(".calib.txt")
        dataset_txt, used_count = write_calib_dataset(
            args.data,
            dataset_txt,
            args.splits,
            args.calib_count,
            args.seed,
        )
        print(f"Calibration dataset: {dataset_txt}")
        print(f"Calibration images: {used_count}")

    rknn = RKNN(verbose=args.verbose)

    print("--> Config RKNN")
    config_kwargs = {
        "mean_values": [args.mean],
        "std_values": [args.std],
        "target_platform": args.target,
        "optimization_level": args.optimization_level,
        "output_optimize": args.output_optimize,
    }
    if args.custom_string:
        config_kwargs["custom_string"] = args.custom_string
    rknn.config(**config_kwargs)

    print("--> Load ONNX")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Load ONNX failed: {ret}")

    print("--> Build INT8 RKNN")
    ret = rknn.build(
        do_quantization=True,
        dataset=str(dataset_txt),
    )
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Build RKNN failed: {ret}")

    print("--> Export RKNN")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Export RKNN failed: {ret}")

    rknn.release()
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
