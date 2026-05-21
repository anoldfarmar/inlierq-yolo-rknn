import os
import cv2
import yaml
import time
import argparse
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def load_runtime(rknn_path, target="rk3588", device_id=None):
    try:
        from rknnlite.api import RKNNLite

        rknn = RKNNLite()
        ret = rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        return rknn, "rknn_lite2"

    except ImportError:
        from rknn.api import RKNN

        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        kwargs = {"target": target}
        if device_id:
            kwargs["device_id"] = device_id

        ret = rknn.init_runtime(**kwargs)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        return rknn, "rknn_toolkit2"


def resolve_dataset_path(data_yaml, split):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    root = data.get("path", yaml_dir)

    if not os.path.isabs(root):
        root = os.path.abspath(os.path.join(yaml_dir, root))

    split_path = data.get(split)
    if split_path is None:
        raise KeyError(f"data.yaml has no split: {split}")

    if isinstance(split_path, list):
        paths = split_path
    else:
        paths = [split_path]

    resolved = []
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        resolved.append(os.path.abspath(p))

    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    return resolved, names


def collect_images(paths):
    images = []

    for path in paths:
        if os.path.isfile(path):
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        img = line.strip()
                        if img:
                            images.append(os.path.abspath(img))
            else:
                images.append(path)

        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in files:
                    if os.path.splitext(name.lower())[1] in IMG_EXTS:
                        images.append(os.path.join(root, name))

    return sorted(images)


def image_to_label_path(image_path):
    base, _ = os.path.splitext(image_path)

    parts = image_path.replace("\\", "/").split("/")
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_path = "/".join(parts)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        return label_path

    return base + ".txt"


def load_label(label_path, img_shape):
    h, w = img_shape[:2]

    if not os.path.exists(label_path):
        return np.empty((0, 5), dtype=np.float32)

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) < 5:
                continue

            cls, x, y, bw, bh = map(float, arr[:5])

            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h

            labels.append([cls, x1, y1, x2, y2])

    return np.asarray(labels, dtype=np.float32)


def letterbox(img, new_shape=1280, color=(114, 114, 114)):
    h, w = img.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return img, r, (left, top)


def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    area1 = np.maximum(0, box1[:, 2] - box1[:, 0]) * np.maximum(0, box1[:, 3] - box1[:, 1])
    area2 = np.maximum(0, box2[:, 2] - box2[:, 0]) * np.maximum(0, box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])

    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / np.maximum(union, 1e-6)


def nms(boxes, scores, iou_thres=0.6):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        order = order[np.where(iou <= iou_thres)[0] + 1]

    return keep


def decode_postprocessed_output(pred, conf_thres=0.001, max_det=300):
    pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[1] < 6:
        return np.empty((0, 6), dtype=np.float32)

    boxes = pred[:, :4].astype(np.float32, copy=False)
    scores = pred[:, 4].astype(np.float32, copy=False)
    class_ids = pred[:, 5].astype(np.float32, copy=False)

    valid = np.isfinite(pred[:, :6]).all(axis=1) & (scores >= conf_thres)
    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    dets = np.concatenate([boxes, scores[:, None], class_ids[:, None]], axis=1).astype(np.float32)
    order = np.argsort(-dets[:, 4])
    return dets[order[:max_det]]


def decode_yolo_output(outputs, conf_thres=0.001, iou_thres=0.6, max_det=300, multi_label=True, output_format="auto"):
    if len(outputs) == 2:
        first = np.asarray(outputs[0])
        second = np.asarray(outputs[1])
        if first.ndim == second.ndim == 3:
            if first.shape[1] == 4 and second.shape[1] >= 1 and first.shape[2] == second.shape[2]:
                pred = np.concatenate([first, second], axis=1)
            elif first.shape[2] == 4 and second.shape[2] >= 1 and first.shape[1] == second.shape[1]:
                pred = np.concatenate([first, second], axis=2)
            else:
                pred = np.concatenate([x.reshape(x.shape[0], -1, x.shape[-1]) for x in outputs], axis=1)
        else:
            pred = np.concatenate([x.reshape(x.shape[0], -1, x.shape[-1]) for x in outputs], axis=1)
    elif len(outputs) == 1:
        pred = outputs[0]
    else:
        pred = np.concatenate([x.reshape(x.shape[0], -1, x.shape[-1]) for x in outputs], axis=1)

    pred = np.asarray(pred)

    if pred.ndim == 3:
        pred = pred[0]

    if output_format not in {"auto", "raw", "postprocessed"}:
        raise ValueError(f"unknown output_format: {output_format}")

    if output_format == "postprocessed" or (
        output_format == "auto"
        and pred.ndim == 2
        and pred.shape[1] == 6
        and pred.shape[0] <= max_det
    ):
        return decode_postprocessed_output(pred, conf_thres, max_det)

    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
        pred = pred.T

    if pred.shape[1] < 5:
        return np.empty((0, 6), dtype=np.float32)

    boxes = pred[:, :4]

    if pred.shape[1] == 5:
        scores = pred[:, 4]
        class_ids = np.zeros_like(scores, dtype=np.int32)

        mask = scores >= conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
    else:
        cls_scores = pred[:, 4:]
        if multi_label:
            box_ids, class_ids = np.where(cls_scores >= conf_thres)
            boxes = boxes[box_ids]
            scores = cls_scores[box_ids, class_ids]
            class_ids = class_ids.astype(np.int32)
        else:
            class_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
            mask = scores >= conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = xywh2xyxy(boxes)

    dets = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_thres)

        cls_dets = np.concatenate(
            [
                cls_boxes[keep],
                cls_scores[keep, None],
                np.full((len(keep), 1), cls, dtype=np.float32),
            ],
            axis=1,
        )
        dets.append(cls_dets)

    dets = np.concatenate(dets, axis=0).astype(np.float32)
    order = np.argsort(-dets[:, 4])
    return dets[order[:max_det]]


def describe_outputs(outputs, prefix=""):
    lines = []
    for i, out in enumerate(outputs):
        arr = np.asarray(out)
        if arr.size:
            min_v = float(arr.min())
            max_v = float(arr.max())
            mean_v = float(arr.mean())
        else:
            min_v = max_v = mean_v = 0.0
        lines.append(
            f"{prefix}output[{i}]: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}"
        )

        pred = arr
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
            pred = pred.T
        if pred.ndim == 2 and pred.shape[1] >= 5:
            score_col = pred[:, 4]
            lines.append(
                f"{prefix}output[{i}] score_col: min={float(score_col.min()):.6f}, "
                f"max={float(score_col.max()):.6f}, mean={float(score_col.mean()):.6f}, "
                f"gt_conf={int(np.sum(score_col >= 0.001))}"
            )
        elif pred.ndim == 2 and pred.shape[1] == 1:
            score_col = pred[:, 0]
            lines.append(
                f"{prefix}output[{i}] score_tensor: min={float(score_col.min()):.6f}, "
                f"max={float(score_col.max()):.6f}, mean={float(score_col.mean()):.6f}, "
                f"gt_conf={int(np.sum(score_col >= 0.001))}"
            )
    return lines


def maybe_save_outputs(outputs, save_dir, image_path):
    if not save_dir:
        return

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for i, out in enumerate(outputs):
        out_path = os.path.join(save_dir, f"{stem}.output{i}.npy")
        np.save(out_path, np.asarray(out))


def scale_boxes(boxes, ratio, pad, orig_shape):
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

    return boxes


def process_batch(detections, labels):
    correct = np.zeros((len(detections), len(IOU_THRESHOLDS)), dtype=bool)

    if len(labels) == 0 or len(detections) == 0:
        return correct

    gt_cls = labels[:, 0].astype(np.int32)
    gt_boxes = labels[:, 1:5]

    pred_boxes = detections[:, :4]
    pred_cls = detections[:, 5].astype(np.int32)

    ious = box_iou(gt_boxes, pred_boxes)
    correct_class = gt_cls[:, None] == pred_cls[None, :]
    ious = ious * correct_class

    for t_idx, iou_thres in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(ious >= iou_thres)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                match_ious = ious[matches[:, 0], matches[:, 1]]
                matches = matches[match_ious.argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), t_idx] = True

    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return ap


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_metrics(stats, num_classes):
    correct, conf, pred_cls, target_cls = stats

    if len(conf) == 0:
        return 0.0, 0.0, 0.0, 0.0

    order = np.argsort(-conf)
    correct = correct[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    unique_classes, nt = np.unique(target_cls.astype(np.int32), return_counts=True)
    if len(unique_classes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    x = np.linspace(0, 1, 1000)
    ap = np.zeros((len(unique_classes), correct.shape[1]))
    p_curve = np.zeros((len(unique_classes), 1000))
    r_curve = np.zeros((len(unique_classes), 1000))

    for ci, cls in enumerate(unique_classes):
        cls_pred = pred_cls == cls
        cls_target_count = nt[ci]

        if np.sum(cls_pred) == 0:
            continue

        cls_correct = correct[cls_pred]
        cls_conf = conf[cls_pred]

        tpc = cls_correct.cumsum(0)
        fpc = (1 - cls_correct).cumsum(0)

        recall = tpc / (cls_target_count + 1e-16)
        r_curve[ci] = np.interp(-x, -cls_conf, recall[:, 0], left=0)

        precision = tpc / (tpc + fpc + 1e-16)
        p_curve[ci] = np.interp(-x, -cls_conf, precision[:, 0], left=1)

        for j in range(correct.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    best_i = smooth(f1_curve.mean(0), 0.1).argmax()

    mp = float(p_curve[:, best_i].mean())
    mr = float(r_curve[:, best_i].mean())
    map50 = float(ap[:, 0].mean())
    map5095 = float(ap.mean())

    return mp, mr, map50, map5095


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", required=True, help="rknn model path")
    parser.add_argument("--data", required=True, help="data.yaml path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--output-format",
        default="auto",
        choices=["auto", "raw", "postprocessed"],
        help="raw: [1,5,8400] or split boxes/scores; postprocessed: [1,300,6] xyxy+score+class",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--debug-outputs", type=int, default=0, help="print raw output stats for first N images")
    parser.add_argument("--dump-output-dir", default=None, help="save raw outputs as .npy for debugging")
    args = parser.parse_args()

    image_roots, names = resolve_dataset_path(args.data, args.split)
    images = collect_images(image_roots)

    if not images:
        raise FileNotFoundError(f"no images found from data.yaml split={args.split}")

    num_classes = len(names) if names else 1

    rknn, backend = load_runtime(args.rknn, args.target, args.device_id)

    first = cv2.imread(images[0])
    warm_img, _, _ = letterbox(first, args.imgsz)
    warm_img = cv2.cvtColor(warm_img, cv2.COLOR_BGR2RGB)
    warm_img = np.expand_dims(warm_img, axis=0)

    for _ in range(args.warmup):
        rknn.inference(inputs=[warm_img])

    preprocess_times = []
    inference_times = []
    postprocess_times = []

    stats_correct = []
    stats_conf = []
    stats_pred_cls = []
    stats_target_cls = []

    for idx, image_path in enumerate(images, 1):
        img0 = cv2.imread(image_path)
        if img0 is None:
            print(f"skip unreadable image: {image_path}")
            continue

        label_path = image_to_label_path(image_path)
        labels = load_label(label_path, img0.shape)

        t0 = time.perf_counter()
        img, ratio, pad = letterbox(img0, args.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        t1 = time.perf_counter()

        outputs = rknn.inference(inputs=[img])
        t2 = time.perf_counter()

        if idx <= args.debug_outputs:
            for line in describe_outputs(outputs, prefix=f"[{idx}/{len(images)}] "):
                print(line)
        if idx == 1 and args.dump_output_dir:
            maybe_save_outputs(outputs, args.dump_output_dir, image_path)

        detections = decode_yolo_output(
            outputs,
            args.conf,
            args.iou,
            args.max_det,
            output_format=args.output_format,
        )
        t3 = time.perf_counter()

        if len(detections):
            detections[:, :4] = scale_boxes(detections[:, :4], ratio, pad, img0.shape)

        correct = process_batch(detections, labels)

        preprocess_times.append((t1 - t0) * 1000)
        inference_times.append((t2 - t1) * 1000)
        postprocess_times.append((t3 - t2) * 1000)

        if len(detections):
            stats_correct.append(correct)
            stats_conf.append(detections[:, 4])
            stats_pred_cls.append(detections[:, 5])

        if len(labels):
            stats_target_cls.append(labels[:, 0])

        print(
            f"[{idx}/{len(images)}] "
            f"{os.path.basename(image_path)} "
            f"labels={len(labels)} preds={len(detections)}"
        )

    if stats_correct:
        stats_correct = np.concatenate(stats_correct, axis=0)
        stats_conf = np.concatenate(stats_conf, axis=0)
        stats_pred_cls = np.concatenate(stats_pred_cls, axis=0)
    else:
        stats_correct = np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
        stats_conf = np.zeros((0,), dtype=np.float32)
        stats_pred_cls = np.zeros((0,), dtype=np.float32)

    if stats_target_cls:
        stats_target_cls = np.concatenate(stats_target_cls, axis=0)
    else:
        stats_target_cls = np.zeros((0,), dtype=np.float32)

    mp, mr, map50, map5095 = compute_metrics(
        [stats_correct, stats_conf, stats_pred_cls, stats_target_cls],
        num_classes,
    )

    pre = float(np.mean(preprocess_times))
    infer = float(np.mean(inference_times))
    post = float(np.mean(postprocess_times))
    total = pre + infer + post

    print("\n--- 准度指标 (Accuracy) ---")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map5095:.4f}")
    print(f"Precision: {mp:.4f}")
    print(f"Recall: {mr:.4f}")

    print("\n--- 速度指标 (Speed per image) ---")
    print(f"Backend: {backend}")
    print(f"Target: {args.target}")
    print(f"Images: {len(preprocess_times)}")
    print(f"Pre-process: {pre:.2f} ms")
    print(f"Inference: {infer:.2f} ms")
    print(f"Post-process: {post:.2f} ms")
    print(f"Total: {total:.2f} ms")
    print(f"FPS: {1000.0 / total:.2f}")

    rknn.release()


if __name__ == "__main__":
    main()
