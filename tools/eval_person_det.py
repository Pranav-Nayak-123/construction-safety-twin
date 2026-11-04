"""Person detection on held-out images: AP50, plus recall and precision at an operating point.

Any Ultralytics checkpoint works: the stage-1 open-vocabulary model (its `person` class)
or a single-class fine-tune. Every model is scored by the same matcher on the same images.

    uv run --no-sync python tools/eval_person_det.py data/working/person_det \\
        artifacts/models/pretrained/yoloe-26l-construction.pt runs/person/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (
        boxes[:, 3] - boxes[:, 1]
    )
    return inter / (union - inter + 1e-9)


def truth(label: Path, width: int, height: int) -> np.ndarray:
    rows = (
        [r.split() for r in label.read_text().splitlines() if r.strip()] if label.is_file() else []
    )
    out = []
    for _, cx, cy, w, h in ((r[0], *map(float, r[1:5])) for r in rows):
        out.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
    return np.array(out).reshape(-1, 4)


def score(model_path: Path, data: Path, imgsz: int, operating_conf: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    person = [k for k, v in model.names.items() if str(v).lower() == "person"]
    detections, total_truth = [], 0
    for image_path in sorted((data / "images" / "test").iterdir()):
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        gt = truth(data / "labels" / "test" / f"{image_path.stem}.txt", w, h)
        total_truth += len(gt)
        result = model.predict(image, imgsz=imgsz, conf=0.001, classes=person, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        used = np.zeros(len(gt), bool)
        for i in np.argsort(-confs):
            hit = False
            if len(gt):
                overlap = iou(boxes[i], gt)
                overlap[used] = 0
                j = int(overlap.argmax())
                if overlap[j] >= 0.5:
                    used[j] = hit = True
            detections.append((float(confs[i]), hit))
    detections.sort(key=lambda d: -d[0])
    hits = np.cumsum([d[1] for d in detections])
    precision = hits / np.arange(1, len(detections) + 1)
    recall = hits / max(total_truth, 1)
    # all-point interpolated AP50
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * envelope))
    at = [d for d in detections if d[0] >= operating_conf]
    tp = sum(d[1] for d in at)
    return {
        "model": model_path.name,
        "imgsz": imgsz,
        "people": total_truth,
        "ap50": round(ap, 3),
        f"recall@{operating_conf}": round(tp / max(total_truth, 1), 3),
        f"precision@{operating_conf}": round(tp / max(len(at), 1), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20, help="the pipeline's operating point")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = [score(m, args.data, args.imgsz, args.conf) for m in args.models]
    for r in results:
        print(json.dumps(r))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
