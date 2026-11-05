"""Find where a SARD excerpt sits in its frame-level annotation, for ground-truth evaluation.

SARD's `N.txt` labels every frame of the full recording (up to ~260k frames), while
`N.mp4` is a short excerpt of it. The excerpt's offset is found by sliding our own
detected person boxes (a finished run's pass-1 store) along the annotation: both are
binned into a coarse grid of box centres per frame and cross-correlated over every
offset at once (FFT), then the best candidates are confirmed by box IoU.

SARD is evaluation-only (data/README.md): the aligned labels are never trained on.

    uv run --no-sync python tools/align_sard.py 9 --store <cache>/runs/sard9-final/pass1.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = (16, 9)
CANVAS = (1920.0, 1080.0)  # annotation and pass-1 reference space


def load_labels(path: Path) -> dict[int, list[tuple[str, float, float, float, float]]]:
    """frame -> [(class, x1, y1, x2, y2)]; class keeps the first word ('person', 'helmet')."""
    frames: dict[int, list] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        frame = int(parts[0])
        x1, y1, x2, y2 = map(float, parts[-4:])
        frames[frame].append((parts[2], x1, y1, x2, y2))
    return frames


def load_detections(store: Path) -> dict[int, np.ndarray]:
    """source frame -> (N, 4) person boxes in the 1920x1080 reference space."""
    db = sqlite3.connect(store)
    frames = dict(db.execute("select frame_index, source_index from frames"))
    out: dict[int, list] = defaultdict(list)
    for frame_index, record in db.execute(
        "select frame_index, record_json from tracks where class_name = 'person'"
    ):
        box = json.loads(record)["box_reference"]
        out[frames[frame_index]].append((box["x1"], box["y1"], box["x2"], box["y2"]))
    return {k: np.array(v) for k, v in out.items()}


def grid_of(boxes: np.ndarray) -> np.ndarray:
    cells = np.zeros(GRID[0] * GRID[1], np.float32)
    if len(boxes):
        cx = (
            ((boxes[:, 0] + boxes[:, 2]) / 2 / CANVAS[0] * GRID[0]).astype(int).clip(0, GRID[0] - 1)
        )
        cy = (boxes[:, 3] / CANVAS[1] * GRID[1]).astype(int).clip(0, GRID[1] - 1)
        np.add.at(cells, cy * GRID[0] + cx, 1.0)
    return cells


def iou_matches(a: np.ndarray, b: np.ndarray, threshold: float = 0.4) -> int:
    if not len(a) or not len(b):
        return 0
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
    return int((iou.max(axis=1) >= threshold).sum())


def align(labels: dict, detections: dict[int, np.ndarray], *, stride: int = 1) -> list[dict]:
    """Best offsets o such that video frame k shows annotation frame o + stride * k."""
    persons = {
        f: np.array([b[1:] for b in boxes if b[0] == "person"]) for f, boxes in labels.items()
    }
    last = max(persons)
    cells = GRID[0] * GRID[1]
    label_grid = np.zeros((last + 1, cells), np.float32)
    for f, boxes in persons.items():
        label_grid[f] = grid_of(boxes)
    span = max(detections) * stride + 1
    det_grid = np.zeros((span, cells), np.float32)
    for k, boxes in detections.items():
        det_grid[k * stride] = grid_of(boxes)
    # score[o] = sum_t sum_c label[o + t, c] * det[t, c], every offset at once via FFT
    size = 1 << int(np.ceil(np.log2(last + 1 + span)))
    score = np.zeros(size, np.float64)
    for c in range(cells):
        if not det_grid[:, c].any():
            continue
        score += np.fft.irfft(
            np.fft.rfft(label_grid[:, c], size) * np.conj(np.fft.rfft(det_grid[:, c], size)), size
        )
    score = score[: last + 1 - span + 1]
    candidates = np.argsort(score)[::-1][:20]
    sample = sorted(detections)[:: max(1, len(detections) // 150)]
    results = []
    for offset in candidates:
        matched = total = 0
        for k in sample:
            truth = persons.get(int(offset) + k * stride, np.zeros((0, 4)))
            matched += iou_matches(detections[k], truth)
            total += len(detections[k])
        results.append(
            {
                "offset": int(offset),
                "stride": stride,
                "grid_score": round(float(score[offset]), 1),
                "iou_match_rate": round(matched / max(total, 1), 3),
            }
        )
    return sorted(results, key=lambda r: -r["iou_match_rate"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", help="SARD clip number, e.g. 9")
    parser.add_argument("--store", type=Path, required=True, help="a finished run's pass1.sqlite")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("evaluation/labels"))
    args = parser.parse_args(argv)
    labels_path = args.labels or Path(f"data/external/sard/{args.clip}.txt")
    labels = load_labels(labels_path)
    detections = load_detections(args.store)
    best = []
    for stride in (1, 2):
        found = align(labels, detections, stride=stride)
        print(f"stride {stride}: top {found[:3]}")
        best += found[:1]
    best.sort(key=lambda r: -r["iou_match_rate"])
    winner = best[0]
    print(
        f"best: offset {winner['offset']} stride {winner['stride']} "
        f"({winner['iou_match_rate']:.0%} of our person boxes match a labelled person)"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"sard-{args.clip}.alignment.json").write_text(
        json.dumps({"clip": args.clip, **winner}, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if winner["iou_match_rate"] >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
