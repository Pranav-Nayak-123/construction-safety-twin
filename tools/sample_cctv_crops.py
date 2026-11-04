"""Sample worker crops from real CCTV runs for hand-labelled PPE ground truth.

Distinct tracks, spread over the clip, tall enough to judge (>= 60 px), cropped from the
unblurred source video. Writes the crops, numbered contact sheets for labelling, and a
`crops.jsonl` whose `head_state` / `torso_state` are filled in by hand afterwards.
Local only: unblurred faces never leave the workstation.

    uv run --no-sync python tools/sample_cctv_crops.py --out data/working/cctv_ppe
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

import cv2
import numpy as np

STORES = {
    "sard9": ("data/external/sard/9.mp4", "sard9-final"),
    "sard8": ("data/external/sard/8.mp4", "sard8-v3"),
    "sard7": ("data/external/sard/7.mp4", "sard7-v3"),
}
CACHE = Path.home() / "AppData/Local/safety-twin/runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-clip", type=int, default=40)
    parser.add_argument("--sheet", type=int, default=20)
    args = parser.parse_args()
    rng = random.Random(7)
    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip, (video, run) in STORES.items():
        db = sqlite3.connect(CACHE / run / "pass1.sqlite")
        frames = dict(db.execute("select frame_index, source_index from frames"))
        best: dict[int, tuple] = {}
        for frame, track, rec in db.execute(
            "select frame_index, track_id, record_json from tracks where class_name = 'person'"
        ):
            r = json.loads(rec)
            if r.get("interpolated"):
                continue
            b = r["box_reference"]
            h = b["y2"] - b["y1"]
            if h >= 60 and (track not in best or rng.random() < 0.02):
                best[track] = (frame, b)
        picks = rng.sample(sorted(best), min(args.per_clip, len(best)))
        cap = cv2.VideoCapture(video)
        for n, track in enumerate(sorted(picks)):
            frame, b = best[track]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[frame])
            ok, image = cap.read()
            if not ok:
                continue
            image = cv2.resize(image, (1920, 1080))
            x1, y1, x2, y2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
            pad = int(0.08 * (y2 - y1))
            crop = image[max(0, y1 - pad) : y2, max(0, x1 - pad) : x2 + pad]
            crop_id = f"{clip}-{n:02d}"
            cv2.imwrite(str(crops_dir / f"{crop_id}.jpg"), crop)
            rows.append({"crop_id": crop_id, "image_path": f"crops/{crop_id}.jpg", "split": "test",
                         "clip": clip, "track": track, "person_height_px": y2 - y1,
                         "head_state": None, "torso_state": None})
    (args.out / "crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    # numbered contact sheets: each tile 200 px tall
    for start in range(0, len(rows), args.sheet):
        tiles = []
        for r in rows[start : start + args.sheet]:
            img = cv2.imread(str(args.out / r["image_path"]))
            img = cv2.resize(img, (max(1, int(img.shape[1] * 260 / img.shape[0])), 260))
            canvas = np.full((290, 170, 3), 255, np.uint8)
            w = min(img.shape[1], 170)
            canvas[30:290, :w] = img[:, :w]
            cv2.putText(canvas, r["crop_id"], (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            tiles.append(canvas)
        while len(tiles) % 10:
            tiles.append(np.full((290, 170, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i : i + 10]) for i in range(0, len(tiles), 10)])
        cv2.imwrite(str(args.out / f"sheet_{start // args.sheet:02d}.jpg"), sheet)
    print(f"{len(rows)} crops, {(len(rows) + args.sheet - 1) // args.sheet} sheets in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
