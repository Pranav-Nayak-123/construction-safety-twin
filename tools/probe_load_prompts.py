"""Which text prompts make YOLOE see suspended loads? A dev-split probe for R7.

For every SynthSite dev clip (never the test split), a dozen frames are run through the
base YOLOE model prompted with every candidate at once, and each prompt's highest
confidence per clip is kept. A useful prompt fires on clips where a worker is under a load
and stays quiet on the rest; the crane structure itself must not count as a load.

    uv run --no-sync python tools/probe_load_prompts.py --out <results>/r7_prompt_probe.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.synthsite_eval import split  # noqa: E402

CANDIDATES = [
    "crane hook",
    "suspended load",
    "load hanging from a crane",
    "crane lifting a load",
    "hoisted load",
    "hanging load",
    "lifted cargo",
    "hook block",
    "lifting sling",
    "steel beam",
    "bundle of steel bars",
    "rebar bundle",
    "concrete bucket",
    "pallet",
    "precast concrete panel",
    "shipping container",
    "wooden crate",
    "load on crane hook",
    "construction material lifted by crane",
    "steel pipe bundle",
]
CONTROLS = ["tower crane", "mobile crane", "person"]  # compete for boxes, never counted


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/external/synthsite"))
    parser.add_argument(
        "--weights", type=Path, default=Path("artifacts/models/pretrained/yoloe-26l-seg.pt")
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from ultralytics import YOLOE

    names = CANDIDATES + CONTROLS
    weights = args.weights.resolve()
    with working_directory(weights.parent):
        model = YOLOE(str(weights))
        model.set_classes(names, model.get_text_pe(names))
    rows = list(csv.DictReader((args.data / "synthetic_video_labels.csv").open(encoding="utf-8")))
    dev = [r for r in rows if split(r["filename"]) == "dev"]
    clips = []
    for n, row in enumerate(dev, 1):
        cap = cv2.VideoCapture(str(args.data / "videos" / row["filename"]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        best = dict.fromkeys(CANDIDATES, 0.0)
        for index in np.linspace(0, total - 1, args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            result = model.predict(frame, conf=0.01, imgsz=960, verbose=False)[0]
            for cls, conf in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
            ):
                name = names[int(cls)]
                if name in best:
                    best[name] = max(best[name], conf)
        cap.release()
        unsafe = row["resolved_label"] == "True_Positive"
        clips.append({"filename": row["filename"], "unsafe": unsafe, "max_conf": best})
        print(f"[{n}/{len(dev)}] {row['filename']} unsafe={unsafe}", flush=True)

    summary = {}
    for prompt in CANDIDATES:
        pos = np.array([c["max_conf"][prompt] for c in clips if c["unsafe"]])
        neg = np.array([c["max_conf"][prompt] for c in clips if not c["unsafe"]])
        rows_t = {}
        for t in (0.05, 0.1, 0.15, 0.2, 0.3):
            rows_t[str(t)] = {
                "unsafe_hit": round(float((pos >= t).mean()), 3),
                "safe_hit": round(float((neg >= t).mean()), 3),
            }
        summary[prompt] = rows_t
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"clips": clips, "summary": summary}, indent=1) + "\n", "utf-8")
    ranked = sorted(
        summary, key=lambda p: -(summary[p]["0.1"]["unsafe_hit"] - summary[p]["0.1"]["safe_hit"])
    )
    for p in ranked:
        s = summary[p]
        print(
            f"{p:<42} @0.10 unsafe {s['0.1']['unsafe_hit']:.2f} safe {s['0.1']['safe_hit']:.2f} | "
            f"@0.20 unsafe {s['0.2']['unsafe_hit']:.2f} safe {s['0.2']['safe_hit']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
