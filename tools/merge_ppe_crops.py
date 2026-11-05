"""Merge PPE crop sets from two dataset editions without test leakage.

The November 2026 edition of Construction-PPE renamed every image, so the same photo
can sit in our old test split and in the new training split under different names. Any
new crop within a few bits (64-bit difference hash) of an old test crop is dropped.
The old test split stays the test split, so scores compare with the earlier model; half
of the new edition's validation images become a second, fresh test set (`--fresh-test`).

    uv run --no-sync python tools/merge_ppe_crops.py data/working/ppe data/working/ppe_2026 \\
        --out data/working/ppe_v3 --fresh-test data/working/ppe_2026_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

NEAR_DUPLICATE_BITS = 6


def dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]


def rebased(record: dict, root: Path, out: Path, split: str) -> dict:
    path = os.path.relpath(root / record["image_path"], out).replace("\\", "/")
    return {**record, "image_path": path, "split": split}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fresh-test", type=Path, required=True)
    args = parser.parse_args()

    old, new = records(args.old), records(args.new)
    test_hashes = [dhash(args.old / r["image_path"]) for r in old if r["split"] == "test"]
    test_array = np.array(test_hashes, dtype=np.uint64)

    def near_test(record: dict) -> bool:
        h = np.uint64(dhash(args.new / record["image_path"]))
        distance = np.array([bin(int(x)).count("1") for x in (test_array ^ h)])
        return bool((distance <= NEAR_DUPLICATE_BITS).any())

    merged = [rebased(r, args.old, args.out, r["split"]) for r in old]
    fresh, dropped = [], 0
    for r in new:
        if r["split"] == "val":
            first = hashlib.sha256(r["source_image"].encode()).digest()[0]
            if first >= 128:  # fresh test: never used for training or model selection
                fresh.append(rebased(r, args.new, args.fresh_test, "test"))
                continue
        if near_test(r):
            dropped += 1
            continue
        merged.append(rebased(r, args.new, args.out, r["split"]))

    for root, rows in ((args.out, merged), (args.fresh_test, fresh)):
        root.mkdir(parents=True, exist_ok=True)
        (root / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    counts = {s: sum(r["split"] == s for r in merged) for s in ("train", "val", "test")}
    print(
        f"merged {counts}; fresh test {len(fresh)}; dropped {dropped} near-duplicates of old test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
