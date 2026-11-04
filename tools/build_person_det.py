"""A single-class (person) YOLO dataset from Construction-PPE, split like the PPE crops.

train = the edition's train split; val/test = its val split halved by the same image hash
as tools/merge_ppe_crops.py, so the person detector and the PPE classifier are tested on
the same untouched images.

    uv run --no-sync python tools/build_person_det.py data/external/construction-ppe \\
        --out data/working/person_det
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = yaml.safe_load((args.dataset / "data.yaml").read_text(encoding="utf-8"))
    names = meta["names"]
    person = next(int(k) for k, v in names.items() if str(v).lower() == "person")
    counts: dict[str, int] = {}
    for source_split in ("train", "val"):
        for image in sorted((args.dataset / "images" / source_split).iterdir()):
            split = source_split
            if source_split == "val":
                first = hashlib.sha256(image.name.encode()).digest()[0]
                split = "test" if first >= 128 else "val"
            label = args.dataset / "labels" / source_split / f"{image.stem}.txt"
            lines = []
            if label.is_file():
                for row in label.read_text(encoding="utf-8").splitlines():
                    parts = row.split()
                    if parts and int(parts[0]) == person:
                        lines.append("0 " + " ".join(parts[1:5]))
            for kind, payload in (("images", None), ("labels", lines)):
                folder = args.out / kind / split
                folder.mkdir(parents=True, exist_ok=True)
                if payload is None:
                    shutil.copy2(image, folder / image.name)
                else:
                    (folder / f"{image.stem}.txt").write_text("\n".join(payload) + "\n", "utf-8")
            counts[split] = counts.get(split, 0) + 1
    (args.out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(args.out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )
    print(f"images per split: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
