"""Score PPE classifier checkpoints on the same test crops, at full size and CCTV sizes.

    uv run --no-sync python tools/eval_ppe.py data/working/ppe_2026_test \\
        artifacts/models/ppe/best.pt artifacts/checkpoints/ppe/effb0-v3/promoted.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.vision.ppe_model import TwoHeadPPEClassifier  # noqa: E402
from tools.train_ppe import SEED, evaluate, load_split  # noqa: E402

HEIGHTS = (("original", None), ("cctv_96px", 96), ("cctv_64px", 64), ("cctv_48px", 48))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="crop set whose 'test' split is scored")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    records = [json.loads(line) for line in (args.data / "records.jsonl").read_text().splitlines()]
    splits = {name: load_split(args.data, records, "test", h, SEED + 1) for name, h in HEIGHTS}
    report = {}
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = TwoHeadPPEClassifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        t = checkpoint["metadata"]["temperatures"]
        temps = (t["head"], t["torso"])
        report[str(path)] = {
            name: evaluate(model, data, device, temps) for name, data in splits.items()
        }
        row = " | ".join(
            f"{name}: helmet {r['head']['macro_f1']:.3f} vest {r['torso']['macro_f1']:.3f}"
            for name, r in report[str(path)].items()
        )
        print(f"{path}\n  {row}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
