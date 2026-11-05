"""False alarms of PPE classifiers on hand-labelled real CCTV crops (tools/sample_cctv_crops.py).

Real site footage rarely shows a bare head or a hi-vis vest, so the useful numbers are how
often a helmeted worker is called "no helmet" (a false R1 alarm), how often a worker
without a vest is called "vest", and how often each is left "unknown" (no alarm, no
evidence).

    uv run --no-sync python tools/eval_ppe_cctv.py data/working/cctv_ppe \\
        artifacts/models/ppe/best.pt artifacts/checkpoints/ppe/effb0-v3/promoted.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data.ppe_crops import HEAD_LABELS, TORSO_LABELS  # noqa: E402
from pipeline.vision.ppe_model import TwoHeadPPEClassifier  # noqa: E402
from tools.train_ppe import SEED, collect_logits, load_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    records = [json.loads(line) for line in (args.data / "records.jsonl").read_text().splitlines()]
    images, heads, torsos = load_split(args.data, records, "test", None, SEED)
    report = {}
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = TwoHeadPPEClassifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        t = checkpoint["metadata"]["temperatures"]
        head_logits, torso_logits = collect_logits(model, images, device)
        head_pred = (head_logits / t["head"]).argmax(1).tolist()
        torso_pred = (torso_logits / t["torso"]).argmax(1).tolist()
        result = {}
        for name, truth, pred, labels in (
            ("head", heads, head_pred, HEAD_LABELS),
            ("torso", torsos, torso_pred, TORSO_LABELS),
        ):
            by_truth: dict[str, Counter] = {}
            for y, p in zip(truth.tolist(), pred, strict=True):
                if 0 <= y < len(labels):
                    by_truth.setdefault(str(labels[y]), Counter())[str(labels[p])] += 1
            result[name] = {k: dict(v) for k, v in by_truth.items()}
        helmet = Counter(result["head"].get("helmet", {}))
        no_vest = Counter(result["torso"].get("no_vest", {}))

        def rate(counts: Counter, key: str) -> float:
            return round(counts[key] / max(sum(counts.values()), 1), 3)

        result["false_no_helmet_rate"] = rate(helmet, "no_helmet")
        result["helmet_recognised_rate"] = rate(helmet, "helmet")
        result["false_vest_rate"] = rate(no_vest, "vest")
        result["no_vest_recognised_rate"] = rate(no_vest, "no_vest")
        report[str(path)] = result
        print(path, json.dumps({k: v for k, v in result.items() if "rate" in k}))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
