"""R7 "worker under a suspended load" on SynthSite: 227 labelled clips, against published baselines.

SynthSite (huggingface.co/datasets/govtech/SynthSite, GovTech Singapore, MIT-based license)
is 227 synthetic construction videos (Sora 2 Pro, Veo 3.1, Wan 2.2/2.6), 5–10 s at 720p,
each labelled unsafe (a worker stays in the fall zone of a suspended load) or safe by 2–3
reviewers. Tier 1 (150) is unanimous; Tier 2 (77) is where reviewers disagreed. The
dataset ships YOLO-World and Grounding DINO results, which are scored here the same way.

A clip is predicted unsafe when the full pipeline (`ProcessJob.analyse()`: detection,
tracking, rules) confirms at least one R7 incident, i.e. a worker held the drop-zone
condition for the R7 debounce. The clips are generated, not CCTV, so this measures the
rule and the open-vocabulary detector on clean hazards, not deployment accuracy.

    uv run --no-sync python -m evaluation.synthsite_eval --data data/external/synthsite
    uv run --no-sync python -m evaluation.synthsite_eval --data ... --limit 10   # smoke run

Per-clip results are appended to evaluation/results/synthsite_r7_clips.jsonl, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

RESULTS = Path("evaluation/results")

BASELINES = {
    "YOLO-World": "results/detector_outcome/results_yolo_world.csv",
    "Grounding DINO": "results/detector_outcome/results_grounding_dino.csv",
}


def metrics(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """(actual_unsafe, predicted_unsafe) pairs -> confusion and scores (unsafe = positive)."""
    tp = fp = tn = fn = 0
    for actual, predicted in pairs:
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(beta: float) -> float:
        b2 = beta * beta
        denominator = b2 * precision + recall
        return (1 + b2) * precision * recall / denominator if denominator else 0.0

    total = tp + fp + tn + fn
    return {
        "clips": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "f1": round(f_beta(1.0), 4),
        "f2": round(f_beta(2.0), 4),
    }


def bootstrap_ci(
    pairs: list[tuple[bool, bool]], key: str, *, rounds: int = 2000, seed: int = 7
) -> list[float]:
    """95 % percentile bootstrap interval of one score, resampling clips."""
    if not pairs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    scores = sorted(metrics(rng.choices(pairs, k=len(pairs)))[key] for _ in range(rounds))
    return [round(scores[int(0.025 * rounds)], 4), round(scores[int(0.975 * rounds) - 1], 4)]


def scored(pairs_by_tier: Mapping[str, list[tuple[bool, bool]]]) -> dict[str, Any]:
    out = {}
    for tier, pairs in pairs_by_tier.items():
        out[tier] = metrics(pairs) | {"f2_ci95": bootstrap_ci(pairs, "f2")}
    return out


DEV_SHARE = 0.3


def split(filename: str) -> str:
    """Fixed dev/test split by filename hash: tuning may only look at 'dev'."""
    digest = hashlib.sha256(filename.encode("utf-8")).digest()
    return "dev" if digest[0] < round(DEV_SHARE * 256) else "test"


def groups(rows: Iterable[tuple[str, str, bool, bool]]) -> dict[str, list[tuple[bool, bool]]]:
    """(filename, tier, actual, predicted) rows -> pairs per split and tier."""
    out: dict[str, list[tuple[bool, bool]]] = {
        "overall": [],
        "tier1": [],
        "tier2": [],
        "dev": [],
        "test": [],
    }
    for filename, tier, actual, predicted in rows:
        for key in ("overall", f"tier{tier}", split(filename)):
            out[key].append((actual, predicted))
    return out


def load_labels(data: Path) -> list[dict[str, str]]:
    with (data / "synthetic_video_labels.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_rows(
    data: Path, name: str, tiers: Mapping[str, str]
) -> list[tuple[str, str, bool, bool]]:
    with (data / BASELINES[name]).open(encoding="utf-8", newline="") as handle:
        return [
            (
                r["filename"],
                tiers[r["filename"]],
                r["ground_truth"] == "UNSAFE",
                r["prediction"] == "UNSAFE",
            )
            for r in csv.DictReader(handle)
            if r["filename"] in tiers
        ]


def score_clip(clip: Path, scratch: Path, config: Any = None) -> dict[str, Any]:
    from jobs.process_job import ProcessJob
    from shared.enums import RuleId, RuleStatus

    job = ProcessJob(
        clip, output_root=scratch / "out", run_id=f"synthsite-{clip.stem}", config=config
    )
    started = time.perf_counter()
    try:
        found = job.analyse()
        r7 = [i for i in found.pass2.rules.incidents if i.rule_id is RuleId.R7]
        coverage = next(c for c in found.pass2.rules.coverage if c.rule_id is RuleId.R7)
        counts = {
            str(k.value if hasattr(k, "value") else k): v for k, v in coverage.status_counts.items()
        }
        load_frames = sum(v for k, v in counts.items() if k != RuleStatus.NOT_APPLICABLE.value)
        found.store.close()
        return {
            "predicted_unsafe": bool(r7),
            "r7_incidents": len(r7),
            "r7_status": coverage.status.value,
            "frames_with_load": load_frames,
            "frames": found.pass1.frames,
            "camera": found.decision.mode.value,
            "seconds": round(time.perf_counter() - started, 1),
        }
    finally:
        shutil.rmtree(job.work_dir, ignore_errors=True)


def experiment_config(detector: Path | None, r7_min_confidence: float | None = None) -> Any:
    """The default config, or one with an experimental detector and/or R7 threshold."""
    if detector is None and r7_min_confidence is None:
        return None
    from shared.config import load_config

    config = load_config()
    if detector is not None:
        stage1 = config.stage1.model_copy(update={"pretrained_fallback": str(detector)})
        config = config.model_copy(update={"stage1": stage1})
    if r7_min_confidence is not None:
        rules = config.rules.model_copy(update={"r7_min_confidence": r7_min_confidence})
        config = config.model_copy(update={"rules": rules})
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/external/synthsite"))
    parser.add_argument("--limit", type=int, default=None, help="score only the first N clips")
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--detector", type=Path, help="experimental detector weights")
    parser.add_argument("--tag", default="", help="suffix for result files, e.g. prompts-v2")
    parser.add_argument("--r7-min-confidence", type=float, help="experimental R7 load threshold")
    args = parser.parse_args(argv)

    config = experiment_config(args.detector, args.r7_min_confidence)
    suffix = f"-{args.tag}" if args.tag else ""
    clips_file = RESULTS / f"synthsite_r7_clips{suffix}.jsonl"
    labels = [
        r
        for r in load_labels(args.data)
        if args.split == "all" or split(r["filename"]) == args.split
    ]
    clips = {p.name: p for p in args.data.rglob("*.mp4")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = {}
    if clips_file.is_file():
        for line in clips_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["filename"]] = row
    todo = [r for r in labels if r["filename"] not in done and r["filename"] in clips]
    if args.limit is not None:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"{len(labels)} labelled, {len(done)} already scored, {len(todo)} to score")
    with tempfile.TemporaryDirectory() as scratch, clips_file.open("a", encoding="utf-8") as sink:
        for n, row in enumerate(todo, 1):
            try:
                result = score_clip(clips[row["filename"]], Path(scratch), config)
            except Exception as error:  # noqa: BLE001 - one bad clip must not stop the benchmark
                result = {
                    "error": f"{type(error).__name__}: {error}"[:300],
                    "predicted_unsafe": False,
                }
            entry = {
                "filename": row["filename"],
                "tier": row["tier"],
                "actual_unsafe": row["resolved_label"] == "True_Positive",
                **result,
            }
            sink.write(json.dumps(entry) + "\n")
            sink.flush()
            done[row["filename"]] = entry
            print(
                f"[{n}/{len(todo)}] {row['filename']}: actual "
                f"{'UNSAFE' if entry['actual_unsafe'] else 'safe'}, predicted "
                f"{'UNSAFE' if entry['predicted_unsafe'] else 'safe'}"
                + (f" ({entry['error']})" if "error" in entry else f" in {entry['seconds']} s")
            )

    wanted = {r["filename"] for r in labels}
    scored_names = {n for n, r in done.items() if "error" not in r and n in wanted}
    tiers = {r["filename"]: r["tier"] for r in labels if r["filename"] in scored_names}
    ours = [
        (n, done[n]["tier"], done[n]["actual_unsafe"], done[n]["predicted_unsafe"]) for n in tiers
    ]
    summary: dict[str, Any] = {
        "benchmark": "SynthSite (govtech/SynthSite), worker under suspended load",
        "split": args.split,
        "detector": str(args.detector) if args.detector else "default",
        "clips_scored": len(ours),
        "clips_failed": sorted(n for n, r in done.items() if "error" in r and n in wanted),
        "systems": {"This pipeline (R7)": scored(groups(ours))},
    }
    for name in BASELINES:
        summary["systems"][name] = scored(groups(baseline_rows(args.data, name, tiers)))
    (RESULTS / f"synthsite_r7{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", "utf-8"
    )
    lines = [
        f"SynthSite, {len(ours)} clips (same clips for every system), split {args.split}",
        "",
        "| System | Subset | P | R | Spec. | F1 | F2 (95 % CI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for system, subsets in summary["systems"].items():
        for subset, m in subsets.items():
            if not m["clips"]:
                continue
            lo, hi = m["f2_ci95"]
            lines.append(
                f"| {system} | {subset} ({m['clips']}) | {m['precision']:.2f} | "
                f"{m['recall']:.2f} | {m['specificity']:.2f} | {m['f1']:.2f} | "
                f"{m['f2']:.2f} ({lo:.2f}–{hi:.2f}) |"
            )
    (RESULTS / f"synthsite_r7{suffix}.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
