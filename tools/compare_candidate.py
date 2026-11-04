"""Run a clip through the current and a candidate model setup and compare incidents.

No ground truth is needed: it reports what changes (incidents per rule, and which
incidents are new or gone), so every change on real footage can be inspected by eye.

    uv run --no-sync python tools/compare_candidate.py data/external/sard/9.mp4 \\
        --site config/sites/sard-9-demo.json \\
        --detector artifacts/models/pretrained/yoloe-26l-construction-r7v4.pt \\
        --ppe artifacts/checkpoints/ppe/effb0-v3/promoted.pt --r7-min-confidence 0.25
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def analyse(video: Path, site: Path | None, config, tag: str) -> list[dict]:  # noqa: ANN001
    from jobs.process_job import ProcessJob
    from shared.identity import display_track_label

    scratch = Path(tempfile.mkdtemp(prefix=f"compare-{tag}-"))
    job = ProcessJob(video, output_root=scratch, run_id=f"compare-{tag}-{video.stem}",
                     site_config=site, config=config)
    shutil.rmtree(job.work_dir, ignore_errors=True)
    try:
        found = job.analyse()
        out = [
            {"rule": i.rule_id.value, "worker": display_track_label(i.canonical_track_id or ""),
             "start": round(i.first_seen_s, 1), "evidence": i.evidence_quality}
            for i in found.pass2.rules.incidents
        ]
        found.store.close()
        return out
    finally:
        shutil.rmtree(job.work_dir, ignore_errors=True)


def main() -> int:
    from shared.config import load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--ppe", type=Path)
    parser.add_argument("--r7-min-confidence", type=float)
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    base = load_config()
    candidate = base
    if args.detector:
        candidate = candidate.model_copy(update={"stage1": candidate.stage1.model_copy(
            update={"pretrained_fallback": str(args.detector)})})
    if args.ppe:
        candidate = candidate.model_copy(update={"ppe": candidate.ppe.model_copy(
            update={"model_path": str(args.ppe)})})
    if args.r7_min_confidence is not None:
        candidate = candidate.model_copy(update={"rules": candidate.rules.model_copy(
            update={"r7_min_confidence": args.r7_min_confidence})})
    results = {"candidate": analyse(args.video, args.site, candidate, "candidate")}
    if not args.skip_current:
        # the current setup: original detector, PPE and R7 threshold, people-only tracking
        current = base.model_copy(update={"tracker": base.tracker.model_copy(
            update={"hazard_new_track_threshold": None})})
        results["current"] = analyse(args.video, args.site, current, "current")
    for name, incidents in results.items():
        print(name, dict(Counter(i["rule"] for i in incidents)))
    if "current" in results:
        key = lambda i: (i["rule"], i["worker"], round(i["start"]))  # noqa: E731
        now = {key(i) for i in results["current"]}
        cand = {key(i) for i in results["candidate"]}
        print("new in candidate:", sorted(cand - now))
        print("gone in candidate:", sorted(now - cand))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
