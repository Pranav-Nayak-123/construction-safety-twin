"""Incident precision from several reviewers, with their agreement (Cohen's kappa).

Reviewers mark incidents on the hosted report (real / false alarm / unsure); the server
stores each verdict in Supabase. This turns those rows into:

* each reviewer's latest verdict per incident;
* a consensus per incident: the majority of decided verdicts (real vs false alarm); a tie
  or only "unsure" leaves the incident undecided;
* precision per rule = real / (real + false alarm) over decided incidents;
* agreement: Cohen's kappa for each pair of reviewers over incidents both decided, and the
  raw share of identical verdicts. Kappa corrects for chance agreement: a reviewer who
  marks everything "real" agrees often but tells you little.

    uv run python -m evaluation.reviews output/sard9-r7fix          # rows from Supabase
    uv run python -m evaluation.reviews output/sard9-r7fix --rows reviews.json

Needs SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment (or .env) for Supabase.
Set SAFETY_REVIEWERS="name1,name2" to count only those reviewers.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

DECIDED = ("real", "false_alarm")


def allowed_reviewers(env: Mapping[str, str] | None = None) -> set[str] | None:
    raw = (env if env is not None else os.environ).get("SAFETY_REVIEWERS", "")
    names = {n.strip().lower() for n in raw.split(",") if n.strip()}
    return names or None


def latest_by_reviewer(
    rows: Iterable[Mapping[str, Any]], allowed: set[str] | None = None
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """reviewer -> incident_id -> that reviewer's latest row (rows in time order)."""
    out: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    rows = [r for r in rows if r.get("incident_id") and r.get("verdict")]  # skip partial rows
    # ids are compared as text: integer and uuid primary keys must not crash the sort
    for row in sorted(rows, key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or ""))):
        name = str(row.get("reviewer") or "").strip().lower()
        if not name or (allowed is not None and name not in allowed):
            continue
        out[name][str(row["incident_id"])] = row
    return dict(out)


def consensus(verdicts: Sequence[str]) -> str | None:
    counts = Counter(v for v in verdicts if v in DECIDED)
    if not counts or counts["real"] == counts["false_alarm"]:
        return None
    return "real" if counts["real"] > counts["false_alarm"] else "false_alarm"


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same items; None when undefined."""
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label] for label in labels) / (n * n)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def summarize(
    rows: Iterable[Mapping[str, Any]],
    incident_rules: Mapping[str, str],
    *,
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    """Precision per rule from the consensus, plus reviewer agreement."""
    latest = latest_by_reviewer(
        (r for r in rows if str(r.get("incident_id")) in incident_rules), allowed
    )
    per_incident: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for by_incident in latest.values():
        for incident_id, row in by_incident.items():
            per_incident[incident_id].append(row)

    rules: dict[str, Counter[str]] = defaultdict(Counter)
    for incident_id, rule in incident_rules.items():
        verdict_rows = per_incident.get(incident_id, [])
        if not verdict_rows:
            rules[rule]["unreviewed"] += 1
            continue
        decided = consensus([str(r["verdict"]) for r in verdict_rows])
        rules[rule][decided or "undecided"] += 1
        if decided == "false_alarm":
            reasons = Counter(
                str(r.get("reason") or "other")
                for r in verdict_rows
                if r["verdict"] == "false_alarm"
            )
            rules[rule][f"reason:{reasons.most_common(1)[0][0]}"] += 1

    pairs = []
    for first, second in combinations(sorted(latest), 2):
        shared = [
            i
            for i in latest[first]
            if i in latest[second]
            and latest[first][i]["verdict"] in DECIDED
            and latest[second][i]["verdict"] in DECIDED
        ]
        if len(shared) < 2:
            continue
        a = [str(latest[first][i]["verdict"]) for i in shared]
        b = [str(latest[second][i]["verdict"]) for i in shared]
        kappa = cohen_kappa(a, b)
        pairs.append(
            {
                "reviewers": [first, second],
                "incidents": len(shared),
                "agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 3),
                "kappa": None if kappa is None else round(kappa, 3),
            }
        )
    kappas = [p["kappa"] for p in pairs if p["kappa"] is not None]
    by_rule = {}
    for rule, counts in sorted(rules.items()):
        decided = counts["real"] + counts["false_alarm"]
        by_rule[rule] = {
            **dict(counts),
            "precision": round(counts["real"] / decided, 3) if decided else None,
        }
    real = sum(c["real"] for c in rules.values())
    false = sum(c["false_alarm"] for c in rules.values())
    return {
        "reviewers": len(latest),
        "incidents": len(incident_rules),
        "reviewed": len(per_incident),
        "decided": real + false,
        "precision": round(real / (real + false), 3) if real + false else None,
        "by_rule": by_rule,
        "agreement": {
            "pairs": pairs,
            "mean_kappa": round(sum(kappas) / len(kappas), 3) if kappas else None,
        },
    }


def supabase_settings(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """URL and key, trimmed: a value pasted into a dashboard often brings a stray space or
    line break, which made every request fail with InvalidURL on the hosted demo."""
    env = env if env is not None else os.environ
    url = (env.get("SUPABASE_URL") or "").strip().strip('"').strip("'").rstrip("/")
    key = (env.get("SUPABASE_SERVICE_KEY") or "").strip().strip('"').strip("'")
    if url and not url.startswith(("https://", "http://")):
        url = "https://" + url
    return url or None, key or None


def fetch_supabase(run_id: str, env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """All review rows for one run through Supabase's REST API (server-side key)."""
    env = env if env is not None else os.environ
    url, key = supabase_settings(env)
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are needed")
    query = urllib.parse.urlencode(
        {"run_id": f"eq.{run_id}", "select": "*", "order": "created_at.asc"}
    )
    headers = {"apikey": key}
    if key.startswith("eyJ"):  # legacy service_role JWT
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{url.rstrip('/')}/rest/v1/incident_reviews?{query}", headers=headers
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_env_file(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    from pipeline.output.bundle import read_model
    from shared.schemas.bundle import INCIDENTS_FILE, RUN_MANIFEST_FILE, IncidentsFile
    from shared.schemas.run import RunManifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rows", type=Path, help="review rows as JSON instead of Supabase")
    args = parser.parse_args(argv)
    _load_env_file()
    manifest = read_model(args.run_dir / RUN_MANIFEST_FILE, RunManifest)
    incidents = read_model(args.run_dir / INCIDENTS_FILE, IncidentsFile).incidents
    rows = (
        json.loads(args.rows.read_text(encoding="utf-8"))
        if args.rows
        else fetch_supabase(manifest.run_id)
    )
    summary = summarize(
        rows, {i.incident_id: i.rule_id.value for i in incidents}, allowed=allowed_reviewers()
    )
    out = Path(f"evaluation/results/team_review_{manifest.run_id}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{summary['reviewers']} reviewers, {summary['reviewed']}/{summary['incidents']} "
        f"incidents reviewed, {summary['decided']} decided, precision {summary['precision']}"
    )
    for rule, row in summary["by_rule"].items():
        print(f"  {rule}: {row}")
    for pair in summary["agreement"]["pairs"]:
        print(f"  agreement {pair}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
