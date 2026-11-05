<img src="docs/brand/ground-truth-mark.svg" alt="Ground Truth logo" width="72" align="left">

# Ground Truth

**Live demo: https://safety-twin.onrender.com** · CS3235 Team 5, *AI-Based Construction Safety Inspection and Compliance*

<br clear="left">

Upload a fixed-camera construction-site video on the site, or process one on the workstation, and get:
- hazard incidents for eight rules, each with an evidence clip and a "why it fired" trace;
- leading indicators;
- an interactive 3D digital twin of the site;
- a 5×5 risk register;
- an assistant that answers questions about the run and cites every fact it uses.

The name is the job: what is actually true on the ground, checked against the footage. The GT logo hides it. The amber T drops onto a ground line, and the dot is a worker on the ground plane.

This is a heuristic triage tool, not a certified safety system and not legal advice. Distances are relative risk bands in worker heights, not metres. The twin does not reconstruct hidden geometry or BIM.

## At a glance (November 2026)

| Area | State |
|---|---|
| Offline pipeline | Runs end to end on three real CCTV clips (SARD 7, 8, 9) at 1.1–2.0× real time on a GTX 1660 Ti |
| Hazard rules | R1–R8: PPE, restricted zones, machinery, open edges, worker down, under a suspended load, fire or smoke |
| Hosted demo | Live on free tiers: Render + Hugging Face + Gemini + Supabase. CI/CD deploys after tests pass; $0 a month |
| Video uploads | Upload on the site with an access code. The workstation's GPU worker runs the full pipeline and publishes the report, and the raw upload is deleted. Live and tested end to end |
| Digital twin | Calibrated from the workers' own heights. A steady camera that cannot be calibrated (short or sparse clips) gets an approximate, clearly labelled layout instead of none |
| Risk assistant | Gemini 3.6 Flash over precomputed facts (counts, timelines, rankings). Every claim must cite a fact, and every number and worker is checked against it. Legal and off-topic questions are answered without the model. 35/35 on a graded question set, up from 30/35 (`evaluation/chat_eval.py`) |
| Incident narrator | Qwen2.5-1.5B LoRA, trained and evaluated against base 0-shot and 4-shot. Not promoted: 0.90 pass rate vs 0.93 for 4-shot on the test split. Guarded template text is shown instead |
| Accuracy (November 2026) | **PPE classifier v3b** (both Construction-PPE editions, 3,513 crops): helmet/vest macro-F1 0.946/0.950 on the original test set (v2: 0.944/0.933), 0.924/0.971 on a fresh held-out set (v2: 0.698/0.860). Missing-helmet recall on unseen images went from 0.39 to 0.85. On 100 hand-labelled real SARD crops it raises 0 false "no helmet" alarms; vest calls on real CCTV are still weak (52% false "vest", from 89%). **Person detection**: AP50 0.989 on held-out Construction-PPE, so no fine-tune was needed. **R7**: loads now get their own tracker, and static structures (masts, booms, ledges) are filtered out. on the held-out SynthSite test split (160 clips) recall rose from 0.04 to 0.33 and F2 from 0.05 to 0.37 (level with Grounding DINO), with 0 false R7 alarms on real SARD footage; YOLO-World reaches 0.87 recall (`evaluation/results/synthsite_r7-final-test.md`). Assistant: 35/35 and 34/34 on graded question sets. Incident precision and kappa come from team review on the live site |

```text
video ─► intake ─► pass 1: detect (YOLOE-26l, every 2nd frame) · track (ByteTrack) · PPE · pose
      ─► calibrate camera from workers ─► stitch identities ─► rules R1–R8 (debounce, inconclusive)
      ─► insights · traces · evidence grades ─► pass 3: render (faces pixelated) · 3D twin · clips
      ─► validated run bundle + report.html ─► LoRA narration behind a guardrail
      ─► publish (privacy gate) ─► cloud app: gallery · reports · risk register · grounded chat
site upload ─► private inbox ─► GPU worker (this pipeline) ─► publish ─► report on the site
```

## Components

The offline pipeline runs end to end on real footage:

- **Detection**: open-vocabulary YOLOE-26l with a baked construction vocabulary (workers, hard hats, excavators, cranes, loaders, trucks, crane hooks, fire, smoke), plus yolo26n-pose to confirm a worker lying down.
- **Tracking**: ByteTrack.
- **PPE (R1/R2)**: a two-head EfficientNet-B0 classifier, temperature-calibrated, smoothed per track, with a detected hard hat able to veto a "no helmet".
- **Camera calibration**: fitted from the workers themselves, giving R3–R5 distances in worker heights with bootstrap intervals.
- **Rules**: the R1–R8 engine: PPE, zones, machinery, edges, a worker down, a worker under a suspended load, fire or smoke.
- **Outputs**: a side-by-side MP4 with a live 3D site twin, an offline HTML report, and the [run bundle](docs/contracts/run-bundle.md).

An optional LoRA-tuned narrator (Qwen2.5-1.5B) rewrites incident text behind an eight-check guardrail. When a rule cannot be judged it reports `unsupported` or `inconclusive` with the reason, never a fake verdict.

### What v3 adds

v3 is built against what commercial site-safety analytics and open-source projects commonly miss. The comparison and its sources are in [docs/benchmark/industry-comparison.md](docs/benchmark/industry-comparison.md).

- **Leading indicators**, not just incident counts (`insights.json`):
  - exposure in worker-seconds per rule, zone and worker;
  - brief exposures (near-misses that never reached the alert threshold);
  - closest machine approaches;
  - an exposure hotspot map.
- **One worker, one identity.** When a worker is hidden longer than the tracker's memory, they reappear under a new ID. Stitching re-links tracks whose motion, size and clothing colour agree. On SARD 9 it turns 192 track IDs into 146 identities, and 34 incidents become 25.
- **Explainable, triaged alerts.** Every incident carries a decision trace ("why it fired": condition, signal, debounce, visibility) and an evidence grade (strong / moderate / weak). The report puts *Act now* before *Verify first*, which counters alert fatigue.
- **Evidence clips.** Each incident gets a short annotated clip that plays in its report card.
- **Privacy by default.**
  - Workers' faces are pixelated in every exported frame, crop and clip; hard hats stay visible.
  - No identity or biometrics are used, and track IDs are per clip.
  - The report says what is and is not monitored, which supports the notice and oversight duties for workplace monitoring.
- **Zone health.** Each run checks whether the site has visibly changed where a zone was drawn, and flags stale zones for review.
- **Hazards beyond PPE.**
  - R6 *possible worker down*: a worker lying on the ground for 3 s. The box shape nominates, a pose model confirms the body axis is near horizontal, so crouching over rebar does not count.
  - R7 *worker under a suspended load*: a worker in the drop zone below a crane hook or hanging load. Masts and other tall, frame-cropped columns are rejected as loads.
  - R8 *possible fire or smoke*: one incident per fire or smoke region, not one per worker.
- **Interactive 3D twin in the report.** A Three.js view of the site in worker heights, synced to the video:
  - click a worker to follow them and see their path, exposure and incidents;
  - click a zone for its exposure;
  - incident markers on the timeline, an exposure heat layer and a hazards-only filter;
  - keyboard: Space, ←/→, 1/2/3 for camera presets, F to follow, Esc.
- **Risk assessment and a grounded chatbot.**
  - Every run gets a 5×5 risk register: likelihood from what the clip showed × severity from the catalogue, and controls in hierarchy-of-controls order.
  - Rules that could not be checked are listed as *not assessed*, never as low risk.
  - The *Ask about this run* chat answers from the run's facts and cites a fact for every claim. Each number and worker in an answer must appear in the facts that sentence cites, or the answer is replaced by one composed from the register.
- **Re-evaluate the past.** `safety-twin reprocess output/<run>` re-runs geometry, rules and rendering on stored detections, with new zones or settings and no GPU.
- **Faster.**
  - The detector runs on every 2nd frame, with tracks interpolated offline in between.
  - The 3D twin renders in a parallel process.
  - Intake samples keyframes instead of seeking.

Full specification: [`docs/plan/construction-safety-2.5d-twin-plan-v7.md`](docs/plan/construction-safety-2.5d-twin-plan-v7.md)

## Constraints

- Team of four, 12 weeks; developed on Apple M1 Pro (16 GB) and Windows + GTX 1660 Ti (6 GB)
- Input: prerecorded fixed-camera video
- Output: 1920×1080 side-by-side H.264 MP4
- Twin is **2.5D**, not a surveyed reconstruction
- Gemini is optional and cached; demo must run offline
- LLM fine-tuning is a graded experiment, not the rule authority

## Hazard rules

| Id | Meaning |
|---|---|
| R1 | Apparent missing helmet |
| R2 | Apparent missing hi-vis near machinery or traffic |
| R3 | Restricted-zone intrusion |
| R4 | Approximate machinery proximity |
| R5 | Possible missing fall protection near an elevated/open edge |
| R6 | Possible worker down (lying on the ground) |
| R7 | Worker under a suspended load |
| R8 | Possible fire or smoke |

## Layout

```text
app/           desktop entrypoint and the hosted demo (app/cloud.py, FastAPI)
jobs/          pipeline jobs: process, reprocess, analyse (rules without rendering)
pipeline/      intake, vision, scene, mapping, rules, render, report, outputs (risk, facts)
twin/          2.5D renderer
shared/        config, coordinates, schemas (the run-bundle contract)
llm/           narrator (LoRA + guardrail) and the risk-assessment chatbot
evaluation/    PPE, narration, team review and SynthSite benchmarks
tools/         dataset fetch, labelling, publishing, profilers
config/        runtime defaults, class map, site configs
docs/          plan, run-bundle contract, industry comparison, cloud deployment
Dockerfile, render.yaml, .github/workflows/   container, Render Blueprint, CI
```

## Setup

Requires Python 3.11.9, [uv](https://docs.astral.sh/uv/) and ffmpeg on `PATH`.

```bash
uv sync --group dev                    # rules, schemas, tests (no GPU stack)
uv sync --extra vision --group dev     # + torch, ultralytics for the real pipeline
uv run safety-tools fetch-models yolo26l.pt yolo26n-pose.pt
uv run python tools/build_open_vocab.py          # one-time: YOLOE-26l with the construction vocabulary
uv run pytest tests
```

The `vision` extra pulls CUDA 12.8 PyTorch on Windows and the default wheels on macOS/Linux; `device: auto` in `config/defaults.yaml` picks CUDA, then Apple MPS, then CPU.

## Running the pipeline

```bash
uv run safety-twin process path/to/clip.mp4                       # → output/<run_id>/
uv run safety-twin process clip.mp4 --site config/sites/yard.json  # with zones/edges (R3, R5)
uv run safety-twin reprocess output/<run_id> --site new_zones.json  # rules + render again, no detection
uv run safety-twin process clip.mp4 --synthetic                    # week-1 placeholder, no models
uv run safety-twin app                                             # local desk UI (synthetic)
```

Each run writes a validated bundle — `run_manifest.json`, `incidents.json`, `tracks_summary.json`, `scene.json`, evidence images, `safety_twin.mp4` (1920×1080) and `safety_twin_720p.mp4` — described in [docs/contracts/run-bundle.md](docs/contracts/run-bundle.md). The manifest records stage timings and a warning for every capability the run lacked.

**Zones and edges (R3/R5)** are drawn once per fixed camera:

```bash
uv run python tools/annotate_site.py clip.mp4 --camera yard-east --time 10
```

Zones only apply to shots whose view still matches the frame they were drawn on.

**Performance** (GTX 1660 Ti, i5-10300H, 173 s 1440p clip, on mains power): 5m46s end to end with YOLOE-26l, the PPE classifier, pose, stitching, R1–R8, the 1080p video and the report, or 2.0× real time. On battery the GPU clocks down about 3×; the manifest warns when a run was timed on battery. Encoding uses NVENC or Intel Quick Sync when available (`video_codec: auto`). Detector trade-offs: `uv run python tools/benchmark_stage1.py clip.mp4 --models yolo26m.pt yolo26l.pt yolo26x.pt`, results in `evaluation/results/`.

**Report.** `report.html` is a single offline web page with a sidebar and seven sections: Overview, Digital twin, Incidents, Risk assessment, Leading indicators, Site & camera, Run details. It adapts to light and dark mode and contains:

- a privacy and oversight notice;
- the 720p replay, where clicking any R1–R8 timeline span or card seeks the video;
- the interactive 3D twin, synced to the replay;
- an inbox-style incident view: a triaged list, and a detail panel with the evidence clip, the "why it fired" trace, the facts and the recommended action;
- review buttons on each incident (real / false alarm with a reason / unsure). *Export review* saves them, and `tools/review_incidents.py --import` turns them into precision per rule;
- the risk register, a 5×5 matrix and the chat (offline it answers the suggested questions from the register; hosted, it answers anything about the run);
- leading indicators, with an exposure chart, brief exposures, closest approaches and the hotspot map;
- rule coverage;
- camera calibration;
- zone health;
- stage timings.

**Measuring accuracy.**
- `uv run python tools/label_ppe.py output/<run>`: blind-label a fixed, stratified sample of 400 worker crops (about 20 min).
- `uv run python -m evaluation.ppe_eval output/<run>`: PPE precision and recall by worker size.
- `uv run python tools/review_incidents.py output/<run>`: mark each incident real or false alarm, giving precision per rule.
- On the live site, reviewers mark incidents under their name. `uv run python -m evaluation.reviews output/<run>` then gives precision per rule from the majority verdict, and Cohen's kappa between reviewers.
- `uv run --no-sync python -m evaluation.synthsite_eval`: R7 on the [SynthSite](https://huggingface.co/datasets/govtech/SynthSite) benchmark (227 clips labelled unsafe/safe), with P/R/F1/F2 and 95 % CIs, next to the dataset's YOLO-World and Grounding DINO results. Tuning is allowed on the fixed dev split only.

**Incident narration (optional, L4 + L5).**

```bash
uv sync --extra vision --extra narration --group dev
uv run python -m llm.corpus                    # train / val / test / blind-holdout corpus
uv run python -m llm.train_lora                # LoRA r8/α16 on q,v of the top 8 layers: 311,296 params
uv run python -m llm.evaluate                  # template vs base 4-shot vs LoRA, all through the guardrail
uv run safety-llm narrate output/<run_id>      # add narrations to a finished run
```

A narration is shown only if the guardrail passes it. The guardrail checks:
- shape;
- no invented numbers or IDs;
- no citations;
- no legal verdicts;
- no metres;
- the action keeps its subject;
- no overclaiming or downplaying.

If any check fails, the deterministic template text is used and the failed checks are recorded.

**Models.**

| Model | How it is built | Evaluation |
|---|---|---|
| Detector | `tools/build_open_vocab.py` bakes the construction vocabulary into YOLOE-26l | — |
| PPE classifier | `tools/build_ppe_crops.py` then `tools/train_ppe.py` | `evaluation/results/ppe_effb0_v2.json`; SARD audit in `ppe_sard_audit.md` |

Demo clip assignments are frozen in `evaluation/demo_inventory.yaml`.

Or: `./run.sh status`

Copy `.env.example` to `.env` if you later enable Gemini scene bootstrap. The pipeline must still work with that key unset.

## Hosted demo

**Live: https://safety-twin.onrender.com**

The public demo runs on free tiers only: Render (Docker web service), a Hugging Face dataset for run bundles, the Gemini API for the chatbot, and Supabase for chat logs and reviews. Processing stays on the GPU workstation; finished runs are published with `tools/publish_run.py`, which refuses any run without face pixelation.

```bash
uv sync --extra serve
SAFETY_RUNS_DIR=output SAFETY_LLM_PROVIDER=ollama uv run uvicorn app.cloud:app --port 8080   # local
```

**Video uploads.** Anyone with the access code can upload a clip at `/upload` (up to 200 MB and 5 minutes). Render has no GPU, so the site only queues the job in a private Hugging Face inbox. The worker on the GPU workstation picks it up, runs the full pipeline and publishes the run through the same privacy gate. The status page shows live progress and links the report. Start the worker with:

```bash
SAFETY_WORKER_KEY=... uv run python tools/worker.py --site https://safety-twin.onrender.com \
    --inbox <user>/safety-twin-inbox --runs <user>/safety-twin-runs
```

The worker pauses on battery, and jobs wait in the queue while it is offline. The site is redesigned (Ground Truth): a verdict-first overview, a grouped incident inbox, and the assistant as a drawer on every page. Every published report is re-rendered with the current design at startup.

The architecture, the filled-in Section 11 table, step-by-step deployment, upload setup, rollback and the shutdown plan are in [docs/deploy/cloud.md](docs/deploy/cloud.md).

## Data

Datasets and working videos are local-only (about 30 GB on the development machine). Do not commit them. See [`data/README.md`](data/README.md).

```bash
uv run python tools/fetch_sard.py --list complementary
uv run python tools/fetch_sard.py complementary 9.mp4 9.txt   # resumable
uv run python tools/cctv.py vet data/source/your_clip.mp4
```

## License

AGPL-3.0-or-later. Ultralytics / YOLO components used for academic open-source work require AGPL.
