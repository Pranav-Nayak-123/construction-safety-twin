"""R7 — worker under a suspended load (v3).

Struck-by from a dropped load is one of construction's fatal four. The detector finds the
crane hook or the load hanging from it; the drop zone is everything directly beneath it.
A hanging load has no ground contact to calibrate from, so the test is in image space and
deliberately generous: a worker whose box overlaps the load's horizontal extent (widened
by `SIDE_MARGIN` either side, since loads swing) and who stands below it in the frame,
within `MAX_DROP_HEIGHTS` load heights. Heuristic by design; the evidence grade and the
review loop catch the cases perspective gets wrong.

A detection only counts as a load if it looks like one: it hangs below its cable, so it is
not cut off by the top of the frame, and it is compact, not a tall column. On SARD 9 the
open-vocabulary detector labels the tower-crane mast "suspended load" in every frame;
these two checks reject it without touching real loads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pipeline.rules.base import ResultFactory, most_relevant
from shared.enums import RuleId, RuleStatus
from shared.schemas.frame import FrameContext, HazardObject
from shared.schemas.incidents import RuleResult

ALERT_REASON = "worker_under_suspended_load"
LOAD_CLASS = "suspended_load"
SIDE_MARGIN = 0.25  # of the load's width
MAX_DROP_HEIGHTS = 8.0
TOP_EDGE_PX = 3.0
MAX_LOAD_ASPECT = 2.2  # height / width; masts and columns are taller
FIXED_AFTER_S = 15.0
FIXED_SHIFT = 0.2  # of the load's width
PLACE_OVERLAP = 0.5  # median boxes this alike are the same place


def plausible_load(load: HazardObject) -> bool:
    b = load.box_reference
    width, height = b.x2 - b.x1, b.y2 - b.y1
    return b.y1 > TOP_EDGE_PX and width > 0 and height / width <= MAX_LOAD_ASPECT


def fixed_structures(tracks: Mapping[str, Sequence]) -> set[str]:
    """Load tracks that never move: structure the detector mistook for a load.

    A suspended load is being lifted, so it moves. On SARD 7 a concrete pump's boom was
    tracked as a "suspended load" in the same place for the whole clip, and on SARD 8 a
    building ledge was, split into several short tracks. Tracks are therefore pooled by
    place (their median boxes overlap): a place occupied for at least `FIXED_AFTER_S` whose
    box centre stays within `FIXED_SHIFT` of its width is structure, and every track there
    is dropped. Short clips (SynthSite is 5-10 s) are untouched.
    """

    def box(o):  # noqa: ANN001, ANN202
        return o.box_oriented or o.box_reference

    def median_box(observations: Sequence) -> tuple[float, float, float, float]:
        boxes = [box(o) for o in observations]
        mid = len(boxes) // 2
        return tuple(sorted(getattr(b, k) for b in boxes)[mid] for k in ("x1", "y1", "x2", "y2"))

    def overlap(a: tuple, b: tuple) -> float:
        w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
        return w * h / union if union > 0 else 0.0

    places: list[tuple[tuple, list[str]]] = []  # (representative box, track ids)
    for track_id, observations in tracks.items():
        if not observations:
            continue
        here = median_box(observations)
        for representative, members in places:
            if overlap(representative, here) >= PLACE_OVERLAP:
                members.append(track_id)
                break
        else:
            places.append((here, [track_id]))

    fixed: set[str] = set()
    for _, members in places:
        observations = sorted((o for t in members for o in tracks[t]), key=lambda o: o.video_time_s)
        if len(observations) < 2:
            continue
        if observations[-1].video_time_s - observations[0].video_time_s < FIXED_AFTER_S:
            continue
        boxes = [box(o) for o in observations]
        cx = [(b.x1 + b.x2) / 2 for b in boxes]
        cy = [(b.y1 + b.y2) / 2 for b in boxes]
        width = sorted(b.x2 - b.x1 for b in boxes)[len(boxes) // 2]
        if max(max(cx) - min(cx), max(cy) - min(cy)) <= FIXED_SHIFT * max(width, 1.0):
            fixed.update(members)
    return fixed


def beneath(worker_box, load: HazardObject) -> bool:
    lb = load.box_reference
    width = lb.x2 - lb.x1
    left, right = lb.x1 - SIDE_MARGIN * width, lb.x2 + SIDE_MARGIN * width
    overlaps = worker_box.x2 > left and worker_box.x1 < right
    below = worker_box.y2 > lb.y2
    near = worker_box.y1 - lb.y2 <= MAX_DROP_HEIGHTS * max(lb.y2 - lb.y1, 1.0)
    return overlaps and below and near


class R7SuspendedLoad:
    rule_id = RuleId.R7

    def __init__(self, *, basis: str, min_confidence: float) -> None:
        self.basis = basis
        self.min_confidence = min_confidence

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        make = ResultFactory(self.rule_id, self.basis, context)
        if LOAD_CLASS not in context.capabilities.hazard_classes:
            return make.frame(RuleStatus.UNSUPPORTED, "suspended_load_detection_unavailable")
        loads = [
            h
            for h in context.hazards
            if h.class_name == LOAD_CLASS
            and (h.confidence or 0.0) >= self.min_confidence
            and plausible_load(h)
        ]
        if not loads:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_suspended_load_observed")
        workers = context.workers()
        if not workers:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_person_tracks")
        results = []
        for worker in workers:
            track_id = worker.canonical_track_id
            if worker.box_reference is None:
                results.append(
                    make.make(RuleStatus.INCONCLUSIVE, "worker_box_unavailable", track_id=track_id)
                )
                continue
            per_load = []
            for load in loads:
                hit = beneath(worker.box_reference, load)
                per_load.append(
                    make.make(
                        RuleStatus.EVALUATED_ALERT if hit else RuleStatus.EVALUATED_CLEAR,
                        ALERT_REASON if hit else "not_beneath_load",
                        track_id=track_id,
                        object_id=load.object_id,
                        confidence=load.confidence,
                    )
                )
            results.append(most_relevant(per_load))
        return results
