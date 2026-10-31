"""Paths, the rule under comparison, and the ground truth every step of this experiment shares.

The rule and the scorer are imported read-only from the rule-route worktree (branch feature/rule-eval-tolerance,
commit 4201f17) so both routes are judged by the same code: scripts/eval_rule_events.py, tolerance 1.25 s.
"""
import json
import sys
from pathlib import Path

RULE_ROOT = Path(r"E:\Learning\yolo-ped-rulefix")
RULE_COMMIT = "4201f17"
DATA_ROOT = Path(r"E:\Learning\Yolo-Pedestrian-Detection-for-Right-of-way-Recognition")
VIDEOS = DATA_ROOT / "datasets" / "來源影片"
RUN2 = DATA_ROOT / "datasets" / "eval-run2"          # <stem>/seg/masks.npz and <stem>/tracks/raw_tracks.json
LABELS = DATA_ROOT / "labels"                          # <stem>.jsonl, 17 labels (9 manual, 8 confirmed_candidate)
VERDICTS = RULE_ROOT / ".amp" / "in" / "scratch" / "verdicts_after.json"  # A/B/C/D on the rule's 28 events
TOLERANCE_S = 1.25
NEGATIVE_MARGIN_S = 2.5  # a candidate this far from every label and judged event is a negative

sys.dont_write_bytecode = True  # never write into the read-only rule worktree
sys.path[:0] = [str(RULE_ROOT), str(RULE_ROOT / "scripts")]
import yield_image_rule as rule  # noqa: E402,F401


def stems():
    return sorted(p.stem for p in LABELS.glob("*.jsonl"))


def video_path(stem):
    return next(VIDEOS.glob(stem + ".*"))


def load_labels(stem):
    path = LABELS / f"{stem}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_verdicts():
    return json.loads(VERDICTS.read_text(encoding="utf-8"))


def ground_truth(candidate, labels, verdicts, tolerance=TOLERANCE_S, margin=NEGATIVE_MARGIN_S):
    """1 (failure to yield), 0 (not one) or None (unknown: left out of training and of the training metrics).

    A human verdict on the same vehicle at the same moment decides first: A (real) and B (real, but already reported)
    are positive, C negative, D unknown. Otherwise a label within the tolerance is positive. Otherwise a candidate with
    no label and no judged event within `margin` is negative: the labels are incomplete (6 of the 15 A events have no
    label), so this negative is noisy, and anything between the tolerance and the margin is left out.
    """
    t = candidate["time_s"]
    for v in verdicts:
        if v["vehicle"] == candidate["vehicle_id"] and abs(v["time_s"] - t) <= tolerance:
            return {"A": 1, "B": 1, "C": 0}.get(v["verdict"])
    if any(abs(label["time_s"] - t) <= tolerance for label in labels):
        return 1
    if any(abs(label["time_s"] - t) <= margin for label in labels) \
            or any(abs(v["time_s"] - t) <= margin and v["verdict"] != "C" for v in verdicts):
        return None
    return 0
