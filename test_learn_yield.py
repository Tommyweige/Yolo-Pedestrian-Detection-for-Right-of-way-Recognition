"""Synthetic tests for scripts/learn_yield.py (no real videos, no GPU)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import learn_yield as ly  # noqa: E402
from yield_features import FEATURE_NAMES  # noqa: E402

H, W = 360, 640
FPS = 60.0
SRC = (1920, 1080)

NEAR = (200, 300, 20, 620)  # horizontal crosswalk band in mask pixels

# Source-pixel boxes: the mask is 640x360 (scale 1/3 of 1920x1080).
PED_ON = (1230, 700, 1290, 850)       # foot inside the near crosswalk
CAR_ON = (700, 500, 1000, 870)        # wheels on the near crosswalk
CAR_BEFORE = (700, 300, 1000, 540)    # stopped before the crosswalk
WAITING = (1230, 480, 1290, 585)      # foot just above the crosswalk edge
FAR_PERSON = (900, 20, 940, 110)      # nowhere near the crosswalk


def mask_with(*rects):
    mask = np.zeros((H, W), bool)
    for y1, y2, x1, x2 in rects:
        mask[y1:y2, x1:x2] = True
    return mask


def track(identity, kind, box, frames):
    return {"id": identity, "kind": kind, "boxes": [[f, *box] for f in frames]}


def synthetic_entries():
    mask = mask_with(NEAR)
    on_crosswalk = [track(1, "person", PED_ON, range(1, 31)),
                    track(10, "car", CAR_ON, range(1, 31))]
    waiting = [track(1, "person", WAITING, range(1, 31)),
               track(10, "car", CAR_ON, range(1, 31))]
    quiet = [track(1, "person", FAR_PERSON, range(1, 31)),
             track(10, "car", CAR_BEFORE, range(1, 31))]
    return [
        {"name": "v1", "mask_at": lambda f: mask, "tracks": on_crosswalk,
         "fps": FPS, "src_size": SRC, "truth_times": [round(1 / FPS, 3)]},
        {"name": "v2", "mask_at": lambda f: mask, "tracks": waiting,
         "fps": FPS, "src_size": SRC, "truth_times": []},
        {"name": "v3", "mask_at": lambda f: mask, "tracks": quiet,
         "fps": FPS, "src_size": SRC, "truth_times": [2.0]},
    ]


def test_candidates_are_a_superset_of_the_current_rule():
    entries = synthetic_entries()
    for entry in entries:
        cands = ly.candidates(entry["mask_at"], entry["tracks"], entry["fps"], entry["src_size"])
        rules = ly.rule_events(entry["mask_at"], entry["tracks"], entry["fps"], entry["src_size"])
        assert isinstance(cands, list)
        assert all(isinstance(c, dict) for c in cands)
        assert len(cands) >= len(rules)

    by_name = {entry["name"]: entry for entry in entries}
    waiting = by_name["v2"]
    assert ly.rule_events(waiting["mask_at"], waiting["tracks"], waiting["fps"], waiting["src_size"]) == []
    assert len(ly.candidates(waiting["mask_at"], waiting["tracks"], waiting["fps"], waiting["src_size"])) == 1


def test_candidates_honour_an_explicit_last_frame():
    entry = synthetic_entries()[0]
    assert ly.candidates(entry["mask_at"], entry["tracks"], entry["fps"], entry["src_size"], last_frame=0) == []


def test_features_are_read_from_the_tracks_the_rule_judged(monkeypatch):
    """detect() merges duplicate vehicle tracks and reports the merged id; the features must see the merged track too."""
    import yield_image_rule
    seen = []
    real = ly.yield_features.index_tracks
    monkeypatch.setattr(ly.yield_features, "index_tracks", lambda tracks: seen.append(tracks) or real(tracks))
    entries = synthetic_entries()
    entries[0]["tracks"] = entries[0]["tracks"] + [track(11, "car", CAR_ON, range(1, 31))]  # a second id, same car
    ly.build_dataset(entries)
    assert len(seen[0]) == 2  # person + one car, not the raw three tracks
    for entry, indexed in zip(entries, seen):
        params = yield_image_rule.Params(fps=entry["fps"], src_size=entry["src_size"], **ly.LOOSE)
        assert indexed == yield_image_rule.merge_duplicate_tracks(entry["tracks"], params)


def test_build_dataset_shapes_labels_and_truth_counts():
    dataset = ly.build_dataset(synthetic_entries())
    n = len(dataset["times"])

    assert dataset["X"].shape == (n, 21)
    assert dataset["X"].shape[1] == len(FEATURE_NAMES)
    assert len(dataset["y"]) == n
    assert len(dataset["groups"]) == n
    assert dataset["truth_counts"] == {"v1": 1, "v2": 0, "v3": 1}
    assert dataset["feature_names"] == FEATURE_NAMES

    # v1 contributes a positive candidate; v2 contributes an unlabelled candidate;
    # v3 has no candidates but is still recorded in truth_counts.
    assert n == 2
    assert int(dataset["y"].sum()) == 1
    assert dataset["groups"] == ["v1", "v2"]
    assert set(dataset["baseline"]) == {"v1", "v2", "v3"}
    assert dataset["baseline"]["v2"] == []
    assert dataset["baseline"]["v1"][0]["time_s"] == dataset["times"][0]


def test_baseline_score_true_positive_is_hand_counted():
    dataset = {"baseline": {"v1": [
        {"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 1.0},
        {"kind": "vehicle_on_crosswalk_with_pedestrian", "time_s": 5.0},
    ]}}
    result = ly.baseline_score(dataset, {"v1": [1.2]}, tolerance=0.5)
    assert result["predicted"] == 2
    assert result["labelled"] == 1
    assert result["true_positive"] == 1
    assert result["false_alarm"] == 1
    assert result["miss"] == 0


def test_baseline_score_empty_is_empty_dict():
    assert ly.baseline_score({"baseline": {}}, {}, tolerance=1.0) == {}


def test_main_reads_a_cache_and_prints_a_json_summary(tmp_path, capsys):
    dataset = ly.build_dataset(synthetic_entries())
    truth = {entry["name"]: entry["truth_times"] for entry in synthetic_entries()}
    payload = {
        "X": np.asarray(dataset["X"]).tolist(),
        "y": np.asarray(dataset["y"]).tolist(),
        "groups": dataset["groups"],
        "truth_counts": dataset["truth_counts"],
        "times": dataset["times"],
        "baseline": dataset["baseline"],
        "feature_names": dataset["feature_names"],
        "truth": truth,
    }
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # None of these paths exist: reading the cache must never touch videos or labels.
    code = ly.main([
        "--cache", str(cache),
        "--run", str(tmp_path / "missing-run"),
        "--videos", str(tmp_path / "missing-videos"),
        "--labels", str(tmp_path / "missing-labels"),
        "--l2", "1.0", "0.1",
    ])

    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    summary = json.loads(lines[-1])
    assert set(summary) == {"baseline", "logo"}
    assert set(summary["logo"]) == {"1.0", "0.1"}
    assert summary["baseline"]["true_positive"] == 1
