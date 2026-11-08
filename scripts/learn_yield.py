"""Loose candidate generation, feature learning and leave-one-video-out scoring for yield events.

Candidate generation re-runs yield_image_rule with loosened parameters so that events the
current rule discards are kept for a learned scorer. Features come from yield_features and
the scorer/validation from yield_learn. Everything is deterministic and needs no GPU.

Run as a CLI from the repository root (paths are relative to it):

  python scripts/learn_yield.py --cache datasets/learn-cache.json --model-out datasets/yield-model.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import clip_artifacts  # noqa: E402
import eval_events  # noqa: E402
import yield_features  # noqa: E402
import yield_image_rule  # noqa: E402
import yield_learn  # noqa: E402

LOOSE = dict(require_crossing=False, lateral_vehicle_widths=3.0, min_seconds=0.05, edge_margin_person_heights=1.5)

KIND = "suspected_failure_to_yield"


def _frames(tracks, last_frame):
    if last_frame is None:
        last_frame = max((int(box[0]) for track in tracks for box in track["boxes"]), default=0)
    return range(1, int(last_frame) + 1)


def _run(mask_at, tracks, params, last_frame):
    return [event.to_dict() for event in yield_image_rule.detect(mask_at, tracks, params, _frames(tracks, last_frame))]


def candidates(mask_at, tracks, fps, src_size, last_frame=None):
    """Loose candidates as event dicts; last_frame None means the tracks' largest frame."""
    params = yield_image_rule.Params(fps=fps, src_size=src_size, **LOOSE)
    return _run(mask_at, tracks, params, last_frame)


def rule_events(mask_at, tracks, fps, src_size, last_frame=None):
    """The current rule's output (the baseline) as event dicts."""
    params = yield_image_rule.Params(fps=fps, src_size=src_size)
    return _run(mask_at, tracks, params, last_frame)


def _people_by_frame(index):
    counts = {}
    for track in index.values():
        if track["kind"] == "person":
            for frame in track["frames"]:
                frame = int(frame)
                counts[frame] = counts.get(frame, 0) + 1
    return counts


def build_dataset(entries, tolerance=1.0):
    """Turn entries ({"name","mask_at","tracks","fps","src_size","truth_times"}) into a dataset."""
    rows, labels, groups, times = [], [], [], []
    truth_counts, baseline = {}, {}
    for entry in entries:
        name = entry["name"]
        fps = float(entry["fps"])
        src_size = entry["src_size"]
        tracks = entry["tracks"]
        truth_times = list(entry["truth_times"])

        found = candidates(entry["mask_at"], tracks, fps, src_size)
        index = yield_features.index_tracks(tracks)
        people_by_frame = _people_by_frame(index)
        for candidate in found:
            features = yield_features.candidate_features(index, candidate, fps, src_size, people_by_frame)
            rows.append([features[key] for key in yield_features.FEATURE_NAMES])

        candidate_times = [candidate["time_s"] for candidate in found]
        times.extend(float(value) for value in candidate_times)
        labels.extend(yield_learn.label_candidates(candidate_times, truth_times, tolerance).tolist())
        groups.extend([name] * len(found))
        truth_counts[name] = len(truth_times)
        baseline[name] = rule_events(entry["mask_at"], tracks, fps, src_size)

    if rows:
        X = np.asarray(rows, dtype=float)
    else:
        X = np.zeros((0, len(yield_features.FEATURE_NAMES)), dtype=float)
    y = np.asarray(labels, dtype=int) if labels else np.zeros(0, dtype=int)

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "truth_counts": truth_counts,
        "times": times,
        "baseline": baseline,
        "feature_names": yield_features.FEATURE_NAMES,
    }


def baseline_score(dataset, truth, tolerance=1.0):
    """Event-level baseline score, treating every baseline event as a suspected failure to yield."""
    names = list(truth)
    for name in dataset.get("baseline", {}):
        if name not in names:
            names.append(name)

    pairs = []
    for name in names:
        predicted = [{**event, "kind": KIND} for event in dataset.get("baseline", {}).get(name, [])]
        expected = [{"kind": KIND, "time_s": float(value)} for value in truth.get(name, [])]
        pairs.append((predicted, expected))
    return eval_events.aggregate(pairs, tolerance).get(KIND, {})


def load_entries(run_dir, videos_dir, labels_dir):
    """Disk-backed entries for every video that has labels, masks and tracks."""
    run_dir, videos_dir, labels_dir = Path(run_dir), Path(videos_dir), Path(labels_dir)
    entries = []
    for video in sorted(videos_dir.iterdir(), key=lambda path: path.name):
        if video.suffix.lower() not in clip_artifacts.VIDEO_SUFFIXES:
            continue
        stem = video.stem
        label_path = labels_dir / f"{stem}.jsonl"
        mask_path = run_dir / stem / clip_artifacts.MASKS
        tracks_path = run_dir / stem / clip_artifacts.TRACKS
        if not (label_path.exists() and mask_path.exists() and tracks_path.exists()):
            continue

        info = clip_artifacts.video_info(video)
        mask_at, _ = clip_artifacts.load_masks(mask_path)
        tracks = clip_artifacts.load_tracks(tracks_path)
        truth_times = [float(row["time_s"]) for row in eval_events.load(label_path) if row["kind"] == KIND]
        entries.append({
            "name": stem,
            "mask_at": mask_at,
            "tracks": tracks,
            "fps": info.fps,
            "src_size": info.size,
            "truth_times": truth_times,
        })
    return entries


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _cache_payload(dataset, truth):
    return {
        "X": [[float(value) for value in row] for row in dataset["X"]],
        "y": [int(value) for value in dataset["y"]],
        "groups": list(dataset["groups"]),
        "truth_counts": {name: int(count) for name, count in dataset["truth_counts"].items()},
        "times": [float(value) for value in dataset["times"]],
        "baseline": dataset["baseline"],
        "feature_names": list(dataset["feature_names"]),
        "truth": {name: [float(value) for value in values] for name, values in truth.items()},
    }


def _dataset_from_cache(payload):
    feature_names = list(payload.get("feature_names", yield_features.FEATURE_NAMES))
    X = np.asarray(payload.get("X", []), dtype=float)
    if X.size == 0:
        X = np.zeros((0, len(feature_names)), dtype=float)
    return {
        "X": X,
        "y": np.asarray(payload.get("y", []), dtype=int),
        "groups": list(payload.get("groups", [])),
        "truth_counts": {name: int(count) for name, count in payload.get("truth_counts", {}).items()},
        "times": [float(value) for value in payload.get("times", [])],
        "baseline": payload.get("baseline", {}),
        "feature_names": feature_names,
    }


def _format(values):
    def number(value):
        return "None" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))

    fields = (("predicted", "predicted"), ("labelled", "labelled"), ("true_positive", "TP"),
              ("precision", "precision"), ("recall", "recall"), ("f1", "f1"))
    return "  ".join(f"{label}={number(values.get(key))}" for key, label in fields)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="datasets/eval-run2", help="folder holding <stem>/seg/masks.npz and <stem>/tracks/raw_tracks.json")
    parser.add_argument("--videos", default="datasets/來源影片")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--l2", type=float, nargs="+", default=[1.0], help="one or more L2 strengths, each validated with LOGO")
    parser.add_argument("--cache", help="dataset cache JSON: read when present, otherwise build and write it")
    parser.add_argument("--model-out", help="write the model trained on all candidates as JSON")
    args = parser.parse_args(argv)

    cache_path = _repo_path(args.cache) if args.cache else None
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        dataset = _dataset_from_cache(payload)
        truth = {name: [float(value) for value in times] for name, times in payload.get("truth", {}).items()}
    else:
        entries = load_entries(_repo_path(args.run), _repo_path(args.videos), _repo_path(args.labels))
        dataset = build_dataset(entries, args.tolerance)
        truth = {entry["name"]: list(entry["truth_times"]) for entry in entries}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(_cache_payload(dataset, truth), ensure_ascii=False), encoding="utf-8")

    baseline = baseline_score(dataset, truth, args.tolerance)
    print("baseline: " + _format(baseline))

    positives = int(np.sum(dataset["y"]))
    print(f"candidates={len(dataset['times'])}  positives={positives}  covered_labels={positives}")

    logo_summary = {}
    for l2 in args.l2:
        result = yield_learn.leave_one_group_out(
            dataset["X"], dataset["y"], dataset["groups"], dataset["truth_counts"], l2=l2)
        total = result["total"]
        logo_summary[str(l2)] = total
        print(f"l2={l2} total: " + _format(total))
        for group, values in result["per_group"].items():
            print(f"  {group}: predicted={values['predicted']} TP={values['true_positive']} labelled={values['labelled']}")

    if args.model_out:
        if len(dataset["times"]):
            model = yield_learn.train(
                dataset["X"], dataset["y"], dataset["groups"], dataset["truth_counts"],
                l2=args.l2[0], feature_names=dataset["feature_names"])
            model_path = _repo_path(args.model_out)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text(json.dumps(model.to_dict(), ensure_ascii=False), encoding="utf-8")
        else:
            print("no candidates: model not written", file=sys.stderr)

    print(json.dumps({"baseline": baseline, "logo": logo_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
