"""Re-run yield_image_rule on already-computed masks/tracks and score it against human labels (rules only, no GPU).

For every labelled video: events = detect(masks, tracks) with each --geometry, then event-level precision/recall.
False alarms are split into `rejected` (a human reviewed that candidate and said no: certain false alarm) and
`unreviewed` (nobody looked: may be a real event the labeller missed). Read the numbers with that in mind.
Everything here is in-sample until clips nobody tuned on are labelled.

  python scripts/eval_rule.py --run datasets/eval-run2 --videos datasets/來源影片 --geometry image crosswalk
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import clip_artifacts  # noqa: E402
import eval_events as ev  # noqa: E402
import yield_image_rule as rule  # noqa: E402

KIND = "suspected_failure_to_yield"


def run_one(video, run_dir, geometry, include_waiting=False, overrides=None):
    info = clip_artifacts.video_info(video)
    params = rule.Params(fps=info.fps, src_size=info.size, require_crossing=not include_waiting, geometry=geometry,
                         **(overrides or {}))
    tracks = clip_artifacts.load_tracks(run_dir / video.stem / clip_artifacts.TRACKS)
    return [e.to_dict() for e in rule.run_on_clip(run_dir / video.stem / clip_artifacts.MASKS, tracks, params)]


def rejected_times(labels_dir, stem):
    path = labels_dir / f"{stem}.review.json"
    if not path.exists():
        return []
    return [r["time_s"] for r in json.loads(path.read_text(encoding="utf-8")) if r.get("decision") == "rejected"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="datasets/eval-run2", help="folder holding <stem>/seg/masks.npz and <stem>/tracks/raw_tracks.json")
    parser.add_argument("--videos", default="datasets/來源影片")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--geometry", nargs="+", default=["image"], choices=("image", "crosswalk"))
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--include-waiting", action="store_true")
    parser.add_argument("--param", nargs="*", default=[], metavar="NAME=VALUE", help="override yield_image_rule.Params fields (ablations)")
    parser.add_argument("--save-events", help="optional folder to write each geometry's events as <geometry>/<stem>.jsonl")
    args = parser.parse_args(argv)
    run_dir, labels_dir = ROOT / args.run, ROOT / args.labels
    overrides = {}
    for item in args.param:
        name, value = item.split("=", 1)
        overrides[name] = type(getattr(rule.Params(), name))(json.loads(value))
    videos = sorted(p for p in (ROOT / args.videos).iterdir() if p.suffix.lower() in clip_artifacts.VIDEO_SUFFIXES)
    for geometry in args.geometry:
        pairs, lines = [], []
        rejected_fa = unreviewed_fa = 0
        for video in videos:
            label_path = labels_dir / f"{video.stem}.jsonl"
            if not label_path.exists() or not (run_dir / video.stem / clip_artifacts.MASKS).exists():
                continue
            predicted = [{**e, "kind": ev.ALIASES.get(e["kind"], e["kind"])} for e in run_one(video, run_dir, geometry, args.include_waiting, overrides)]
            truth = ev.load(label_path)
            if args.save_events:
                out = ROOT / args.save_events / geometry
                out.mkdir(parents=True, exist_ok=True)
                clip_artifacts.write_events(out / f"{video.stem}.jsonl", predicted)
            pairs.append((predicted, truth))
            c = ev.counts(predicted, truth, args.tolerance).get(KIND, {"predicted": 0, "labelled": 0, "true_positive": 0})
            bad = rejected_times(labels_dir, video.stem)
            matched = {t["time_s"] for t in truth}
            fa_times = [p["time_s"] for p in predicted if all(abs(p["time_s"] - m) > args.tolerance for m in matched)]
            rej = sum(1 for t in fa_times if any(abs(t - b) <= args.tolerance for b in bad))
            rejected_fa += rej
            unreviewed_fa += max(0, c["predicted"] - c["true_positive"]) - rej
            lines.append(f"  {video.stem[:16]:<18} predicted {c['predicted']:>2}  labelled {c['labelled']:>2}  hit {c['true_positive']:>2}")
        total = ev.aggregate(pairs, args.tolerance).get(KIND, {})
        print(f"== geometry={geometry}  videos={len(pairs)}  tolerance={args.tolerance}s")
        print("\n".join(lines))
        print("  TOTAL " + "  ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in total.items()))
        print(f"  false alarms: {rejected_fa} human-rejected, {unreviewed_fa} never reviewed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
