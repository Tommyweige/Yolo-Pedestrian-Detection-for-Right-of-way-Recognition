"""One command, seconds: for every video of a pipeline run, say which rule stage blocks each human label, and how far each
prediction is from the nearest label. Reads datasets/<run>/<stem>/{seg,tracks} and labels/<stem>.jsonl; no detection or GPU.

    python scripts/diagnose_run.py --run datasets/eval-run2 --videos "datasets/來源影片"
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402
import diagnose_misses  # noqa: E402
import yield_image_rule as rule  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def read_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []


def relaxed(video, run, stem, label_s, pad=4.0):
    """Which rule filter, once relaxed, produces a prediction within 1 s of the label? Runs the rule on label +- pad seconds."""
    capture = cv2.VideoCapture(str(video))
    fps, size = capture.get(cv2.CAP_PROP_FPS), (int(capture.get(3)), int(capture.get(4)))
    capture.release()
    mask_at, _ = rule.load_mask_provider(str(run / stem / "seg" / "masks.npz"))
    tracks = json.loads((run / stem / "tracks" / "raw_tracks.json").read_text(encoding="utf-8"))
    frames = range(max(1, round((label_s - pad) * fps)), round((label_s + pad) * fps) + 1)
    found = []
    for name, changes in (("base", {}), ("no_crossing_filter", {"require_crossing": False}), ("min_seconds=0", {"min_seconds": 0.0}),
                          ("both", {"require_crossing": False, "min_seconds": 0.0})):
        params = rule.Params(fps=fps, src_size=size, **changes)
        times = [e.time_s for e in rule.detect(mask_at, tracks, params, frames)]
        if any(abs(t - label_s) <= 1.0 for t in times):
            found.append(name)
    return found[0] if found else "none even with both relaxed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--window", type=float, default=1.0)
    args = parser.parse_args(argv)
    run = Path(args.run)
    summary = {}
    for video in sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS):
        stem = video.stem
        labels_path = Path(args.labels) / f"{stem}.jsonl"
        events = read_jsonl(run / "events" / f"{stem}.jsonl")
        labels = read_jsonl(labels_path)
        print(f"\n== {stem}")
        if labels:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                diagnose_misses.main(["--video", str(video), "--masks", str(run / stem / "seg" / "masks.npz"),
                                      "--tracks", str(run / stem / "tracks" / "raw_tracks.json"),
                                      "--labels", str(labels_path), "--window", str(args.window)])
            for line in buffer.getvalue().splitlines():
                row = json.loads(line)
                near = min((abs(e["time_s"] - row["label_s"]) for e in events), default=None)
                matched = near is not None and near <= 1.0
                blocker = "MATCHED" if matched else row["blocker"]
                summary[blocker] = summary.get(blocker, 0) + 1
                stages = " ".join(f"{k}={row[k]}" for k in ("frames", "crosswalk", "person", "person_at_crosswalk", "vehicle_on", "pair", "deep"))
                print(f"  label {row['label_s']:7.3f}s  {blocker:<22} nearest prediction: {'-' if near is None else f'{near:.2f}s'}  [{stages}]")
                if not matched:
                    print(f"      first relaxation that yields a prediction: {relaxed(video, run, stem, row['label_s'])}")
        else:
            print("  (no human labels)")
        for e in events:
            near = min((abs(l["time_s"] - e["time_s"]) for l in labels), default=None)
            if near is None or near > 1.0:
                print(f"  FALSE ALARM {e['time_s']:7.3f}s  nearest label: {'-' if near is None else f'{near:.2f}s'}")
    print("\nblockers over all labels:", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
