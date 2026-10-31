"""Build the sample table and the clips: candidates (positives, hard negatives), vehicle-pass negatives, shifted windows.

Samples, all grouped by clip (`stem`) so no clip is ever on both sides of a split:
- candidate: every generator candidate, labelled by common.ground_truth (1 / 0 / None = left out). Only these are scored.
- pass: auto-cut negatives. A vehicle crossing a crosswalk (the rule's vehicle_on, no pedestrian condition), one window
  at the middle of its time on the crosswalk, with the pedestrian nearest to it if any. Taken only when nothing labelled
  or judged-not-C is within NEGATIVE_MARGIN_S and the vehicle has no candidate within the tolerance, so it is a vehicle
  pass the generator did not flag. Training only.
- shift: each positive candidate again at +-0.4 s (one frame step of the window). Training only.

Writes OUT/samples.json and OUT/clips_<n>.npz (uint8 9x224x224x4). The clips are video frames: keep OUT out of git.
Usage: python dataset.py --candidates DIR --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clips  # noqa: E402
from common import (NEGATIVE_MARGIN_S, RUN2, TOLERANCE_S, ground_truth, load_labels, load_verdicts, rule,  # noqa: E402
                    stems, video_path)


def vehicle_passes(stem, params):
    """[(vehicle_id, middle frame on the crosswalk, nearest person id or None)] for every vehicle seen on it."""
    tracks = rule.merge_duplicate_tracks(
        json.loads((RUN2 / stem / "tracks" / "raw_tracks.json").read_text(encoding="utf-8")), params)
    mask_at, mask_frames = rule.load_mask_provider(str(RUN2 / stem / "seg" / "masks.npz"))
    by_frame = {}
    for t in tracks:
        for f, *b in t["boxes"]:
            by_frame.setdefault(int(f), []).append((t["id"], t["kind"], tuple(b)))
    on = {}
    for f in mask_frames:
        mask = mask_at(f)
        sx, sy = mask.shape[1] / params.src_size[0], mask.shape[0] / params.src_size[1]
        components = rule.crosswalk_components(mask, params)
        for vid, kind, box in by_frame.get(f, []):
            if kind in rule.VEHICLE_KINDS and any(rule.vehicle_on(rule.scaled(box, sx, sy), c, params) for c in components):
                on.setdefault(vid, []).append(f)
    passes = []
    for vid, frames in on.items():
        if len(frames) < 3:  # under ~0.3 s on the crosswalk: a box flicker, not a pass
            continue
        middle = frames[len(frames) // 2]
        vbox = next(b for i, _, b in by_frame[middle] if i == vid)
        people = [(abs((b[0] + b[2]) / 2 - (vbox[0] + vbox[2]) / 2), i) for i, k, b in by_frame[middle] if k == "person"]
        passes.append((vid, middle, min(people)[1] if people else None))
    return passes


def build(candidates_dir, out):
    verdicts = load_verdicts()
    samples = []
    for index, stem in enumerate(stems()):
        capture = cv2.VideoCapture(str(video_path(stem)))
        fps, size = capture.get(cv2.CAP_PROP_FPS), (int(capture.get(3)), int(capture.get(4)))
        capture.release()
        params = rule.Params(fps=fps, src_size=size)
        labels = load_labels(stem)
        judged = [v for v in verdicts if v["stem"] == stem]
        cands = [json.loads(line) for line in (Path(candidates_dir) / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = []
        for k, c in enumerate(cands):
            y = ground_truth(c, labels, judged)
            rows.append({"id": f"{stem}|c{k}", "stem": stem, "role": "candidate", "y": y, "fps": fps, **c})
            if y == 1:
                for d in (-1, 1):
                    frame = c["frame"] + round(d * clips.HALF_S / 4 * fps)
                    rows.append({"id": f"{stem}|c{k}s{d}", "stem": stem, "role": "shift", "y": 1, "fps": fps,
                                 "frame": frame, "time_s": round(frame / fps, 3), "vehicle_id": c["vehicle_id"],
                                 "pedestrian_id": c["pedestrian_id"]})
        for vid, frame, pid in vehicle_passes(stem, params):
            t = frame / fps
            if any(abs(lab["time_s"] - t) <= NEGATIVE_MARGIN_S for lab in labels) \
                    or any(abs(v["time_s"] - t) <= NEGATIVE_MARGIN_S and v["verdict"] != "C" for v in judged) \
                    or any(c["vehicle_id"] == vid and abs(c["time_s"] - t) <= TOLERANCE_S for c in cands):
                continue
            rows.append({"id": f"{stem}|p{vid}", "stem": stem, "role": "pass", "y": 0, "fps": fps, "frame": frame,
                         "time_s": round(t, 3), "vehicle_id": vid, "pedestrian_id": pid})
        arrays = clips.extract(stem, rows, params)
        for n, r in enumerate(rows):
            r["clip_file"], r["key"] = f"clips_{index}.npz", f"s{n}"
        np.savez_compressed(Path(out) / f"clips_{index}.npz", **{r["key"]: arrays[r["id"]] for r in rows})
        samples += rows
        count = lambda role, y: sum(1 for r in rows if r["role"] == role and r["y"] == y)  # noqa: E731
        print(f"{stem[:16]} candidates +{count('candidate', 1)} -{count('candidate', 0)} ?{count('candidate', None)} "
              f"passes -{count('pass', 0)} shifts +{count('shift', 1)}", flush=True)
    (Path(out) / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    build(args.candidates, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
