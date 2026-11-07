"""Shared harness: run the rule with a completed crosswalk mask used only for the pedestrian test.
Local only (reads private data). fill(kind, ...) returns the pedestrian mask for one stored frame."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import yield_image_rule as rule
import eval_events as ev
RUN, VIDS, LABELS = ROOT / "datasets/eval-run2", ROOT / "datasets/來源影片", ROOT / "labels"
SHAPE = (360, 640)
_orig_scan = rule.scan_frame


class PedMask(np.ndarray):
    ped = None


def scan_frame_ped(mask, people, vehicles, params):
    """Like rule.scan_frame (image geometry) but the pedestrian state is judged on mask.ped when present."""
    ped = getattr(mask, "ped", None)
    base = np.asarray(mask)
    if ped is None or params.geometry != "image":
        return _orig_scan(base, people, vehicles, params)
    components = rule.crosswalk_components(base, params)
    closed = cv2.morphologyEx(ped.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((params.close_px,) * 2, np.uint8))
    n, lab = cv2.connectedComponents(closed, connectivity=8)
    motos = [b for _, k, b in vehicles if k in params.rider_kinds]
    pairs, deep = [], set()
    for component in components:
        ids = np.unique(lab[component]); ids = ids[ids > 0]
        pc = np.isin(lab, ids) | component
        distance = cv2.distanceTransform((~pc).astype(np.uint8), cv2.DIST_L2, 5)
        depth = cv2.distanceTransform(pc.astype(np.uint8), cv2.DIST_L2, 5)
        standing = [(i, b, rule.person_state(b, pc, distance, depth, params)) for i, b in people if not rule.is_rider(b, motos, params)]
        standing = [s for s in standing if s[2]]
        deep.update(i for i, _, s in standing if s == "on_crosswalk")
        axis = rule.crosswalk_axis(component)
        for vid, kind, vbox in vehicles:
            if not standing or not rule.vehicle_on(vbox, component, params):
                continue
            for pid, pbox, state in standing:
                gap, extent = rule.along_gap(pbox, vbox, axis)
                if gap <= params.lateral_vehicle_widths * extent:
                    pairs.append((vid, pid, state))
    return pairs, deep


rule.scan_frame = scan_frame_ped


def clips():
    for label in sorted(LABELS.glob("*.jsonl")):
        stem = label.stem
        if not (RUN / stem / "seg" / "masks.npz").exists():
            continue
        video = next(p for p in VIDS.iterdir() if p.stem == stem)
        cap = cv2.VideoCapture(str(video)); fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4)); cap.release()
        tracks = json.loads((RUN / stem / "tracks" / "raw_tracks.json").read_text(encoding="utf-8"))
        yield stem, label, fps, w, h, tracks


def load_masks(stem):
    data = np.load(RUN / stem / "seg" / "masks.npz")
    frames = sorted(int(k[1:]) for k in data.files)
    size = SHAPE[0] * SHAPE[1]
    return frames, {f: np.unpackbits(data[f"f{f}"])[:size].astype(bool).reshape(SHAPE) for f in frames}


def box_region(tracks, w, h, dilate=3, kinds=None):
    """frame -> list of boxes in mask pixels (all person/vehicle detections)."""
    sx, sy = SHAPE[1] / w, SHAPE[0] / h
    out = {}
    for t in tracks:
        if kinds and t["kind"] not in kinds:
            continue
        if t["kind"] != "person" and t["kind"] not in rule.VEHICLE_KINDS:
            continue
        for f, *b in t["boxes"]:
            x1, y1, x2, y2 = rule.scaled(b, sx, sy)
            out.setdefault(int(f), []).append((max(0, int(x1) - dilate), max(0, int(y1) - dilate),
                                               min(SHAPE[1], int(np.ceil(x2)) + dilate), min(SHAPE[0], int(np.ceil(y2)) + dilate)))
    return out


def unknown_at(boxes, f):
    u = np.zeros(SHAPE, bool)
    for x1, y1, x2, y2 in boxes.get(f, []):
        u[y1:y2, x1:x2] = True
    return u


import os as _os, json as _json
_HERE = Path(__file__).parent
ORIG_ONLY = _os.environ.get("CW_ORIG") == "1"
_SPANS = _json.loads((_HERE / "replay_spans.json").read_text(encoding="utf-8")) if ORIG_ONLY else {}


def run_variant(make_ped, overrides=None, label=""):
    """make_ped(stem, frames, masks, boxes, tracks) -> {stored frame: ped mask}. Returns totals and per-clip rows."""
    tot, per = [0, 0, 0], {}
    for stem, lab, fps, w, h, tracks in clips():
        frames, masks = load_masks(stem)
        boxes = box_region(tracks, w, h)
        peds = make_ped(stem, frames, masks, boxes, tracks) if make_ped else {}

        def mask_at(frame, _fr=frames):
            i = np.searchsorted(_fr, frame, side="right") - 1
            if i < 0:
                return None
            f = _fr[i]
            m = masks[f].view(PedMask)
            m.ped = peds.get(f)
            return m
        last = max(int(b[0]) for t in tracks for b in t["boxes"])
        params = rule.Params(fps=fps, src_size=(w, h), **(overrides or {}))
        events = [dict(e.to_dict(), kind="suspected_failure_to_yield") for e in rule.detect(mask_at, tracks, params, range(1, last + 1))]
        truth = ev.load(lab)
        if ORIG_ONLY:  # original footage only: drop events in zoomed replays, use labels mapped back to the original
            sp = next((v for k, v in _SPANS.items() if stem.startswith(k)), [])
            events = [e for e in events if not any(a <= e["time_s"] < b for a, b in sp)]
            truth = ev.load(_HERE / "labels_orig" / (stem + ".jsonl"))
        c = ev.counts(events, truth, 1.0).get("suspected_failure_to_yield", {"predicted": 0, "true_positive": 0, "labelled": 0})
        r = [c["predicted"], c["true_positive"], c["labelled"]]
        per[stem] = r
        tot = [a + b for a, b in zip(tot, r)]
    p, tp, l = tot
    print(f"{label:40s} P {p} TP {tp} L {l} F1 {2 * tp / (p + l):.3f}  per-clip {[v for v in per.values()]}", flush=True)
    return tot, per


def temporal_union(k):
    def make(stem, frames, masks, boxes, tracks):
        out = {}
        for i, f in enumerate(frames):
            u = unknown_at(boxes, f)
            if not u.any():
                continue
            acc = np.zeros(SHAPE, bool)
            for j in range(max(0, i - k), min(len(frames), i + k + 1)):
                acc |= masks[frames[j]]
            out[f] = masks[f] | (acc & u)
        return out
    return make


if __name__ == "__main__":
    run_variant(None, label="default (harness, no ped mask)")
    run_variant(lambda *a: {f: m for f, m in zip(a[1], [a[2][f] for f in a[1]])}, label="identity ped mask")
    for k in (2, 5, 10):
        run_variant(temporal_union(k), label=f"temporal union +-{k} stored frames")
