"""Full-video annotation: watch a clip end to end, press a key at every failure to yield, and export complete labels.

The labels in labels/<stem>.jsonl were written "when seen", so a clip's labels are not its full answer and precision
measured against them is not precision. This tool produces a clip's full answer: the annotator watches the whole clip,
marks every incident, and presses "這支標完了" (done). Only done clips are exported, so a clip nobody finished is never
scored (scripts/eval_rule_events.py globs <labels>/*.jsonl, so a clip without a file is simply not in the score).

Inputs come from a newclip run (run.json from scripts/newclip_loop.py): per clip the source video, masks, tracks and
the system's events. The video is only read. At every mark the server records which vehicles are on the crosswalk and
which pedestrians are on or at it at that frame (same geometry as yield_image_rule), so the annotator does not have to
identify the car.

Work state:   <out>/work/<stem>.json   (marks + done flag, saved on every change)
Export:       <out>/labels/<stem>.jsonl (done clips only; the format scripts/eval_events.load reads)

Frame convention: a mark at video frame index k (0-based) is stored as frame k+1 and time_s (k+1)/fps, the same
one-based frame and time the tracks and the rule use, so the label-vs-event residual has no built-in one-frame shift.

Usage:
  python scripts/annotate_full.py --run E:\\...\\dev-run --out E:\\Learning\\yolo-ped-analysis\\full-labels [--port 8765]
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_events as ev  # noqa: E402
import yield_image_rule as r  # noqa: E402

PAGE = Path(__file__).with_name("annotate_full.html")
KIND = "suspected_failure_to_yield"
REQUIRED = ("frame", "kind", "source", "time_s", "video")
KIND_ZH = {"car": "汽車", "motorcycle": "機車", "bus": "公車", "truck": "貨車", "bicycle": "腳踏車"}


def resolve(path, run_dir):
    """run.json stores some paths relative to the repo that wrote it (".amp\\in\\dev-run\\..."): find the file."""
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (run_dir, *run_dir.parents):
        if (base / path).exists():
            return base / path
    return run_dir / path.name


class Clip:
    """One clip: its video, fps, the system's events and, loaded on first use, its tracks and masks."""

    def __init__(self, entry, run_dir):
        self.stem = entry["stem"]
        self.video = Path(entry["video_path"])
        self.video_name = entry["video_name"]
        self.fps = float(entry["fps"])
        self.size = tuple(entry["size"])
        self.frames = int(entry["frames"])
        self.masks_path = Path(entry["masks"])
        self.tracks_path = Path(entry["tracks"])
        events_path = resolve(entry["events_jsonl"], run_dir)
        self.events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()] if events_path.exists() else []
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        with self._lock:
            if self._loaded:
                return
            self.params = r.Params(fps=self.fps, src_size=self.size)
            tracks = json.loads(self.tracks_path.read_text(encoding="utf-8"))
            tracks = r.merge_duplicate_tracks(tracks, self.params)   # same vehicle ids as the rule's events
            self.riders = r.rider_frames(tracks, self.params)
            self.by_frame = {}
            for t in tracks:
                for f, *box in t["boxes"]:
                    self.by_frame.setdefault(int(f), []).append((t["id"], t["kind"], tuple(box)))
            self.mask_at, _ = r.load_mask_provider(str(self.masks_path))
            self._loaded = True

    def scene(self, frame):
        """Vehicles on the crosswalk and pedestrians on/at it at this one-based frame, boxes in source pixels.

        Vehicles are ordered by how close they are to a pedestrian on the crosswalk (the likeliest offender first),
        which is the order the number keys pick them in.
        """
        self._load()
        mask = self.mask_at(frame)
        here = self.by_frame.get(frame, [])
        if mask is None or not here:
            return {"frame": frame, "vehicles": [], "pedestrians": []}
        sx, sy = mask.shape[1] / self.size[0], mask.shape[0] / self.size[1]
        riding = {pid for pid, seats in self.riders.items() if frame in seats}
        people = [(i, b) for i, k, b in here if k == "person" and i not in riding]
        vehicles = [(i, k, b) for i, k, b in here if k in r.VEHICLE_KINDS]
        on_vehicles, peds = {}, {}
        for component in r.crosswalk_components(mask, self.params):
            distance = cv2.distanceTransform((~component).astype("uint8"), cv2.DIST_L2, 5)
            depth = cv2.distanceTransform(component.astype("uint8"), cv2.DIST_L2, 5)
            for pid, box in people:
                state = r.person_state(r.scaled(box, sx, sy), component, distance, depth, self.params)
                if state and peds.get(pid) != "on_crosswalk":
                    peds[pid] = state
            for vid, kind, box in vehicles:
                if r.vehicle_on(r.scaled(box, sx, sy), component, self.params):
                    on_vehicles[vid] = (kind, box)
        boxes = {i: b for i, _, b in here}

        def gap(vbox):
            on = [boxes[p] for p, s in peds.items() if s == "on_crosswalk"] or [boxes[p] for p in peds]
            if not on:
                return float("inf")
            cx = (vbox[0] + vbox[2]) / 2
            return min(abs(cx - (p[0] + p[2]) / 2) + abs(vbox[3] - p[3]) for p in on)

        ordered = sorted(on_vehicles.items(), key=lambda kv: gap(kv[1][1]))
        return {"frame": frame,
                "vehicles": [{"id": vid, "kind": kind, "kind_zh": KIND_ZH.get(kind, kind),
                              "box": [round(v, 1) for v in box]} for vid, (kind, box) in ordered],
                "pedestrians": [{"id": pid, "state": state, "box": [round(v, 1) for v in boxes[pid]]}
                                for pid, state in peds.items()]}

    def scene_near(self, frame, window_s=0.5):
        """The scene at this frame, or, when no vehicle is on the crosswalk there, at the nearest frame within
        `window_s` that has one: a key press lands a few frames off the moment the vehicle is on the stripes."""
        reach = round(window_s * self.fps)
        for offset in sorted(range(-reach, reach + 1), key=abs):
            if 1 <= frame + offset <= self.frames:
                found = self.scene(frame + offset)
                if found["vehicles"]:
                    return {**found, "frame": frame, "vehicle_frame": frame + offset}
        return {**self.scene(frame), "vehicle_frame": None}


def label_rows(clip, work):
    """The export rows of one done clip, in the labels format (eval_events.load) plus who was on the crosswalk."""
    rows = []
    for mark in sorted(work["marks"], key=lambda m: m["frame"]):
        frame = int(mark["frame"])
        row = {"frame": frame, "kind": KIND, "source": "manual", "time_s": round(frame / clip.fps, 3),
               "video": clip.video_name, "annotation": "full_pass"}
        if mark.get("vehicle"):
            row["vehicle_id"], row["vehicle_kind"] = mark["vehicle"]["id"], mark["vehicle"]["kind"]
        if mark.get("pedestrians"):
            row["pedestrian_ids"] = mark["pedestrians"]
        rows.append(row)
    return rows


def check_export(path, rows):
    """Read the written file back the way the scorer does and refuse it unless it round-trips."""
    loaded = ev.load(path)
    if len(loaded) != len(rows):
        raise ValueError(f"{path.name}: wrote {len(rows)} rows, read back {len(loaded)}")
    for row in loaded:
        missing = [k for k in REQUIRED if k not in row]
        if missing or row["kind"] != KIND or not isinstance(row["time_s"], (int, float)):
            raise ValueError(f"{path.name}: bad row {row} (missing {missing})")
    return loaded


class Store:
    def __init__(self, out):
        self.work_dir = out / "work"
        self.labels_dir = out / "labels"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def get(self, stem):
        path = self.work_dir / f"{stem}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"stem": stem, "done": False, "marks": [], "updated_at": None}

    def put(self, stem, work):
        work = {"stem": stem, "done": bool(work.get("done")), "marks": work.get("marks", []),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        path = self.work_dir / f"{stem}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(work, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return work

    def export(self, clips):
        """Write <stem>.jsonl for every done clip and remove it for every clip that is not done."""
        result = []
        for clip in clips.values():
            work = self.get(clip.stem)
            path = self.labels_dir / f"{clip.stem}.jsonl"
            if not work["done"]:
                if path.exists():
                    path.unlink()
                result.append({"stem": clip.stem, "done": False, "exported": False})
                continue
            rows = label_rows(clip, work)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            check_export(path, rows)
            result.append({"stem": clip.stem, "done": True, "exported": True, "rows": len(rows), "path": str(path)})
        return result


def make_handler(clips, store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def clip(self, query):
            stem = query.get("stem", [""])[0]
            if stem not in clips:
                self.send_json({"error": f"unknown clip {stem}"}, 404)
                return None
            return clips[stem]

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/clips":
                self.send_json([{"stem": c.stem, "fps": c.fps, "frames": c.frames, "duration": c.frames / c.fps,
                                 "size": c.size, "events": len(c.events), "done": store.get(c.stem)["done"],
                                 "marks": len(store.get(c.stem)["marks"])} for c in clips.values()])
            elif url.path == "/api/clip" and (c := self.clip(query)):
                self.send_json({"stem": c.stem, "fps": c.fps, "frames": c.frames, "size": c.size,
                                "events": [{"time_s": e["time_s"], "frame": e["frame"], "vehicle_id": e["vehicle_id"],
                                            "vehicle_kind": KIND_ZH.get(e["vehicle_kind"], e["vehicle_kind"]),
                                            "pedestrian_id": e["pedestrian_id"]} for e in c.events],
                                "work": store.get(c.stem)})
            elif url.path == "/api/scene" and (c := self.clip(query)):
                frame = int(query.get("frame", ["1"])[0])
                self.send_json(c.scene_near(frame) if query.get("near") else c.scene(frame))
            elif url.path == "/video" and (c := self.clip(query)):
                self.send_video(c.video)
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if url.path == "/api/work" and (c := self.clip(query)):
                self.send_json(store.put(c.stem, body))
            elif url.path == "/api/export":
                try:
                    self.send_json({"labels_dir": str(store.labels_dir), "clips": store.export(clips)})
                except ValueError as error:
                    self.send_json({"error": str(error)}, 500)
            else:
                self.send_json({"error": "not found"}, 404)

        def send_video(self, path):
            """The clip with Range support (the browser cannot seek without it). Read only."""
            size = path.stat().st_size
            start, end = 0, size - 1
            ranged = self.headers.get("Range", "").startswith("bytes=")
            if ranged:
                a, _, b = self.headers["Range"][6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = min(int(b), size - 1) if a and b else end
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass

    return Handler


def load_clips(run_dir):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {entry["stem"]: Clip(entry, run_dir) for entry in run["clips"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="newclip run folder with run.json (videos, masks, tracks, events)")
    parser.add_argument("--out", required=True, help="work/ and labels/ are kept here")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    clips = load_clips(Path(args.run))
    store = Store(Path(args.out))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(clips, store))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(clips)} clips, work in {store.work_dir}, export to {store.labels_dir}")
    print(f"open {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
