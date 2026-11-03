"""Export contract of scripts/annotate_full.py: only clips marked done are written, in the format the scorer reads."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import annotate_full as a  # noqa: E402
import eval_events as ev  # noqa: E402


def clip(stem, fps):
    return SimpleNamespace(stem=stem, fps=fps, video_name=stem + ".mp4")


def test_only_done_clips_are_exported_and_the_scorer_reads_them(tmp_path):
    store = a.Store(tmp_path)
    clips = {"done": clip("done", 59.94), "open": clip("open", 30.0)}
    # marks out of order; the second has no vehicle (nothing identified on the crosswalk)
    store.put("done", {"done": True, "marks": [
        {"frame": 300, "vehicle": {"id": 7, "kind": "bus"}, "pedestrians": [3]},
        {"frame": 120, "vehicle": None, "pedestrians": []}]})
    store.put("open", {"done": False, "marks": [{"frame": 10}]})

    result = {row["stem"]: row for row in store.export(clips)}

    assert result["done"]["exported"] and not result["open"]["exported"]
    assert not (tmp_path / "labels" / "open.jsonl").exists()
    rows = ev.load(tmp_path / "labels" / "done.jsonl")
    assert [row["frame"] for row in rows] == [120, 300]
    assert [row["time_s"] for row in rows] == [round(120 / 59.94, 3), round(300 / 59.94, 3)]   # 2.002, 5.005
    assert all(row["kind"] == "suspected_failure_to_yield" and row["source"] == "manual" for row in rows)
    assert rows[1]["vehicle_id"] == 7 and "vehicle_id" not in rows[0]


def test_unmarking_a_clip_removes_its_export(tmp_path):
    store = a.Store(tmp_path)
    clips = {"c": clip("c", 25.0)}
    store.put("c", {"done": True, "marks": [{"frame": 50}]})
    store.export(clips)
    assert (tmp_path / "labels" / "c.jsonl").exists()
    store.put("c", {"done": False, "marks": [{"frame": 50}]})
    store.export(clips)
    assert not (tmp_path / "labels" / "c.jsonl").exists()   # an unfinished clip must not be scored
