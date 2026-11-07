import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import eval_public_detection as epd  # noqa: E402


VOC = """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <object>
    <name>pedestrian</name>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>42</ymax></bndbox>
  </object>
  <object>
    <name>traffic light</name>
    <bndbox><xmin>5</xmin><ymin>6</ymin><xmax>7</xmax><ymax>8</ymax></bndbox>
  </object>
</annotation>
"""


def test_parse_voc_keeps_only_mapped_classes():
    objects = epd.parse_voc(VOC)
    assert objects == [{"group": "person", "box": [1.0, 2.0, 30.0, 42.0], "occluded": None}]


def test_parse_bdd_item_maps_gts_and_collects_rider_ignore_boxes():
    item = {
        "name": "scene.jpg",
        "attributes": {"timeofday": "day", "weather": "clear", "scene": "city street"},
        "labels": [
            {"category": "person", "attributes": {"occluded": False, "truncated": False},
             "box2d": {"x1": 1, "y1": 2, "x2": 11, "y2": 22}},
            {"category": "rider", "attributes": {"occluded": False, "truncated": False},
             "box2d": {"x1": 20, "y1": 30, "x2": 40, "y2": 60}},
            {"category": "car", "attributes": {"occluded": True, "truncated": True},
             "box2d": {"x1": 50, "y1": 60, "x2": 90, "y2": 100}},
            {"category": "traffic sign", "attributes": {"occluded": False, "truncated": False},
             "box2d": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}},
            {"category": "bus", "attributes": {"occluded": False, "truncated": False}},
        ],
    }
    attrs, gts, ignore = epd.parse_bdd_item(item)
    assert attrs == {"timeofday": "day", "weather": "clear", "scene": "city street"}
    assert gts == [
        {"group": "person", "box": [1.0, 2.0, 11.0, 22.0], "occluded": False},
        {"group": "vehicle", "box": [50.0, 60.0, 90.0, 100.0], "occluded": True},
    ]
    assert ignore == [[20.0, 30.0, 40.0, 60.0]]


def test_match_is_greedy_per_group_and_suppresses_person_on_ignore_boxes():
    gts = [
        {"group": "person", "box": [0, 0, 10, 10]},
        {"group": "vehicle", "box": [20, 20, 40, 40]},
    ]
    preds = [
        {"group": "person", "box": [0, 0, 10, 10], "conf": 0.9},
        {"group": "vehicle", "box": [30, 30, 50, 50], "conf": 0.8},
        {"group": "person", "box": [50, 50, 60, 60], "conf": 0.7},
    ]
    result = epd.match(gts, preds, ignore_boxes=[[50, 50, 60, 60]])
    assert result["gt_hit"] == [True, False]
    assert result["fp"]["person"] == 0
    assert result["fp"]["vehicle"] == 1


def test_match_does_not_assign_one_gt_to_two_predictions():
    gts = [{"group": "person", "box": [0, 0, 10, 10]}]
    preds = [
        {"group": "person", "box": [0, 0, 10, 10], "conf": 0.9},
        {"group": "person", "box": [0, 0, 10, 10], "conf": 0.8},
    ]
    result = epd.match(gts, preds)
    assert result["gt_hit"] == [True]
    assert result["fp"]["person"] == 1


def test_size_bucket_boundaries():
    assert epd.size_bucket([0, 0, 10, 4.9], 100) == "small"
    assert epd.size_bucket([0, 0, 10, 5], 100) == "medium"
    assert epd.size_bucket([0, 0, 10, 15], 100) == "large"


def test_summarize_computes_rates_and_skips_none_conditions():
    records = [
        {"dataset": "d", "group": "person", "hit": True, "size": "small", "timeofday": "day", "occluded": False},
        {"dataset": "d", "group": "person", "hit": False, "size": "large", "timeofday": "night", "occluded": None},
        {"dataset": "d", "group": "vehicle", "hit": True, "size": "small", "timeofday": "day", "occluded": False},
        {"kind": "fp", "dataset": "d", "group": "person", "fp": 1, "timeofday": "day"},
        {"kind": "fp", "dataset": "d", "group": "vehicle", "fp": 0, "timeofday": "day"},
    ]
    summary = epd.summarize(records)
    assert summary["d"]["person"] == {"gt": 2, "hit": 1, "recall": 0.5, "fp": 1, "precision": 0.5}
    assert summary["d"]["vehicle"] == {"gt": 1, "hit": 1, "recall": 1.0, "fp": 0, "precision": 1.0}
    assert summary["by_size"]["person"]["small"] == {"gt": 1, "hit": 1, "recall": 1.0}
    assert summary["by_size"]["person"]["large"] == {"gt": 1, "hit": 0, "recall": 0.0}
    assert summary["by_timeofday"]["person"]["day"] == {"gt": 1, "hit": 1, "recall": 1.0}
    assert summary["by_timeofday"]["person"]["night"] == {"gt": 1, "hit": 0, "recall": 0.0}
    assert summary["by_occluded"]["person"][False] == {"gt": 1, "hit": 1, "recall": 1.0}
    assert None not in summary["by_occluded"]["person"]


def _write_ivs(root):
    annotations = root / "Annotations"
    images = root / "JPEGImages"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        assert cv2.imwrite(str(images / name), np.zeros((100, 100, 3), np.uint8))
        (annotations / f"{Path(name).stem}.xml").write_text(VOC, encoding="utf-8")


def test_main_uses_cache_without_importing_ultralytics(tmp_path, capsys, monkeypatch):
    # Other tests in the same pytest process may already have imported torch, so checking sys.modules afterwards
    # says nothing; instead make both imports fail for the duration of this call.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    ivs = tmp_path / "ivs"
    _write_ivs(ivs)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "ivs": {
            "a.jpg": [{"group": "person", "box": [1, 2, 30, 42], "conf": 0.9}],
            "b.jpg": [{"group": "person", "box": [1, 2, 30, 42], "conf": 0.9}],
        }
    }), encoding="utf-8")
    output = tmp_path / "summary.json"
    code = epd.main(["--ivs", str(ivs), "--cache", str(cache), "--output", str(output)])
    assert code == 0
    assert output.exists()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
