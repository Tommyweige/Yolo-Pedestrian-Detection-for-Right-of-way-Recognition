"""Tests for trajectory features of yield-to-pedestrian candidate events."""
import math

import pytest

import yield_features as yf

FPS = 30.0
SRC = (1000.0, 1000.0)
EVENT_FRAME = 30

EXPECTED_FEATURES = (
    "is_car,is_motorcycle,is_bus,is_truck,ped_on,entered,v_h,v_w,v_bottom,v_cx,"
    "p_h,p_bottom,rel_dx,rel_dy,v_speed,v_grow,v_dy,p_speed,v_track_s,p_track_s,n_people"
).split(",")


def make_track(track_id, kind, boxes):
    return {"id": track_id, "kind": kind, "boxes": boxes}


def moving_vehicle(track_id=1, kind="car", frames=range(0, 61)):
    """Car 100 px wide driving right at 10 px per frame."""
    return make_track(track_id, kind, [[f, 300 + 10 * f, 400, 400 + 10 * f, 500] for f in frames])


def grown_vehicle(track_id=1, frames=range(0, 61)):
    """Car whose box height grows by 1% every frame."""
    return make_track(track_id, "car", [[f, 300, 0, 400, 100 * 1.01 ** f] for f in frames])


def static_vehicle(track_id=1):
    return make_track(track_id, "car", [[f, 400, 400, 500, 500] for f in range(0, 61)])


def static_pedestrian(track_id=2):
    return make_track(track_id, "person", [[f, 600, 450, 660, 550] for f in range(0, 61)])


def make_event(vehicle_id=1, pedestrian_id=2, frame=EVENT_FRAME, vehicle_kind="car",
               pedestrian_state="on_crosswalk", entered=True):
    return {
        "frame": frame,
        "vehicle_id": vehicle_id,
        "pedestrian_id": pedestrian_id,
        "vehicle_kind": vehicle_kind,
        "pedestrian_state": pedestrian_state,
        "pedestrian_entered_crosswalk": entered,
    }


def features(tracks, event=None, people_by_frame=None, src_size=SRC):
    index = yf.index_tracks(tracks)
    return yf.candidate_features(index, event or make_event(), FPS, src_size, people_by_frame)


def test_feature_names_order_and_result_keys():
    assert len(yf.FEATURE_NAMES) == 21
    assert yf.FEATURE_NAMES == EXPECTED_FEATURES

    result = features([moving_vehicle(), static_pedestrian()])
    assert list(result) == yf.FEATURE_NAMES
    assert all(isinstance(value, float) for value in result.values())


def test_v_speed_of_car_moving_ten_px_per_frame():
    result = features([moving_vehicle(), static_pedestrian()])
    assert result["v_speed"] == pytest.approx(10 * FPS / 100, abs=1e-6)


def test_static_car_has_zero_motion_features():
    result = features([static_vehicle(), static_pedestrian()])
    assert result["v_speed"] == 0.0
    assert result["v_grow"] == 0.0
    assert result["v_dy"] == 0.0


def test_v_grow_of_vehicle_growing_one_percent_per_frame():
    result = features([grown_vehicle(), static_pedestrian()])
    assert result["v_grow"] == pytest.approx(FPS * math.log(1.01), rel=1e-3)


def test_single_frame_tracks_have_zero_motion_features():
    tracks = [
        make_track(1, "car", [[EVENT_FRAME, 400, 400, 500, 500]]),
        make_track(2, "person", [[EVENT_FRAME, 600, 450, 660, 550]]),
    ]
    result = features(tracks)
    assert result["v_speed"] == 0.0
    assert result["v_grow"] == 0.0
    assert result["v_dy"] == 0.0
    assert result["p_speed"] == 0.0


def test_missing_event_frame_uses_nearest_vehicle_box():
    frames = [f for f in range(0, 61) if f not in (30, 31)]
    boxes = [[f, 300 + 10 * f, 480 + f - 100, 400 + 10 * f, 480 + f] for f in frames]
    tracks = [make_track(1, "car", boxes), static_pedestrian()]
    result = features(tracks)
    assert result["v_bottom"] == pytest.approx((480 + 29) / SRC[1], abs=1e-9)


@pytest.mark.parametrize("kind", ["car", "motorcycle", "bus", "truck"])
def test_vehicle_kind_one_hot(kind):
    result = features(
        [moving_vehicle(kind=kind), static_pedestrian()],
        make_event(vehicle_kind=kind),
    )
    for name in ("is_car", "is_motorcycle", "is_bus", "is_truck"):
        assert result[name] == (1.0 if name == "is_" + kind else 0.0)


def test_ped_on_reflects_pedestrian_state():
    tracks = [moving_vehicle(), static_pedestrian()]
    assert features(tracks, make_event(pedestrian_state="on_crosswalk"))["ped_on"] == 1.0
    assert features(tracks, make_event(pedestrian_state="waiting_at_edge"))["ped_on"] == 0.0


def test_entered_reflects_event_flag():
    tracks = [moving_vehicle(), static_pedestrian()]
    assert features(tracks, make_event(entered=True))["entered"] == 1.0
    assert features(tracks, make_event(entered=False))["entered"] == 0.0


def test_rel_dx_is_offset_between_pedestrian_and_vehicle_centre():
    # Vehicle x centre 450, width 100; pedestrian bottom centre x = 630.
    result = features([static_vehicle(), static_pedestrian()])
    assert result["rel_dx"] == pytest.approx(abs(630 - 450) / 100, abs=1e-9)


def test_rel_dy_is_signed_bottom_offset():
    result = features([static_vehicle(), static_pedestrian()])
    assert result["rel_dy"] == pytest.approx((550 - 500) / 100, abs=1e-9)


def test_n_people_counts_person_boxes_at_event_frame():
    extra_one = make_track(3, "person", [[EVENT_FRAME, 100, 500, 140, 600]])
    extra_other = make_track(4, "person", [[EVENT_FRAME + 1, 200, 500, 240, 600]])
    tracks = [static_vehicle(), static_pedestrian(), extra_one, extra_other]
    assert features(tracks)["n_people"] == 2.0


def test_n_people_uses_supplied_frame_counts():
    result = features([static_vehicle(), static_pedestrian()], people_by_frame={EVENT_FRAME: 5})
    assert result["n_people"] == 5.0


def test_unknown_vehicle_id_raises_key_error():
    index = yf.index_tracks([moving_vehicle(), static_pedestrian()])
    with pytest.raises(KeyError):
        yf.candidate_features(index, make_event(vehicle_id=99), FPS, SRC)


def test_unknown_pedestrian_id_raises_key_error():
    index = yf.index_tracks([moving_vehicle(), static_pedestrian()])
    with pytest.raises(KeyError):
        yf.candidate_features(index, make_event(pedestrian_id=99), FPS, SRC)


def test_p_speed_of_moving_pedestrian():
    boxes = [[f, 600 + 10 * f, 450, 660 + 10 * f, 550] for f in range(0, 61)]
    result = features([static_vehicle(), make_track(2, "person", boxes)])
    assert result["p_speed"] == pytest.approx(10 * FPS / 100, abs=1e-6)


def test_index_tracks_and_box_at():
    index = yf.index_tracks([make_track(7, "person", [[3.0, 0, 0, 10, 20], [1.0, 5, 5, 15, 25]])])
    entry = index[7]
    assert entry["kind"] == "person"
    assert entry["frames"].tolist() == [1, 3]
    assert entry["boxes"].shape == (2, 4)
    assert yf.box_at(entry, 2).tolist() == [5.0, 5.0, 15.0, 25.0]
    assert yf.box_at(entry, 99).tolist() == [0.0, 0.0, 10.0, 20.0]
    assert yf.box_at(yf.index_tracks([make_track(8, "car", [])])[8], 0) is None
