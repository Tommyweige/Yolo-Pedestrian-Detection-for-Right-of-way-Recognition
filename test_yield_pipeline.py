"""Run with a Python environment containing opencv-python and numpy (detection_backend imports cv2 lazily)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import desktop_bridge
import detection_backend
import yield_pipeline


def step_script(work, events):
    """A stand-in step that writes the rule's .jsonl output, so the protocol can be tested without a GPU."""
    path = Path(work) / "events.jsonl"
    return (
        "import json, pathlib\n"
        f"pathlib.Path(r'{work}', 'events.jsonl').write_text("
        f"''.join(json.dumps(e) + chr(10) for e in {events!r}), encoding='utf-8')\n"
    )


class PlanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"")
        self.work = self.root / "out" / "yield" / "clip"
        self.model = self.root / "model"
        self.model.mkdir()
        self.weights = self.root / "yolo26x.pt"
        self.weights.write_bytes(b"")

    def plan(self, **environ):
        environ.setdefault("TRAFFIC_CROSSWALK_MODEL", str(self.model))
        environ.setdefault("TRAFFIC_DETECTOR_WEIGHTS", str(self.weights))
        return yield_pipeline.plan(self.video, self.work, environ)

    def test_three_steps_in_order_with_the_documented_flags(self):
        segmentation, tracking, rule = self.plan()
        self.assertEqual([step.label for step in (segmentation, tracking, rule)],
                         ["斑馬線分割", "偵測與追蹤", "不禮讓規則"])
        self.assertIn("segment_crosswalk.py", segmentation.command[1])
        self.assertEqual(segmentation.command[segmentation.command.index("--every-seconds") + 1], "0.1")
        self.assertEqual(segmentation.command[segmentation.command.index("--output") + 1],
                         str(self.work / "seg"))
        self.assertEqual(segmentation.command[segmentation.command.index("--model") + 1], str(self.model))
        self.assertIn("track_video.py", tracking.command[1])
        self.assertEqual(tracking.command[tracking.command.index("--weights") + 1], str(self.weights))
        self.assertEqual(tracking.command[tracking.command.index("--conf") + 1], "0.15")
        self.assertEqual(tracking.command[tracking.command.index("--imgsz") + 1], "1280")
        self.assertEqual(tracking.command[tracking.command.index("--device") + 1], "0")
        self.assertEqual(tracking.command[tracking.command.index("--output") + 1],
                         str(self.work / "tracks"))
        self.assertIn("yield_image_rule.py", rule.command[1])
        self.assertEqual(rule.command[rule.command.index("--masks") + 1],
                         str(self.work / "seg" / "masks.npz"))
        self.assertEqual(rule.command[rule.command.index("--tracks") + 1],
                         str(self.work / "tracks" / "raw_tracks.json"))
        self.assertEqual(rule.command[rule.command.index("--output") + 1],
                         str(self.work / "events.jsonl"))

    def test_the_detector_defaults_to_yolo26x_under_runtime(self):
        default = self.root / "runtime" / "smoke-assets" / "yolo26x.pt"
        default.parent.mkdir(parents=True)
        default.write_bytes(b"")
        self.assertEqual(yield_pipeline.detector_weights({}, root=self.root), default)

    def test_missing_detector_weights_are_reported_before_anything_runs(self):
        with self.assertRaises(FileNotFoundError) as caught:
            yield_pipeline.detector_weights({}, root=self.root)
        self.assertIn("TRAFFIC_DETECTOR_WEIGHTS", str(caught.exception))

    def test_the_rule_step_reads_the_video_for_fps_and_size(self):
        rule = self.plan()[2]
        self.assertEqual(rule.command[rule.command.index("--video") + 1], str(self.video))

    def test_each_environment_variable_chooses_its_interpreter(self):
        steps = self.plan(TRAFFIC_LABELING_PYTHON="/envs/labeling/python",
                          TRAFFIC_MODERN_PYTHON="/envs/modern/python")
        self.assertEqual(steps[0].command[0], "/envs/labeling/python")
        self.assertTrue(all(step.command[0] == "/envs/modern/python" for step in steps[1:]))

    def test_without_the_variables_the_current_interpreter_is_used(self):
        self.assertTrue(all(step.command[0] == sys.executable for step in self.plan()))

    def test_device_can_be_overridden(self):
        tracking = self.plan(TRAFFIC_DEVICE="cpu")[1]
        self.assertEqual(tracking.command[tracking.command.index("--device") + 1], "cpu")

    def test_a_missing_crosswalk_model_is_reported_before_anything_runs(self):
        with self.assertRaises(FileNotFoundError) as caught:
            yield_pipeline.plan(self.video, self.work, {"TRAFFIC_CROSSWALK_MODEL": str(self.root / "nope")})
        self.assertIn("TRAFFIC_CROSSWALK_MODEL", str(caught.exception))

    def test_relative_model_paths_are_read_from_the_repository_root(self):
        relative = mock.patch.object(yield_pipeline, "ROOT", self.root)
        relative.start()
        self.addCleanup(relative.stop)
        steps = yield_pipeline.plan(self.video, self.work, {"TRAFFIC_CROSSWALK_MODEL": "model"})
        self.assertEqual(steps[0].command[steps[0].command.index("--model") + 1], str(self.model))


class ConstructionTest(unittest.TestCase):
    """The rule step leaves the construction to the rule's defaults, so the GUI reports what the rule reports."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "model"
        self.model.mkdir()
        self.weights = Path(self.temp.name) / "yolo26x.pt"
        self.weights.write_bytes(b"")

    def rule(self):
        steps = yield_pipeline.plan("clip.mp4", Path(self.temp.name) / "work",
                                    {"TRAFFIC_CROSSWALK_MODEL": str(self.model),
                                     "TRAFFIC_DETECTOR_WEIGHTS": str(self.weights)})
        return steps[2]

    def test_the_rule_step_passes_no_construction_flag(self):
        rule = self.rule()
        self.assertNotIn("--one-event-per-vehicle", rule.command)
        self.assertNotIn("--contact-dedup", rule.command)
        self.assertNotIn("--include-waiting", rule.command)

    def test_the_defaults_the_gui_inherits_are_the_candidate_list(self):
        import yield_image_rule

        params = yield_image_rule.Params(fps=30.0, src_size=(1920, 1080))
        self.assertTrue(params.report_every_contact)
        self.assertEqual(params.contact_dedup, "pair")
        self.assertAlmostEqual(params.contact_dedup_s, 1.25)
        self.assertAlmostEqual(params.min_inside_person_heights, 0.03)


class WorkspaceTest(unittest.TestCase):
    def test_each_video_gets_a_folder_named_after_it(self):
        self.assertEqual(yield_pipeline.workspace("out", "clips/影片 1.mp4"),
                         Path("out") / "yield" / "影片 1")


class ExecuteTest(unittest.TestCase):
    def test_a_failing_step_reports_its_last_lines(self):
        with self.assertRaises(RuntimeError) as caught:
            yield_pipeline.execute(
                [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], "不禮讓規則")
        self.assertIn("boom", str(caught.exception))
        self.assertIn("3", str(caught.exception))
        self.assertIn("不禮讓規則", str(caught.exception))

    def test_cancellation_terminates_the_step(self):
        seen = []
        outcome = yield_pipeline.execute([sys.executable, "-c", "import time; time.sleep(60)"],
                                         cancelled=lambda: True, on_process=seen.append)
        self.assertEqual(outcome, "cancelled")
        self.assertIsNotNone(seen[0].returncode)  # the child was terminated, not waited out

    def test_the_child_is_handed_over_and_released_around_the_step(self):
        seen = []
        yield_pipeline.execute([sys.executable, "-c", "pass"], on_process=seen.append)
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[-1])


class WriteRunMetaTest(unittest.TestCase):
    def test_run_json_records_timings_counts_and_the_commands(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path(tmp.name)
        steps = [yield_pipeline.Step(1, "斑馬線分割", ("python", "segment.py"))]
        events = [{"time_s": 1.0}, {"time_s": 2.5}]
        meta = yield_pipeline.write_run_meta(work, "clips/clip.mp4", events, {"斑馬線分割": 12.5}, steps,
                                             {"frames": 300, "fps": 30.0, "size": (640, 360)})
        written = json.loads((work / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(written["frames"], 300)
        self.assertEqual(written["events"], 2)
        self.assertEqual(written["event_times_s"], [1.0, 2.5])
        self.assertEqual(written["step_seconds"], {"斑馬線分割": 12.5})
        self.assertEqual(written["seconds"], 12.5)
        self.assertEqual(written["commands"], [["python", "segment.py"]])
        self.assertEqual(meta["video"], "clip.mp4")


class RunYieldTest(unittest.TestCase):
    """The protocol the GUI consumes: progress -> event -> done, with cancellation and failure."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.output = self.root / "out"
        self.output.mkdir()
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"")
        self.work = self.output / "yield" / "clip"
        self.events = [{"time_s": 6.974, "frame": 418, "vehicle_id": 1494, "vehicle_kind": "car",
                        "pedestrian_id": 345, "pedestrian_state": "waiting_at_edge",
                        "pedestrian_entered_crosswalk": True, "needs_review": False}]
        patcher = mock.patch.object(detection_backend, "_video_metadata",
                                    return_value={"frames": 722, "fps": 59.94, "size": (1920, 1080)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def plan_with(self, source):
        def plan(video, work, environ):
            script = self.root / "step.py"
            script.write_text(source, encoding="utf-8")
            return [yield_pipeline.Step(1, "不禮讓規則", (sys.executable, str(script)))]
        return plan

    def run_yield(self, plan, cancelled=lambda: False):
        with mock.patch.object(yield_pipeline, "plan", plan):
            return list(detection_backend.run_yield([str(self.video)], str(self.output),
                                                    cancelled=cancelled))

    def test_events_carry_the_video_and_are_followed_by_done(self):
        generated = self.run_yield(self.plan_with(step_script(self.work, self.events)))
        kinds = [event["type"] for event in generated]
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(kinds.count("event"), 1)
        event = next(event for event in generated if event["type"] == "event")
        self.assertEqual(event["video"], str(self.video))
        self.assertEqual(event["video_name"], "clip.mp4")
        self.assertEqual(event["time_s"], 6.974)
        self.assertEqual(event["pedestrian_id"], 345)

    def test_progress_reports_the_step_count_and_the_step_name(self):
        generated = self.run_yield(self.plan_with(step_script(self.work, self.events)))
        progress = [event for event in generated if event["type"] == "progress"]
        self.assertEqual(progress[0]["current"], 0)
        self.assertEqual(progress[0]["total"], 1)
        self.assertTrue(any("不禮讓規則" in event["message"] for event in progress))
        self.assertEqual(progress[-1]["current"], 1)
        self.assertIn("1 件事件", progress[-1]["message"])

    def test_run_json_is_left_next_to_the_events(self):
        self.run_yield(self.plan_with(step_script(self.work, self.events)))
        meta = json.loads((self.work / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["events"], 1)
        self.assertEqual(meta["frames"], 722)
        self.assertEqual(set(meta["step_seconds"]), {"不禮讓規則"})

    def test_a_failing_step_stops_the_video_and_raises(self):
        plan = self.plan_with("import sys\nprint('no model')\nsys.exit(2)\n")
        with self.assertRaises(RuntimeError) as caught:
            self.run_yield(plan)
        self.assertIn("no model", str(caught.exception))

    def test_cancellation_before_the_first_step_yields_cancelled(self):
        generated = self.run_yield(self.plan_with(step_script(self.work, self.events)),
                                   cancelled=lambda: True)
        self.assertEqual(generated, [{"type": "cancelled"}])

    def test_videos_and_output_are_required(self):
        with self.assertRaises(ValueError):
            list(detection_backend.run_yield([], str(self.output)))
        with self.assertRaises(ValueError):
            list(detection_backend.run_yield([str(self.video)], str(self.root / "nope")))
        with self.assertRaises(FileNotFoundError):
            list(detection_backend.run_yield([str(self.root / "missing.mp4")], str(self.output)))


class BridgeDispatchTest(unittest.TestCase):
    """The Rust UI sends one request object; the bridge picks the pipeline from its task field."""

    def test_the_yield_task_runs_the_image_rule(self):
        with mock.patch.object(desktop_bridge, "run_yield",
                               return_value=iter([{"type": "done"}])) as image_rule, \
                mock.patch.object(desktop_bridge, "run_detection") as legacy:
            events = list(desktop_bridge.stream(
                {"task": "yield", "videos": ["a.mp4"], "output": "out"}, lambda: False))
        self.assertEqual(events, [{"type": "done"}])
        image_rule.assert_called_once_with(["a.mp4"], "out", cancelled=mock.ANY)
        legacy.assert_not_called()

    def test_other_tasks_keep_the_legacy_predictors(self):
        with mock.patch.object(desktop_bridge, "run_yield") as image_rule, \
                mock.patch.object(desktop_bridge, "run_detection",
                                  return_value=iter([{"type": "done"}])) as legacy:
            events = list(desktop_bridge.stream(
                {"task": "zebra", "videos": ["a.mp4"], "output": "out", "model": "m", "angle": 15},
                lambda: False))
        self.assertEqual(events, [{"type": "done"}])
        legacy.assert_called_once_with(["a.mp4"], "out", "m", "zebra", 15, mock.ANY)
        image_rule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
