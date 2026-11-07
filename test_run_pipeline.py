import argparse
import contextlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import run_pipeline  # noqa: E402


def make_args(**overrides):
    values = {
        "folder": "datasets/來源影片",
        "run": "eval-run2",
        "weights": None,
        "conf": 0.15,
        "imgsz": 1280,
        "device": "0",
        "reuse_tracks": False,
        "geometry": "image",
        "mode": "strict",
        "serial": False,
        "only": [],
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def joined(commands):
    return [" ".join(command) for command in commands]


class BuildCommandsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = self.root / "datasets" / "來源影片" / "clip.mp4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"")
        self.dir = self.root / "datasets" / "eval-run2" / "clip"

    def make_masks(self):
        masks = self.dir / "seg" / "masks.npz"
        masks.parent.mkdir(parents=True, exist_ok=True)
        masks.write_bytes(b"")
        return masks

    def make_tracks(self):
        tracks = self.dir / "tracks" / "raw_tracks.json"
        tracks.parent.mkdir(parents=True, exist_ok=True)
        tracks.write_bytes(b"")
        return tracks

    def commands(self, **overrides):
        return run_pipeline.build_commands(self.video, make_args(**overrides), self.root, {})

    def test_step_one_is_omitted_when_masks_already_exist(self):
        self.make_masks()
        commands = self.commands()
        self.assertEqual(len(commands), 4)
        self.assertFalse(any("segment_crosswalk.py" in command[1] for command in commands))

    def test_step_one_segments_when_masks_are_missing(self):
        commands = self.commands()
        self.assertEqual(len(commands), 5)
        segment = commands[0]
        self.assertIn("segment_crosswalk.py", segment[1])
        self.assertEqual(segment[segment.index("--model") + 1], "runtime/models/mask2former-mapillary")
        self.assertEqual(segment[segment.index("--every-seconds") + 1], "0.1")
        self.assertEqual(segment[segment.index("--output") + 1], str(self.dir / "seg"))

    def test_reuse_tracks_refilters_the_existing_result(self):
        self.make_masks()
        tracks = self.make_tracks()
        commands = self.commands(reuse_tracks=True)
        track = commands[0]
        self.assertIn("track_video.py", track[1])
        self.assertEqual(track[track.index("--refilter") + 1], str(tracks))
        self.assertNotIn("--output", track)
        self.assertNotIn("--conf", track)

    def test_reuse_tracks_runs_detection_when_no_result_exists(self):
        self.make_masks()
        commands = self.commands(reuse_tracks=True)
        track = commands[0]
        self.assertNotIn("--refilter", track)
        self.assertEqual(track[track.index("--output") + 1], str(self.dir / "tracks"))
        self.assertEqual(track[track.index("--conf") + 1], "0.15")
        self.assertEqual(track[track.index("--imgsz") + 1], "1280")
        self.assertEqual(track[track.index("--device") + 1], "0")

    def test_the_geometry_flag_reaches_the_rule_step(self):
        self.make_masks()
        rule = [command for command in self.commands() if "yield_image_rule.py" in command[1]][0]
        self.assertEqual(rule[rule.index("--geometry") + 1], "image")
        rule = [command for command in self.commands(geometry="crosswalk") if "yield_image_rule.py" in command[1]][0]
        self.assertEqual(rule[rule.index("--geometry") + 1], "crosswalk")

    def test_weights_are_only_passed_when_given(self):
        self.make_masks()
        self.assertFalse(any("--weights" in command for command in joined(self.commands())))
        track = self.commands(weights="runtime/smoke-assets/yolo26x.pt")[0]
        self.assertEqual(track[track.index("--weights") + 1], "runtime/smoke-assets/yolo26x.pt")

    def test_environment_variables_choose_the_python_interpreter(self):
        environment = {
            "TRAFFIC_MODERN_PYTHON": "/envs/traffic-modern/python",
            "TRAFFIC_LABELING_PYTHON": "/envs/traffic-labeling/python",
        }
        commands = run_pipeline.build_commands(self.video, make_args(), self.root, environment)
        self.assertEqual(commands[0][0], "/envs/traffic-labeling/python")
        self.assertTrue(all(command[0] == "/envs/traffic-modern/python" for command in commands[1:]))

    def test_without_environment_variables_sys_executable_is_used(self):
        commands = self.commands()
        self.assertTrue(all(command[0] == sys.executable for command in commands))

    def test_later_steps_reference_the_run_outputs(self):
        self.make_masks()
        commands = self.commands()
        events = self.root / "datasets" / "eval-run2" / "events" / "clip.jsonl"
        overlay = self.root / "datasets" / "eval-run2" / "overlay" / "clip.mp4"
        labels = self.root / "labels" / "clip.jsonl"
        self.assertEqual(commands[1][commands[1].index("--output") + 1], str(events))
        render = commands[2]
        self.assertEqual(render[render.index("--events") + 1], str(events))
        self.assertEqual(render[render.index("--labels") + 1], str(labels))
        self.assertEqual(render[render.index("--output") + 1], str(overlay))

    def test_strict_mode_is_the_default_and_review_mode_adds_waiting_pedestrians(self):
        # strict: the rule's candidate list already finds 14/17 with 50 reports; review adds one hit for 71 more reports
        self.assertEqual(run_pipeline._parse_args([]).mode, "strict")
        self.make_masks()
        rule = [command for command in self.commands() if "yield_image_rule.py" in command[1]][0]
        self.assertNotIn("--include-waiting", rule)
        rule = [command for command in self.commands(mode="review") if "yield_image_rule.py" in command[1]][0]
        self.assertIn("--include-waiting", rule)

    def test_evidence_step_cuts_padded_clips_into_the_run_folder(self):
        self.make_masks()
        evidence = self.commands()[3]
        self.assertIn("export_evidence.py", evidence[1])
        events = self.root / "datasets" / "eval-run2" / "events" / "clip.jsonl"
        self.assertEqual(evidence[evidence.index("--events") + 1], str(events))
        self.assertEqual(evidence[evidence.index("--output") + 1], str(self.root / "datasets" / "eval-run2" / "evidence" / "clip"))
        self.assertEqual(evidence[evidence.index("--pad-seconds") + 1], "2")

    def test_the_candidate_list_is_built_once_for_the_whole_run(self):
        command = run_pipeline.candidate_list_command(make_args(), self.root, {})
        self.assertIn("list_candidates.py", command[1])
        self.assertEqual(command[command.index("--run-dir") + 1], str(self.root / "datasets" / "eval-run2"))


class SpeedTest(unittest.TestCase):
    def test_speed_is_reported_against_the_video_length(self):
        self.assertEqual(run_pipeline.speed_text(60.0, 360.0), "影片 60 秒，處理 360 秒（即時的 6.0 倍）")

    def test_unknown_video_length_reports_only_the_processing_time(self):
        self.assertEqual(run_pipeline.speed_text(None, 12.4), "處理 12 秒")


class FakePopen:
    """Stands in for subprocess.Popen; every instance is recorded so tests can see what ran and in what order."""

    started = []
    codes = {}  # script name -> exit code; default 0

    def __init__(self, command, cwd=None):
        self.command, self.cwd, self.terminated = command, cwd, False
        script = Path(command[1]).name
        self.code = FakePopen.codes.get(script, 0)
        FakePopen.started.append(self)

    def poll(self):
        return -15 if self.terminated else self.code

    def wait(self):
        return self.poll()

    def terminate(self):
        self.terminated = True
        self.code = None

    @classmethod
    def scripts(cls):
        return [Path(process.command[1]).name for process in cls.started]


class SlowFakePopen(FakePopen):
    """Still running until terminated."""

    def poll(self):
        return -15 if self.terminated else None


class MainTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(run_pipeline, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = self.root / "videos"
        self.folder.mkdir()
        FakePopen.started, FakePopen.codes = [], {}
        popen = mock.patch("run_pipeline.subprocess.Popen", FakePopen)
        popen.start()
        self.addCleanup(popen.stop)
        nap = mock.patch("run_pipeline.time.sleep")
        nap.start()
        self.addCleanup(nap.stop)

    def run_main(self, argv):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = run_pipeline.main(argv)
        return code, output.getvalue(), errors.getvalue()

    def add_video(self, name):
        path = self.folder / name
        path.write_bytes(b"")
        return path

    def test_dry_run_prints_commands_only_and_creates_nothing(self):
        self.add_video("b.mp4")
        self.add_video("a.MOV")
        code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(FakePopen.started, [])
        self.assertFalse((self.root / "datasets").exists())
        lines = output.splitlines()
        self.assertEqual(len(lines), 11)  # 2 videos x 5 steps, then the run's candidate list
        self.assertIn("list_candidates.py", lines[-1])
        self.assertIn("a.MOV", lines[0])
        self.assertNotIn(str(self.folder / "b.mp4"), lines[0])

    def test_segmentation_and_tracking_run_at_the_same_time_then_the_rest(self):
        self.add_video("a.mp4")
        code, _, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 0)
        self.assertEqual(FakePopen.scripts(), ["segment_crosswalk.py", "track_video.py", "yield_image_rule.py",
                                               "render_overlay.py", "export_evidence.py", "list_candidates.py"])

    def test_a_failing_step_stops_its_partner_and_every_later_step(self):
        self.add_video("a.mp4")
        self.add_video("b.mp4")
        FakePopen.codes = {"track_video.py": 1}
        with mock.patch("run_pipeline.subprocess.Popen", side_effect=lambda command, cwd=None: (
                FakePopen if "track_video.py" in command[1] else SlowFakePopen)(command, cwd)):
            code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 1)
        self.assertEqual(FakePopen.scripts(), ["segment_crosswalk.py", "track_video.py"])
        self.assertTrue(FakePopen.started[0].terminated)  # the still-running segmentation is stopped
        self.assertIn("失敗", output)
        self.assertIn("track_video.py", output)

    def waits_for_b(self, script, code=0):
        """Popen whose `script` for video a keeps running until b's segmentation has started (or gives up)."""

        class Waits(FakePopen):
            gave_up = False

            def poll(self):
                for _ in range(2000):
                    if any("segment_crosswalk.py" in p.command[1] and "b.mp4" in " ".join(p.command)
                           for p in FakePopen.started):
                        return -15 if self.terminated else code
                    if self.terminated:
                        return -15
                    threading.Event().wait(0.001)  # time.sleep is mocked out in these tests
                Waits.gave_up = True
                return code

        def make(command, cwd=None):
            mine = script in command[1] and "a.mp4" in " ".join(command)
            return (Waits if mine else FakePopen)(command, cwd)
        return make, Waits

    def test_the_next_video_is_segmented_while_the_previous_one_finishes(self):
        self.add_video("a.mp4")
        self.add_video("b.mp4")
        make, waits = self.waits_for_b("yield_image_rule.py")
        with mock.patch("run_pipeline.subprocess.Popen", side_effect=make):
            code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 0)
        self.assertFalse(waits.gave_up)  # a's rule was still running when b's segmentation started
        self.assertEqual(len(FakePopen.started), 11)  # 2 x 5 steps + the candidate list, which comes last
        self.assertIn("list_candidates.py", FakePopen.started[-1].command[1])
        self.assertIn("整批", output)

    def test_a_failing_later_step_stops_the_next_videos_gpu_steps(self):
        self.add_video("a.mp4")
        self.add_video("b.mp4")
        make, _ = self.waits_for_b("render_overlay.py", code=1)

        def popen(command, cwd=None):
            if "b.mp4" in " ".join(command):
                return SlowFakePopen(command, cwd)  # b's segmentation and tracking would run until stopped
            return make(command, cwd)
        with mock.patch("run_pipeline.subprocess.Popen", side_effect=popen):
            code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 1)
        b = [p for p in FakePopen.started if "b.mp4" in " ".join(p.command)]
        self.assertEqual([Path(p.command[1]).name for p in b], ["segment_crosswalk.py", "track_video.py"])
        self.assertTrue(all(p.terminated for p in b))
        self.assertNotIn("list_candidates.py", FakePopen.scripts())
        self.assertIn("render_overlay.py", output)
        self.assertEqual(output.count("失敗"), 1)  # the stopped GPU steps are not reported as failures of their own

    def test_serial_runs_one_step_at_a_time(self):
        self.add_video("a.mp4")
        FakePopen.codes = {"segment_crosswalk.py": 1}
        code, _, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch", "--serial"])
        self.assertEqual(code, 1)
        self.assertEqual(FakePopen.scripts(), ["segment_crosswalk.py"])

    def test_only_keeps_videos_whose_name_contains_a_given_text(self):
        self.add_video("apple.mp4")
        self.add_video("banana.mp4")
        code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "x", "--dry-run", "--only", "nan"])
        self.assertEqual(code, 0)
        self.assertNotIn("apple", output)
        self.assertIn("banana", output)

    def test_a_missing_folder_or_one_without_videos_returns_two(self):
        self.assertEqual(self.run_main(["--folder", str(self.root / "nope"), "--run", "x"])[0], 2)
        self.assertEqual(self.run_main(["--folder", str(self.folder), "--run", "x"])[0], 2)

    def test_every_step_runs_from_the_repository_root(self):
        self.add_video("a.mp4")
        code, _, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 0)
        self.assertEqual(len(FakePopen.started), 6)  # 5 steps + the candidate list
        for process in FakePopen.started:
            self.assertEqual(process.cwd, run_pipeline.ROOT)

    def test_success_runs_every_step_and_appends_the_log(self):
        self.add_video("a.mp4")
        code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 0)
        self.assertIn("=== [1/1] a.mp4", output)
        self.assertIn("全部完成", output)
        log = self.root / "datasets" / "scratch.log"
        self.assertTrue(log.exists())
        text = log.read_text(encoding="utf-8")
        self.assertIn("=== [1/1] a.mp4", text)
        self.assertIn("全部完成", text)
        self.assertIn("candidates.csv", text)
        self.assertIn("處理", text)  # the per-video processing time goes into the log


if __name__ == "__main__":
    unittest.main()
