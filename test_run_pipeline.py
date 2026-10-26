import argparse
import contextlib
import io
import sys
import tempfile
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
        self.assertEqual(len(commands), 3)
        self.assertFalse(any("segment_crosswalk.py" in command[1] for command in commands))

    def test_step_one_segments_when_masks_are_missing(self):
        commands = self.commands()
        self.assertEqual(len(commands), 4)
        segment = commands[0]
        self.assertIn("segment_crosswalk.py", segment[1])
        self.assertEqual(segment[segment.index("--model") + 1], "runtime/models/mask2former-mapillary")
        self.assertEqual(segment[segment.index("--stride") + 1], "3")
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
        with mock.patch("run_pipeline.subprocess.run") as run:
            code, output, _ = self.run_main(
                ["--folder", str(self.folder), "--run", "scratch", "--dry-run"]
            )
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertFalse((self.root / "datasets").exists())
        lines = output.splitlines()
        self.assertEqual(len(lines), 8)
        self.assertIn("a.MOV", lines[0])
        self.assertNotIn(str(self.folder / "b.mp4"), lines[0])

    def test_a_failing_step_returns_one_and_stops_before_the_next_video(self):
        self.add_video("a.mp4")
        self.add_video("b.mp4")
        calls = []

        def fake_run(command, check=False):
            calls.append(command)
            return mock.Mock(returncode=1)

        with mock.patch("run_pipeline.subprocess.run", side_effect=fake_run):
            code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("a.mp4", " ".join(calls[0]))
        self.assertIn("失敗", output)

    def test_a_missing_folder_or_one_without_videos_returns_two(self):
        self.assertEqual(self.run_main(["--folder", str(self.root / "nope"), "--run", "x"])[0], 2)
        self.assertEqual(self.run_main(["--folder", str(self.folder), "--run", "x"])[0], 2)

    def test_success_runs_every_step_and_appends_the_log(self):
        self.add_video("a.mp4")
        with mock.patch("run_pipeline.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0)
            code, output, _ = self.run_main(["--folder", str(self.folder), "--run", "scratch"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 4)
        self.assertIn("=== [1/1] a.mp4", output)
        self.assertIn("全部完成", output)
        log = self.root / "datasets" / "scratch.log"
        self.assertTrue(log.exists())
        text = log.read_text(encoding="utf-8")
        self.assertIn("=== [1/1] a.mp4", text)
        self.assertIn("全部完成", text)


if __name__ == "__main__":
    unittest.main()
