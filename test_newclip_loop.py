"""Run with a Python that has opencv-python, numpy and Pillow (the rule and the page builder import them)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import newclip_loop  # noqa: E402


class PlanTest(unittest.TestCase):
    """`plan_clip` decides which of the three steps still have to run, and never changes the rule's command."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = self.root / "videos" / "clip.mp4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"")
        self.out = self.root / "run"
        (self.root / "model").mkdir()
        self.environ = {"TRAFFIC_MODERN_PYTHON": "modern-python", "TRAFFIC_LABELING_PYTHON": "labeling-python",
                        "TRAFFIC_CROSSWALK_MODEL": str(self.root / "model")}

    def plant(self, root, masks=True, tracks=True):
        if masks:
            (root / "clip" / "seg").mkdir(parents=True, exist_ok=True)
            (root / "clip" / "seg" / "masks.npz").write_bytes(b"")
        if tracks:
            (root / "clip" / "tracks").mkdir(parents=True, exist_ok=True)
            (root / "clip" / "tracks" / "raw_tracks.json").write_text("[]", encoding="utf-8")

    def test_a_new_clip_runs_all_three_steps_with_the_documented_flags(self):
        steps, masks, tracks = newclip_loop.plan_clip(self.video, self.out, self.environ)
        commands = {number: command for number, _, command in steps}
        self.assertEqual(sorted(commands), [1, 2, 3])
        segmentation, tracking, rule = commands[1], commands[2], commands[3]
        self.assertEqual(segmentation[0], "labeling-python")
        self.assertIn("segment_crosswalk.py", segmentation[1])
        self.assertEqual(segmentation[segmentation.index("--stride") + 1], "3")
        self.assertEqual(tracking[0], "modern-python")
        self.assertIn("track_video.py", tracking[1])
        self.assertEqual(tracking[tracking.index("--conf") + 1], "0.15")
        self.assertEqual(tracking[tracking.index("--imgsz") + 1], "1280")
        self.assertEqual(rule[0], "modern-python")
        self.assertTrue(rule[1].endswith("yield_image_rule.py"))
        self.assertEqual(masks, self.out / "clip" / "seg" / "masks.npz")
        self.assertEqual(tracks, self.out / "clip" / "tracks" / "raw_tracks.json")

    def test_the_rule_step_passes_no_construction_flags_so_the_defaults_apply(self):
        rule = newclip_loop.plan_clip(self.video, self.out, self.environ)[0][2][2]
        self.assertNotIn("--one-event-per-vehicle", rule)
        self.assertNotIn("--min-inside-person-heights", rule)
        self.assertEqual([part for part in rule if part.startswith("--")],
                         ["--video", "--masks", "--tracks", "--output"])

    def test_existing_masks_and_tracks_in_the_run_folder_are_reused(self):
        self.plant(self.out)
        steps, _, _ = newclip_loop.plan_clip(self.video, self.out, self.environ)
        self.assertIsNone(steps[0][2])
        self.assertIsNone(steps[1][2])
        self.assertIsNotNone(steps[2][2])

    def test_reuse_from_reads_masks_and_tracks_from_the_other_run(self):
        earlier = self.root / "earlier"
        self.plant(earlier)
        steps, masks, tracks = newclip_loop.plan_clip(self.video, self.out, self.environ, reuse=earlier)
        self.assertIsNone(steps[0][2])
        self.assertIsNone(steps[1][2])
        self.assertEqual(masks, earlier / "clip" / "seg" / "masks.npz")
        self.assertEqual(tracks, earlier / "clip" / "tracks" / "raw_tracks.json")
        self.assertEqual(steps[2][2][steps[2][2].index("--masks") + 1], str(masks))

    def test_force_redoes_masks_and_tracks(self):
        self.plant(self.out)
        steps, masks, _ = newclip_loop.plan_clip(self.video, self.out, self.environ, force=True)
        self.assertIsNotNone(steps[0][2])
        self.assertIsNotNone(steps[1][2])
        self.assertEqual(masks, self.out / "clip" / "seg" / "masks.npz")


class VideosTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)

    def test_videos_are_found_by_extension_in_name_order(self):
        for name in ("b.mp4", "a.MOV", "notes.txt", "c.mkv"):
            (self.folder / name).write_bytes(b"")
        self.assertEqual([path.name for path in newclip_loop.videos_in(self.folder)], ["a.MOV", "b.mp4", "c.mkv"])

    def test_a_folder_without_videos_is_an_error(self):
        (self.folder / "notes.txt").write_text("", encoding="utf-8")
        with self.assertRaises(SystemExit):
            newclip_loop.videos_in(self.folder)


class ManifestTest(unittest.TestCase):
    def clip(self, stem, **extra):
        return {"stem": stem, "video_name": stem + ".mp4", "events": 1, "event_times_s": [1.0], **extra}

    def test_the_instruction_sheet_records_the_defaults_and_the_role(self):
        clips = [self.clip("a"), self.clip("b")]
        manifest = newclip_loop.build_manifest(clips, "dev", "C:/videos", newclip_loop.rule_defaults())
        self.assertEqual(manifest["schema"], newclip_loop.SCHEMA)
        self.assertEqual(manifest["role"], "dev")
        self.assertEqual(manifest["tolerance_s"], 1.25)
        self.assertEqual(manifest["params"]["min_inside_person_heights"], 0.03)
        self.assertTrue(manifest["params"]["report_every_contact"])
        self.assertEqual(manifest["params"]["contact_dedup"], "pair")
        self.assertEqual(manifest["params"]["contact_dedup_s"], 1.25)
        # fps and size are per-clip facts, so the rule's defaults must not claim the 60 fps placeholder
        self.assertNotIn("fps", manifest["params"])
        self.assertNotIn("src_size", manifest["params"])
        self.assertEqual([clip["role"] for clip in clips], ["dev", "dev"])

    def test_a_clip_that_the_development_labels_know_is_flagged(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        labels = Path(tmp.name)
        (labels / "a.jsonl").write_text("", encoding="utf-8")
        clips = [self.clip("a"), self.clip("new")]
        manifest = newclip_loop.build_manifest(clips, "validation", "C:/videos",
                                               newclip_loop.rule_defaults(), dev_labels=labels)
        self.assertEqual([clip["dev_overlap"] for clip in manifest["clips"]], [True, False])


if __name__ == "__main__":
    unittest.main()
