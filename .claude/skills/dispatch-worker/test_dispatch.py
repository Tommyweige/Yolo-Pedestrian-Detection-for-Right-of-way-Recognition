import os
import unittest
from unittest import mock

import dispatch as d


class OutOfScope(unittest.TestCase):
    def test_exact_and_glob_and_dir(self):
        allow = ["src/a.py", "tests/*.py", "docs/"]
        self.assertEqual(d.out_of_scope(["src/a.py", "tests/test_a.py", "docs/x/y.md"], allow), [])

    def test_detects_extra_and_untracked(self):
        bad = d.out_of_scope(["src/a.py", "src/b.py", "out.txt"], ["src/a.py"])
        self.assertEqual(bad, ["src/b.py", "out.txt"])

    def test_windows_separators_and_dot_prefix(self):
        self.assertEqual(d.out_of_scope([".\\src\\a.py"], ["src/a.py"]), [])

    def test_prefix_is_not_substring(self):
        self.assertEqual(d.out_of_scope(["srcx/a.py"], ["src/"]), ["srcx/a.py"])


class Unavailable(unittest.TestCase):
    def test_rate_limit_without_changes(self):
        self.assertTrue(d.classify_unavailable(1, "Error: 429 Too Many Requests", []))
        self.assertTrue(d.classify_unavailable(0, "free quota exceeded", []))

    def test_successful_run_is_not_unavailable_even_if_text_mentions_quota(self):
        self.assertFalse(d.classify_unavailable(0, "I handled the quota logic", ["a.py"]))

    def test_v2_rate_limit_message(self):
        self.assertTrue(d.classify_unavailable(1, "Error: Rate limit exceeded. Please try again later.", []))
        self.assertTrue(d.classify_unavailable(1, "Error: Upstream request failed: Endpoint is unavailable.", []))
        self.assertTrue(d.classify_unavailable(0, "Error: Endpoint is unavailable.", []))

    def test_unrelated_failure(self):
        self.assertFalse(d.classify_unavailable(1, "syntax error in a.py", []))


class Env(unittest.TestCase):
    def test_claude_credentials_env_stripped(self):
        fake = {"ANTHROPIC_API_KEY": "x", "CLAUDE_CODE_OAUTH_TOKEN": "y", "CLAUDECODE": "1", "PATH": "p"}
        with mock.patch.dict(os.environ, fake, clear=True):
            env = d.worker_env()
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDECODE"):
            self.assertNotIn(k, env)
        self.assertEqual(env["PATH"], "p")
        self.assertEqual(env["OPENCODE_DISABLE_CLAUDE_CODE"], "1")

    def test_free_model_first_then_go(self):
        self.assertTrue(d.MODELS[0].startswith("opencode/") and d.MODELS[0].endswith("-free"))
        self.assertEqual(d.MODELS[1:], ("opencode-go/deepseek-v4.1-flash",))

    def test_variant_is_high_and_separate_from_model(self):
        self.assertEqual(d.VARIANT, "high")
        for m in d.MODELS:
            self.assertNotIn("#", m)

    def test_midrun_outage_with_partial_changes_falls_back(self):
        out = "wrote a.py\nError: Streaming response failed: [server_error] Internal server error occurred."
        self.assertTrue(d.classify_unavailable(1, out, ["tools/a.py"]))
        self.assertTrue(d.should_try_next_model(True))
        self.assertFalse(d.should_try_next_model(False))

    def test_variant_option_defaults_high_and_allows_max(self):
        self.assertEqual(d.VARIANTS, ("high", "max"))
        parser = d.build_parser()
        args = parser.parse_args(["run", "--id", "x", "--ticket", "t", "--integration", "i", "--allow", "a"])
        self.assertEqual(args.variant, "high")
        args = parser.parse_args(["run", "--id", "x", "--ticket", "t", "--integration", "i", "--allow", "a", "--variant", "max"])
        self.assertEqual(args.variant, "max")
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "--id", "x", "--ticket", "t", "--integration", "i", "--allow", "a", "--variant", "low"])

    def test_run_worker_passes_variant_to_model(self):
        captured = {}

        class FakeProc:
            stdout = iter(())
            returncode = 0

            def wait(self, timeout=None):
                return 0

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeProc()

        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(d, "find_opencode", return_value="opencode"), \
             mock.patch.object(d.subprocess, "Popen", side_effect=fake_popen):
            try:
                d.run_worker(Path(tmp), Path(tmp) / "t.md", Path(tmp) / "log.txt", 60, d.MODELS[0], "max")
            except Exception:
                pass
        self.assertIn(f"{d.MODELS[0]}#max", captured.get("cmd", []))

    def test_opencode_must_be_pinned_version(self):
        fake = mock.Mock(stdout="opencode v2.0.32\n")
        with mock.patch.object(d.Path, "exists", return_value=True), \
             mock.patch.object(d.subprocess, "run", return_value=fake):
            with self.assertRaises(RuntimeError):
                d.find_opencode()


class SameAsCheckout(unittest.TestCase):
    def test_unchanged_modified_and_new_files(self):
        import subprocess
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            run = lambda *a: subprocess.run(["git", *a], cwd=wt, check=True, capture_output=True)
            run("init", "-q")
            (wt / "a.txt").write_text("hello\n", encoding="utf-8")
            (wt / "b.txt").write_text("keep\n", encoding="utf-8")
            run("add", "-A")
            run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
            (wt / "a.txt").write_text("changed\n", encoding="utf-8")
            (wt / "new.txt").write_text("x\n", encoding="utf-8")
            self.assertTrue(d.same_as_checkout(wt, "HEAD", "b.txt"))
            self.assertFalse(d.same_as_checkout(wt, "HEAD", "a.txt"))
            self.assertFalse(d.same_as_checkout(wt, "HEAD", "new.txt"))


if __name__ == "__main__":
    unittest.main()
