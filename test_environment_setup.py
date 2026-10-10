"""Public CLI safety regressions. Run with the validated modern interpreter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class EnvironmentCommands(unittest.TestCase):
    def command(self, script, *arguments, env=None):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                              cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")

    def test_existing_environment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "legacy"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("unchanged")
            result = self.command("setup_environment.py", "legacy", "--env-root", folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_protected_runtime_cannot_be_used_as_environment_root(self):
        protected = ROOT / "runtime" / "detection-env"
        result = self.command("setup_environment.py", "legacy", "--env-root", protected)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("protected legacy runtime", result.stderr)

    def test_legacy_pickle_override_is_rejected_by_modern_smoke(self):
        env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        result = self.command("check_environment.py", "modern", env=env)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout.splitlines()[-1])
        self.assertIn("must not leak", report["error"])

    def test_unverified_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "untrusted.pt"
            checkpoint.write_bytes(b"not a checkpoint")
            env = os.environ.copy()
            env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            result = self.command("check_environment.py", "modern", "--image", ROOT / "temp.jpg",
                                  "--checkpoint", checkpoint, "--sha256", "0" * 64, env=env)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(result.stdout.splitlines()[-1])
            self.assertIn("SHA-256 mismatch", report["error"])
            self.assertNotIn("model_smoke", report)


if __name__ == "__main__":
    unittest.main()
