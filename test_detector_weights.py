import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from detector_weights import ENV_VAR, FINETUNED, STOCK, resolve_weights, verify_sha256


class ResolveWeightsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_argument_wins_over_the_environment(self):
        self.assertEqual(resolve_weights("my.pt", self.root, {ENV_VAR: "env.pt"}), ("my.pt", "argument"))

    def test_environment_variable_is_used_when_there_is_no_argument(self):
        self.assertEqual(resolve_weights(None, self.root, {ENV_VAR: "env.pt"}), ("env.pt", "environment"))

    def test_empty_environment_variable_counts_as_unset(self):
        self.assertEqual(resolve_weights(None, self.root, {ENV_VAR: ""}), (STOCK, "stock"))

    def test_finetuned_weights_are_used_when_the_file_exists(self):
        (self.root / FINETUNED).parent.mkdir(parents=True)
        (self.root / FINETUNED).touch()
        self.assertEqual(resolve_weights(None, self.root, {}), (str(self.root / FINETUNED), "finetuned"))

    def test_stock_name_is_the_last_resort(self):
        self.assertEqual(resolve_weights(None, self.root, {}), (STOCK, "stock"))


class VerifySha256Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.payload = b"not-really-yolo-weights" * 4096
        self.path = self.dir / "yolo26n.pt"
        self.path.write_bytes(self.payload)
        self.manifest = self.dir / "smoke-assets.json"
        self.expected = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        self._tmp.cleanup()

    def write_manifest(self, sha256):
        self.manifest.write_text(json.dumps({"artifacts": [{"name": "yolo26n.pt", "sha256": sha256}]}), encoding="utf-8")

    def test_listed_name_with_a_matching_digest_is_verified(self):
        self.write_manifest(self.expected)
        self.assertEqual(verify_sha256(self.path, self.manifest), "verified")

    def test_listed_name_with_a_different_digest_raises(self):
        wrong = "0" * 64
        self.write_manifest(wrong)
        with self.assertRaises(ValueError) as caught:
            verify_sha256(self.path, self.manifest)
        for part in ("yolo26n.pt", wrong, self.expected):
            self.assertIn(part, str(caught.exception))

    def test_name_missing_from_the_manifest_is_unlisted(self):
        self.write_manifest(self.expected)
        other = self.dir / "other.pt"
        other.write_bytes(self.payload)
        self.assertEqual(verify_sha256(other, self.manifest), "unlisted")


if __name__ == "__main__":
    unittest.main()
