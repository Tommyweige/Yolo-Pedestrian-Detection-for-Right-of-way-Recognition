"""Where the YOLO detector weights come from, and whether their bytes match the manifest.

weights/ is git-ignored, so a fresh clone has no fine-tuned model; Ultralytics downloads `STOCK` by name instead.
requirements/smoke-assets.json pins the SHA-256 of every artefact the smoke test uses, so a listed file that has
been swapped or corrupted fails loudly rather than silently skewing detections.
"""
import hashlib
import json
from pathlib import Path
from typing import Mapping

STOCK = "yolo26n.pt"  # Ultralytics downloads it by name
FINETUNED = Path("weights") / "yolo26n-traffic.pt"
ENV_VAR = "TRAFFIC_DETECTOR_WEIGHTS"
MANIFEST = Path("requirements") / "smoke-assets.json"
CHUNK = 1024 * 1024  # hash in 1 MiB blocks so multi-hundred-MB checkpoints do not load into memory


def resolve_weights(explicit: str | None, root: Path, environ: Mapping[str, str]) -> tuple[str, str]:
    """Return (weights path or name, source). Source is "argument" / "environment" / "finetuned" / "stock"."""
    if explicit:
        return explicit, "argument"
    from_environment = environ.get(ENV_VAR)
    if from_environment:
        return from_environment, "environment"
    finetuned = root / FINETUNED
    if finetuned.exists():
        return str(finetuned), "finetuned"
    return STOCK, "stock"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, manifest: Path) -> str:
    """Return "verified" or "unlisted"; raise ValueError when the name is listed but the digest differs."""
    path = Path(path)
    artifacts = json.loads(Path(manifest).read_text(encoding="utf-8")).get("artifacts", [])
    listed = next((artifact for artifact in artifacts if artifact.get("name") == path.name), None)
    if listed is None:
        return "unlisted"
    actual = _file_sha256(path)
    expected = listed.get("sha256")
    if actual != expected:
        raise ValueError(f"{path.name}: sha256 mismatch (expected {expected}, actual {actual})")
    return "verified"
