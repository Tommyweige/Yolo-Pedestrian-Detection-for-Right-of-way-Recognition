"""Check an isolated dependency profile; optional model forward is a smoke, not evaluation."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("legacy", "labeling", "modern"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", type=Path, help="Run one labeling/modern model forward")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256", help="Trusted checkpoint digest, verified before loading")
    parser.add_argument("--config-sha256", help="Trusted DINO cached Python config digest")
    args = parser.parse_args()
    result = {"profile": args.profile, "python": sys.version, "interpreter": sys.executable,
              "platform": platform.platform(), "status": "fail"}
    try:
        if sys.prefix == sys.base_prefix:
            raise ValueError("Run this command using an isolated environment interpreter")
        config_directory = Path(sys.prefix) / ".config" / "ultralytics"
        config_directory.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
        if args.profile != "legacy" and os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "").lower() in {"1", "true", "y", "yes"}:
            raise ValueError("Unsafe legacy pickle override must not leak into labeling/modern")
        import torch
        import torchvision
        import numpy as np
        import cv2
        result.update(torch=torch.__version__, torchvision=torchvision.__version__,
                      numpy=np.__version__, opencv=cv2.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), isolated=sys.prefix != sys.base_prefix)
        if not result["isolated"]:
            raise ValueError("Run this command using an isolated environment interpreter")
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU verification did not run")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
        # Exercise both torch kernels and the compiled torchvision operator.
        assert (torch.eye(2, device=device) @ torch.ones(2, device=device)).tolist() == [1.0, 1.0]
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device=device)
        assert torchvision.ops.nms(boxes, torch.tensor([.9, .8], device=device), .5).tolist() == [0]
        result["kernel_device"] = device
        if args.profile == "legacy":
            import ultralytics
            from ultralytics.yolo.engine.predictor import BasePredictor
            import moviepy.editor
            from PyQt5 import QtCore
            assert ultralytics.__version__ == "8.0.3"
            result["ultralytics"] = ultralytics.__version__
        elif args.profile == "modern":
            import ultralytics
            from ultralytics import YOLO
            assert ultralytics.__version__ == "8.4.175"
            result["ultralytics"] = ultralytics.__version__
            if not args.image:
                model = YOLO("yolo26n.yaml")
                result["architecture"] = type(model.model).__name__
        else:
            from autodistill_grounding_dino import GroundingDINO
            from autodistill.detection import CaptionOntology
            import groundingdino
            import segment_anything
            result["packages"] = {name: metadata.version(name) for name in
                ("autodistill", "autodistill-grounding-dino", "rf-groundingdino", "rf-segment-anything", "transformers", "supervision")}
            try:
                metadata.version("autodistill-yolov8")
            except metadata.PackageNotFoundError:
                pass
            else:
                raise ValueError("autodistill-yolov8 must not be installed")
        if args.image:
            if args.profile == "legacy":
                raise ValueError("Use scripts/validate_detection.py for legacy model regression")
            if not args.checkpoint or not args.sha256:
                raise ValueError("Provide a trusted checkpoint and its independent SHA-256")
            with args.checkpoint.open("rb") as checkpoint_file:
                digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
            if digest != args.sha256.lower():
                raise ValueError("Checkpoint SHA-256 mismatch; model was not loaded")
            result["checkpoint"] = {"name": args.checkpoint.name, "sha256": digest}
            if args.profile == "modern":
                model = YOLO(str(args.checkpoint.resolve()))
                predictions = model.predict(str(args.image.resolve()), device=device, imgsz=320, verbose=False)
                result["detections"] = len(predictions[0].boxes)
            else:
                cache = Path.home() / ".cache" / "autodistill" / "groundingdino"
                config = cache / "GroundingDINO_SwinT_OGC.py"
                if args.checkpoint.resolve() != (cache / "groundingdino_swint_ogc.pth").resolve():
                    raise ValueError("Preseed the plugin's cache with the verified DINO checkpoint")
                if not args.config_sha256:
                    raise ValueError("Verify the executable DINO Python config before constructing the model")
                with config.open("rb") as config_file:
                    config_digest = hashlib.file_digest(config_file, "sha256").hexdigest()
                if config_digest != args.config_sha256.lower():
                    raise ValueError("DINO config SHA-256 mismatch; config was not executed")
                model = GroundingDINO(ontology=CaptionOntology({"car": "car"}))
                predictions = model.predict(str(args.image.resolve()))
                result["detections"] = len(predictions)
                result["config_sha256"] = config_digest
            result["model_smoke"] = True
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        result["pip_check"] = check.stdout.strip() or check.stderr.strip()
        if check.returncode:
            raise RuntimeError("pip check failed")
        result["status"] = "pass"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
