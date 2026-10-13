"""Fine-tune YOLO26n on the assembled 7-class set. GPU only: refuses to start without CUDA.

Run in the modern env with a checkpoint whose SHA-256 you verified (docs/environments.md).
Outputs go to --project (keep it outside Git); the best.pt must be re-validated before any claim.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="data.yaml from build_training_set.py")
    parser.add_argument("--weights", required=True, help="verified yolo26n.pt")
    parser.add_argument("--sha256", required=True, help="expected SHA-256 of --weights")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="yolo26n-traffic")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        print("CUDA is unavailable; refusing to train on CPU.", file=sys.stderr)
        return 2
    if sha256(args.weights) != args.sha256.lower():
        print("Checkpoint SHA-256 mismatch; not loading it.", file=sys.stderr)
        return 2
    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                device=0, amp=True, cache="ram", seed=args.seed, deterministic=False, patience=20,
                project=args.project, name=args.name, exist_ok=False, plots=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
