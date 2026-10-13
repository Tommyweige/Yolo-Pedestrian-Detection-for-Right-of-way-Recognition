"""Per-frame crosswalk segmentation with a Mapillary-Vistas Mask2Former (GPU only), as an overlay montage + stats.

Classes used: 23 'Lane Marking - Crosswalk' (zebra stripes) and 8 'Crosswalk - Plain'. The model is pinned by revision
and loaded from safetensors in --model. Mapillary Vistas is CC BY-NC-SA: research/demo use only.
Pinned: facebook/mask2former-swin-large-mapillary-vistas-semantic @ 4772b6bf101d91f2534c106dc524d906aeb3c68a,
model.safetensors SHA-256 72721afb6894c0b1239b9259c428996d9ca0aecbebc4efb4a2deaae0403e2ea7.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

STRIPES, PLAIN = 23, 8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="folder holding config.json + model.safetensors")
    parser.add_argument("--frames", type=int, nargs="+", help="one-based frame numbers")
    parser.add_argument("--stride", type=int, help="instead of --frames: every Nth frame of the whole video; also saves masks.npz")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--short-edge", type=int, default=720)
    args = parser.parse_args(argv)
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    if not torch.cuda.is_available():
        print("CUDA is unavailable; refusing to run on CPU.", file=sys.stderr)
        return 2
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model, use_safetensors=True).to("cuda").half().eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    if args.stride:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        args.frames = list(range(1, total + 1, args.stride))
    if not args.frames:
        print("give --frames or --stride", file=sys.stderr)
        return 2
    masks = {}
    tiles, stats = [], []
    for number in args.frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        scale = args.short_edge / min(height, width)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)))
        inputs = processor(images=cv2.cvtColor(small, cv2.COLOR_BGR2RGB), return_tensors="pt", do_resize=False)
        inputs = {k: (v.to("cuda").half() if v.dtype == torch.float32 else v.to("cuda")) for k, v in inputs.items()}
        started = time.time()
        with torch.no_grad():
            result = model(**inputs)
        labels = processor.post_process_semantic_segmentation(result, target_sizes=[small.shape[:2]])[0].cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.time() - started
        mask = np.isin(labels, (STRIPES, PLAIN))
        overlay = small.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(overlay, f"f{number}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(cv2.resize(overlay, (640, 360)))
        masks[f"f{number}"] = np.packbits(cv2.resize(mask.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST).astype(bool))
        rows = np.where(mask.any(axis=1))[0]
        stats.append({"frame": number, "crosswalk_pixel_fraction": float(mask.mean()), "seconds": round(elapsed, 3),
                      "lowest_crosswalk_row_fraction": float(rows.max() / mask.shape[0]) if len(rows) else None})
    if args.stride:
        np.savez_compressed(out / "masks.npz", **masks)  # packed 640x360 booleans, key f<frame>
        tiles = tiles[:30]  # montage only needs a sample
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "montage.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({"frames": len(stats), "mean_seconds": round(float(np.mean([s["seconds"] for s in stats])), 3)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
