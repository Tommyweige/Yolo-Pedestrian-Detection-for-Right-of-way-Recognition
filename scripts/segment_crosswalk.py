"""Per-frame crosswalk segmentation with a Mapillary-Vistas Mask2Former (GPU only), as an overlay montage + stats.

Classes used: 23 'Lane Marking - Crosswalk' (zebra stripes) and 8 'Crosswalk - Plain'. The model is pinned by revision
and loaded from safetensors in --model. Mapillary Vistas is CC BY-NC-SA: research/demo use only.
Pinned: facebook/mask2former-swin-large-mapillary-vistas-semantic @ 4772b6bf101d91f2534c106dc524d906aeb3c68a,
model.safetensors SHA-256 72721afb6894c0b1239b9259c428996d9ca0aecbebc4efb4a2deaae0403e2ea7.

Speed: frames are decoded in order (skipped ones only grabbed) on a background thread while the GPU works, and the
image normalisation runs on the GPU. Seeking with CAP_PROP_POS_FRAMES returned the same frames but cost ~0.1 s each.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

STRIPES, PLAIN = 23, 8
MONTAGE_TILES = 30  # with --stride the montage only needs a sample


def sampled_frames(capture, numbers):
    """Yield (number, frame) for the wanted one-based frame numbers, reading the video front to back once."""
    position = 0
    for number in sorted(set(numbers)):
        while position < number - 1:
            if not capture.grab():
                return
            position += 1
        ok, frame = capture.read()
        position += 1
        if not ok:
            return
        yield number, frame


def stride_for(fps, seconds):
    """Frames between masks so one is taken every `seconds` (rounded, at least 1; unknown fps -> every frame)."""
    return max(1, int(fps * seconds + 0.5))


def prefetch(items, depth=8):
    """Run the iterator on a background thread so decoding overlaps the GPU; errors are re-raised here."""
    buffer, done = queue.Queue(maxsize=depth), object()

    def fill():
        try:
            for item in items:
                buffer.put(item)
        except BaseException as error:  # noqa: BLE001 - handed to the consumer
            buffer.put((done, error))
            return
        buffer.put((done, None))

    threading.Thread(target=fill, daemon=True).start()
    while True:
        item = buffer.get()
        if isinstance(item, tuple) and len(item) == 2 and item[0] is done:
            if item[1] is not None:
                raise item[1]
            return
        yield item


def normalize_constants(config):
    """(scale, mean, std) as float32 arrays from preprocessor_config.json, same maths as the HF image processor."""
    mean = np.array(config["image_mean"], np.float32)
    std = np.array(config["image_std"], np.float32)
    return np.float32(config["rescale_factor"]), mean, std


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="folder holding config.json + model.safetensors")
    parser.add_argument("--frames", type=int, nargs="+", help="one-based frame numbers")
    parser.add_argument("--stride", type=int, help="instead of --frames: every Nth frame of the whole video; also saves masks.npz")
    parser.add_argument("--every-seconds", type=float, help="like --stride, but one mask per this many seconds of video "
                        "whatever the fps (0.1 s: stride 3 at 25-30 fps, 6 at 60 fps)")
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
    config = json.loads((Path(args.model) / "preprocessor_config.json").read_text(encoding="utf-8"))
    scale, mean, std = normalize_constants(config)
    mean = torch.from_numpy(mean).view(1, 3, 1, 1).to("cuda")
    std = torch.from_numpy(std).view(1, 3, 1, 1).to("cuda")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    if args.every_seconds:
        args.stride = stride_for(capture.get(cv2.CAP_PROP_FPS), args.every_seconds)
        print(f"stride {args.stride} (one mask per {args.every_seconds} s)", flush=True)
    if args.stride:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        args.frames = list(range(1, total + 1, args.stride))
    if not args.frames:
        print("give --frames or --stride", file=sys.stderr)
        return 2

    def prepared():
        for number, frame in sampled_frames(capture, args.frames):
            height, width = frame.shape[:2]
            factor = args.short_edge / min(height, width)
            small = cv2.resize(frame, (round(width * factor), round(height * factor)))
            yield number, small, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    masks = {}
    tiles, stats = [], []
    for number, small, rgb in prefetch(prepared()):  # one frame per call: batching was 4% faster but changed masks
        started = time.time()
        rgb = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1).unsqueeze(0).float()
        # fp32 maths then fp16, as the HF processor path did; contiguous because a channels-last tensor makes cuDNN
        # pick other fp16 kernels, which moved crosswalk edges in two thirds of the masks
        pixels = ((rgb * float(scale) - mean) / std).half().contiguous()
        with torch.no_grad():
            result = model(pixel_values=pixels)
        labels = processor.post_process_semantic_segmentation(result, target_sizes=[small.shape[:2]])[0].cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.time() - started
        mask = np.isin(labels, (STRIPES, PLAIN))
        if not args.stride or len(tiles) < MONTAGE_TILES:
            overlay = small.copy()
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.putText(overlay, f"f{number}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            tiles.append(cv2.resize(overlay, (640, 360)))
        masks[f"f{number}"] = np.packbits(cv2.resize(mask.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST).astype(bool))
        rows = np.where(mask.any(axis=1))[0]
        stats.append({"frame": number, "crosswalk_pixel_fraction": float(mask.mean()), "seconds": round(elapsed, 3),
                      "lowest_crosswalk_row_fraction": float(rows.max() / mask.shape[0]) if len(rows) else None})
    capture.release()
    if not stats:
        print("no frames could be read", file=sys.stderr)
        return 2
    if args.stride:
        np.savez_compressed(out / "masks.npz", **masks)  # packed 640x360 booleans, key f<frame>
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "montage.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({"frames": len(stats), "mean_seconds": round(float(np.mean([s["seconds"] for s in stats])), 3)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
