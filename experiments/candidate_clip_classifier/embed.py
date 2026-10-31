"""Frozen Kinetics-400 video backbones as feature extractors: R(2+1)D-18 (torchvision) and TimeSformer-base (HF).

With ~50 positive windows from 7 clips nothing this size can be fine-tuned (README: sample size), so the backbones stay
frozen and only a linear head is fit per fold. Two inputs per backbone:
- rgb: the 9 RGB frames.
- dim: the thesis's V1 prior, RGB with everything outside the prior channel dimmed to 30%. A frozen RGB backbone has
  no weights for a 4th channel, so V1 is the only way the prior reaches it.
Writes DATA/emb_<backbone>_<input>.npy, rows in samples.json order.
Usage: python embed.py --data DIR [--backbones r2p1d,timesformer]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DIM = 0.3


def load_clips(data):
    samples = json.loads((Path(data) / "samples.json").read_text(encoding="utf-8"))
    files = {}
    for s in samples:
        files.setdefault(s["clip_file"], np.load(Path(data) / s["clip_file"]))
    return samples, (files[s["clip_file"]][s["key"]] for s in samples)


def as_input(clip, variant):
    rgb = clip[..., :3].astype(np.float32) / 255
    if variant == "dim":
        rgb = rgb * np.where(clip[..., 3:4] > 0, 1.0, DIM).astype(np.float32)
    return rgb  # (T, H, W, 3) float32


def r2p1d(device):
    import torchvision
    model = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)

    def run(x):  # (B, T, H, W, 3) -> (B, 512)
        x = x.permute(0, 4, 1, 2, 3)
        x = torch.nn.functional.interpolate(x, size=(x.shape[2], 112, 112), mode="trilinear", align_corners=False)
        return model((x - mean) / std)
    return run


def timesformer(device):
    from transformers import TimesformerModel
    model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400").eval().to(device)
    pick = torch.from_numpy(np.linspace(0, 8, model.config.num_frames).round().astype(int))  # 9 -> 8 frames

    def run(x):  # (B, T, H, W, 3) -> (B, 768): the CLS token
        x = x[:, pick].permute(0, 1, 4, 2, 3)
        return model(pixel_values=(x - 0.45) / 0.225).last_hidden_state[:, 0]
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--backbones", default="r2p1d,timesformer")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.backbones.split(","):
        run = {"r2p1d": r2p1d, "timesformer": timesformer}[name](device)
        for variant in ("rgb", "dim"):
            samples, clips = load_clips(args.data)
            out, batch = [], []
            for i, clip in enumerate(clips):
                batch.append(as_input(clip, variant))
                if len(batch) == args.batch or i == len(samples) - 1:
                    with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
                        out.append(run(torch.from_numpy(np.stack(batch)).to(device)).float().cpu().numpy())
                    batch.clear()
                if i % 100 == 0:
                    print(f"{name} {variant} {i}/{len(samples)}", flush=True)
            np.save(Path(args.data) / f"emb_{name}_{variant}.npy", np.concatenate(out))
            print(f"{name} {variant} done {np.concatenate(out).shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
