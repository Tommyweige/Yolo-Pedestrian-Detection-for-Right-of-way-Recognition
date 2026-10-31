"""Small 3D CNN trained from scratch, leave-one-video-out: RGB (3 channels) vs RGB + prior (4-channel early fusion).

The thesis's early fusion concatenates the prior as a 4th input channel and trains the network. A pretrained backbone
cannot be fine-tuned on ~50 positives, so this is the lower-bound version of that experiment: ~0.2M parameters, 9 x
112 x 112, trained per fold on the 6 other clips (candidates with a known label, vehicle-pass negatives, shifted
positive windows), class-balanced loss, decision threshold 0.5 (a nested threshold would need 6 more trainings per
fold; the threshold-free AUROC/AP and the @28 budget in score.py do not depend on it). Probabilities are the mean of
3 seeds.
Writes OUT/oof_cnn_<rgb|fused>.json in lovo.py's format.
Usage: python cnn.py --data DIR --out DIR [--epochs 25] [--seeds 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

SIZE = 112


def net(channels):
    def block(i, o, k, s):
        return [nn.Conv3d(i, o, k, s, padding=tuple(x // 2 for x in k)), nn.BatchNorm3d(o), nn.ReLU(inplace=True)]
    return nn.Sequential(*block(channels, 16, (3, 5, 5), (1, 2, 2)), *block(16, 32, (3, 3, 3), (1, 2, 2)),
                         *block(32, 64, (3, 3, 3), (2, 2, 2)), *block(64, 64, (3, 3, 3), (2, 2, 2)),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 1))


def load(data):
    samples = json.loads((Path(data) / "samples.json").read_text(encoding="utf-8"))
    files, clips = {}, []
    for s in samples:
        files.setdefault(s["clip_file"], np.load(Path(data) / s["clip_file"]))
        clip = torch.from_numpy(files[s["clip_file"]][s["key"]]).permute(3, 0, 1, 2).float() / 255  # (4, T, H, W)
        clips.append(nn.functional.interpolate(clip, size=(SIZE, SIZE), mode="area"))
    x = torch.stack(clips)
    x[:, :3] = (x[:, :3] - 0.45) / 0.225
    x[:, 3] = x[:, 3] * 2 - 1
    return samples, x


def train_predict(x_train, y_train, x_test, channels, epochs, seed, device):
    torch.manual_seed(seed)
    model = net(channels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    positive = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - positive) / max(positive, 1), device=device))
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(y_train)).split(16):
            xb = x_train[batch, :channels].to(device)
            flip = torch.rand(len(batch), device=device) < 0.5
            xb[flip] = xb[flip].flip(-1)
            loss = loss_fn(model(xb).squeeze(1), y_train[batch].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([model(b[:, :channels].to(device)) for b in x_test.split(32)]).squeeze(1)).cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, x = load(args.data)
    stems = np.array([s["stem"] for s in samples])
    role = np.array([s["role"] for s in samples])
    y = np.array([-1 if s["y"] is None else s["y"] for s in samples])
    for name, channels in (("rgb", 3), ("fused", 4)):
        result = {"scores": {}, "thresholds": {}, "C": {}}
        for held in sorted(set(stems)):
            train = np.nonzero((y >= 0) & (stems != held))[0]
            test = np.nonzero((stems == held) & (role == "candidate"))[0]
            p = np.mean([train_predict(x[train], torch.tensor(y[train], dtype=torch.float32), x[test], channels,
                                       args.epochs, seed, device) for seed in range(args.seeds)], axis=0)
            result["scores"].update({samples[i]["id"]: float(v) for i, v in zip(test, p)})
            result["thresholds"][held], result["C"][held] = 0.5, None
            print(f"cnn {name} held {held[:10]} train {len(train)} test {len(test)}", flush=True)
        (Path(args.out) / f"oof_cnn_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
