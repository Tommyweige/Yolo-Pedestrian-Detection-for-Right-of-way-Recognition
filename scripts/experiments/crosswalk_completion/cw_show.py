"""Render synthetic training samples: frame | input (visible mask white, unknown red) | target | model fill (if a fold model exists).
Usage: python cw_show.py <out.jpg> [fold_model.pt clip_index]"""
import sys, ast
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cw_complete as cc
src = (HERE / "cw_train.py").read_text(encoding="utf-8")
tree = ast.parse(src)
for n in tree.body:
    if (isinstance(n, ast.ClassDef) and n.name == "UNet") or (isinstance(n, ast.FunctionDef) and n.name == "holes"):
        exec(compile(ast.Module([n], []), "cw_train", "exec"), globals())
H, W = 180, 320
out_path = sys.argv[1]
net = None
clip_i = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if len(sys.argv) > 2:
    net = UNet().cuda(); net.load_state_dict(torch.load(sys.argv[2])); net.eval()
stem, lab, fps, w, h, tracks = list(cc.clips())[clip_i]
frames, masks = cc.load_masks(stem)
boxes = cc.box_region(tracks, w, h)
video = next(p for p in cc.VIDS.iterdir() if p.stem == stem)
cap = cv2.VideoCapture(str(video))
rng = np.random.default_rng(7)
sizes = np.array([((x2 - x1) / 2, (y2 - y1) / 2) for bl in boxes.values() for x1, y1, x2, y2 in bl], np.float32)
good = [f for f in frames if masks[f].mean() > 0.02]
rows = []
for f in rng.choice(good, 6, replace=False):
    m = cv2.resize(masks[f].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    r = cv2.resize(cc.unknown_at(boxes, f).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    hole = holes(m, sizes, rng); unk = hole | r; vis = m & ~unk
    cap.set(1, int(f) - 1); ok, im = cap.read()
    im = cv2.resize(im, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
    for x1, y1, x2, y2 in boxes.get(f, []):
        cv2.rectangle(im, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 255), 1)
    inp = np.zeros((H, W, 3), np.uint8); inp[vis] = 255
    inp[hole] = (0, 0, 200); inp[r & ~hole] = (0, 120, 255)
    tgt = np.zeros((H, W, 3), np.uint8); tgt[m] = 255; tgt[hole & ~r & m] = (0, 220, 0)
    panels = [im, inp, tgt]
    if net is not None:
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(np.stack([vis, unk]).astype(np.float32))[None].cuda()))[0, 0].cpu().numpy() > 0.5
        pr = np.zeros((H, W, 3), np.uint8); pr[vis] = 255
        pr[p & unk & m] = (0, 220, 0); pr[p & unk & ~m] = (0, 0, 255)
        panels.append(pr)
    rows.append(np.hstack([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=60) for x in panels]))
sheet = np.vstack(rows)
heads = ["frame + boxes", "input: white=visible, red=synthetic hole, orange=real box", "target (green = scored hole)", "model fill: green right, red wrong"]
bar = np.full((22, sheet.shape[1], 3), 30, np.uint8)
for i, t in enumerate(heads[:len(rows[0][0]) and (4 if net is not None else 3)]):
    cv2.putText(bar, t, (6 + i * (W + 4), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
cv2.imwrite(out_path, np.vstack([bar, sheet]))
print("wrote", out_path)
