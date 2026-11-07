"""Train a small mask-completion UNet per held-out clip (7 folds) on synthetic rectangular holes.
Input: [visible mask, unknown]; target: the stored mask; loss only on synthetic-hole pixels not under a real detection box.
Holes are sized like real detection boxes and centred mostly on the crosswalk edge, so the model sees band ends / kerbs.
Usage: python cw_train.py <out_dir> [steps]. Local only (reads private data)."""
import sys, time, math
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import cw_complete as cc

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
H, W = 180, 320
dev = "cuda"
log = open(OUT / "train.log", "a", encoding="utf-8")


def say(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()


def down(m):
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


# ---- data: per clip, stored masks + real-box unknown region + box sizes, at 180x320
data = {}
for stem, lab, fps, w, h, tracks in cc.clips():
    frames, masks = cc.load_masks(stem)
    boxes = cc.box_region(tracks, w, h)
    M = np.stack([down(masks[f]) for f in frames])
    R = np.stack([down(cc.unknown_at(boxes, f)) for f in frames])
    sizes = np.array([((x2 - x1) / 2, (y2 - y1) / 2) for bl in boxes.values() for x1, y1, x2, y2 in bl], np.float32)
    keep = M.reshape(len(M), -1).mean(1) > 0.003
    data[stem] = (M[keep], R[keep], sizes)
    say(f"clip {stem[:10]} stored {len(frames)} usable {keep.sum()} boxes {len(sizes)}")
stems = list(data)
RF = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if RF:  # Roboflow screenshots: extra crosswalk shapes, training only (never a held-out fold)
    import json
    z = np.load(RF / "rf_masks.npz"); bx = json.loads((RF / "rf_boxes.json").read_text(encoding="utf-8"))
    Ms, Rs, sz = [], [], []
    for k in z.files:
        m = np.unpackbits(z[k])[:360 * 640].astype(bool).reshape(360, 640)
        if m.mean() <= 0.003:
            continue
        r = np.zeros((360, 640), bool)
        for x1, y1, x2, y2 in bx.get(k, []):
            r[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = True
        Ms.append(down(m)); Rs.append(down(r))
        sz += [((x2 - x1) / 2, (y2 - y1) / 2) for x1, y1, x2, y2 in bx.get(k, [])]
    data["roboflow"] = (np.stack(Ms), np.stack(Rs), np.concatenate([data[s][2] for s in stems]))  # dashcam box sizes for holes
    say(f"roboflow usable {len(Ms)}")


def holes(m, sizes, rng, n=None):
    """Synthetic rectangular holes: real box sizes, 60% centred on the mask edge, 25% inside, 15% anywhere."""
    hole = np.zeros((H, W), bool)
    edge = np.argwhere(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
    inside = np.argwhere(m)
    for _ in range(n or rng.integers(1, 5)):
        bw, bh = sizes[rng.integers(len(sizes))]
        r = rng.random()
        pool = edge if r < 0.6 and len(edge) else inside if r < 0.85 and len(inside) else None
        if pool is not None:
            cy, cx = pool[rng.integers(len(pool))]
            cy += rng.integers(-int(bh * 0.6) - 1, int(bh * 0.6) + 1)  # boxes straddle the edge: bottom (foot) near it
        else:
            cy, cx = rng.integers(H), rng.integers(W)
        x1, x2 = int(max(0, cx - bw / 2)), int(min(W, cx + bw / 2 + 1))
        y1, y2 = int(max(0, cy - bh)), int(min(H, cy + 1))
        hole[y1:y2, x1:x2] = True
    return hole


def batch(clip_list, bs, rng):
    xs, ys, ws = [], [], []
    for _ in range(bs):
        M, R, sizes = data[clip_list[rng.integers(len(clip_list))]]
        i = rng.integers(len(M))
        m, r = M[i], R[i]
        if rng.random() < 0.5:
            m, r = m[:, ::-1], r[:, ::-1]
        hole = holes(m, sizes, rng)
        unk = hole | r
        xs.append(np.stack([m & ~unk, unk]).astype(np.float32)); ys.append(m.astype(np.float32)); ws.append((hole & ~r).astype(np.float32))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).to(dev)
    return t(xs), t(ys)[:, None], t(ws)[:, None]


class UNet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        ch = [c, 2 * c, 4 * c, 8 * c, 8 * c]
        s.enc = nn.ModuleList([blk(2, ch[0])] + [blk(ch[i], ch[i + 1]) for i in range(4)])
        s.dec = nn.ModuleList([blk(ch[i + 1] + ch[i], ch[i]) for i in range(4)])
        s.head = nn.Conv2d(ch[0], 1, 1)

    def forward(s, x):
        feats = []
        for i, e in enumerate(s.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); feats.append(x)
        for i in reversed(range(4)):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = s.dec[i](torch.cat([x, feats[i]], 1))
        return s.head(x)


def train(train_clips, tag, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = UNet().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
    t0 = time.time(); run = 0
    for step in range(1, STEPS + 1):
        x, y, wt = batch(train_clips, 16, rng)
        loss = (F.binary_cross_entropy_with_logits(net(x), y, reduction="none") * wt).sum() / wt.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0:
            say(f"[{tag}] step {step}/{STEPS} loss {run:.4f} {time.time() - t0:.0f}s")
    torch.save(net.state_dict(), OUT / f"{tag}.pt")
    return net


def hull_fill(vis, unk):
    out = vis.copy()
    n, lab = cv2.connectedComponents(cv2.morphologyEx(vis.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8)))
    for i in range(1, n):
        pts = np.argwhere(lab == i)[:, ::-1].astype(np.int32)
        if len(pts) < 20:
            continue
        hm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(hm, cv2.convexHull(pts), 1)
        out |= hm.astype(bool) & unk
    return out


@torch.no_grad()
def evaluate(net, clip, tag):
    M, R, sizes = data[clip]
    rng = np.random.default_rng(123)
    idx = rng.choice(len(M), min(300, len(M)), replace=False)
    acc = {k: np.zeros(4) for k in ("zero", "close", "hull", "model")}  # TP, FP, FN in scored hole; FP only
    net.eval()
    for i in idx:
        m, r = M[i], R[i]
        hole = holes(m, sizes, rng); unk = hole | r; score = hole & ~r
        vis = m & ~unk
        x = torch.from_numpy(np.stack([vis, unk]).astype(np.float32))[None].to(dev)
        pm = (torch.sigmoid(net(x))[0, 0].cpu().numpy() > 0.5) & unk | vis
        preds = {"zero": vis, "close": vis | (cv2.morphologyEx(vis.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8)).astype(bool) & unk),
                 "hull": hull_fill(vis, unk), "model": pm}
        for k, p in preds.items():
            acc[k] += [(p & m & score).sum(), (p & ~m & score).sum(), (~p & m & score).sum(), score.sum()]
    net.train()
    out = {}
    for k, (tp, fp, fn, n) in acc.items():
        out[k] = (tp / max(1, tp + fp + fn), fp / max(1, tp + fp))  # overfill = share of filled hole pixels that are not crosswalk
    say(f"[{tag}] held-out {clip[:10]}: " + "  ".join(f"{k} IoU {v[0]:.3f} overfill {v[1]:.4f}" for k, v in out.items()))
    return out


if __name__ == "__main__":
    res = {}
    for held in stems:
        tag = f"fold_{stems.index(held)}"
        net = train([s for s in stems if s != held] + (["roboflow"] if RF else []), tag)
        res[held] = evaluate(net, held, tag)
    say("MEAN over folds: " + "  ".join(
        f"{k} IoU {np.mean([r[k][0] for r in res.values()]):.3f} overfill {np.mean([r[k][1] for r in res.values()]):.4f}" for k in res[stems[0]]))
    say("done")
