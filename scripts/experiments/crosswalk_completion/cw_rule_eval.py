"""Rule F1 with the learned completion used for the pedestrian test. Each clip uses the fold model that never saw it
(fold order = cc.clips() order, as in cw_train). Threshold fixed at 0.5 in advance.
Usage: python cw_rule_eval.py <train_out>"""
import sys, ast
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cw_complete as cc

# reuse the UNet definition from cw_train without running its training code
src = (HERE / "cw_train.py").read_text(encoding="utf-8")
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "UNet")
exec(compile(ast.Module([cls], []), "cw_train_UNet", "exec"), globals())

OUT = Path(sys.argv[1]); H, W = 180, 320
stems = [c[0] for c in cc.clips()]


def learned(kinds_fill=None):
    def make(stem, frames, masks, boxes, tracks):
        net = UNet().cuda(); net.load_state_dict(torch.load(OUT / f"fold_{stems.index(stem)}.pt")); net.eval()
        fill_boxes = boxes if kinds_fill is None else cc.box_region(tracks, *size[stem], kinds=kinds_fill)
        out = {}
        with torch.no_grad():
            for f in frames:
                u_full = cc.unknown_at(boxes, f)
                if not u_full.any():
                    continue
                m = masks[f]
                md = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                ud = cv2.resize(u_full.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                x = torch.from_numpy(np.stack([md & ~ud, ud]).astype(np.float32))[None].cuda()
                p = torch.sigmoid(net(x))[0, 0].cpu().numpy()
                pf = cv2.resize(p, (cc.SHAPE[1], cc.SHAPE[0]), interpolation=cv2.INTER_LINEAR) > 0.5
                out[f] = m | (pf & cc.unknown_at(fill_boxes, f))
        return out
    return make


size = {stem: (w, h) for stem, _, _, w, h, _ in cc.clips()}
if __name__ == "__main__":
    cc.run_variant(None, label="default")
    cc.run_variant(learned(), label="learned fill, all boxes, ped test only")
    cc.run_variant(learned({"person"}), label="learned fill, person boxes only")
