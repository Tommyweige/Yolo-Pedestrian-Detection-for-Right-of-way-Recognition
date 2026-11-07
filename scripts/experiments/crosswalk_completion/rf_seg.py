"""Crosswalk masks for the Roboflow images of train-v1 (letterboxed 640x640) with the pipeline's Mask2Former.
Keeps the central 16:9 band (rows 140-500) as a 640x360 mask, plus the person/vehicle label boxes in that band.
Output: <out>/rf_masks.npz (packed bool, key = file stem), <out>/rf_boxes.json. Run in the traffic-labeling env."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "datasets/train-v1"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
Y0, Y1 = 140, 500
proc = AutoImageProcessor.from_pretrained(ROOT / "runtime/models/mask2former-mapillary")
model = Mask2FormerForUniversalSegmentation.from_pretrained(ROOT / "runtime/models/mask2former-mapillary", use_safetensors=True).to("cuda").half().eval()
paths = sorted(DS.glob("images/*/rf_*.jpg"))
masks, boxes, t0 = {}, {}, time.time()
log = open(OUT / "rf_seg.log", "w", encoding="utf-8")
for n, p in enumerate(paths, 1):
    im = cv2.imread(str(p))
    inputs = proc(images=cv2.cvtColor(im, cv2.COLOR_BGR2RGB), return_tensors="pt", do_resize=False)
    inputs = {k: (v.to("cuda").half() if v.dtype == torch.float32 else v.to("cuda")) for k, v in inputs.items()}
    with torch.no_grad():
        lab = proc.post_process_semantic_segmentation(model(**inputs), target_sizes=[im.shape[:2]])[0].cpu().numpy()
    m = np.isin(lab, (23, 8))[Y0:Y1]
    masks[p.stem] = np.packbits(m)
    bl = []
    lp = DS / "labels" / p.parent.name / (p.stem + ".txt")
    for line in (lp.read_text().splitlines() if lp.exists() else []):
        c, cx, cy, w, h = line.split()
        if int(c) > 4:
            continue
        x1, x2 = (float(cx) - float(w) / 2) * 640, (float(cx) + float(w) / 2) * 640
        y1, y2 = (float(cy) - float(h) / 2) * 640 - Y0, (float(cy) + float(h) / 2) * 640 - Y0
        if y2 > 0 and y1 < Y1 - Y0:
            bl.append([round(max(0, x1)), round(max(0, y1)), round(min(640, x2)), round(min(360, y2))])
    boxes[p.stem] = bl
    if n % 100 == 0 or n == len(paths):
        s = f"{n}/{len(paths)} {time.time() - t0:.0f}s  with crosswalk: {sum(np.unpackbits(v).mean() > 0.003 for v in masks.values())}"
        print(s, flush=True); log.write(s + "\n"); log.flush()
np.savez_compressed(OUT / "rf_masks.npz", **masks)
(OUT / "rf_boxes.json").write_text(json.dumps(boxes), encoding="utf-8")
print("done", flush=True); log.write("done\n")
