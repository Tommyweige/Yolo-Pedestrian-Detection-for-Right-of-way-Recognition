"""Find, for each label inside a zoomed replay, the matching moment in the original footage (SIFT + homography inliers).
Writes replay_map.json and replay_map/<clip>_<t>.jpg (replay frame | best original frame). Local only."""
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
import cw_complete as cc
HERE = Path(__file__).parent; OUTD = HERE / "replay_map"; OUTD.mkdir(exist_ok=True)
spans = json.loads((HERE / "replay_spans.json").read_text(encoding="utf-8"))
sift = cv2.SIFT_create(1500); bf = cv2.BFMatcher()

def grab(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    return cv2.resize(im, (640, 360)) if ok else None

def feats(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); g[300:] = 0  # ignore bottom overlay strip
    return sift.detectAndCompute(g, None)

def inliers(a, b):
    if a[1] is None or b[1] is None or len(a[0]) < 10 or len(b[0]) < 10: return 0
    good = [m for m, n in (p for p in bf.knnMatch(a[1], b[1], k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8: return len(good) // 2
    H, msk = cv2.findHomography(np.float32([a[0][m.queryIdx].pt for m in good]), np.float32([b[0][m.trainIdx].pt for m in good]), cv2.RANSAC, 5.0)
    return int(msk.sum()) if msk is not None else 0

res = {}
for stem, lab, fps, w, h, tracks in cc.clips():
    sp = next((v for k, v in spans.items() if stem.startswith(k)), [])
    reps = [t for t in (r["time_s"] for r in cc.ev.load(lab)) if any(a <= t < b for a, b in sp)]
    if not reps: continue
    video = next(p for p in cc.VIDS.iterdir() if p.stem == stem); cap = cv2.VideoCapture(str(video))
    end = int(sp[0][0] * fps)
    orig = [(f, feats(im)) for f in range(0, end, 2) if (im := grab(cap, f)) is not None]
    for t in reps:
        rows = []
        for dt in (-0.5, 0.0, 0.5):  # three replay frames, to check speed / consistency
            q = grab(cap, int(round((t + dt) * fps)))
            sc = np.array([inliers(feats(q), o[1]) for o in orig])
            bi = int(sc.argmax()); rows.append((dt, orig[bi][0] / fps, int(sc[bi]), int(np.sort(sc)[-10]) if len(sc) > 10 else 0))
        best = rows[1]
        res.setdefault(stem, []).append({"replay_t": t, "orig_t": round(best[1], 3), "inliers": best[2], "10th_best": best[3],
                                          "neighbours": [(d, round(o, 2), s) for d, o, s, _ in rows]})
        print(stem[:8], t, "->", round(best[1], 2), "inl", best[2], "10th", best[3], "nbrs", [(d, round(o, 2), s) for d, o, s, _ in rows], flush=True)
        a = grab(cap, int(round(t * fps))); b = grab(cap, int(round(best[1] * fps)))
        cv2.imwrite(str(OUTD / f"{stem[:6]}_{t}.jpg"), np.hstack([a, b]))
(HERE / "replay_map.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
