"""Leave-one-video-out scores for every candidate, from linear heads on hand-made or frozen-backbone features.

The thesis's discipline is grouping folds by video; with 7 clips the strictest version is one clip per fold. For each
held-out clip the head is fit on the other 6 only, and everything chosen from data is chosen inside those 6 by an inner
leave-one-video-out: the L2 strength C (by average precision) and the decision threshold (by F1). The held-out clip's
labels are never read until scoring.

Training rows: candidates with a known label, plus (with --extra) the vehicle-pass negatives and the shifted positive
windows of the training clips. Scored rows: every candidate of the held-out clip, including unknown ones (at inference
nothing is known). Hand-made features exist only for candidates, so the `hand` head never uses the extras.
Writes OUT/oof_<model>.json: {"scores": {sample id: p}, "thresholds": {stem: t}, "C": {stem: C}}.
Usage: python lovo.py --data DIR --out DIR [--models hand,r2p1d_rgb,...] [--extra]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID_C = (0.003, 0.03, 0.3, 3.0)


def features(samples, data, model):
    if model == "hand":
        names = sorted(next(s for s in samples if s["role"] == "candidate")["features"])
        rows = [[s["features"][n] for n in names] if s["role"] == "candidate" else None for s in samples]
        return np.array([r if r is not None else [np.nan] * len(names) for r in rows], float)
    return np.load(Path(data) / f"emb_{model}.npy")


def fit(x, y, c):
    head = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
    return head.fit(x, y)


def best_threshold(y, p):
    grid = np.unique(np.round(p, 4))
    scores = [f1_score(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))]) if len(grid) else 0.5


def inner_select(x, y, groups, scored):
    """C by mean inner-fold AP, then the F1-best threshold on the pooled inner out-of-fold scores of `scored` rows."""
    best = None
    for c in GRID_C:
        oof = np.full(len(y), np.nan)
        for g in np.unique(groups):
            train, test = groups != g, (groups == g) & scored
            if test.sum() == 0 or len(np.unique(y[train])) < 2:
                continue
            oof[test] = fit(x[train], y[train], c).predict_proba(x[test])[:, 1]
        known = scored & ~np.isnan(oof)
        ap = average_precision_score(y[known], oof[known]) if len(np.unique(y[known])) == 2 else 0
        if best is None or ap > best[0]:
            best = (ap, c, best_threshold(y[known], oof[known]))
    return best[1], best[2]


def run(samples, x, extra):
    stems = np.array([s["stem"] for s in samples])
    role = np.array([s["role"] for s in samples])
    y = np.array([-1 if s["y"] is None else s["y"] for s in samples])
    usable = (y >= 0) & ~np.isnan(x).any(axis=1) & ((role == "candidate") | extra)
    result = {"scores": {}, "thresholds": {}, "C": {}}
    for held in sorted(set(stems)):
        train = usable & (stems != held)
        c, threshold = inner_select(x[train], y[train], stems[train], role[train] == "candidate")
        head = fit(x[train], y[train], c)
        test = (stems == held) & (role == "candidate")
        for i, p in zip(np.nonzero(test)[0], head.predict_proba(x[test])[:, 1]):
            result["scores"][samples[i]["id"]] = float(p)
        result["thresholds"][held], result["C"][held] = threshold, c
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="hand,r2p1d_rgb,r2p1d_dim,timesformer_rgb,timesformer_dim")
    parser.add_argument("--extra", action="store_true", help="also train on vehicle-pass negatives and shifted windows")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    samples = json.loads((Path(args.data) / "samples.json").read_text(encoding="utf-8"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        result = run(samples, features(samples, args.data, model), args.extra)
        name = model + ("+extra" if args.extra else "")
        (Path(args.out) / f"oof_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: C {sorted(set(result['C'].values()))} thresholds "
              f"{[round(t, 3) for t in result['thresholds'].values()]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
