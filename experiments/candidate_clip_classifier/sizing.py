"""How many samples would make this measurable, and how far the current data is from it.

1. Uncertainty of what we have: Wilson 95% intervals of the rule's numbers, and a clip-level bootstrap interval of a
   classifier's out-of-fold AUROC/AP (resampling whole clips, since samples within a clip are not independent).
2. Required sample sizes (normal approximation, 95% two-sided, 80% power):
   - estimate a recall or precision to +-d: n = z^2 p(1-p) / d^2,
   - tell precision p0 from p1 (two independent proportions),
   - inflate by the design effect 1 + (m - 1) ICC for m events per clip.
3. Learning curve: out-of-fold AP of the linear head when it is trained on k of the 6 other clips.
Usage: python sizing.py --data DIR --oof DIR [--model timesformer_rgb]
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lovo  # noqa: E402

Z, ZB = 1.959964, 0.841621


def wilson(k, n):
    p, z2 = k / n, Z * Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return round(centre - half, 3), round(centre + half, 3)


def n_estimate(p, d):
    return math.ceil(Z * Z * p * (1 - p) / (d * d))


def n_two_proportions(p0, p1):
    pbar = (p0 + p1) / 2
    top = Z * math.sqrt(2 * pbar * (1 - pbar)) + ZB * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return math.ceil(top * top / (p1 - p0) ** 2)


def clip_bootstrap(samples, scores, rounds=2000, seed=0):
    rng = np.random.default_rng(seed)
    by_stem = {}
    for s in samples:
        if s["role"] == "candidate" and s["y"] is not None:
            by_stem.setdefault(s["stem"], []).append((s["y"], scores[s["id"]]))
    stems = sorted(by_stem)
    aucs, aps = [], []
    for _ in range(rounds):
        rows = [r for stem in rng.choice(stems, len(stems)) for r in by_stem[stem]]
        y, p = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        if len(set(y)) == 2:
            aucs.append(roc_auc_score(y, p))
            aps.append(average_precision_score(y, p))
    return [round(float(v), 3) for v in np.percentile(aucs, [2.5, 97.5])], \
           [round(float(v), 3) for v in np.percentile(aps, [2.5, 97.5])]


def learning_curve(samples, x):
    """Pooled out-of-fold AP when each held-out clip's head sees only k of the other clips (all k-subsets)."""
    stems = np.array([s["stem"] for s in samples])
    role = np.array([s["role"] for s in samples])
    y = np.array([-1 if s["y"] is None else s["y"] for s in samples])
    known = (y >= 0) & (role == "candidate")
    out = {}
    for k in range(1, 7):
        aps = []
        for subset_index in range(10):  # up to 10 random k-subsets per held-out clip, same draw for every clip
            ys, ps = [], []
            for held in sorted(set(stems)):
                others = sorted(set(stems) - {held})
                subsets = list(itertools.combinations(others, k))
                pick = subsets[subset_index % len(subsets)]
                train = known & np.isin(stems, pick)
                test = known & (stems == held)
                if len(set(y[train])) < 2 or test.sum() == 0:
                    continue
                head = lovo.fit(x[train], y[train], 0.003)
                ys += list(y[test])
                ps += list(head.predict_proba(x[test])[:, 1])
            aps.append(average_precision_score(ys, ps))
        out[k] = (round(float(np.mean(aps)), 3), round(float(np.std(aps)), 3))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--model", default="timesformer_rgb")
    args = parser.parse_args(argv)
    samples = json.loads((Path(args.data) / "samples.json").read_text(encoding="utf-8"))
    report = {
        "rule_in_sample_wilson95": {"incident_recall_11_15": wilson(11, 15), "manual_5_9": wilson(5, 9),
                                    "human_precision_A_15_28": wilson(15, 28), "label_precision_12_28": wilson(12, 28)},
        "n_to_estimate": {f"p{p}_pm{d}": n_estimate(p, d) for p in (0.5, 0.75) for d in (0.1, 0.05)},
        "n_per_arm_to_tell_precision": {f"{p0}->{p1}": n_two_proportions(p0, p1)
                                        for p0, p1 in ((0.536, 0.75), (0.536, 0.65), (0.536, 0.85))},
        "design_effect_m4": {f"icc{icc}": round(1 + 3 * icc, 2) for icc in (0.1, 0.2, 0.3)},
    }
    oof = json.loads((Path(args.oof) / f"oof_{args.model}.json").read_text(encoding="utf-8"))
    auc_ci, ap_ci = clip_bootstrap(samples, oof["scores"])
    report[f"{args.model}_clip_bootstrap95"] = {"auroc": auc_ci, "ap": ap_ci}
    known = [s for s in samples if s["role"] == "candidate" and s["y"] is not None]
    report["candidate_base_rate"] = round(sum(s["y"] for s in known) / len(known), 3)
    report[f"{args.model}_learning_curve_ap_by_training_clips"] = learning_curve(samples, lovo.features(samples, args.data, args.model))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
