"""預測評分: how a list of predictions is scored against human labels (docs/adr/0003-fixed-rule-evaluation.md).

The one place that defines a hit, the tolerance, an incident and a duplicate report, so every script that reports a
number (the fixed evaluation, tuning, learning, diagnosis) counts the same way:

- A prediction matches a label of the same kind within the tolerance, one-to-one, closest first (`matched_pairs`).
  `vehicle_on_crosswalk_with_pedestrian` (yield_image_rule.py) is scored as `suspected_failure_to_yield`.
- The tolerance is derived from the labels-vs-predictions residual distribution (`derive_tolerance`), not chosen by
  hand (`scripts/eval_rule_events.py` derives it on every run). Scripts that tune or diagnose the rule still take a
  fixed `--tolerance`; which fixed value matches ADR 0003 is an open question (docs/yield-image-rule.md).
- Labels closer than the tolerance are one incident (`incident_labels`), so recall is reported per label and per
  incident.
- Two events of one physical vehicle (`yield_image_rule.vehicle_groups`) within the tolerance are a duplicate report;
  two vehicles at one pedestrian are `coincident`, left to a human (`score_clip`, `summarise`).

Rows are dicts with at least "kind" and "time_s"; labels may carry "source" ("manual" or "confirmed_candidate").
"""
import json
from pathlib import Path

KIND = "suspected_failure_to_yield"
# The fixed evaluation's tolerance (eval_rule_events, eval_constructions), frozen on 2026-11-08 by the ADR 0003
# amendment: derive_tolerance found 1.25 s on the residuals at the time, and re-deriving it on every run would move the
# yardstick whenever the rule changes. derive_tolerance is still reported next to it so a drift stays visible.
TOLERANCE_S = 1.25
SOURCES = ("manual", "confirmed_candidate")
ALIASES = {"vehicle_on_crosswalk_with_pedestrian": "suspected_failure_to_yield"}


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["kind"] = ALIASES.get(row["kind"], row["kind"])
    return rows


def matched_pairs(predicted, truth, tolerance):
    """[(prediction index, label index)] matched one-to-one, closest first."""
    candidates = sorted((abs(p["time_s"] - t["time_s"]), i, j) for i, p in enumerate(predicted)
                        for j, t in enumerate(truth) if p["kind"] == t["kind"] and abs(p["time_s"] - t["time_s"]) <= tolerance)
    used_p, used_t, out = set(), set(), []
    for _, i, j in candidates:
        if i not in used_p and j not in used_t:
            used_p.add(i), used_t.add(j)
            out.append((i, j))
    return out


def match(predicted, truth, tolerance):
    return len(matched_pairs(predicted, truth, tolerance))


def residuals(predicted, truth, search_s=5.0):
    """Signed (prediction - label) seconds for each label's nearest same-kind prediction, within `search_s`.

    This is the distribution the tolerance is derived from: it is the only place the human label time and the rule's
    event time meet. A label with no prediction in range gets None (a miss at any tolerance up to `search_s`).
    """
    out = []
    for t in truth:
        same = [p for p in predicted if p["kind"] == t["kind"]]
        if not same:
            out.append({"label": t, "predicted": None, "delta_s": None})
            continue
        nearest = min(same, key=lambda p: abs(p["time_s"] - t["time_s"]))
        delta = nearest["time_s"] - t["time_s"]
        out.append({"label": t, "predicted": nearest["time_s"],
                    "delta_s": delta if abs(delta) <= search_s else None})
    return out


def derive_tolerance(magnitudes, min_gap_s=0.5, min_ratio=2.0, resolution_s=0.05, fallback_s=1.0):
    """Tolerance, in seconds, from the sorted |label - prediction| residuals: the edge of the dense cluster.

    Two timestamps refer to one incident when the distance comes from label placement and from the rule's event-time
    definition, i.e. when it sits in the same cluster as every other match. Beyond that cluster are pairs that only
    happen to be nearest to each other (a different vehicle's event), and a tolerance that reaches them inflates
    recall without any detection gain. So: walk the sorted magnitudes and stop at the first gap that is both
    `min_gap_s` wide and at least `min_ratio` times the previous value; the tolerance is that gap's midpoint.
    With no such gap (too few labels, or a smooth spread) the labeler's documented +-1 s is used and reported as such.
    """
    values = sorted(m for m in magnitudes if m is not None)
    for a, b in zip(values, values[1:]):
        if a > 0 and b - a >= min_gap_s and b / a >= min_ratio:
            midpoint = (a + b) / 2
            return round(-(-midpoint // resolution_s) * resolution_s, 3), {
                "method": "cluster_edge", "below_s": a, "above_s": b, "gap_s": round(b - a, 3),
                "ratio": round(b / a, 2), "sorted": values}
    return fallback_s, {"method": "fallback_labeler_resolution", "sorted": values}


def incident_labels(truth, gap_s):
    """Group label indices whose times are within `gap_s`: one incident needs one prediction, not one per label.

    Two labels closer than the tolerance are not two independent requirements (the labeller marked two moments of the
    same pass, or two candidates for it); counting them separately would make recall depend on how often someone pressed
    the key. The per-label numbers stay the headline; this is the second view.
    """
    groups = []
    for i in sorted(range(len(truth)), key=lambda i: truth[i]["time_s"]):
        if groups and truth[i]["time_s"] - truth[groups[-1][-1]]["time_s"] <= gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def counts(predicted, truth, tolerance):
    result = {}
    for kind in sorted({e["kind"] for e in predicted} | {e["kind"] for e in truth}):
        p = [e for e in predicted if e["kind"] == kind]
        t = [e for e in truth if e["kind"] == kind]
        tp = match(p, t, tolerance)
        result[kind] = {"predicted": len(p), "labelled": len(t), "true_positive": tp,
                        "false_alarm": len(p) - tp, "miss": len(t) - tp}
    return result


def with_rates(per_kind):
    out = {}
    for kind, c in per_kind.items():
        precision = c["true_positive"] / c["predicted"] if c["predicted"] else None
        recall = c["true_positive"] / c["labelled"] if c["labelled"] else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if c["predicted"] and c["labelled"] else None)
        out[kind] = {**c, "precision": precision, "recall": recall, "f1": f1}
    return out


def score(predicted, truth, tolerance):
    return with_rates(counts(predicted, truth, tolerance))


def aggregate(pairs, tolerance):
    """pairs: [(predicted, truth)] -> per-kind totals summed over videos (micro average)."""
    total = {}
    for predicted, truth in pairs:
        for kind, c in counts(predicted, truth, tolerance).items():
            t = total.setdefault(kind, {k: 0 for k in c})
            for k, v in c.items():
                t[k] += v
    return with_rates(total)


def predicted_of(events):
    """The events as eval_events sees them: rule event kinds aliased to the label vocabulary."""
    return [{"kind": ALIASES.get(e["kind"], e["kind"]), "time_s": e["time_s"]} for e in events]


def duplicate_clusters(events, groups, tolerance):
    """[(indices)] groups of events within the tolerance that are one physical vehicle: each cluster of k events is k-1
    redundant reports. Measured on the tracks, so it shows what the rule would still emit without the merge."""
    clusters = []
    for i, event in enumerate(events):
        for cluster in clusters:
            if groups.get(event["vehicle_id"]) == groups.get(events[cluster[0]]["vehicle_id"]) \
                    and any(abs(event["time_s"] - events[j]["time_s"]) <= tolerance for j in cluster):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return [cluster for cluster in clusters if len(cluster) > 1]


def score_clip(events, labels, tolerance, groups, riders):
    """One clip's matches, per-source hits, duplicates and flags."""
    predicted = predicted_of(events)
    pairs = matched_pairs(predicted, labels, tolerance)
    hit_events = {i for i, _ in pairs}
    hit_labels = {j for _, j in pairs}
    by_source = {source: {"labelled": 0, "hit": 0} for source in SOURCES}
    for j, label in enumerate(labels):
        source = label.get("source", "manual")
        by_source.setdefault(source, {"labelled": 0, "hit": 0})
        by_source[source]["labelled"] += 1
        if j in hit_labels:
            by_source[source]["hit"] += 1
    incidents = incident_labels(labels, tolerance)
    incident_hits = sum(1 for group in incidents if hit_labels & set(group))
    unmatched = [e["time_s"] for i, e in enumerate(events) if i not in hit_events]
    duplicates = []
    for a in range(len(events)):
        for b in range(a + 1, len(events)):
            if abs(events[a]["time_s"] - events[b]["time_s"]) > tolerance:
                continue
            same = groups.get(events[a]["vehicle_id"]) == groups.get(events[b]["vehicle_id"])
            if same:
                duplicates.append({"time_s": [events[a]["time_s"], events[b]["time_s"]],
                                   "vehicles": [events[a]["vehicle_id"], events[b]["vehicle_id"]],
                                   "pedestrians": [events[a]["pedestrian_id"], events[b]["pedestrian_id"]]})
    coincident = [{"time_s": [events[a]["time_s"], events[b]["time_s"]],
                   "vehicles": [events[a]["vehicle_id"], events[b]["vehicle_id"]],
                   "pedestrian": events[a]["pedestrian_id"]}
                  for a in range(len(events)) for b in range(a + 1, len(events))
                  if abs(events[a]["time_s"] - events[b]["time_s"]) <= tolerance
                  and events[a]["pedestrian_id"] == events[b]["pedestrian_id"]
                  and groups.get(events[a]["vehicle_id"]) != groups.get(events[b]["vehicle_id"])]
    return {
        "event_count": len(events), "label_count": len(labels), "matched": len(pairs),
        "by_source": by_source,
        "false_alarm": unmatched,
        "missed_labels": [labels[j]["time_s"] for j in sorted(set(range(len(labels))) - hit_labels)],
        "incidents": len(incidents), "incident_hit": incident_hits,
        "duplicates": duplicates, "coincident": coincident,
        "rider_pedestrian_events": [e["time_s"] for e in events
                                    if e["frame"] in riders.get(e["pedestrian_id"], ())],
        "pairs": [[events[i]["time_s"], labels[j]["time_s"], round(abs(events[i]["time_s"] - labels[j]["time_s"]), 3)]
                  for i, j in pairs],
    }


def summarise(clips, tolerance):
    """Totals over clips for one tolerance: label hits per source, incident hits, events, false alarms, duplicates."""
    total = {"labels": 0, "hit": 0, "events": 0, "duplicates": 0, "duplicate_reports": 0,
             "incidents": 0, "incident_hit": 0}
    by_source = {source: {"labelled": 0, "hit": 0} for source in SOURCES}
    for clip in clips:
        events = clip["events"]
        labels = clip["labels"]
        pairs = matched_pairs(predicted_of(events), labels, tolerance)
        hit_labels = {j for _, j in pairs}
        total["labels"] += len(labels)
        total["hit"] += len(hit_labels)
        total["events"] += len(events)
        for j, label in enumerate(labels):
            source = label.get("source", "manual")
            by_source.setdefault(source, {"labelled": 0, "hit": 0})
            by_source[source]["labelled"] += 1
            by_source[source]["hit"] += j in hit_labels
        groups = clip["groups"]
        total["duplicates"] += sum(1 for a in range(len(events)) for b in range(a + 1, len(events))
                                   if abs(events[a]["time_s"] - events[b]["time_s"]) <= tolerance
                                   and groups.get(events[a]["vehicle_id"]) == groups.get(events[b]["vehicle_id"]))
        total["duplicate_reports"] += sum(len(cluster) - 1 for cluster in duplicate_clusters(events, groups, tolerance))
        incidents = incident_labels(labels, tolerance)
        total["incidents"] += len(incidents)
        total["incident_hit"] += sum(1 for group in incidents if hit_labels & set(group))
    per_source = {source: dict(**counts, recall=counts["hit"] / counts["labelled"] if counts["labelled"] else None)
                  for source, counts in by_source.items()}
    return {**total, "by_source": per_source,
            "precision": total["hit"] / total["events"] if total["events"] else None,
            "incident_recall": total["incident_hit"] / total["incidents"] if total["incidents"] else None}
