"""Score a judged batch: precision from the A/B/C/D page export, recall only where independent labels exist.

The loop is `scripts/newclip_loop.py` (videos -> events -> review page); this reads that page's export back. Three
blocks are printed and never averaged together:

- 驗證集 — the run whose videos were never used to tune anything (`--run`)
- 開發集 — the seven labelled clips (`--dev-run`, their verdicts from `--dev-verdicts`)
- 合併 — both pools added up count by count (micro), with a clip that appears in both counted once

Two instruments, because they answer different questions:

- **Verdicts (A/B/C/D)** judge the events the rule reported, so they give precision: A is a real incident, C is a
  false alarm, B is a repeat of the same incident, D is undecided (kept out of the denominator and listed). They
  cannot give recall: an incident the rule never reported has no card to judge.
- **Labels (`<stem>.jsonl`)** are an independent pass over the whole video, marks included, so they give recall as
  well. `scripts/eval_rule_events.py`'s own `score_clip` / `summarise` do that arithmetic here, at the same 1.25 s
  tolerance the development numbers were made with, so the numbers are comparable instead of merely similar.

A clip with neither verdicts nor labels only contributes "報了幾件"; the output says so and names what is missing.

Verdict files: the page's export (`{"verdicts": [{video, time_s, vehicle_id, pedestrian_id, verdict, ...}]}`) and the
older `yolo-ped-analysis\\review\\verdicts.json` rows (`[clip_index, time_s, vehicle_id, verdict]`, one-based index into
the sorted clip names of the run it is scored against).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_events as ev  # noqa: E402
import eval_rule_events  # noqa: E402
import yield_image_rule as rule  # noqa: E402

TOLERANCE_S = 1.25
VERDICTS = ("A", "B", "C", "D")
ROLE_NAMES = {"dev": "開發集", "validation": "驗證集"}


# ---------------------------------------------------------------- inputs

def read_events(path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_clip(run_dir, clip):
    return {**clip, "events": read_events(clip["events_jsonl"])}


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    clips = [{**load_clip(run_dir, clip), "params": manifest["params"]} for clip in manifest["clips"]]
    return {"dir": run_dir, "manifest": manifest, "clips": clips}


def verdict_rows(payload, stems):
    """Rows normalised to {stem, time_s, vehicle, ped, verdict, note} from either verdict file shape."""
    if isinstance(payload, dict) and "verdicts" in payload:
        return [{"stem": row["video"], "time_s": float(row["time_s"]),
                 "vehicle": row.get("vehicle_id"), "ped": row.get("pedestrian_id"),
                 "verdict": row.get("verdict"), "note": row.get("note") or "", "page": row.get("id")}
                for row in payload["verdicts"]]
    if isinstance(payload, list):
        rows = []
        for row in payload:
            if not isinstance(row, list) or len(row) < 4:
                raise SystemExit(f"看不懂的判定列：{row!r}（舊格式應為 [影片編號, 時間, 車輛, 判定]）")
            index, time_s, vehicle, verdict = row[0], row[1], row[2], row[3]
            if not isinstance(index, int) or not 1 <= index <= len(stems):
                raise SystemExit(f"舊格式的影片編號 {index!r} 對不上這批的 {len(stems)} 支影片；"
                                 f"要沿用舊判定，開發集 run 必須包含全部那幾支影片（依檔名排序）。")
            rows.append({"stem": stems[index - 1], "time_s": float(time_s), "vehicle": vehicle, "ped": None,
                         "verdict": verdict, "note": "", "page": None})
        return rows
    raise SystemExit("判定檔既不是頁面匯出的 JSON，也不是舊的 verdicts.json")


def load_verdicts(paths, stems):
    rows = []
    for path in paths or []:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows += verdict_rows(payload, stems)
    seen, unique = set(), []
    for row in rows:
        # Two vehicles at one moment are two verdicts, so the vehicle is part of the identity; the same row repeated
        # (the old file lists a vehicle's later duplicates separately) is not.
        key = (row["stem"], round(row["time_s"], 3), row["vehicle"], row["ped"], row["verdict"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


# ---------------------------------------------------------------- joins

def join_verdicts(clips, rows):
    """Give each event at most one verdict: same vehicle and pedestrian first, then nearest in time.

    A verdict file made against an older rule run has different vehicle ids (duplicate tracks are merged differently),
    so the fallback is time alone, inside the tolerance, which is the same rule the evaluation matches with.
    """
    by_stem = {clip["stem"]: clip for clip in clips}
    events = {stem: [{"verdict": None, "note": "", "time_s": event["time_s"],
                      "vehicle_id": event["vehicle_id"], "pedestrian_id": event["pedestrian_id"]}
                     for event in clip["events"]]
              for stem, clip in by_stem.items()}
    stray = []
    for row in rows:
        pool = events.get(row["stem"])
        if not pool:
            stray.append(row)
            continue
        best = None
        for index, event in enumerate(pool):
            if event["verdict"] is not None:
                continue
            delta = abs(event["time_s"] - row["time_s"])
            if delta > TOLERANCE_S:
                continue
            same_vehicle = row["vehicle"] is None or event["vehicle_id"] == row["vehicle"]
            same_ped = row["ped"] is None or event["pedestrian_id"] == row["ped"]
            rank = (not (same_vehicle and same_ped), not same_vehicle, delta)
            if best is None or rank < best[0]:
                best = (rank, index)
        if best is None:
            stray.append(row)
            continue
        pool[best[1]]["verdict"] = row["verdict"]
        pool[best[1]]["note"] = row["note"]
    for clip in clips:
        clip["judged"] = events[clip["stem"]]
    return stray


def clip_counts(clip):
    """Reported / judged / verdict counts for one clip."""
    judged = clip.get("judged") or []
    counts = {"events": len(clip["events"]), "judged": 0, **{v: 0 for v in VERDICTS}}
    for event in judged:
        if event["verdict"] in VERDICTS:
            counts["judged"] += 1
            counts[event["verdict"]] += 1
    counts["unjudged"] = counts["events"] - counts["judged"]
    return counts


def pool(clips):
    """Micro totals: counts added up, never averaged."""
    total = {"clips": len(clips), "events": 0, "judged": 0, **{v: 0 for v in VERDICTS}, "unjudged": 0}
    for clip in clips:
        counts = clip_counts(clip)
        for key in total:
            if key != "clips":
                total[key] += counts[key]
    decided = total["A"] + total["B"] + total["C"]
    total["precision"] = total["A"] / decided if decided else None
    total["precision_all"] = total["A"] / total["events"] if total["events"] else None
    return total


def dedupe(clips, preferred):
    """One entry per video name: the copy with more verdicts attached wins, a tie goes to `preferred` (the dev run).

    A clip can be in both sets — the demo/back-test deliberately feeds a development clip in as a "new" one — and the
    pooled number must count its events once. The copy that was judged more completely is the one that knows more about
    those events.
    """
    chosen = {}
    for clip in clips:
        held = chosen.get(clip["stem"])
        if held is None:
            chosen[clip["stem"]] = clip
            continue
        candidate = (clip_counts(clip)["judged"], clip.get("run_role") == preferred)
        incumbent = (clip_counts(held)["judged"], held.get("run_role") == preferred)
        if candidate > incumbent:
            chosen[clip["stem"]] = clip
    return list(chosen.values())


# ---------------------------------------------------------------- labels

def label_context(clip, params):
    """Tracks-derived facts `eval_rule_events.score_clip` needs: vehicle groups and rider frames."""
    tracks = json.loads(Path(clip["tracks"]).read_text(encoding="utf-8"))
    overrides = {k: v for k, v in params.items() if k not in ("fps", "src_size")}
    if isinstance(overrides.get("rider_kinds"), list):
        overrides["rider_kinds"] = tuple(overrides["rider_kinds"])
    rule_params = rule.Params(fps=float(clip["fps"]), src_size=tuple(clip["size"]), **overrides)
    return {"groups": rule.vehicle_groups(tracks, rule_params),
            "riders": rule.rider_frames(rule.merge_duplicate_tracks(tracks, rule_params), rule_params)}


def label_score(clips, labels_dir, tolerance):
    """eval_rule_events' own per-clip scoring for every clip that has <stem>.jsonl, pooled with `summarise`."""
    scored, missing = [], []
    for clip in clips:
        path = Path(labels_dir) / f"{clip['stem']}.jsonl"
        if not path.exists():
            missing.append(clip["stem"])
            continue
        labels = ev.load(path)
        context = label_context(clip, clip["params"])
        result = eval_rule_events.score_clip(clip["events"], labels, tolerance, context["groups"], context["riders"])
        scored.append({"stem": clip["stem"], "events": clip["events"], "labels": labels, **context, **result})
    if not scored:
        return None, missing
    total = eval_rule_events.summarise(scored, tolerance)
    return {"clips": scored, "totals": total}, missing


# ---------------------------------------------------------------- report

def bar(length=78):
    return "─" * length


def clip_line(values, stem):
    """Numbers in fixed columns, the clip name last: a name is mostly Chinese, so it cannot be a fixed-width column."""
    return "  " + values + "  " + stem


def fmt_rate(value):
    return "—" if value is None else f"{value:.3f}"


def print_block(title, clips):
    total = pool(clips)
    print(f"\n{title}")

    def values(counts):
        decided = counts["A"] + counts["B"] + counts["C"]
        precision = counts["A"] / decided if decided else None
        return (f"{counts['events']:>6}{counts['judged']:>6}{counts['A']:>4}{counts['B']:>4}{counts['C']:>4}"
                f"{counts['D']:>4}{counts['unjudged']:>6}{fmt_rate(precision):>11}")
    header = f"{'事件':>6}{'已判':>6}{'A':>4}{'B':>4}{'C':>4}{'D':>4}{'未判':>6}{'precision':>11}"
    print(clip_line(header, "影片"))
    for clip in clips:
        print(clip_line(values(clip_counts(clip)), clip["stem"]))
    print(clip_line(values(total), "小計"))
    decided = total["A"] + total["B"] + total["C"]
    if decided:
        print(f"  precision = {total['A']}/{decided} = {total['precision']:.3f}"
              f"（A ÷ 已判且非 D；B 是重報、C 是誤報）")
    else:
        print("  precision：還沒有任何 A/B/C 判定，算不出來。")
    if total["judged"]:
        print(f"  以「全部報告數」為分母：{total['A']}/{total['events']} = {total['precision_all']:.3f}")
    if total["unjudged"]:
        print(f"  未判 {total['unjudged']} 件沒有分數，也沒被算成誤報。")
    return total


def print_label_block(title, result, missing, tolerance):
    if result is None:
        print(f"\n{title}")
        print("  沒有可用的整片標註（<stem>.jsonl），所以只有「報了幾件」，沒有 recall。")
        print(f"  缺這些檔：{'、'.join(missing)}")
        return None
    totals = result["totals"]
    print(f"\n{title}（eval_rule_events，容差 {tolerance} s）")

    def values(counts, events, labels, hit, missed):
        precision = hit / events if events else None
        recall = counts["incident_hit"] / counts["incidents"] if counts["incidents"] else None
        return (f"{events:>6}{labels:>6}{hit:>6}{fmt_rate(precision):>11}{fmt_rate(recall):>13}{missed:>5}")
    header = f"{'事件':>6}{'標註':>6}{'命中':>6}{'precision':>11}{'事故 recall':>13}{'漏':>5}"
    print(clip_line(header, "影片"))
    for clip in result["clips"]:
        print(clip_line(values(clip, clip["event_count"], clip["label_count"], clip["matched"],
                               len(clip["missed_labels"])), clip["stem"]))
    print(clip_line(values(totals, totals["events"], totals["labels"], totals["hit"],
                           totals["labels"] - totals["hit"]), "小計"))
    manual = totals["by_source"].get("manual", {"labelled": 0, "hit": 0, "recall": None})
    print(f"  其中人工標註（source=manual）：{manual['hit']}/{manual['labelled']}，"
          f"recall {fmt_rate(manual['recall'])}")
    if missing:
        print(f"  沒有標註檔、因此不進 recall 的影片：{'、'.join(missing)}")
    return totals


def stray_lines(stray, tolerance):
    """The verdict rows that could not be attached, all of them: they are the reason a precision is partial."""
    lines = [f"    影片 時間 車輛 → 判定（時間差 > {tolerance} s 或影片不在這批）"]
    for row in stray:
        lines.append(f"    {row['stem']}  {row['time_s']:.3f}s 車 {row['vehicle']} → {row['verdict']}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run folder of the batch being scored (validation)")
    parser.add_argument("--verdicts", action="append", help="its exported verdicts JSON (repeatable)")
    parser.add_argument("--dev-run", help="run folder of the development clips, for the combined number")
    parser.add_argument("--dev-verdicts", action="append", help="the development clips' verdicts (page export or the "
                                                               "older yolo-ped-analysis review/verdicts.json)")
    parser.add_argument("--labels", help="folder of <stem>.jsonl whole-video labels; enables recall")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_S,
                        help=f"matching tolerance in seconds (default {TOLERANCE_S}, the development tolerance)")
    parser.add_argument("--json", help="write the pooled numbers here")
    args = parser.parse_args(argv)

    run = load_run(args.run)
    for clip in run["clips"]:
        clip["run_role"] = run["manifest"]["role"]

    run_rows = load_verdicts(args.verdicts, sorted(clip["stem"] for clip in run["clips"]))
    stray_run = join_verdicts(run["clips"], run_rows)

    dev = None
    stray_dev = []
    if args.dev_run:
        dev = load_run(args.dev_run)
        for clip in dev["clips"]:
            clip["run_role"] = dev["manifest"]["role"]
        dev_rows = load_verdicts(args.dev_verdicts, sorted(clip["stem"] for clip in dev["clips"]))
        stray_dev = join_verdicts(dev["clips"], dev_rows)

    role = run["manifest"]["role"]
    name = ROLE_NAMES.get(role, role)
    dev_name = ROLE_NAMES.get(dev["manifest"]["role"], dev["manifest"]["role"]) if dev else None
    print("=" * 78)
    print(f"{name} run：{run['dir']}")
    print(f"  角色 {role} · 規則預設參數 · 容差 {args.tolerance} s")
    overlap = [clip["stem"] for clip in run["clips"] if clip.get("dev_overlap")]
    if overlap:
        print(f"  ⚠ 這些影片在開發集的標註裡，不是 out-of-sample 的新片：{'、'.join(overlap)}")
        print("    （示範或回測時才會這樣；真正的新片不該出現在這裡）")

    totals = [(name, print_block(f"{name} · 判定分數", run["clips"]))]
    if dev is not None:
        totals.append((dev_name, print_block(f"{dev_name} · 判定分數（in-sample，只供對照）", dev["clips"])))
        combined = dedupe(run["clips"] + dev["clips"], preferred=dev["manifest"]["role"])
        totals.append(("合併", print_block(f"合併 · 判定分數（{name} + {dev_name}，同一支影片只算一次）",
                                          combined)))
    else:
        print(f"\n沒有 --dev-run，所以只報{name}；合併數字需要開發集的 run。")

    print("\n" + bar())
    print("合計（判定分數，micro：分數相加，不是平均）")
    for heading, total in totals:
        print(f"  {heading}：事件 {total['events']}、已判 {total['judged']}（A {total['A']} / B {total['B']} / "
              f"C {total['C']} / D {total['D']}）、未判 {total['unjudged']}、"
              f"precision {fmt_rate(total['precision'])}")

    for stray, label in ((stray_run, name), (stray_dev, dev_name if dev else None)):
        if not stray or label is None:
            continue
        print(f"\n{label}的判定有 {len(stray)} 列對不上這一輪的事件"
              f"（時間差 > {args.tolerance} s 或影片不在這批；判定檔可能是舊版規則的事件）：")
        print("\n".join(stray_lines(stray, args.tolerance)))

    label_results = {}
    if args.labels:
        print("\n" + bar())
        result, missing = label_score(run["clips"], args.labels, args.tolerance)
        label_results[name] = print_label_block(f"{name} · 標註分數（含 recall）", result, missing, args.tolerance)
        if dev is not None:
            result, missing = label_score(dev["clips"], args.labels, args.tolerance)
            label_results[dev_name] = print_label_block(f"{dev_name} · 標註分數（含 recall）", result, missing,
                                                        args.tolerance)
            both = dedupe(run["clips"] + dev["clips"], preferred=dev["manifest"]["role"])
            result, missing = label_score(both, args.labels, args.tolerance)
            label_results["合併"] = print_label_block("合併 · 標註分數（含 recall）", result, missing, args.tolerance)
    else:
        print("\n" + bar())
        print("recall：沒有 --labels，一個 recall 數字都算不出來。")
        print("  recall 需要「整片獨立標註」：對整支影片每一件真實事故都標時間（包含規則漏掉的），")
        print("  一個影片一個 <stem>.jsonl，格式同 scripts/label_events.py（kind=suspected_failure_to_yield + time_s）。")
        print("  只判規則報出來的事件（A/B/C/D）永遠看不到漏掉的那些，所以那樣只有 precision。")

    verdicts_json = {
        "run": str(run["dir"]),
        "dev_run": str(dev["dir"]) if dev else None,
        "tolerance_s": args.tolerance,
        "verdicts": {name: total for name, total in totals},
        "labels": label_results,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(verdicts_json, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
