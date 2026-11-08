"""Tune yield_image_rule parameters with nested leave-one-video-out (LOVO) selection.

For each video, the configs are scored on the other videos only, the best one is
picked (ties prefer the config closest to the yield_image_rule.Params defaults),
and that config's counts on the held-out video are summed into a micro F1. This
reports how a parameter choice made without a video generalises to it.

  python scripts/tune_rule_lovo.py --run datasets/eval-run2 --videos datasets/來源影片 --labels labels
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import clip_artifacts  # noqa: E402
import event_scoring as scoring  # noqa: E402
import eval_rule  # noqa: E402
import yield_image_rule  # noqa: E402

KIND = scoring.KIND
DEFAULT_GRID = {"entry_window_s": [1.0, 1.5, 2.0],                  # every axis holds the Params default, so the
                "lateral_vehicle_widths": [1.5, 2.0, 2.5, 3.0],      # tie-break toward the defaults is a real choice
                "min_inside_person_heights": [0.1, 0.05, 0.03, 0.0]}


def configs(grid: dict) -> list[dict]:
    """Cartesian product of grid values, in grid-key and list order."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def config_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def distance(cfg: dict, default: dict) -> float:
    """Relative L1 distance to the defaults; a zero default falls back to |cfg[k]|."""
    total = 0.0
    for key, value in cfg.items():
        reference = default[key]
        denominator = abs(reference) if reference else abs(value)
        total += abs(value - reference) / denominator if denominator else 0.0
    return total


def micro_f1(rows) -> float:
    """Micro F1 over (predicted, true_positive, labelled) rows; 0.0 when empty."""
    predicted = true_positive = labelled = 0
    for p, tp, l in rows:
        predicted += p
        true_positive += tp
        labelled += l
    denominator = predicted + labelled
    return 2 * true_positive / denominator if denominator else 0.0


def _rates(predicted: int, true_positive: int, labelled: int) -> dict:
    return {
        "precision": true_positive / predicted if predicted else None,
        "recall": true_positive / labelled if labelled else None,
        "f1": 2 * true_positive / (predicted + labelled) if predicted + labelled else None,
    }


def nested_lovo(table: dict, cfg_list: list[dict], default: dict) -> dict:
    entries = []
    for cfg in cfg_list:
        key = config_key(cfg)
        if key in table:
            entries.append((cfg, table[key]))
    videos = sorted({video for _, per_video in entries for video in per_video})

    def rows(per_video, selected):
        return [tuple(per_video[video]) for video in selected if video in per_video]

    def choose(selected):
        best = None
        for cfg, per_video in entries:
            f1 = micro_f1(rows(per_video, selected))
            dist = distance(cfg, default)
            if best is None or f1 > best[1] or (f1 == best[1] and dist < best[2]):
                best = (cfg, f1, dist)
        return best

    per_video = {}
    total_predicted = total_true_positive = total_labelled = 0
    for video in videos:
        chosen, _, _ = choose([other for other in videos if other != video])
        predicted, true_positive, labelled = table[config_key(chosen)][video]
        per_video[video] = {"chosen": chosen, "predicted": predicted,
                            "true_positive": true_positive, "labelled": labelled}
        total_predicted += predicted
        total_true_positive += true_positive
        total_labelled += labelled

    total = {"predicted": total_predicted, "true_positive": total_true_positive,
             "labelled": total_labelled,
             **_rates(total_predicted, total_true_positive, total_labelled)}
    best_cfg, best_f1, _ = choose(videos)
    default_key = config_key(default)
    default_f1 = micro_f1(rows(table[default_key], videos)) if default_key in table else None
    return {"per_video": per_video, "total": total,
            "in_sample_best": {"config": best_cfg, "f1": best_f1},
            "default_f1": default_f1}


def count_table(run_dir, videos_dir, labels_dir, cfg_list, geometry="image", tolerance=1.0, log=print) -> dict:
    run_dir, videos_dir, labels_dir = Path(run_dir), Path(videos_dir), Path(labels_dir)
    videos = sorted(p for p in videos_dir.iterdir()
                    if p.suffix.lower() in clip_artifacts.VIDEO_SUFFIXES)
    videos = [video for video in videos
              if (labels_dir / f"{video.stem}.jsonl").exists()
              and (run_dir / video.stem / clip_artifacts.MASKS).exists()]

    table = {}
    total = len(videos) * len(cfg_list)
    index = 0
    for cfg in cfg_list:
        key = config_key(cfg)
        table[key] = {}
        for video in videos:
            index += 1
            events = eval_rule.run_one(video, run_dir, geometry, overrides=cfg)
            events = [{**event, "kind": scoring.ALIASES.get(event["kind"], event["kind"])}
                      for event in events]
            truth = scoring.load(labels_dir / f"{video.stem}.jsonl")
            counts = scoring.counts(events, truth, tolerance).get(
                KIND, {"predicted": 0, "true_positive": 0, "labelled": 0})
            table[key][video.stem] = [counts["predicted"], counts["true_positive"], counts["labelled"]]
            log(f"[{index}/{total}] {key} {video.stem[:16]} "
                f"P={counts['predicted']} TP={counts['true_positive']} L={counts['labelled']}")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="datasets/eval-run2",
                        help="folder holding <stem>/seg/masks.npz and <stem>/tracks/raw_tracks.json")
    parser.add_argument("--videos", default="datasets/來源影片")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--geometry", default="image", choices=("image", "crosswalk"))
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--cache", help="JSON file with {table, grid}; written after a fresh run")
    parser.add_argument("--log", help="optional file to append progress lines to (UTF-8)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    log_handle = open(args.log, "a", encoding="utf-8") if args.log else None

    def log(message):
        print(message, flush=True)
        if log_handle:
            print(message, file=log_handle, flush=True)

    try:
        cache_path = Path(args.cache) if args.cache else None
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            grid, table = cached["grid"], cached["table"]
        else:
            grid = DEFAULT_GRID
            cfg_list = configs(grid)
            table = count_table(ROOT / args.run, ROOT / args.videos, ROOT / args.labels,
                                cfg_list, args.geometry, args.tolerance, log)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"table": table, "grid": grid}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

        cfg_list = configs(grid)
        default = {key: getattr(yield_image_rule.Params(), key) for key in grid}
        result = nested_lovo(table, cfg_list, default)

        for video in sorted(result["per_video"]):
            row = result["per_video"][video]
            print(f"  {video[:16]:<18} chosen {config_key(row['chosen'])}  "
                  f"P={row['predicted']} TP={row['true_positive']} L={row['labelled']}", flush=True)
        total = result["total"]
        print("  TOTAL " + "  ".join(f"{key}={value}" for key, value in total.items()), flush=True)
        print(f"  in_sample_best {config_key(result['in_sample_best']['config'])} "
              f"f1={result['in_sample_best']['f1']}", flush=True)
        print(f"  default_f1 {result['default_f1']}", flush=True)

        summary = {"lovo": total,
                   "default_f1": result["default_f1"],
                   "in_sample_best": result["in_sample_best"],
                   "chosen": {video: result["per_video"][video]["chosen"]
                              for video in sorted(result["per_video"])}}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    finally:
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
