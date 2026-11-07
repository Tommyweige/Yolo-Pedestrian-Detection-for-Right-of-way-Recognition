"""One review list for a run: every candidate event of every video, with its time, priority and evidence files.

Reads datasets/<run>/events/*.jsonl (and datasets/<run>/evidence/<video>/ when present) and writes
datasets/<run>/candidates.csv (UTF-8 with BOM so Excel shows the Chinese headers). Output is a review aid only.

  python scripts/list_candidates.py --run-dir datasets/my-run
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

COLUMNS = ["影片", "時間", "秒", "優先", "車種", "行人狀態", "提醒", "片段", "截圖"]


def _tenths(seconds: float) -> int:
    return math.floor(seconds * 10 + 0.5)  # half up: 75.25 s -> 752.5 -> 753, not Python's half-to-even 752


def _seconds(seconds: float) -> str:
    tenths = _tenths(seconds)
    return f"{tenths // 10}.{tenths % 10}"


def clip_stem(index: int, event: dict) -> str:
    """File name (without extension) of an event's clip and still; index is the event's line in its events file."""
    return f"event{index:03d}_t{_seconds(event['time_s'])}s"


def _clock(seconds: float) -> str:
    minutes, tenths = divmod(_tenths(seconds), 600)
    return f"{minutes}:{tenths // 10:02d}.{tenths % 10}"


def candidate_rows(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    rows = []
    for path in sorted((run_dir / "events").glob("*.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        found = []
        for index, line in enumerate(lines):
            event = json.loads(line)
            name = clip_stem(index, event)
            folder = run_dir / "evidence" / path.stem
            files = {ext: f"evidence/{path.stem}/{name}.{ext}" if (folder / f"{name}.{ext}").exists() else "" for ext in ("mp4", "png")}
            found.append({
                "影片": path.stem,
                "時間": _clock(event["time_s"]),
                "秒": _seconds(event["time_s"]),
                # 高 = the pedestrian was on the crosswalk (a strict run reports it too); 低 = only waiting at its edge
                "優先": "高" if event.get("pedestrian_entered_crosswalk") else "低",
                "車種": event.get("vehicle_kind", ""),
                "行人狀態": event.get("pedestrian_state", ""),
                "提醒": ",".join(event.get("review_reasons", [])),
                "片段": files["mp4"],
                "截圖": files["png"],
                "_t": event["time_s"],
            })
        rows += sorted(found, key=lambda r: r["_t"])
    for row in rows:
        del row["_t"]
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="datasets/<run>")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not (run_dir / "events").is_dir():
        print(f"錯誤：找不到事件資料夾：{run_dir / 'events'}", file=sys.stderr)
        return 2
    rows = candidate_rows(run_dir)
    with open(run_dir / "candidates.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    high = sum(r["優先"] == "高" for r in rows)
    print(f"候選 {len(rows)} 個（高 {high}、低 {len(rows) - high}）：{run_dir / 'candidates.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
