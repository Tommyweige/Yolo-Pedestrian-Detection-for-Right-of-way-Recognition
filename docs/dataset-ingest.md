# Dataset ingestion (`scripts/dataset_ingest.py`)

Needs only `opencv-python` and `numpy` (any of the project environments). Sources are read-only:
the tool copies images / extracts frames into a **new** output folder and re-hashes every source at
the end, failing (exit 3) if any changed. It refuses to overwrite an existing manifest and refuses an
output folder that overlaps an input folder.

```powershell
python scripts/dataset_ingest.py --output datasets/v1 `
  --videos "D:\clips" --seconds-interval 1 `
  --images "D:\street photos" --group-by folder `
  --roboflow-export "D:\export" --split 0.8 0.1 0.1 --seed 42 --dataset-version v1
```

No Roboflow account or token is used; pass an already-unzipped YOLO export. Empty or unsupported
inputs exit with code 2 and a message.

## Output
- `images/` copied images (`<sha256[:16]>.ext`) and video frames (`video-<hash12>_f<frame>.png`)
- `labels/` YOLO label files imported from a Roboflow export
- `manifest.jsonl` one row per image: `image_id`, `sha256`, `file`, `source_group`, `split`,
  `source_type`, `annotations_origin`, `label_file`, `gold`, `dataset_version`, `seed`, size, plus
  `video_id/video_path/video_sha256/frame_no/source_fps/pts_ms` for frames
- `rejected.jsonl` corrupt images, unsupported extensions, unreadable videos, duplicate content
- `dataset.json` run summary

## Leakage rules
- Every frame of a video shares one group (`video-<hash of file>`), so identical videos also merge.
- Loose images are grouped by `--group-by` (`folder` default, `file`, or `single`).
- Groups sharing identical image content are merged and the duplicate copy is dropped (logged).
- Whole groups are split with a seeded shuffle; the same inputs and seed give an identical manifest.
- Split happens at ingestion, before any teacher labeling (#10).
- `gold` is always `false`. Imported or teacher labels are never gold; use `--train-only` when
  val/test will come from separate human gold labels.

## Tests
`python -m unittest test_dataset_ingest -v` (synthetic media, Chinese/space paths, corrupt files,
duplicates, determinism, source immutability, Roboflow labels).
