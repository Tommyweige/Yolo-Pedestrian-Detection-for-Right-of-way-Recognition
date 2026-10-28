# Runtime model weights

Download the project checkpoints from
[TommyPanLab/traffic-violation-yolov8 on Hugging Face](https://huggingface.co/TommyPanLab/traffic-violation-yolov8).
The upstream folders are `traffic-light/`, `right-of-way/`, and `tracking/`.
Place all seven files directly in this directory (without those subfolders)
before starting the GUI:

```text
weights/
├── yolov8s_tf.pt         <- traffic-light/
├── yolov8l_tf.pt         <- traffic-light/
├── yolov8x6_tf.pt        <- traffic-light/
├── yolov8s_zebra.pt      <- right-of-way/
├── yolov8l_zebra.pt      <- right-of-way/
├── yolov8x6_zebra.pt     <- right-of-way/
└── ckpt.t7               <- tracking/
```

## Access (checked 2026-10-28)

The model repository is **public**. No Hugging Face account, `hf auth login`
or token is needed: anonymous `curl` and `hf download` were both verified to
fetch all seven files with matching SHA-256.

Pinned revision: `98f890a251c03aff28e164c6e835715dbc5019a6`. The download
commands (`hf download` and `curl`), the expected size and SHA-256 of each
file, and the environment-variable overrides are in the
[project README](../README.md#3-準備模型權重與測試影片).

`ckpt.t7` is the DeepSORT appearance-model checkpoint. Large weight files are
ignored by Git on purpose.
