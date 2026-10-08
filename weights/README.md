# Runtime model weights

Download the project checkpoints from
[TommyPanLab/traffic-violation-yolov8 on Hugging Face](https://huggingface.co/TommyPanLab/traffic-violation-yolov8).
The upstream folders are `traffic-light/`, `right-of-way/`, and `tracking/`.
Place all seven files directly in this directory (without those subfolders)
before starting the GUI. See the [project README](../README.md#3-準備模型權重與測試影片)
for the current local path and the validated model revision:

```text
weights/
├── yolov8s_tf.pt
├── yolov8l_tf.pt
├── yolov8x6_tf.pt
├── yolov8s_zebra.pt
├── yolov8l_zebra.pt
├── yolov8x6_zebra.pt
└── ckpt.t7
```

`ckpt.t7` is the DeepSORT appearance-model checkpoint. Large weight files are
ignored by Git on purpose. See the project README for environment-variable
overrides when the files are stored elsewhere.
