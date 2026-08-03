# Runtime model weights

The repository does not include model checkpoints. Put the checkpoints supplied
with the project in this directory before starting the GUI:

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
