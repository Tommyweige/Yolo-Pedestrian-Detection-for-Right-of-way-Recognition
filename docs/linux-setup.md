# Linux（CPU-only）安裝與現況

這份文件只記錄**實測**結果，不是推測的安裝指南。

- 實測日期：2026-10-27（安裝、測試與 CPU 量測）、2026-10-28（真實權重端到端）
- 機器：Oracle A1（aarch64）、Oracle Linux Server 9.8、kernel 6.12.0-206.104.4.4.el9uek.aarch64
- 資源：4 vCPU、22 GiB RAM、**無 GPU**、無 sudo（執行身分 `amp`）
- 結論：Python 偵測核心可以只靠 CPU 跑完「影片解碼 → YOLO → DeepSORT → 輸出影片」，兩種模式都用真實權重驗證通過。repo 仍沒有現成的 Linux 安裝腳本；PyQt5 與 Rust 介面在這台機器上都不可用。

## 1. 既有文件為什麼不足

- `README.md` 的「安裝指南」只有 Windows PowerShell 指令（`Python312\python.exe`、`runtime\detection-env\Scripts\python.exe`），並以 `--require-cuda` 驗證，最後用 `python start.py` 啟動 Rust 視窗。權重下載段落另有一行 Linux `wget` 提示，但安裝與啟動步驟仍只有 Windows。
- `docs/environments.md` 明說驗證平台是 Windows x64 + RTX 4060 Laptop，WSL/Linux 一節只寫「沒有 Linux lock、不要把 Windows lock 當成 Linux 可重現性證據」。
- `docs/adr/0002-rust-ui-dual-detection-in-two-stages.md` 說「Rust UI 只承諾支援 Windows；Linux 使用者走同一條命令列流程」，但 repo 沒有任何一份 Linux 或 CPU-only 的實際步驟。
- `scripts/setup_environment.py --locked` 在 Linux 直接拒絕：

```text
setup_environment.py: error: The full locks are validated only on Windows/Python 3.12
```

- `scripts/check_environment.py legacy --require-cuda` 在此機必定失敗，因為 `torch.cuda.is_available()` 是 `False`。

## 2. 系統依賴（實測）

| 項目 | 這台機器 | 是否為必要 |
|---|---|---|
| Python 3.12.15（`/usr/bin/python3.12`） | 有，含 `python3.12-devel` 標頭 | 必要 |
| glibc 2.34、`mesa-libGL`、`glib2` | 有 | 足夠讓 OpenCV wheel 匯入 |
| `gcc` 11.5、`make`、`pkg-config` | 有 | 只有建置 sdist 時用到 |
| `g++` | **沒有** | 只有從 sdist 建置 PyQt5 才需要 |
| `ffmpeg` 執行檔 | **沒有** | **不需要**：`moviepy` 用 `imageio-ffmpeg` 內附的 `ffmpeg-linux-aarch64-v7.0.2`；OpenCV wheel 自帶解碼器 |
| `cargo` / `rustc` | **沒有** | 只有 Rust 介面需要 |
| Qt5 開發套件（`qmake`） | **沒有** | 只有從 sdist 建置 PyQt5 才需要 |
| sudo | **沒有** | 因此無法用 `dnf` 補上面三項 |

Python 這條路徑不需要任何系統編譯工具或 ffmpeg；需要 root 的只有舊版 PyQt5 介面。

## 3. 建立 venv 與安裝依賴

venv 建在 repo 外，避免 repo 出現未追蹤檔案：

```bash
/usr/bin/python3.12 -m venv /srv/projects/.venvs/yolo
/srv/projects/.venvs/yolo/bin/python -m pip install -U pip
```

直接照 repo 的 spec 安裝會失敗：

```bash
/srv/projects/.venvs/yolo/bin/python -m pip install -r requirements/legacy.txt
```

原始錯誤（重點段落）：

```text
Collecting PyQt5==5.15.11 (from -r requirements/legacy.txt (line 15))
  ...
  File "/tmp/pip-build-env-3kwc4xff/overlay/lib/python3.12/site-packages/pyqtbuild/builder.py", line 49, in apply_user_defaults
    raise PyProjectOptionException('qmake',
sipbuild.pyproject.PyProjectOptionException
error: metadata-generation-failed
× Encountered an error while generating package metadata.
╰─> PyQt5
```

單獨探測 wheel 供應：

```text
ERROR: Could not find a version that satisfies the requirement PyQt5==5.15.11 (from versions: none)
ERROR: No matching distribution found for PyQt5==5.15.11
```

原因：**PyQt5 在 PyPI 沒有 aarch64 wheel**（任何版本都沒有），pip 只能抓 sdist，而 sdist 建置需要 Qt5 的 `qmake` 與 C++ 編譯器，這台兩者都沒有、也沒有 sudo 可補。這是環境限制，不是專案程式碼的問題。

拿掉 PyQt5 後，其餘依賴全部從 aarch64 wheel 安裝成功：

```bash
/srv/projects/.venvs/yolo/bin/python -m pip install \
  torch==2.13.0 torchvision==0.28.0 ultralytics==8.0.3 numpy==1.26.4 \
  opencv-python==4.11.0.86 moviepy==1.0.3 setuptools==80.10.2 \
  hydra-core==1.3.2 easydict==1.13 thop==0.1.1.post2209072238 \
  ipython==9.17.1 tensorboard==2.21.0 GitPython==3.2.0
```

（這就是 `requirements/legacy.txt` 去掉 `PyQt5` 那行；`moviepy 1.0.3` 是 sdist，但純 Python，建置成功。）

實測版本：

| 套件 | 版本 |
|---|---|
| torch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0 |
| ultralytics | 8.0.3 |
| moviepy | 1.0.3 |

注意事項：

- venv 大小 **5.2 GB**。PyPI 給 aarch64 的 torch 是 CUDA 13 建置，會一併安裝 `nvidia-*`、`cuda-toolkit`；CPU 執行不需要它們，但可以運作。
- `pip check` 會失敗（原文）：

```text
nvidia-cusparselt-cu13 0.8.1 is not supported on this platform
```

  `scripts/check_environment.py` 把 `pip check` 非零視為錯誤，所以這個檢查在 Linux 上會一直回 `fail`，與偵測本身無關。
- 專案自己的檢查器實測（`check_environment.py legacy`）：

```json
{"status": "fail", "torch": "2.13.0+cu130", "cuda_available": false,
 "kernel_device": "cpu", "error": "ModuleNotFoundError: No module named 'PyQt5'"}
```

  torch 的 CPU 矩陣運算與 torchvision NMS 都已先通過，失敗點只在 PyQt5 匯入。

## 4. 模型權重

**2026-10-28 更新：** 模型庫 `TommyPanLab/traffic-violation-yolov8` 已改為 **public**，不需要 Hugging Face 帳號、token 或 `hf auth login`，七個檔案（六個 `*_tf.pt`／`*_zebra.pt` 與 `ckpt.t7`）都可匿名下載。目前的下載指令、固定 revision 與 SHA-256 見 `README.md` 第 3 節與 PR #30。

`weights/` 只有 `.gitkeep` 與 `README.md`。

固定 revision `98f890a251c03aff28e164c6e835715dbc5019a6`，下載後攤平到 `weights/`：

```bash
rev=98f890a251c03aff28e164c6e835715dbc5019a6
base="https://huggingface.co/TommyPanLab/traffic-violation-yolov8/resolve/$rev"
for f in traffic-light/yolov8s_tf.pt right-of-way/yolov8s_zebra.pt tracking/ckpt.t7; do
  curl -L --fail -o "weights/$(basename "$f")" "$base/$f"
done
```

本次抓了最小組合（`yolov8s_tf.pt`、`yolov8s_zebra.pt`、`ckpt.t7`），三個 SHA-256 都與 README 的表一致：`efc8e3ba…`、`c0146430…`、`df75ddef…`。`weights/` 已被 `.gitignore` 忽略，不會進版控。

缺少權重時，`scripts/validate_detection.py` 的原始錯誤：

```text
FileNotFoundError: 找不到 YOLO 模型權重：/srv/projects/yolo-ped/weights/yolov8s_zebra.pt
請將 yolov8s_zebra.pt 放到 /srv/projects/yolo-ped/weights，或設定環境變數 YOLO_ZEBRA_MODEL 指向權重檔案。
```

另外，`ckpt.t7` 是**兩種模式**的硬前置：`predict_tf.py` 與 `predict_zebra.py` 都在啟動時呼叫 `init_tracker()`，缺檔直接：

```text
FileNotFoundError: DeepSORT checkpoint not found: .../deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

`test_traffic_rules.py` 也因此在 `setUpClass` 失敗（`Ran 0 tests`）。

## 5. 實際可用的執行方式

Rust 介面需要 `cargo build`（此機沒有 cargo，`python start.py` 只會印「請先執行 cargo build…」），PyQt5 介面則是 `ModuleNotFoundError`。**Linux 上唯一可用的入口是命令列**，走 `detection_backend`：

```bash
export YOLO_WEIGHTS_DIR=/path/to/weights
export DEEPSORT_REID_CKPT=/path/to/weights/ckpt.t7
/srv/projects/.venvs/yolo/bin/python scripts/validate_detection.py \
  --video /path/to/video.mp4 --output /path/to/out --model yolov8s --task zebra
```

這條路徑會檢查權重、啟動預測子程序、回報進度、驗證輸出幀數與幀率。

若要直接呼叫 `predict_*.py`，必須自己設環境變數，否則 torch 2.13 的 `weights_only=True` 會讓 ultralytics 8.0.3 載入 `.pt` 失敗：

```text
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options...
	WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default.
```

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

`detection_backend.py` 已經自動設定這個變數，所以走 backend 不用管。同理，`half=True` 在 CPU 上會被 ultralytics 自動關閉（`ultralytics/yolo/engine/predictor.py:120`：`self.args.half &= device.type != 'cpu'`），不需要改。

## 6. 端到端實測（真實權重）

用第 4 節的三個真實權重跑 `scripts/validate_detection.py`；測試影片是本機既有的 `data/people-detection.mp4`（未納入版控）前 96 幀（768x432、12 fps）：

| 模式 | 結果 |
|---|---|
| `--task zebra`（`yolov8s_zebra.pt`） | **pass**：96/96 幀、輸出 12.0 fps、64.4 秒、端到端 1.49 fps |
| `--task tf`（`yolov8s_tf.pt`） | **pass**：96/96 幀、輸出 12.0 fps、63.3 秒、端到端 1.52 fps |
| `predict_zebra.py` 直接執行（backend 同參數） | 96/96 幀；96 幀中 12 幀偵測到 `person` |

逐幀推論時間（由 `events.jsonl` 的每幀訊息統計）：zebra 平均 600.9 ms、中位數 590.8 ms；tf 平均 593.2 ms、中位數 588.8 ms。MoviePy 旋轉轉檔（10°、96 幀）也成功，輸出 96 幀 832x560、3.14 秒。

這是流程驗證：輸出影片、幀數、幀率與事件檔都成立。這段行人影片在 `tf` 模式沒有偵測到號誌或車輛，`zebra` 模式只出現 `person`，所以 96 幀內沒有觸發違規事件；違規辨識準確率需要標註資料另行評估。

## 7. CPU 推論速度（實測）

量測方式：同一張 768x432 影格，預熱 2 次後計時，640/768 取 5 次平均、1920 取 3 次平均，4 執行緒（`torch.get_num_threads() == 4`）。

| 模型 | imgsz | augment | 秒/張 | fps | 峰值 RSS |
|---|---:|---|---:|---:|---:|
| yolov8n | 640 | 否 | 0.162 | 6.16 | 933 MB |
| yolov8n | 768 | 否 | 0.182 | 5.50 | 1029 MB |
| yolov8n | 768 | 是 | 0.372 | 2.69 | 1069 MB |
| yolov8s | 640 | 否 | 0.293 | 3.42 | 1256 MB |
| yolov8s | 768 | 否 | 0.355 | 2.82 | 1551 MB |
| yolov8s | 768 | 是 | 0.676 | 1.48 | 1834 MB |
| yolov8s | 1920 | 否 | 1.677 | 0.60 | 1373 MB |
| yolov8s | 1920 | 是 | 3.714 | 0.27 | 1504 MB |

（yolov8n/yolov8s 是公開 COCO 權重，不是專案的交通模型；上表用來說明這顆 CPU 的量級。真實 `yolov8s_zebra.pt` 的 backend 全流程是 0.60 秒/張（第 6 節），與表中 `yolov8s` 768+augment 的 0.676 秒同量級；`imgsz=1920` + augment 以真實模型量到 3.604 秒/張。）

專案後端的設定會直接決定實際速度（`detection_backend.py:217-220`）：

- `augment=True`：成本約 2–3 倍（TTA）。
- `imgsz={width}`：直接把影片寬度當推論尺寸。1080p 就是 1920。
- `conf=0.7`、`iou=0.3`、`half=True`：`half` 在 CPU 自動失效。

換算：

- 1080p（imgsz=1920）+ augment：**3.60 秒/張、0.28 fps**（真實 `yolov8s_zebra.pt`）。1 分鐘 30 fps 的 1080p 影片（1800 幀）約 **1.8 小時**。
- 768 寬、12 fps 的 596 幀測試影片：以 0.60 秒/張推算約 **6 分鐘**（實測 96 幀 63–64 秒）。
- 記憶體：真實權重全流程 `Maximum resident set size` 為 1,143,408 KB（zebra）與 1,155,420 KB（tf）；22 GB 綽綽有餘。瓶頸是 4 核，推論期間約 346% CPU。

## 8. 測試套件結果

| 指令 | 結果 |
|---|---|
| `pytest -q test_yield_rules.py test_yield_image_rule.py test_eval_events.py test_environment_setup.py` | 39 passed、1 failed |
| `pytest -q test_label_tools.py test_auto_label.py test_build_training_set.py test_dataset_ingest.py` | 19 passed |
| `python test_desktop_bridge.py` | PASS（旋轉、中文／空白路徑、進度、取消、缺權重提示） |
| `pytest -q test_track_video.py` | 5 passed |
| `python test_traffic_rules.py`（`DEEPSORT_REID_CKPT` 指向真實 `ckpt.t7`） | 5 tests OK；沒有 `ckpt.t7` 時 0 tests、error |

唯一的失敗是 `test_environment_setup.py::test_unverified_checkpoint_is_rejected_before_loading`，原因是它用**同一個直譯器**去跑 `scripts/check_environment.py modern`，在 legacy venv 會先撞到 `assert ultralytics.__version__ == "8.4.175"`，得到沒有訊息的 `AssertionError`。該檔 docstring 已寫「Run with the validated modern interpreter」，所以這是環境不匹配，不是 Linux 專屬問題。`test_desktop_bridge.py` 用替身預測腳本，`test_traffic_rules.py` 用真實 `ckpt.t7` 與已知偵測框，兩者都不使用 YOLO 推論。

## 9. 已知限制

1. **PyQt5 無法安裝**：PyPI 沒有 aarch64 wheel，sdist 需要 `qmake` 與 `g++`；此機沒有、也沒有 sudo。`python start.py --legacy` 直接 `ModuleNotFoundError: No module named 'PyQt5'`。
2. **Rust 介面不可用**：沒有安裝 `cargo`，`python start.py` 只印「請先執行 cargo build --manifest-path rust-ui/Cargo.toml」。`rust-ui/src/video.rs` 的原生影片預覽以 `#[cfg(windows)]` 包住，ADR 0002 也只承諾 Windows。
3. **`ckpt.t7` 對兩種模式都是硬需求**：連闖紅燈模式也會在 `init_tracker()` 停下。權重現在可匿名下載，但缺檔就是直接失敗，不會退回無追蹤模式。
4. **CPU 上的預設參數不實用**：`augment=True` 與 `imgsz=width` 都寫死在後端；1080p 只有 0.28 fps。要調整得改程式，不在本文件範圍。
5. **`pip check` 一定失敗**（`nvidia-cusparselt-cu13`），連帶讓 `scripts/check_environment.py` 判 `fail`。
6. 這台機器沒有 GPU，`torch.cuda.is_available()` 為 `False`；安裝的仍是 CUDA 建置的 torch，只是跑 CPU。未驗證 CPU-only index 是否能縮小 venv。

## 10. 要讓它在這台機器上真正跑起來，最少需要什麼

1. Python 3.12 + 第 3 節的 pip 安裝指令（**不要**裝 PyQt5）。
2. 用第 4 節的 `curl` 指令下載權重（公開、免登入），放進 `weights/`，或設 `YOLO_WEIGHTS_DIR`、`DEEPSORT_REID_CKPT`。
3. 用 `scripts/validate_detection.py` 當唯一入口（避開兩個 GUI）；`--task zebra` 與 `--task tf` 都已實測跑完。
4. 接受 CPU 速度：1080p 約 0.28 fps。若要實用，必須調降 `imgsz`、關閉 `augment`（需要改 `detection_backend.py`）。
5. 只有在需要舊 GUI 時，才需要 root 安裝 Qt5 開發套件與 `g++` 後自行建置 PyQt5（此機無 sudo，**未實測**）。
