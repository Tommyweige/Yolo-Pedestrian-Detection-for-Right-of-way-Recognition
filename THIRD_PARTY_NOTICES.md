# 第三方授權聲明

本專案自身採 AGPL-3.0（見 [`LICENSE`](LICENSE)）。下列為本 repo 內含、依賴或執行期取得的第三方元件。

| 元件 | 版本 | 授權 | Upstream | 在本 repo 的形態 |
|---|---|---|---|---|
| Ultralytics YOLOv8 | 8.0.3（見下方判定方式） | GPL-3.0 | https://github.com/ultralytics/ultralytics | vendored：`YOLOv8_DeepSORT_Object_Tracking/ultralytics/`；授權全文：[`LICENSE`](YOLOv8_DeepSORT_Object_Tracking/ultralytics/LICENSE) |
| DeepSORT / deep_sort_pytorch | 未標示版本（2020 年程式碼） | MIT，Copyright (c) 2020 Ziqiang | https://github.com/ZQPei/deep_sort_pytorch | vendored：`YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/deep_sort_pytorch/`；已有該目錄的 `LICENSE` |
| Ultralytics（modern 環境） | 8.4.175（`requirements/modern.txt`） | AGPL-3.0（upstream 自 8.4 起改宣告 AGPL-3.0） | https://github.com/ultralytics/ultralytics | pip 依賴，未 vendored |
| Ultralytics（legacy 環境） | 8.0.3（`requirements/legacy.txt`） | GPL-3.0 | https://github.com/ultralytics/ultralytics | pip 依賴；版本與 vendored 樹相同 |
| 斑馬線分割模型 facebook/mask2former-swin-large-mapillary-vistas-semantic（Mask2Former，以 Mapillary Vistas 訓練） | revision `4772b6bf101d91f2534c106dc524d906aeb3c68a`；`model.safetensors` SHA-256 `72721afb6894c0b1239b9259c428996d9ca0aecbebc4efb4a2deaae0403e2ea7`（以上抄自 `scripts/segment_crosswalk.py`） | CC BY-NC-SA，**只限非商業用途** | https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic | 執行期下載到 `runtime/models/mask2former-mapillary`，不進 Git |
| 模型權重（`*_tf.pt`、`*_zebra.pt`、`ckpt.t7`） | 見 `model_manifest.json` | Hugging Face 標示 `other`；各檔另見 HF 上的 THIRD_PARTY_NOTICES.md | https://huggingface.co/TommyPanLab/traffic-violation-yolov8 | 執行期下載到 `weights/`，不進 Git |

**斑馬線分割模型是限制最嚴的元件（非商業）**，且屬於主流程第一步（見 [`docs/adr/0001-image-rule-replaces-world-rule.md`](docs/adr/0001-image-rule-replaces-world-rule.md)），不是可選元件：使用本專案的不禮讓偵測流程即受其非商業限制。

## Vendored Ultralytics 的版本與修改

- 版本：`YOLOv8_DeepSORT_Object_Tracking/ultralytics/__init__.py` 的 `__version__ = "8.0.3"`，檔頭註解為 `GPL-3.0 license`。本機 legacy venv 安裝的 `ultralytics==8.0.3` 也是同一版本。
- 與 pip 8.0.3 套件比對（唯一依據，未對照 upstream git 歷史），vendored 樹**有被修改**：
  - `nn/tasks.py`：為相容新版 PyTorch，加入 `torch.serialization.add_safe_globals([DetectionModel])`，兩處 `torch.load` 加 `weights_only=False`。
  - 只含部分子目錄（缺少 `yolo/v8/classify`、`yolo/v8/segment`、`yolo/data/datasets` 等）。
  - 新增本專案檔案：`yolo/v8/detect/` 下的 `predict_tf.py`、`predict_zebra.py`、`traffic_events.py`，以及 `deep_sort_pytorch/`。
  - `__init__.py` 內容相同，僅換行符號不同。
- GPL-3.0 授權全文取自 pip 套件 `ultralytics-8.0.3.dist-info/LICENSE`，與 https://www.gnu.org/licenses/gpl-3.0.txt 的 SHA-256 相同（`3972dc97…6986`），為 FSF 原文。

## 備註

- Ultralytics 8.4 以後 upstream 改宣告 AGPL-3.0；8.0.3 的 vendored 樹仍適用 GPL-3.0。
