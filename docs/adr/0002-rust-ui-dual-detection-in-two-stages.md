# Rust UI 為主介面，一鍵同時偵測闖紅燈與不禮讓，闖紅燈分兩階段搬移

Rust UI（`rust-ui/`）是主介面，只承諾支援 Windows；Linux 使用者走同一條命令列流程。按一次「偵測」預設同時跑闖紅燈和不禮讓（可只勾一種），兩邊結果合併成一份事件列表，每筆標明種類。不禮讓走影像規則流程（見 ADR 0001）；闖紅燈的判斷目前寫在 vendored 舊程式碼 `predict_tf.py` 裡，綁著 legacy 環境（Ultralytics 8.0.3 內部 API），因此分兩階段：

1. **第一階段**：照舊在 legacy 環境呼叫 `predict_tf.py`，不修改 vendored 程式碼；黃燈與同一台車重複越線，由 Python 端合併結果時事後過濾。
2. **第二階段**：把闖紅燈規則抽成獨立模組，在 modern 環境用 ByteTrack 跑，之後主流程不再需要 legacy 環境。

## Considered Options

- **主介面只做不禮讓、闖紅燈只留在 `university-project-pyqt5`**：最省事，但使用者要求一鍵雙偵測。
- **一開始就重寫闖紅燈**：不用同時管三套環境，但要先驗證 tf 權重能否在新版 Ultralytics 載入，會延後整個介面的整合。

## Consequences

- 第一階段期間，Windows 上要裝 legacy、labeling、modern 三套環境；Linux 與 CI 不涵蓋闖紅燈。
- 第一階段的闖紅燈預測與 GLOSSARY 的定義有已知差異，直到第二階段才修正：任一號誌（而非管制該車方向的號誌）、以車框底邊中點代替前輪、以最下方斑馬線框的底邊代替停止線。
