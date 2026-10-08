# 交通違規偵測系統 (Traffic Violation Detection System)

## 專案簡介
本專案是一個基於 YOLOv8 物件偵測和 DeepSORT 物件追蹤技術的交通違規偵測系統。它提供了一個圖形使用者介面 (GUI)，讓使用者可以選擇影片檔案或資料夾，並執行闖紅燈偵測或車輛不禮讓行人偵測。系統還包含影片旋轉校正功能，以適應不同角度的影片輸入。

## 功能特色
- **多種偵測模式**:
    - **闖紅燈偵測**: 識別並記錄闖紅燈行為。
    - **車輛不禮讓行人偵測**: 偵測車輛是否在斑馬線前禮讓行人。
- **YOLOv8 模型選擇**: 支援不同大小的 YOLOv8 模型 (yolov8s, yolov8l, yolov8x6)，以平衡偵測速度和準確性。
- **影片輸入與輸出**:
    - 支援單一或多個影片檔案輸入。
    - 可選擇偵測結果的輸出路徑。
- **影片播放與控制**: 在主畫面中預覽單一選定的影片，並提供播放、暫停、停止功能。
- **進度顯示**: 顯示當前影片和所有影片的處理進度。
- **影片旋轉校正**: 提供一個獨立的「校正畫面」頁面，允許使用者預覽影片並調整旋轉角度，以確保偵測的準確性。
- **Rust 桌面介面與影片預覽**: 使用 egui/eframe 與 Windows 原生影片解碼；Python 保留 YOLOv8、DeepSORT 與偵測前的旋轉轉檔。

## 安裝指南

### 1. 克隆專案
首先，請將本專案從 GitHub 克隆到您的本地機器：
```bash
git clone https://github.com/your-repo-link/GUI.git
cd GUI
```

### 2. 安裝依賴
本專案需要 Python 3.9 。建議使用虛擬環境來管理依賴。

```bash
# 創建並激活虛擬環境 (Windows)
python -m venv venv
.\venv\Scripts\activate

# 創建並激活虛擬環境 (macOS/Linux)
python3 -m venv venv
source venv/bin/activate
```

安裝主專案的依賴：
```bash
pip install -r requirements.txt
```

### 3. 準備模型權重與測試影片

`controller.py` 不再依賴特定電腦的 `D:` 磁碟路徑；專案內的程式路徑會以
`controller.py` 所在位置為基準。模型權重因檔案較大且可能包含訓練資料授權，
不會放進 Git。**本專案的模型由 TommyPanLab 提供，下載位置是
[Hugging Face：TommyPanLab/traffic-violation-yolov8](https://huggingface.co/TommyPanLab/traffic-violation-yolov8)。**

模型庫中的檔案位置如下；下載後將七個檔案直接放到專案根目錄的 `weights/`，
不要在本機 `weights/` 內保留模型庫的子資料夾結構：

| 用途 | Hugging Face 中的位置 | 本機檔名 |
|---|---|---|
| 闖紅燈偵測 | `traffic-light/` | `yolov8s_tf.pt`、`yolov8l_tf.pt`、`yolov8x6_tf.pt` |
| 不禮讓行人偵測 | `right-of-way/` | `yolov8s_zebra.pt`、`yolov8l_zebra.pt`、`yolov8x6_zebra.pt` |
| DeepSORT 追蹤 | `tracking/ckpt.t7` | `ckpt.t7` |

如果模型庫要求登入，請先使用 `hf auth login`，並確認帳號有該模型庫的讀取權限。
本次實測使用的固定版本為 `98f890a251c03aff28e164c6e835715dbc5019a6`，七個檔案皆已驗證大小與 SHA-256。

**目前這台電腦的模型存放位置：**

```text
C:\Users\tommy\.codex\worktrees\8588\Yolo-Pedestrian-Detection-for-Right-of-way-Recognition\weights
```

這是本次工作目錄的本機路徑；其他電腦請使用自己的 `<專案根目錄>\weights`。預期結構為：

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

其中 `ckpt.t7` 是 DeepSORT 的 ReID 權重；`*_tf.pt` 用於闖紅燈偵測，
`*_zebra.pt` 用於車輛不禮讓行人偵測。測試影片不必放在專案內，啟動 GUI 後從
「選取要偵測的影片」選擇 `.mp4`、`.avi` 或 `.mkv` 檔案即可。

如果權重放在其他位置，可在啟動前設定環境變數，不需要修改原始碼：

```powershell
$env:YOLO_WEIGHTS_DIR = "D:\models\traffic"
$env:DEEPSORT_REID_CKPT = "D:\models\traffic\ckpt.t7"
python start.py
```

也可以只替換單一偵測模式的檔案：

```powershell
$env:YOLO_TF_MODEL = "D:\models\traffic\redlight.pt"
$env:YOLO_ZEBRA_MODEL = "D:\models\traffic\zebra.pt"
```

程式啟動偵測前會檢查權重、影片與輸出資料夾；若檔案缺少，GUI 會顯示實際檢查的
位置。


## 使用說明

### 啟動應用程式

先安裝 [Rust](https://www.rust-lang.org/tools/install)，Windows 建置需要 C++ Build Tools
（MSVC 工具鏈）或 MinGW（GNU 工具鏈）。在專案根目錄建置一次：

```powershell
cargo build --manifest-path rust-ui/Cargo.toml
python start.py
```

`start.py` 會啟動 Rust 視窗。若本機有 `runtime/detection-env/Scripts/python.exe`，
會優先使用此偵測環境，否則使用目前的 Python；明確設定 `TRAFFIC_PYTHON` 可覆寫。
也可以直接執行 `rust-ui/target/debug/traffic-desktop.exe`；若 Python 不在 PATH，先設定：

```powershell
$env:TRAFFIC_PYTHON = "C:\path\to\python.exe"
```

Rust 介面提供影片多選、輸出資料夾、模型選擇、兩種偵測模式、批次與單片進度、
影片播放／暫停／停止／影格定位，以及 -45° 到 45° 的旋轉校正。
右側先選擇偵測模式、模型與輸出資料夾，再按「開始分析」。影片清單可切換預覽，
完整路徑與執行記錄可展開查看；右上角可切換深色／淺色外觀。
多選影片時預覽第一部；校正參考影片不會改變偵測清單，按「套用旋轉角度」後才套用於偵測。
取消偵測會停止 Python 預測程序；若正在旋轉轉檔，會等待轉檔完成才取消。
預覽使用 Rust 背景執行緒呼叫 Windows Media Foundation 解碼，並將像素直接送至影片紋理；
播放、暫停、停止、定位與旋轉預覽都不啟動 Python。旋轉預覽由 GPU 繪製，
影片最長邊縮至 960 像素，維持無聲預覽；偵測仍使用原始影片。
連續播放按時間推進，附近影格連續解碼，跨越較長時間或往回定位時才重新搜尋。
檔案與資料夾選擇器在背景執行緒開啟，選擇期間主視窗與預覽仍可操作。
拖曳時間軸每 50 ms 提交最新位置、放開時立即提交；解碼器會取消過時的定位請求，
避免先解完舊位置才回應新位置。長 GOP 影片的單次遠距定位仍需從關鍵影格解碼。
預覽時間軸的影格數由長度與幀率估算，可變幀率影片以名義幀率顯示。
原生預覽目前支援 Windows，實際編碼支援取決於系統的影片解碼器；
目前介面使用 OpenGL，由系統／驅動決定顯卡，啟動時會輸出 `UI GPU` 診斷。
Media Foundation 預覽預設建立 D3D11 / DXGI device manager，優先選 NVIDIA，
沒有 NVIDIA 時使用其他硬體顯卡。硬體管線建立或初始解碼失敗會退回軟體模式，並輸出原因。
會先協商 NV12 解碼，再透過影片處理器轉為縮小的 RGB 畫面；可用以下設定比較或排查：

```powershell
$env:TRAFFIC_VIDEO_ACCELERATION = 'auto' # 預設，硬體優先，失敗退回軟體
# 'hardware' 要求 D3D 管線成功；'software' 關閉 DXVA
python start.py
```

啟動記錄的 `Preview device` 與 `GPU output` 顯示預覽裝置及是否輸出 D3D 影格；
特定編碼是否真的使用硬體解碼還取決於系統解碼器與驅動，不以選到顯卡就當作已加速。
目前影格會回讀至 CPU 再上傳 OpenGL，所以連續播放不保證比軟體模式更快。
介面繪製、預覽解碼與 Python CUDA 推論分別選擇裝置，介面可能仍使用 Intel。
無法解碼時會顯示錯誤。Media Foundation 的設定依據
[Microsoft Source Reader 文件](https://learn.microsoft.com/en-us/windows/win32/medfound/processing-media-data-with-the-source-reader)。

模型權重與上述環境變數沿用原有設定。Rust 本身不需要 PyQt5，
影片預覽不需要 Python、OpenCV 或另外安裝 FFmpeg；Python 偵測仍需要原本的依賴，
包括 OpenCV 與偵測前旋轉轉檔用的 MoviePy。
偵測核心使用已安裝的 `ultralytics==8.0.3`，交通規則與 DeepSORT 使用專案腳本。
專案內的 ultralytics 副本缺少匯入檔案，不能作為完整套件使用。
舊版模型含 Python pickle；偵測子程序設定 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
以維持舊模型載入行為，請只使用可信且驗證過來源的權重。
若部署到其他資料夾，設定 `TRAFFIC_PROJECT_ROOT` 指向含 `desktop_bridge.py` 的專案根目錄；
中文預設使用作業系統字型，也可用 `TRAFFIC_FONT` 指定支援繁體中文的字型檔。

驗證方式：

```powershell
cargo check --manifest-path rust-ui/Cargo.toml
cargo test --manifest-path rust-ui/Cargo.toml
python test_desktop_bridge.py
pwsh -File rust-ui/check-ui.ps1 -Python python
```

Python 檢查會建立臨時影片與替代預測腳本，驗證偵測轉檔、中文／空白路徑、
進度、子程序錯誤、取消與缺少權重提示，無須模型權重；這不代表已驗證真實模型推論。
`check-ui.ps1` 僅用 Python 產生測試素材；執行 Rust 視窗時會指定不存在的 Python 路徑，
確認預覽可以獨立運作。它會開啟六種實際介面狀態，檢查程序正常退出與截圖更新，
將空白畫面、影片預覽、播放至最後影格、原生資料夾選擇期間播放、深色旋轉校正與小視窗截圖
存入 `runtime/ui-check/`，供視覺檢查。
Rust 測試會檢查像素通道、正負行距、截斷緩衝區；執行上面的畫面檢查後，
可額外驗證原生解碼、影格內容、前後定位、影片結尾與錯誤檔案：

```powershell
$env:TRAFFIC_TEST_VIDEO = (Resolve-Path runtime/ui-check/preview.avi).Path
cargo test --manifest-path rust-ui/Cargo.toml native_decode_seek_and_end -- --ignored
cargo test --manifest-path rust-ui/Cargo.toml rapid_scrubbing -- --ignored
```

有 D3D11 影片裝置時，可以對 H.264 影片執行硬體／軟體畫面比較與完整解碼測量：

```powershell
$env:TRAFFIC_TEST_VIDEO = 'C:\path\to\traffic.mp4'
cargo test --manifest-path rust-ui/Cargo.toml hardware_matches_software -- --ignored --nocapture
$env:TRAFFIC_VIDEO_ACCELERATION = 'hardware' # 可改成 software 比較
cargo test --manifest-path rust-ui/Cargo.toml decode_throughput -- --ignored --nocapture
```

真實權重檢查會逐幀解碼輸出並檢查幀數、幀率及後端完成事件：

```powershell
python scripts/validate_detection.py --video C:\path\to\traffic.mp4 --output C:\existing\output --model yolov8s --task zebra
```

這是流程驗證，違規辨識準確率需要標註資料另行評估。
本機模型來源、六模型檢查、1080p60 實測及效能改善記錄見
[2026-10-08 驗證報告](docs/performance-validation-2026-10-08.md)。

舊版 PyQt 介面仍可透過以下方式啟動：

```powershell
python start.py --legacy
```

以下為舊版介面操作說明，Rust 版使用相同的偵測設定與主要操作。

### 主畫面 (`主畫面` Tab)

1.  **選取要偵測的影片**:
    *   點擊 `選取要偵測的影片` 按鈕。
    *   選擇一個或多個 `.mp4`, `.avi`, `.mkv` 格式的影片檔案。
    *   如果選擇單一影片，影片將在 `label_videoframe` 區域顯示預覽，並可使用播放、暫停、停止按鈕控制。
    *   如果選擇多個影片，`label_videoframe` 將顯示提示訊息。
    *   選定的影片路徑將顯示在 `ShowFilePath` 標籤中。

2.  **選擇要輸出的路徑**:
    *   點擊 `選擇要輸出的路徑` 按鈕。
    *   選擇一個資料夾，偵測結果（處理後的影片）將儲存到此資料夾中。
    *   選定的資料夾路徑將顯示在 `ShowFilePath` 標籤中。

3.  **選擇 YOLOv8 模型**:
    *   使用下拉選單 (`comboBox`) 選擇要使用的 YOLOv8 模型：
        *   `yolov8s(快，不準確)`: 速度快，但準確性相對較低。
        *   `yolov8l`: 平衡速度和準確性。
        *   `yolov8x6(慢，準確)`: 速度慢，但準確性最高。

4.  **執行偵測**:
    *   **闖紅燈偵測**: 點擊 `闖紅燈偵測` 按鈕開始執行闖紅燈違規偵測。
    *   **車輛不禮讓行人偵測**: 點擊 `車輛不禮讓行人偵測` 按鈕開始執行車輛不禮讓行人違規偵測。
    *   **進度顯示**:
        *   `當前影片進度` (`SingleVideoProgressBar`): 顯示當前正在處理的影片的進度。
        *   `所有影片進度` (`MultiVideoProgessBar`): 顯示所有選定影片的整體處理進度。
    *   **警告與完成提示**: 如果未選擇影片或輸出路徑，將彈出警告訊息。影片處理完成後，將彈出完成提示。

### 校正畫面 (`校正畫面` Tab)

1.  **選擇一部影片做旋轉校正參考**:
    *   點擊 `選擇一部影片做旋轉校正參考` 按鈕。
    *   選擇一個影片檔案，該影片將用於預覽旋轉校正效果。
    *   影片將在 `rotate_screen` 區域顯示。

2.  **調整旋轉角度**:
    *   使用滑塊 (`rotate_screen_slider`) 調整影片的旋轉角度。滑塊範圍通常在 -90 到 0 度之間，預設為 -45 度。
    *   調整滑塊時，`rotate_screen` 區域會即時顯示旋轉後的影片畫面截圖。

3.  **確認旋轉角度**:
    *   調整到滿意的角度後，點擊 `確認` 按鈕。
    *   確認後的旋轉角度將應用於後續的偵測任務。

## 專案結構 (核心檔案)

-   `start.py`: 應用程式的入口點，啟動 Rust GUI；`--legacy` 啟動 PyQt5。
-   `rust-ui/`: Rust 桌面介面與 Cargo 鎖定檔。
-   `rust-ui/src/video.rs`: Rust 原生影片解碼與預覽執行緒。
-   `desktop_bridge.py`: Rust 與 Python 偵測程序的 JSON-lines 通訊。
-   `detection_backend.py`: 兩種介面共用的權重驗證、偵測程序與進度事件。
-   `controller.py`: 包含 GUI 的主要邏輯，處理使用者互動、影片選擇、偵測任務的啟動和進度更新。
-   `UI.py`: 由 `UI.ui` 自動生成的 Python 檔案，定義了 GUI 的介面佈局和元件。
-   `UI.ui`: Qt Designer 介面設計檔案，用於視覺化設計 GUI。
-   `opencv_engine.py`: 提供影片資訊讀取功能，使用 OpenCV 庫。
-   `requirements.txt`: 主專案的 Python 依賴列表。
-   `weights/`: 本機使用的 YOLOv8 與 DeepSORT 權重（不納入 Git）。
-   `YOLOv8_DeepSORT_Object_Tracking/`: 包含 YOLOv8 和 DeepSORT 相關的程式碼和模型。
    -   `YOLOv8_DeepSORT_Object_Tracking/requirements.txt`: 子模組的 Python 依賴列表。
    -   `YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/predict_tf.py`: 處理闖紅燈偵測的核心腳本。
    -   `YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/predict_zebra.py`: 處理車輛不禮讓行人偵測的核心腳本。
    -   `YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/deep_sort_pytorch/`: DeepSORT 追蹤模組。

## 故障排除

-   **`ModuleNotFoundError` 或其他依賴問題**: 確保您已按照「安裝指南」中的步驟正確安裝了所有 `requirements.txt` 檔案中的依賴，並且虛擬環境已激活。
-   **模型權重檔案未找到**: 確保檔名與 `weights/README.md` 相同，或設定
    `YOLO_WEIGHTS_DIR`、`DEEPSORT_REID_CKPT`、`YOLO_TF_MODEL`、`YOLO_ZEBRA_MODEL`。
-   **影片無法播放或處理**: 檢查影片檔案是否損壞，或格式是否受支援。確保您的系統安裝了必要的影片解碼器。
-   **GUI 介面顯示異常**: 嘗試重新生成 `UI.py` 檔案（如果 `UI.ui` 有修改）。

## 貢獻
歡迎任何形式的貢獻！如果您有任何建議、錯誤報告或功能請求，請隨時提交 Issue 或 Pull Request。
