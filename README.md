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
- **使用者友善的 GUI**: 基於 PyQt5 開發，提供直觀的操作介面。

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


## 使用說明

### 啟動應用程式
在專案的根目錄下，執行 `start.py` 來啟動 GUI 應用程式：
```bash
python start.py
```

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

-   `start.py`: 應用程式的入口點，啟動 PyQt5 GUI。
-   `controller.py`: 包含 GUI 的主要邏輯，處理使用者互動、影片選擇、偵測任務的啟動和進度更新。
-   `UI.py`: 由 `UI.ui` 自動生成的 Python 檔案，定義了 GUI 的介面佈局和元件。
-   `UI.ui`: Qt Designer 介面設計檔案，用於視覺化設計 GUI。
-   `opencv_engine.py`: 提供影片資訊讀取功能，使用 OpenCV 庫。
-   `requirements.txt`: 主專案的 Python 依賴列表。
-   `YOLOv8_DeepSORT_Object_Tracking/`: 包含 YOLOv8 和 DeepSORT 相關的程式碼和模型。
    -   `YOLOv8_DeepSORT_Object_Tracking/requirements.txt`: 子模組的 Python 依賴列表。
    -   `YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/predict_tf.py`: 處理闖紅燈偵測的核心腳本。
    -   `YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/predict_zebra.py`: 處理車輛不禮讓行人偵測的核心腳本。
    -   `YOLOv8_DeepSORT_Object_Tracking/ultralytics/yolo/v8/detect/deep_sort_pytorch/`: DeepSORT 追蹤模組。

## 故障排除

-   **`ModuleNotFoundError` 或其他依賴問題**: 確保您已按照「安裝指南」中的步驟正確安裝了所有 `requirements.txt` 檔案中的依賴，並且虛擬環境已激活。
-   **模型權重檔案未找到**: 確保您已將 YOLOv8 和 DeepSORT 的權重檔案放置在正確的路徑下。
-   **影片無法播放或處理**: 檢查影片檔案是否損壞，或格式是否受支援。確保您的系統安裝了必要的影片解碼器。
-   **GUI 介面顯示異常**: 嘗試重新生成 `UI.py` 檔案（如果 `UI.ui` 有修改）。

## 貢獻
歡迎任何形式的貢獻！如果您有任何建議、錯誤報告或功能請求，請隨時提交 Issue 或 Pull Request。
