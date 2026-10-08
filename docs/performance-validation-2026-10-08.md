# Rust 桌面與真實模型驗證（2026-10-08）

## 已修正

- 三種原生選擇器共用背景工作執行緒，不在 egui 更新執行緒等待。取消保留原設定，只限制同時開第二個選擇器。
- 拖曳預覽合併為最新請求，解碼期間可取消過時定位；拖曳每 50 ms 提交一次，放開立即提交。播放請求不會反覆取消解碼而造成飢餓。
- 修正解碼器被取消後的實際 PTS 游標、B-frame 時間戳排序與起始 PTS。驗證 AVI、MP4、MKV、B-frame、1080p、中文與空白路徑。
- 偵測使用已安裝的 `ultralytics==8.0.3`，保留專案交通規則與 DeepSORT。原始 vendored 套件有缺檔及初始化錯誤。
- 修正 Windows IPC 啟動卡住：先初始化 OpenCV/NumPy，再啟動輸入管線監聽；預測子程序使用 `stdin=DEVNULL`，不繼承父介面的取消管線。原先堆疊停在 NumPy DLL 載入，管線關閉前沒有第一個進度事件；修正後回歸測試檢查開啟管線時能收到進度。
- `start.py` 優先使用現有 `runtime/detection-env`，明確的 `TRAFFIC_PYTHON` 設定仍優先。

## 模型與環境

使用使用者提供的 [Hugging Face 模型庫](https://huggingface.co/TommyPanLab/traffic-violation-yolov8)，
固定 revision `98f890a251c03aff28e164c6e835715dbc5019a6`。
六個 YOLO 權重與 `ckpt.t7` 全部下載到專案 `weights/`，逐檔驗證大小及 SHA-256。
完整下載收據在 `runtime/model-receipt.json`；大型權重不納入 Git。

本機為 Windows、RTX 4060 Laptop GPU、Python 3.12、torch 2.13.0+cu130、
torchvision 0.28.0+cu130、ultralytics 8.0.3、NumPy 1.26.4、OpenCV 4.11.0.86、MoviePy 1.0.3。
偵測環境位於忽略的 `runtime/detection-env`，使用本機既有 Conda 環境的 torch；這是本機驗證環境，並非可搬到其他電腦的完整安裝包。
舊 checkpoint 需要 pickle 載入，僅對偵測子程序設定 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；權重應維持可信來源及雜湊驗證。

## 實測

### 主視窗與資料夾選擇

`rust-ui/check-ui.ps1` 開啟實際原生資料夾選擇器，選擇器保持開啟時主視窗完成 248 次更新、預覽播放到第 17 幀。
這個檢查使用不存在的 Python 路徑，證明預覽獨立於 Python。
六種視窗狀態皆通過：空白、預覽、播放結尾、選擇資料夾、深色校正、小視窗。
證據：`runtime/ui-check/folder.json` 與各狀態截圖。

### 拖曳時間軸

比較同一解碼器「等舊定位完成後再提交新位置」與「直接提交最新位置」，各三次。
數字是最新請求至解碼結果的時間，未包含顯示器呈現延遲，不是與 PyQt5 的全面基準比較。

| 素材 | 舊請求串行等待中位數 | 最新請求優先中位數 | 單次遠距定位 |
|---|---:|---:|---:|
| 合成 1080p 長 GOP、30 fps | 438.3 ms | 38.7 ms | 約 61–358 ms |
| 使用者 1080p60 交通影片 | 719.2 ms | 50.7 ms | 約 178–654 ms |

合成素材改善約 91%，使用者影片約 93%。單次遠距定位仍需解碼關鍵影格之後的畫面。
使用者影片測量與 GPU 推論同時進行，數字反映該次本機負載，不能保證所有影片與硬體相同。
原始量測：`runtime/seek-before.txt`、`runtime/seek-after.txt`、`runtime/seek-real.txt`。

### 真實模型與使用者影片

六種模型組合（s/l/x6 × tf/zebra）各處理一支由專案交通照片製成的 6 幀、6 fps 短片。
六組均完成 YOLO 推論、DeepSORT 流程、完成事件及輸出影片解碼；每支輸出均為 6 幀、6 fps。
各組總耗時約 9.0–11.8 秒，短片大部分時間是啟動與模型載入，不能拿此吞吐量作持續推論排名。
證據：`runtime/e2e/six-models/*/summary.json`。

使用者影片：`C:\Users\tommy\Videos\不禮讓行人，就是台灣駕駛日常（抱歉了小黃就你的車牌錄的最清楚，只好檢舉你）_1080p60.mp4`。
以 `yolov8s_zebra.pt`、角度 0、原始設定（imgsz=1920、conf=0.7、iou=0.3、augment=True、half=True）完整處理。
輸入 1920×1080、900 幀、60 fps；輸出逐幀解碼確認 900 幀、60 fps。
後端完成事件耗時 **93.47 秒，9.63 幀／秒**，含冷啟動、推論、追蹤及影片編碼，不含事後逐幀驗證。
證據：`runtime/e2e/real-zebra-s/summary.json`、`events.jsonl` 與同目錄下的輸出影片。

另外直接從 Rust 視窗啟動同一支完整影片，經 JSON-lines bridge 與真實 Python 預測程序處理。
視窗收到完成訊號，單片／批次進度均為 100%，没有錯誤；GUI 檢查總耗時 **106.47 秒**，
含視窗初始化及完成截圖，不能直接當作純推論速度。輸出再次逐幀解碼為 900 幀、60 fps。
證據：`runtime/e2e/gui-real.png`、`gui-real.json`、`gui-real/summary.json` 與該目錄的輸出影片。

進度紀錄中 565 幀有 person、486 幀有 zebra 偵測；此統計是每幀是否有該類別，不是人數或事件數。
本次沒有產生 `output.txt` 違規紀錄。不能因流程完成，就認定已正確抓到使用者指出的不禮讓事件。
沒有逐幀標註與違規時間標記，因此本次不報 precision、recall 或違規正確率。

## 下一步改善順序

### GPU 路徑確認

另以真實權重執行預測腳本，日誌顯示：
`CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)`。
Rust 視窗建立時讀取 OpenGL 的實際 vendor/renderer，顯示：
`UI GPU: Intel / Intel(R) Iris(R) Xe Graphics`。
證據分別在 `runtime/gpu-check/inference.log`、`runtime/gpu-check/ui-gpu.log`。

因此模型推論已走 NVIDIA CUDA，但 OpenGL 介面由系統／驅動選到 Intel。
預覽目前沒有提供 `MF_SOURCE_READER_D3D_MANAGER`，並未接上指定 NVIDIA 的硬體解碼路徑。
原生 Media Foundation 解碼不代表已實作 NVDEC，也不能由 Intel 的 3D 使用率推論 YOLO 在 Intel 上運算。
[Microsoft 的 Source Reader 文件](https://learn.microsoft.com/en-us/windows/win32/medfound/mf-source-reader-d3d-manager)
指出提供 D3D 裝置可讓支援 DXVA 的解碼器使用硬體加速。

若要調整介面顯卡，可在 Windows「設定 → 系統 → 顯示器 → 圖形」為
`rust-ui/target/debug/traffic-desktop.exe` 設定高效能 GPU，再重新啟動並確認 `UI GPU`。
更換執行檔路徑（例如 release）需對該路徑另設。
這只處理介面繪製；影片硬體解碼仍需要另接 D3D device manager 與解碼／色彩轉換管線，
還應量測 CPU 回讀與跨顯卡複製成本，不能承諾切換顯卡就會消除遠距定位延遲。
[Windows 圖形設定說明](https://support.microsoft.com/en-au/windows/hardware/display-graphics/optimizations-for-windowed-games-in-windows-11)。

| 優先 | 發現與影響 | 建議 |
|---|---|---|
| 1 | 未產生違規紀錄；目前依斑馬線框、行人腳底線與車輛軌跡相交判定，容易受視角、遮擋與方向估計影響 | 先標註這支影片的行人、斑馬線、車輛軌跡與事件時間，逐階段定位漏判，再調整交通規則；不要只降低信心門檻 |
| 2 | 兩個舊腳本把時間寫成 `fcount/30`，且無偵測時提早 return 不增加 fcount | 改用實際來源 PTS 或來源幀號與 fps，補 60 fps 與空白影格的檢查，避免違規時間錯位 |
| 3 | 完整 15 秒片段約 93 秒，原始設定用 1920 寬度與測試增強 | 用有標註片段比較較小 imgsz、關閉 augment 的速度與召回；目前保留原設定，沒有為速度默默降低品質 |
| 4 | 最新定位已改善，但單次遠距定位仍約數百 ms | 若仍影響使用，可生成預覽代理影片或快取關鍵影格附近畫面；需要考量代理產生時間與磁碟空間 |
| 5 | Media Foundation 使用整體媒體長度估算此影片為 906 幀，OpenCV 實際輸出為 900 幀 | 預覽改用視訊軌時長或 PTS 時間軸，處理音訊比視訊長與可變幀率；目前 nominal frame 計數已在 README 註明 |
| 6 | 每部影片重新啟動 Python 並載入權重，短片測試約 9–12 秒 | 大量短片才值得改成持續工作程序；每支影片仍須重設追蹤與規則狀態 |

沒有加入代理影片、模型快取或改寫交通規則；先完成有證據的介面阻塞、過時定位與 IPC 修正。

## 可重跑的檢查

- Rust `cargo test`、`cargo clippy -- -D warnings`、`cargo build` 均通過。
- `python test_desktop_bridge.py`：旋轉、中文路徑、進度、錯誤、取消、缺權重與開啟管線時的初始化進度。
- `pwsh -File rust-ui/check-ui.ps1 -Python python`：六種真實視窗狀態。
- `TRAFFIC_TEST_VIDEO` + ignored `native_decode_seek_and_end`：各種編碼素材；ignored `rapid_scrubbing_keeps_the_latest_request`：請求合併。
- `TRAFFIC_SEEK_VIDEO` + ignored `measure_seek_latency --nocapture`：解碼回應時間。
- `scripts/validate_detection.py --video ... --output ... --model yolov8s --task zebra`：真實模型及逐幀輸出驗證；`--verify-only` 可重新檢查已完成的紀錄與影片。

Rust GUI 完整檢查可使用現有截圖入口（輸出資料夾需先建立，使用新的路徑避免混入先前輸出）：

```powershell
$env:TRAFFIC_PYTHON = (Resolve-Path runtime/detection-env/Scripts/python.exe).Path
$env:TRAFFIC_PREVIEW_VIDEO = 'C:\path\to\traffic.mp4'
$env:TRAFFIC_CHECK_OUTPUT = (Resolve-Path C:\existing\output).Path
$env:TRAFFIC_SCREENSHOT = Join-Path $PWD 'runtime/gui-check.png'
$env:TRAFFIC_SCREENSHOT_VIEW = 'detect'
& rust-ui/target/debug/traffic-desktop.exe
& $env:TRAFFIC_PYTHON scripts/validate_detection.py --video $env:TRAFFIC_PREVIEW_VIDEO --output $env:TRAFFIC_CHECK_OUTPUT --gui-report runtime/gui-check.json
Remove-Item Env:TRAFFIC_SCREENSHOT,Env:TRAFFIC_SCREENSHOT_VIEW,Env:TRAFFIC_PREVIEW_VIDEO,Env:TRAFFIC_CHECK_OUTPUT
```

一般使用只需在專案根目錄執行 `python start.py`。
