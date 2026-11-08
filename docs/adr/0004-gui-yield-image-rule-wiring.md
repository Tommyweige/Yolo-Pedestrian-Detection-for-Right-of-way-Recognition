# Rust 介面的不禮讓模式改走影像規則流程

## Context

Rust 桌面介面（`rust-ui/`，ADR 0002）的「未禮讓行人」模式原本呼叫 `predict_zebra.py`
（yolov8s_zebra + DeepSORT，conf=0.7），也就是 `detection_backend.TASKS["zebra"]`。在 7 支現成影片、
17 個人工標記上實測 0 個旗標：zebra 類別在 conf 0.7 下幾乎偵測不到，失敗發生在 DeepSORT 之前。
影像規則那套流程（`docs/yield-image-rule.md`）只跑得起來命令列 `scripts/run_pipeline.py`，介面上一次都沒被執行過。

舊的 PyQt5 介面（`UI.py`、`controller.py` 那條線）已從 main 移除（PR #41）；Rust 介面是之後唯一的介面。

## Decision

Rust 介面 → `desktop_bridge.py` → `detection_backend.run_yield()`，依序跑三支既有腳本，旗標與參數與
`docs/yield-image-rule.md` 的命令列一致：

1. `scripts/segment_crosswalk.py --stride 3`（labeling 環境，需要 transformers；2026-11-08 起改為 `--every-seconds 0.1`，60 fps 影片每 6 幀一次）
2. `scripts/track_video.py --conf 0.15 --imgsz 1280`（modern 環境，Ultralytics 8.4）
3. `yield_image_rule.py`（modern 環境，fps 與畫面尺寸由影片本身讀取）

三個步驟都是**子行程**，直譯器由 `TRAFFIC_LABELING_PYTHON`、`TRAFFIC_MODERN_PYTHON` 決定（沿用
`run_pipeline.py` 既有的變數；沒設時用目前直譯器）。分割模型路徑由 `TRAFFIC_CROSSWALK_MODEL` 覆寫，預設
`runtime/models/mask2former-mapillary`；偵測器權重由 `TRAFFIC_DETECTOR_WEIGHTS` 決定；`TRAFFIC_DEVICE` 選 GPU/CPU。

**不重用 `scripts/run_pipeline.py`**：它是批次驅動器，輸入是一個資料夾的影片與 run 名稱，輸出固定在
`datasets/<run>/…`，而且第四步一定要產生疊圖影片。介面有自己的選檔清單、使用者選的輸出資料夾與進度事件協定，
套用 run_pipeline 得先把影片複製成資料夾、再把結果搬回來，還多花一次疊圖轉檔。兩條路徑共用的是同一組腳本與旗標，
不是同一支驅動程式。

**不直接呼叫 Python 函式**：分割需要 labeling 環境、追蹤需要 modern 環境，介面的 legacy 直譯器（PyQt5、cv2、
`predict_*.py` 那一套）兩者都沒有；而規則本身（`yield_image_rule.py`）讀的是檔案，直接呼叫只是把同一件事換一種
失敗方式。子行程也讓規則保持在介面之外，符合「接線不改判斷邏輯」。

**事件用既有的 stdout JSON-lines 協定回傳**：Rust 的 `Worker` 本來就一行一個 JSON 讀 `desktop_bridge.py` 的
stdout（stderr 進執行記錄，EOF 代表程序結束，取消就是關掉 stdin）。不禮讓模式在同一個通道多送
`{"type": "event", ...}`，內容是規則的原始欄位加上 `video`、`video_name`：

```json
{"type": "event", "index": 0, "video": "…\\clip.mp4", "video_name": "clip.mp4",
 "kind": "vehicle_on_crosswalk_with_pedestrian", "frame": 419, "time_s": 6.99,
 "vehicle_id": 1288, "vehicle_kind": "car", "pedestrian_id": 665,
 "pedestrian_state": "waiting_at_edge", "pedestrian_entered_crosswalk": true, "needs_review": false}
```

選 stdout 而不是「叫 Rust 去讀 `events.jsonl`」：進度、錯誤、事件三種訊息本來就走同一條協定，事件跟著串流送出，
介面不必等檔案落地、不必自己找路徑，也不需要第二種解析程式；`events.jsonl` 仍然是留在輸出資料夾的耐久產物，
供人工或評估腳本使用，但介面不讀它。

## Consequences

- 中間產物放在使用者選的輸出資料夾下，每支影片一個資料夾：
  `yield/<影片檔名>/seg/masks.npz`、`seg/stats.json`、`tracks/raw_tracks.json`、`events.jsonl`、
  `run.json`（影格數、fps、尺寸、各步驟秒數、事件時間、實際指令）。可直接拿同一份 `masks.npz`／`raw_tracks.json`
  重跑規則，不必重做分割。
- 每次按「開始分析」都重算三個步驟，不沿用舊的中間產物，避免改了 stride 或換了影片卻沿用舊遮罩。
- 實測一支 12 秒、722 影格、59.94 fps、1920×1080 的影片：總計 192.7 秒（分割 152.9、追蹤 35.2、規則 4.6），
  2 件事件（4.705 秒、6.99 秒）。分割是主要成本，取樣 241 格約 0.6 秒/格（stride 3）；同一支影片在 GPU 被其他工作
  佔用時跑了 246.5 秒，差距主要在分割。
- 換成規則的新預設（候選清單、`contact_dedup="pair"`、深度 0.03）後，介面上的第一次輸出：944 影格、30 fps、
  1920×1080 的影片跑 181.1 秒（分割 143.5、追蹤 31.4、規則 6.2），13 件事件（2.733、2.733、3.333、3.333、4.2、
  4.433、6.5、7.067、8.8、18.133、24.4、25.767、27.167 秒），清單摺成 7 列。規則步驟直接跑 `yield_image_rule.py`，
  不帶任何建構參數，所以介面看到的就是命令列預設的那一組（7 支標註影片合計 50 件，不是舊建構的 31 件）。
- 介面右欄的「不禮讓事件」分頁列出事件（播放器在左欄，兩者並排，展開不會壓縮播放器）：每列是可點的時間（mm:ss.mmm）、車輛
  （種類與 track id）、行人（id、狀態、是否踏上斑馬線）、需要複審的標記與影片檔名；滑過時間可看完整路徑與影格。
  點時間會把播放器切到該影片、跳到事件影格（規則的影格從 1 起算，播放器從 0）。事件時間取自 `events.jsonl` 的
  `time_s`，與影片 fps 一致。
- **legacy `predict_zebra.py` 保留、但不再接在「未禮讓」模式上**：`detection_backend.TASKS["zebra"]` 沒動，因為它是
  vendored 的第三方程式碼與 `scripts/validate_detection.py` 的既有路徑；Rust 介面的「闖紅燈」模式（`tf`）仍走同一套
  legacy 後端。不禮讓模式則固定送 `task: "yield"`，由 `desktop_bridge.py` 分派到 `run_yield()`。
- 旋轉校正只作用在舊的 `predict_*.py` 路徑。影像規則與畫面方向無關（斑馬線軸向由遮罩的 PCA 求），所以不套用旋轉，
  事件時間不受影響。
- 取消是關掉 bridge 的 stdin：`run_yield()` 在步驟之間與讀影格時檢查取消旗標，正在跑的子行程會被終止，已寫出的
  中間產物留在原處，下次重跑會覆蓋。
