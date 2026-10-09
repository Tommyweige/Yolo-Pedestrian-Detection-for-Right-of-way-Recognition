# 交通偵測修正與驗證（2026-10-08）

規格：[GitHub #1](https://github.com/Tommyweige/Yolo-Pedestrian-Detection-for-Right-of-way-Recognition/issues/1)。
使用 Matt Pocock 的 to-spec → to-tickets → implement / TDD → code-review 流程。
使用者確認四張獨立 ticket 與生產用每幀處理測試邊界，審查基準指定為 `95d7860`，涵蓋 Rust 移植。

## 修正與回歸證據

| Ticket | 修正 | 修正前的失敗 | 修正後的驗證 |
|---|---|---|---|
| [#2](https://github.com/Tommyweige/Yolo-Pedestrian-Detection-for-Right-of-way-Recognition/issues/2) | 裁切保留 BGR，HSV 轉換使用 BGR | 紅、藍、黃三組畫面決策錯誤 | 生產每幀入口處理紅／綠／藍／黃，輸出 Stop／Go 正確；保留黃色政策 |
| [#3](https://github.com/Tommyweige/Yolo-Pedestrian-Detection-for-Right-of-way-Recognition/issues/3) | 兩種模式以來源幀號與 fps 記錄時間，移除只計算有偵測影格的計數器 | 六組 30／60／29.97 fps 事件時間錯誤 | 首幀時間 0、整秒與分數幀率、空白影格後事件皆正確；無效 fps 報錯；紀錄為 UTF-8 |
| [#4](https://github.com/Tommyweige/Yolo-Pedestrian-Detection-for-Right-of-way-Recognition/issues/4) | 每幀更新一次真實 DeepSORT，包含空偵測，並清除過期的軌跡／方向狀態 | 兩種模式超過壽命後仍沿用 ID 1 | 兩個空白影格仍保留 ID；四個空白影格超過測試用壽命後產生新 ID，沒有重複推進時間 |
| [#5](https://github.com/Tommyweige/Yolo-Pedestrian-Detection-for-Right-of-way-Recognition/issues/5) | 對全部斑馬線檢查行人腳底點，不只取最後一框 | 交換相同兩框的順序後，原本應有的事件消失 | 兩種框順序、無斑馬線、腳底點在框外、重疊框皆符合預期 |

`test_traffic_rules.py` 的五個測試方法、21 組子案例通過。測試把已知偵測框送入生產預測器，
使用實際 DeepSORT 與可信 `ckpt.t7`，未用假追蹤器代替內部協作者。
同一組測試的紅／綠結果保存在忽略的 `runtime/matt-red-02.txt` 至 `matt-green-05.txt`；完整結果為 `runtime/matt-full-tests.txt`。

## 整合驗證

- 共用後端檢查通過：旋轉、中文／空白路徑、進度、程序錯誤、取消、缺權重及 IPC 初始化。
- Rust 全部八項測試通過，包含原生解碼、硬體／軟體畫面比較、連續解碼、定位與資料夾背景執行緒；沒有略過測試。Clippy 通過。
- 使用者 1080p60 影片，以原有 yolov8s zebra 設定完整處理：輸入／逐幀解碼輸出均為 900 幀、60 fps；後端完成約 69.72 秒。
- TF 模式交通照片短片：輸入／輸出均為 6 幀、6 fps；完成約 12.65 秒。
- 整合摘要位於 `runtime/e2e/matt-zebra-fixed/summary.json`、`runtime/e2e/matt-tf-fixed/summary.json`。

此次沒有調整模型、信心門檻、推論尺寸或測試增強。耗時受啟動、快取及本機負載影響，不以跨次耗時差異宣稱效能改善。
使用者影片本次仍沒有產生違規紀錄；沒有人工標註，所以不宣稱已辨識出不禮讓事件，也不報辨識準確率。

## 審查與限制

Standards 軸的共用後端 API 文件建議已處理；本機權重路徑是使用者明確要求的內容，審查者撤回可攜性建議。
Spec 軸的獨立代理先前因額度限制中斷，重試後完成 `95d7860` 以來的移植與四張 ticket 審查。
兩個審查軸均沒有未解決的確認問題；Matt 工作流程設定文件屬於使用者指定流程，並非額外功能。
審查代理以程式碼與既有測試證據檢查，未自行重跑推論或重新目視截圖。
六種模型／模式測試使用交通照片短片；完整 900 幀實片使用 yolov8s zebra，沒有宣稱六組皆跑完整實片。

尚未修正安裝依賴、預覽末尾估算幀數、行人方向啟發式、交通號誌／車道配對，亦未建立標註資料集。
上述工作不屬於這四張 ticket。規格 parent 保持不變，不因 ticket 拆分／實作自動關閉。
