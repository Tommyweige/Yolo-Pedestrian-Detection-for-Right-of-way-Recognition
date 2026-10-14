# 怎麼標事件時間(給評估用的人工真值)

只有你標的時間才算「真值」。我(模型/程式)的判讀不算。標註工具是 Rust 寫的桌面程式:`rust-ui/src/bin/event_labeler.rs`。

## 1. 標什麼
- **不禮讓(F)**:在車輛的**前輪壓上斑馬線**的第一幀暫停並標記,條件是行人已經在斑馬線上,或正要踏上去。一台車標一次。
- **有禮讓(Y,可選)**:車輛明確停在斑馬線前讓行人過。
- **不算**:行人只在路緣等、車輛通過(已決定)。
- 時間誤差 ±1 秒就夠了,不用精準到幀。

## 2. 開啟
第一次需要建置(約 5 分鐘),之後直接執行:
```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
cd E:\Learning\Yolo-Pedestrian-Detection-for-Right-of-way-Recognition
cargo build --release --manifest-path rust-ui/Cargo.toml --bin event_labeler
$video = (Get-ChildItem "$env:USERPROFILE\Videos\*_1080p60.mp4" -File | Select-Object -First 1).FullName
.\rust-ui\target\release\event_labeler.exe --video $video
```
- 標註會存到 `labels\<影片檔名>.jsonl`(可用 `--labels 路徑` 指定)。
- 不加 `--video` 也可以,視窗左上角有「開啟影片…」。

## 2.5 一次標很多支影片(快速切換)
```powershell
.ust-ui	argetelease\event_labeler.exe --folder datasets\來源影片 --events-dir datasets\eval-run1\events
```
- 左側「影片清單」用檔名列出資料夾內所有影片,後面顯示 `[未標]`、`[N 個標記]` 或 `[無事件]`;點一下就切換。
- 快捷鍵:`]` 或 PageDown 下一支,`[` 或 PageUp 上一支;上方也有「上一支 / 下一支」按鈕。
- `--events-dir` 資料夾內放 `<影片檔名(不含副檔名)>.jsonl`,切到該影片時自動載入偵測候選。不加就不自動載入。
- 畫面上方也有「開啟資料夾…」。

## 3. 畫面
- **中間**:影片。**右邊**:你標的所有標記(點一下跳過去,可刪除)和偵測候選。
- **下方兩條時間軸**:上面是整支影片(點擊或拖曳跳轉),下面是目前位置前後 3 秒(精準對位)。
  紅線=沒讓行人、綠線=有讓行人;圓點=偵測候選(橘=待確認、綠=已確認、灰=已否決)。
- 每個標記按下去就立刻存檔,中途關掉視窗也不會掉;再開同一支影片會接著標。

| 鍵 | 功能 |
|---|---|
| 空白鍵 | 播放 / 暫停 |
| A / D 或 ← / → | 退 / 進 1 幀(加 Shift 為 10 幀) |
| Z / C | 退 / 進 10 幀 |
| J / L | 退 / 進 1 秒 |
| Home / End | 到開頭 / 結尾 |
| 1 / 2 / 3 / 4 | 0.25 倍 / 0.5 倍 / 1 倍 / 2 倍 |
| F | 標「沒讓行人」 |
| Y | 標「有讓行人」 |
| U | 復原上一個標記 |
| N | 宣告「這支影片沒有任何事件」(已有標記時會拒絕) |

## 4. 用偵測結果加速(建議)
先跑偵測得到 `events.jsonl`(見 `docs/yield-image-rule.md`),再:
```powershell
.\rust-ui\target\release\event_labeler.exe --video $video --events out\events.jsonl
```
右邊「偵測候選」每一個都有「確認」/「否決」:
- **確認** = 你同意這是一個沒讓行人事件,會寫進標註檔(來源記為 `confirmed_candidate`)。
- **否決** = 這是誤報,**不會**寫進標註檔,只記在 `labels\<影片檔名>.review.json`。
- 點候選的時間會跳到那一幀,你自己看畫面判斷,不要憑感覺按確認。
- **仍要自己看完整支影片**,因為程式漏掉的事件不會出現在候選裡,要用 F 自己補標。

## 5. 一定要做:沒有事件的影片也要標
看完整支影片,確定都沒有事件,按 **N**。這會留下一個空檔,代表「人看過,沒有事件」。
沒有這種影片,就算不出誤報(車輛有禮讓卻被誤判為不禮讓)。

## 6. 算分數
```powershell
$py = "$env:USERPROFILE\anaconda3\envs\traffic-modern\python.exe"
# 每支影片一組「事件檔 標註檔」;沒有事件的影片要加 --allow-empty-truth
& $py scripts/eval_events.py --pair out\clip1\events.jsonl labels\clip1.jsonl --pair out\clip2\events.jsonl labels\clip2.jsonl --allow-empty-truth
```

## 7. 要標多少才可信
- 一兩個事件的數字沒有統計意義。建議至少十個以上不同事件、不同路口與時間(含夜間),和同樣多的「有禮讓」或「沒事件」的片段。
- 規則門檻要在一批影片上調,再用另一批沒看過的測。

## 已知限制
- 視窗的**畫面顯示**我用截圖檢查過(影片、標記、候選、兩條時間軸、中文字型);**鍵盤與滑鼠互動**我沒辦法替你點,只測了存檔/復原/刪除/候選審核/時間軸座標的邏輯(11 個單元測試)。
- 影片總幀數是解碼器用「片長 × fps」估的,可能比實際多幾幀(測試影片顯示 906,OpenCV 數到 900)。影片最後幾幀的時間可能差不到 0.1 秒,中間的標記不受影響。
- 只支援 Windows(影片解碼用 Media Foundation,與既有桌面程式共用)。
