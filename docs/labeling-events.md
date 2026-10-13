# 怎麼標事件時間(給評估用的人工真值)

只有你標的時間才算「真值」。我(模型/程式)的判讀不算。

## 1. 標什麼
- **不禮讓(按 F)**:在車輛的**前輪壓上斑馬線**的第一幀暫停並標記,條件是行人已經在斑馬線上,或正要踏上去。一台車標一次。
- **有禮讓(按 Y,可選)**:車輛明確停在斑馬線前讓行人過。標了才能算「有禮讓」的判斷對不對。
- **不算**:行人只在路緣等、車輛通過。(你已決定這種情況不算不禮讓。)
- 時間誤差 ±1 秒就夠了,不用精準到幀。

## 2. 操作
```powershell
$py = "$env:USERPROFILE\anaconda3\envs\traffic-modern\python.exe"
& $py scripts/label_events.py --video "C:\path\clip1.mp4" --output labels\clip1.jsonl
```
先點一下影片視窗,再用鍵盤:
| 鍵 | 功能 |
|---|---|
| 空白鍵 | 播放 / 暫停 |
| A / D | 退 / 進 1 幀 |
| Z / C | 退 / 進 10 幀 |
| J / L | 退 / 進 1 秒 |
| 1 / 2 / 3 | 0.25 倍 / 0.5 倍 / 正常速度播放 |
| F | 標「沒讓行人」 |
| Y | 標「有讓行人」 |
| U | 復原上一個標記 |
| N | 宣告「這支影片沒有任何事件」(已有標記時會拒絕,請先 U 復原) |
| Q | 存檔離開 |

- 每個標記按下就立刻存檔,中途關掉視窗也不會掉。再開同一個 `--output` 會接著標。
- **一支影片一個標註檔。**
- 標記的時間 = 幀號 ÷ fps,跟偵測程式用的同一套算法。

## 3. 一定要做:沒有事件的影片也要標
看完整支影片,確定都沒有事件,就按 **N** 再 Q。這會留下一個空檔,代表「人看過,沒有事件」。
沒有這種影片,就算不出誤報(車輛有禮讓卻被誤判為不禮讓)。

## 4. 算分數
```powershell
# 偵測(每支影片各跑一次,見 docs/yield-image-rule.md)
& $py yield_image_rule.py --masks out\clip1\seg\masks.npz --tracks out\clip1\tracks\raw_tracks.json --output out\clip1\events.jsonl
# 評估:每支影片一組「事件檔 標註檔」;沒有事件的影片要加 --allow-empty-truth
& $py scripts/eval_events.py --pair out\clip1\events.jsonl labels\clip1.jsonl --pair out\clip2\events.jsonl labels\clip2.jsonl --allow-empty-truth
```
輸出會有每支影片的結果,以及全部加總的 Precision / Recall / F1、誤報數、漏報數。

## 5. 要標多少才可信
- 一兩個事件的數字沒有統計意義,只能說「抓到/沒抓到」。
- 建議至少:十個以上不同的事件、不同路口與時間(含夜間)、和一樣多的「車輛有禮讓」或「沒事件」的片段。
- 規則的門檻(例如 0.1 個人高)要在一批影片上調,再用另一批沒看過的影片測,不要用同一批。

## 限制
標註視窗只做了啟動測試(開啟 8 秒不當掉),按鍵的互動行為我沒辦法替你測;存檔、復原、時間換算和評估邏輯有單元測試。
