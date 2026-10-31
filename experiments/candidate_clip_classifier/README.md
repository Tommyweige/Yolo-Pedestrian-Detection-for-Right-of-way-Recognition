# 規則產生候選 → 時空模型分類（借用電影暴力辨識論文的做法）

實驗分支 `experiment/candidate-clip-classifier`，從 `origin/main`（e3d7f36）開出。目的是試另一條路線：規則不再判到底，只負責產生高召回的候選片段，再交給時空模型分類。評分沿用規則路線同一支 scorer，兩邊數字可以直接比。

**結論先講：用現有資料，分類器路線贏不過規則。** 7 支影片只有約 20 個正例事件，連拿來量測的量都不夠，更別說訓練 TimeSformer。下面的數字只能當成可行性下界，不是準確率。

## 跟規則路線共用的東西（只讀）

- 規則與 scorer 都直接 import 自 `E:\Learning\yolo-ped-rulefix`（branch `feature/rule-eval-tolerance`，commit `4201f17`）：`yield_image_rule.py`、`scripts/eval_rule_events.py`，容差固定 **1.25 秒**。見 `common.py`。
- 影片、遮罩、軌跡、標記：`E:\Learning\Yolo-Pedestrian-Detection-for-Right-of-way-Recognition` 底下的 `datasets\來源影片`、`datasets\eval-run2\<stem>\{seg\masks.npz, tracks\raw_tracks.json}`、`labels\<stem>.jsonl`。共 7 支影片、17 個標記（9 manual、8 confirmed），合成 15 個事件。
- 人工判定：`yolo-ped-rulefix\.amp\in\scratch\verdicts_after.json`，對規則的 28 個事件給 A15 B3 C6 D4。

## 1. 候選產生器：放寬到多少、代價多少（`candidates.py`）

規則每個 frame 的幾何判斷全部沿用（`scan_frame`、重複軌跡合併、騎士排除），只拿掉兩個「判斷」，改交給分類器做：

1. **每台車只報一次（取最早的行人）→ 每組（車、行人）接觸都報。** 單獨放寬門檻時，一個早出現、不該報的接觸會把那台車「用掉」，後面真正的事件反而被吃掉（全部門檻一起放寬時，事件召回 13/15 反而掉到 12/15）。
2. **`require_crossing=True` → False**：在邊緣等的行人也報。

召回掃描（容差 1.25 秒，in-sample）：

| 設定 | 事件／候選數 | manual | confirmed | 事件召回 | label precision |
|---|---|---|---|---|---|
| 規則（4201f17） | 28 | 5/9 | 7/8 | 11/15 | 0.429 |
| 規則 + `require_crossing=False` | 39 | 7/9 | 7/8 | 13/15 | 0.359 |
| 規則 + lateral 6 / min_seconds 0.1 / flank 0.15+band 0.3 / edge 1.5（逐項） | 28–35 | 5/9 | 7/8 | 11/15 | 0.34–0.43 |
| 規則 + 以上全部 | 61 | 6/9 | 7/8 | 12/15 | 0.213 |
| 每組接觸都報 | 53 | 5/9 | 8/8 | 11/15 | 0.245 |
| **每組接觸都報 + `require_crossing=False`（採用）** | **162** | **8/9** | **8/8** | **14/15** | 0.099 |
| 再加 lateral 5 | 173 | 8/9 | 8/8 | 14/15 | 0.092 |
| 再加全部門檻放寬 | 364 | 8/9 | 8/8 | 14/15 | 0.044 |

代價是事件召回 11/15 → 14/15，候選數 28 → 162（5.8 倍）。其他門檻再怎麼放寬都不會多抓到任何事件，只會讓候選數變多。剩下沒抓到的 v6 19.953 秒，所有門檻都放寬後附近仍然沒有任何車與行人的接觸，所以問題出在上游（偵測、追蹤或遮罩），不是規則門檻。

## 2. 第 4 通道怎麼組（`clips.py`）

輸入跟論文一樣是 9 frame × 224 × 224 × 4（RGB 加 1 個二值先驗通道）。

- **取窗**：以事件時間為中心，前後各 1.6 秒，每 0.4 秒取一 frame，共 9 frame。規則和標記者對事件時間點的判讀最多差到容差 1.25 秒，取 ±1.6 秒，候選中心偏掉時標記的那一刻仍在窗內。車子通過斑馬線約 2–3 秒，這個窗也看得到接近、通過、離開。
- **裁切**：不整張縮到 224。1920 寬縮到 224 時，行人只剩約 10 px 高。改成以候選的那一組車與行人為中心，取 9 frame 內兩者框的聯集放大 1.5 倍、至少 224 px 的正方形，9 frame 共用同一個裁切範圍，保留兩者在畫面中的相對運動。
- **先驗遮罩**（逐 frame，二值）= 斑馬線遮罩 ∪ 車輛框 ∪ 膨脹後的行人框。
  - 斑馬線：**不膨脹**。要判的是「車有沒有開上斑馬線」，證據就在斑馬線的邊界；論文讓武器保持銳利邊界，也是同樣的理由。
  - 車輛框：**不膨脹**，理由相同：車輪那一線和斑馬線邊緣的相對位置就是證據。
  - 行人框：**四邊各膨脹 0.5 倍行人身高**。論文固定膨脹 50 px，但鏡頭遠近不同時，50 px 對應的實際距離也不同，所以改成跟著人的大小縮放。膨脹出來的區域就是行人前後左右、讓行車輛不該進入的空間。
  - 只放這一組候選的車和行人，不放畫面裡所有物件：先驗要告訴模型「問的是哪台車、哪個人」。
- **V1 變體**：遮罩外的背景壓暗到 30%，給只吃 3 通道的預訓練骨幹用。

## 3. 訓練資料：正負樣本怎麼來（`dataset.py`、`common.ground_truth`）

每個候選依序套用下列規則決定標籤：

1. 同一台車、相差 1.25 秒內有人工判定時，照判定：A／B → 正，C → 負，D → 不知道。
2. 沒有判定，但 1.25 秒內有標記 → 正。標記不指定是哪台車，所以這是弱標籤，跟 scorer 的配對方式一致。
3. 2.5 秒內都沒有任何標記，也沒有非 C 的判定 → 負。
4. 其他 → 不知道，不參與訓練、也不算進 AUROC／AP。

自動切出的負樣本：

- **困難負樣本**：符合上面第 3 條的候選本身（產生器報了，但附近沒有任何事件）。
- **一般車流負樣本**：車開上斑馬線（只用規則的 `vehicle_on`，不要求附近有行人），在它停留在斑馬線上那段時間的中點切一個窗，搭配最近的行人。條件是 2.5 秒內沒有標記或非 C 判定，而且這台車 1.25 秒內沒有候選。只用於訓練。
- **時間平移**：每個正例另外在 ±0.4 秒處各切一窗，只用於訓練。

最後得到：162 個候選，其中正 51／負 78／不知道 33；另有一般車流負樣本 15 個、平移正例 102 個。

這些負樣本有雜訊：28 個規則事件裡，有 6 個 A 判定附近沒有任何標記，代表標記不完整。所以「附近沒標記」不等於「沒違規」。

## 4. 樣本量：要多少才有意義（`sizing.py`）

| 用途 | 需要 | 現在 |
|---|---|---|
| 召回率估到 ±0.10（95%，p≈0.5） | 97 個事件 | 15 個標記事件；加上沒有標記的 A，約 20 個 |
| 召回率估到 ±0.05 | 385 個事件 | 同上 |
| 區分精確率 0.536（規則）與 0.75（α=0.05、power 0.8） | 每組 78 個事件 | 每組 28 個 |
| 區分 0.536 與 0.65 | 每組 291 個事件 | 每組 28 個 |
| 影片群聚的設計效應（每片約 4 件，ICC 0.1–0.3） | 上面的數字再乘 1.3–1.9 | |
| 微調 TimeSformer、做 4 通道早期融合 | 論文規模約 1,000 個片段的量級（見下），而且影片數要多到 5-fold 群組 CV 每折都有約 10 支影片，也就是 ≥50 支 | 162 個候選、7 支影片 |

論文規模是推算的。88% 準確率、FP 64 時，若 FP≈FN，錯誤數約 128，總片段數約 128 / 0.12 ≈ 1,070。這是依假設推出的量級，不是論文寫的數字。

現有資料的不確定性：

- 規則 in-sample 的事件召回 11/15，Wilson 95% 區間 [0.48, 0.89]；人工精確率 15/28，區間 [0.36, 0.71]。
- 最好的分類器（TimeSformer-RGB）AUROC 0.739，以影片為單位 bootstrap 的 95% 區間 [0.54, 0.92]；AP 0.722，區間 [0.38, 0.96]，下界碰到隨機基準 0.395。
- 學習曲線（TimeSformer-RGB 線性頭，用 k 支影片訓練）：k=2 時 AP 0.56、k=3 0.63、k=4 0.66、k=5 0.68、k=6 0.72。還在上升，沒有飽和，但每增加一支影片的幅度和雜訊差不多大。

**結論：資料要多 5–20 倍才量得出差異，要多約 50 倍才談得上照論文的做法訓練。** 所以這裡不微調任何預訓練模型，只做下面幾種做得到的驗證。

## 5. 實際跑到哪裡：leave-one-video-out

折法是論文紀律（依影片分組）最嚴格的版本：**每折留一支影片**，共 7 折。每折只用另外 6 支影片訓練；L2 強度 C 和決策門檻也只在這 6 支裡用內層 leave-one-video-out 選出（`lovo.py`）。另外寫了一支臨時檢查：在 175 次訓練（外層加內層）裡，被預測的那支影片從沒出現在訓練資料中。

模型：

- `hand`：邏輯迴歸，用 10 個手工幾何特徵（接觸時長、行人在斑馬線上的比例、沿斑馬線方向的車人間距、腳的深度、車種等），當下界。
- `r2p1d_*`／`timesformer_*`：凍結的 Kinetics-400 R(2+1)D-18 與 TimeSformer-base 取特徵，接線性頭。`rgb` 是原圖，`dim` 是 V1 壓暗背景；`+extra` 表示訓練時加入一般車流負樣本與時間平移窗。TimeSformer 只吃 8 frame，從 9 frame 均勻取 8。
- `cnn_rgb`／`cnn_fused`：約 0.2M 參數的 3D CNN 從頭訓練，112×112，3 通道對 4 通道早期融合，門檻 0.5，3 個 seed 平均（`cnn.py`）。

同基準比較（`score.py`，`eval_rule_events.py --events-root`，容差 1.25 秒，7 支影片）。「@28」是全部影片分數最高的 28 個候選（每台車只取一個），也就是給規則同樣的事件數。A 判定只能對到已判定的事件，沒判過的一律不算對，所以分類器這一欄是下界。

| 模型 | 樣本 | 事件 | manual | confirmed | 事件召回 | label precision | A | 未判 | AUROC | AP |
|---|---|---|---|---|---|---|---|---|---|---|
| **規則（4201f17）** | **in-sample** | 28 | 5/9 | 7/8 | **11/15** | **0.429** | **15** | 0 | – | – |
| 產生器全部候選 | in-sample | 162 | 8/9 | 8/8 | 14/15 | 0.099 | 36 | 98 | – | – |
| hand（折內門檻） | LOVO | 30 | 3/9 | 6/8 | 8/15 | 0.300 | 12 | 8 | 0.562 | 0.454 |
| r2p1d_rgb @28 | LOVO | 28 | 2/9 | 3/8 | 4/15 | 0.179 | 8 | 13 | 0.666 | 0.605 |
| r2p1d_dim @28 | LOVO | 28 | 2/9 | 3/8 | 4/15 | 0.179 | 10 | 10 | 0.518 | 0.449 |
| timesformer_rgb（折內門檻） | LOVO | 21 | 3/9 | 5/8 | 7/15 | 0.381 | 11 | 6 | **0.739** | **0.722** |
| timesformer_rgb @28 | LOVO | 28 | 3/9 | 5/8 | 7/15 | 0.286 | 12 | 11 | 0.739 | 0.722 |
| timesformer_rgb+extra @28 | LOVO | 28 | 5/9 | 5/8 | 9/15 | 0.357 | 14 | 9 | 0.675 | 0.601 |
| timesformer_dim @28 | LOVO | 28 | 4/9 | 6/8 | 9/15 | 0.357 | 14 | 9 | 0.697 | 0.593 |
| cnn_rgb @28 | LOVO | 28 | 3/9 | 6/8 | 8/15 | 0.321 | 10 | 12 | 0.503 | 0.488 |
| cnn_fused @28 | LOVO | 28 | 3/9 | 6/8 | 8/15 | 0.321 | 12 | 11 | 0.664 | 0.666 |

隨機基準：候選中正例比例 0.395（AP 的隨機值）、AUROC 0.5。完整表格（含每種模型的折內門檻版本）可用 `score.py` 重跑。

先驗通道效果，以影片為單位做配對 bootstrap（AUROC 差，95% 區間）：

- `cnn_fused − cnn_rgb` = +0.161 [−0.028, +0.277]：4 通道對從頭訓練的小 CNN 有幫助的跡象，但區間跨過 0。
- `timesformer_dim − timesformer_rgb` = −0.042 [−0.304, +0.186]：看不出差別。
- `r2p1d_dim − r2p1d_rgb` = −0.148 [−0.377, −0.019]：V1 壓暗反而傷害凍結的 R(2+1)D。
- `timesformer_rgb − hand` = +0.177 [+0.009, +0.459]：預訓練的時空特徵確實比手工幾何特徵好。

讀法：

- 同樣報 28 件，最好的分類器版本是事件召回 9/15、A 14 件，規則是 11/15、A 15 件。規則的數字是 in-sample（它的門檻就是在這 7 支影片上調出來的），分類器是 out-of-fold，所以這個比較對規則有利。但即使把這點算進去，分類器也沒有贏。
- 分類器挑出的事件裡，有 9–13 件沒人判過，A 欄是下界。要知道它真正的精確率，得先把這些事件拿去人工判定。
- 論文的結論是「4 通道早期融合對 Transformer 有效、對 3D CNN 無效」。這裡兩個方向都無法驗證：TimeSformer 沒有微調，只能用 V1 壓暗，看不出差別；小 CNN 是 4 通道略好，但區間跨過 0。

## 重跑

```powershell
# Python 環境：需要 torch、torchvision、opencv、scikit-learn、transformers
$env:PYTHONIOENCODING = "utf-8"
python experiments\candidate_clip_classifier\candidates.py --out .amp\in\candidates          # 約 75 秒
python experiments\candidate_clip_classifier\dataset.py --candidates .amp\in\candidates --out .amp\in\data
python experiments\candidate_clip_classifier\embed.py --data .amp\in\data                    # GPU，約 3 分鐘
python experiments\candidate_clip_classifier\lovo.py --data .amp\in\data --out .amp\in\oof
python experiments\candidate_clip_classifier\lovo.py --data .amp\in\data --out .amp\in\oof --extra --models r2p1d_rgb,r2p1d_dim,timesformer_rgb,timesformer_dim
python experiments\candidate_clip_classifier\cnn.py --data .amp\in\data --out .amp\in\oof     # GPU，約 10 分鐘
python experiments\candidate_clip_classifier\score.py --data .amp\in\data --oof .amp\in\oof --out .amp\in\scored
python experiments\candidate_clip_classifier\sizing.py --data .amp\in\data --oof .amp\in\oof
```

`.amp\in\data` 存的是影片畫面，不能進版控。
