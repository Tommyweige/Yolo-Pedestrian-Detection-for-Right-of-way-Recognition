---
name: dispatch-worker
description: /implement-spec 的所有 ticket（含 code-review 修正），以及主對話要跑的實驗（生成、渲染檢查、試做腳本）：把一張寫清楚的任務派給 DeepSeek v4.1 flash（先用免費的 opencode/deepseek-v4.1-flash-free，額度用完自動改用 OpenCode Go 的 opencode-go/deepseek-v4.1-flash；推理強度預設 high；找根因的票與重做時加 `--variant max`）實作，在獨立 git worktree 上從 integration branch 開分支、TDD 實作，並回傳範圍檢查與測試結果。同一張 ticket 重做兩次仍失敗才改派 sonnet-implementer。
---

# dispatch-worker（/implement-spec 的實作路徑）

取代 /implement-spec 步驟 4 與步驟 7 修正的「派一個 implementer 子代理」。派工規則見專案 CLAUDE.md 的
「/implement-spec 分流規則」。

opencode 固定用 `@opencode/cli` 2.1.1，另裝在 `%LOCALAPPDATA%\opencode-dispatch`（不用全域安裝），以
`opencode run --standalone --auto -m <模型>#<variant>` 執行（`--variant` 預設 `high`，可選 `max`）。1.x 用不了 Go 模型，且免費額度用完時會卡在
`> build` 不回錯誤；2.1.1 會立刻回 `Rate limit exceeded`。腳本找不到或版本不符會直接報錯並印出安裝指令：
`npm i --prefix "%LOCALAPPDATA%\opencode-dispatch" @opencode/cli@2.1.1`。

`dispatch.py` 只啟動 `opencode run`，**不讀、不轉送任何 Claude 憑證**（子程序環境清掉
`ANTHROPIC_*` / `CLAUDE*`，並關閉 opencode 的 Claude Code 相容模式，worker 看不到 CLAUDE.md、
`~/.claude` 規則與技能，所以 TDD 做法由腳本直接寫進任務）。模型依 `MODELS` 順序：每次派工先試免費模型，
它不可用且還沒動到檔案時，在同一個 worktree 改用 Go 模型；報告的 `model`／`models_tried` 記錄實際用了哪個。

## 派工前：ticket 檔

Markdown，繁中即可，**只放這張 ticket 需要的最少內容**（目標、介面/簽名、範例輸入輸出、驗收條件、
不能動的地方、必要的相關片段）；spec、研究筆記、相關 commit 給「位置」就好，不要整包貼。
worker 不該做任何設計決定——有設計空間時由調度者在 ticket 檔裡寫明決定，檔案太多就拆成幾張。

## 執行（前景執行，worker 輸出即時顯示並寫 log）

```
python .claude/skills/dispatch-worker/dispatch.py run \
  --id T1 --ticket path/to/ticket.md --integration <integration branch> \
  --allow src/a.py --allow tests/test_a.py \
  --test "<測試指令>" [--repo DIR] [--timeout 540] [--reset]
```

**預設用可見視窗跑**（使用者要能看著 worker 做事）：

```
powershell -ExecutionPolicy Bypass -File .claude/skills/dispatch-worker/run-visible.ps1 \
  -TaskId T1 -DispatchArgs "--ticket path/to/ticket.md --integration <branch> --allow src/a.py --test 'cmd /c path\to\test.cmd' --timeout 480"
```

它開一個標題為「DeepSeek worker: T1」的新 PowerShell 視窗跑上面的 `dispatch.py run`，同時寫 `<repo 旁>\<repo>-logs\dispatch-T1.log`。調度者用 Monitor 監看這個 log，看到 `EXIT=` 即結束（`DISPATCH REPORT` 在它前面）。測試指令含引號或空白路徑時，寫成一個 `.cmd` 檔，`--test` 只傳 `cmd /c <檔案>`，避免 PowerShell 5.1 吃掉內層引號。`--test` 只能是單一指令（不經過 shell，`&&`、管線、重新導向都不支援）；要多個步驟就寫成 `.cmd`（Windows）或 `.sh` 檔。

腳本做的事（與 implement-spec 的 implementer 規則一致）：
1. `git worktree add -b worker/<id> <repo 旁>/<repo>-workers/<id> <integration branch tip>`
2. 以 opencode 在該 worktree 執行，任務含 TDD（紅→綠→整理）、允許修改清單、測試指令。
3. 以 integration tip 為基準檢查改動範圍（含未追蹤檔、worker 自行 commit），在 worker 分支上 commit。
4. **回報前**把 integration branch 最新 tip 合併進 worker 分支（衝突 → `MERGE_CONFLICT`）。
5. 自己執行測試指令（不採信 worker 自述）並輸出 `DISPATCH REPORT` JSON。

退回重做：同一個 `--id` 加 `--reset`（丟棄舊 worktree/分支重來），ticket 末尾補上失敗原因。
收尾（implement-spec 步驟 9）：`python .../dispatch.py cleanup --id T1`。

## status

| status | 意義 | 調度者動作 |
|---|---|---|
| `PASS` | 範圍內、測試通過 | 仍須自己看 diff + 跑測試，再交 merger |
| `TEST_FAILED` / `OUT_OF_SCOPE` / `NO_CHANGES` / `TIMEOUT` / `MERGE_CONFLICT` | 失敗 | 退回重做（計入重做次數） |
| `MAIN_TAMPERED` | 主 checkout 在派工期間被改 | 停下檢查並告知使用者 |
| `MODEL_UNAVAILABLE` | 免費與 Go 模型都被拒/額度用完（結束碼 3） | **暫停剩餘 ticket、告訴使用者，不要改派其他模型** |

## 已知限制

- 隔離是「軟性防護」：opencode 的 `external_directory: deny` 擋 read/bash/edit 的外部路徑（已實測），
  不是作業系統層級沙箱；另外腳本事後檢查主 checkout 是否被改。
- opencode 以環境變數 `PWD` 決定專案目錄；腳本已覆寫 `PWD` 並以 worktree 為 cwd。從別的目錄直接手動執行
  `opencode run` 而不設 `PWD`，worker 可能改到錯的 checkout。
- Go 模型需要 OpenCode workspace 的 Privacy 設為 Global region，否則回 `requires Global regions`。
- worker 的程式碼與 ticket 內容會送到模型供應商——**不要把機密放進 ticket 或 worktree**。
- 範圍檢查依賴 repo 的 `.gitignore`（建置/快取產物須已被忽略，否則會被判為範圍外）。
