## Agent skills

### Issue tracker

GitHub Issues (`gh` CLI), repository inferred from `git remote -v`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### /implement-spec 分流規則

**只在使用者執行 `/implement-spec` 時適用**；其他時候照平常方式做。外掛（Matt Pocock skills）本身不改，規則都在這個 repo。

調度者（主 session）只讀 spec／tickets、判斷任務圖、派工、合併、驗收，**不自己寫實作程式碼**。取代 `/implement-spec` 步驟 4「每張 ticket 派一個 implementer」這一步；integration branch、每張 ticket 各自一個 worktree、merger 合併、全部完成後 `code-review`、最後清掉所有 worktree，照原流程。

**全部交給 DeepSeek（high）**：每張 ticket，以及步驟 7 code-review 的修正，都用 Skill `dispatch-worker`（`.claude/skills/dispatch-worker/`，模型 `opencode/deepseek-v4.1-flash-free`，推理強度預設 `high`）實作，不再依大小分流。**找根因的票**（任務是查出 bug 的原因）與**任何重做**（第 1、2 次退回後再派）加 `--variant max`；其他一律 high。DeepSeek 讀不到 `tdd` skill，TDD 做法（紅→綠→整理）已由腳本寫進任務。code-review 本身、merger 等不寫實作的子代理照原流程。

**派工單必備四項**：`/to-tickets` 產出的 ticket 不含檔案路徑，所以調度者讀過程式碼後，在派工單（`dispatch.py --ticket` 的檔案）寫出下列四項；不要求 tracker 上的 ticket 本身寫。

1. 要改的檔案寫得清楚，一到三個
2. 具體、可逐條驗證的驗收條件
3. 要跑的測試指令（本專案用 `python -m pytest <測試檔>`；測試檔在 repo 根目錄，`test_*.py`）
4. 哪些地方不能動（例如 `weights/`、`datasets/`、`labels/`、`runs/` 不進版控；`YOLOv8_DeepSORT_Object_Tracking/` 為第三方程式碼，除非票明說否則不碰）

worker 不做設計決定。ticket 需要設計決定時，由調度者在派工單裡寫明決定；超過三個檔案就拆成幾張依序派工的派工單（同一張 ticket，各自一個 `--id`）。

**分支規則**：worktree 以 integration branch tip 為基底、回報前把 integration branch 最新 tip 合併進自己的分支（腳本自動做）。

**派工前先列派工表**：每一輪派工前（含前沿推進後解鎖的 ticket），先輸出一張表給使用者：每張 ticket 的派工單數、各自要改的檔案數，以及調度者替它做的設計決定（沒有就寫「無」）。列完即開始派工，不等確認；使用者喊停再調整。

**派工內容**：只給 context pointers（spec、ticket、研究筆記、相關 commit 的位置）加上這張 ticket 需要的最少內容；不整包塞過去。

**驗收與重做**：任一 worker 完成後，**合併前**必須看 diff 並自己跑測試。改到 ticket 範圍外的檔案、或測試沒過 → 退回重做（附失敗原因）。DeepSeek 同一張 ticket 重做兩次仍失敗 → 改交 `.claude/agents/sonnet-implementer.md`（`model: sonnet`，只當這個備援）。Sonnet 仍失敗就停下向使用者回報，不自己接手寫實作。

**模型順序**：每次派工先用免費的 `opencode/deepseek-v4.1-flash-free`；它被拒或額度用完時，`dispatch.py` 自動改用使用者訂閱的 OpenCode Go `opencode-go/deepseek-v4.1-flash`（推理強度照該次派工的 `--variant`）。兩個都不可用（回 `MODEL_UNAVAILABLE`，結束碼 3）→ 把剩下的 ticket 全部先暫停並告訴使用者；不偷偷改派別的模型（包括 Sonnet）。已在跑的 Sonnet 備援不受影響。

**底線**：Claude 這邊只用官方 Claude Code，不用 claude-code-router、sub2api、LiteLLM 等會碰 Claude 憑證的代理工具；OpenCode Go 只用使用者已訂閱的 DeepSeek v4.1 flash 當免費模型的備援，不自行升級方案或換用其他 Go 模型；機密不放進 ticket 或 worker 的 worktree（內容會送到模型供應商）。**影片與資料集（`datasets/`、`labels/`、`runs/`）含私人素材，不放進 ticket，也不要求 worker 讀取。**

### 環境

編輯含中文的檔案時用 UTF-8 工具（Edit/Write 或 Python `encoding='utf-8'`），不用 Windows PowerShell 5.1 的 `Set-Content`／`Out-File`，否則中文會亂碼。
