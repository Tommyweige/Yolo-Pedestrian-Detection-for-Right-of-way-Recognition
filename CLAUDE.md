## Agent skills

### Issue tracker

GitHub Issues (`gh` CLI), repository inferred from `git remote -v`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### /implement-spec 分流規則

模型、派工方式、驗收與重做、底線：一律照全域 `~/.claude/CLAUDE.md` 的「三層分工與 /implement-spec 分流規則」。派工用全域的 `~/.claude/skills/dispatch-worker/dispatch.py`，不用本 repo `.claude/skills/dispatch-worker/` 的舊副本；DeepSeek 重做兩次仍失敗改由 Claude Opus 5.5 接手，不用 `.claude/agents/sonnet-implementer.md`。

本專案特有（寫派工單時照這裡）：
- 測試指令：`python -m pytest <測試檔>`；測試檔在 repo 根目錄，`test_*.py`。
- 不能動：`weights/`、`datasets/`、`labels/`、`runs/` 不進版控；`YOLOv8_DeepSORT_Object_Tracking/` 為第三方程式碼，除非票明說否則不碰。
- 影片與資料集（`datasets/`、`labels/`、`runs/`）含私人素材，不放進 ticket，也不要求 worker 讀取。

### 環境

編輯含中文的檔案時用 UTF-8 工具（Edit/Write 或 Python `encoding='utf-8'`），不用 Windows PowerShell 5.1 的 `Set-Content`／`Out-File`，否則中文會亂碼。
