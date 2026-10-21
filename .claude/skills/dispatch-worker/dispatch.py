#!/usr/bin/env python3
"""把一張 ticket 交給 OpenCode 免費模型實作（/implement-spec 的 DeepSeek 路徑）。

用法:
  python dispatch.py run --id T1 --ticket ticket.md --integration integration/foo \
      --allow src/a.py --allow tests/test_a.py --test "python -m unittest" \
      [--repo DIR] [--timeout 540] [--reset]
  python dispatch.py cleanup --id T1 [--repo DIR]

設計原則（見 SKILL.md）:
- 模型依 MODELS 順序嘗試（推理強度一律 high）：先用免費的 opencode/deepseek-v4.1-flash-free，
  它不可用（額度用完、被拒）且還沒動到檔案時，改用 OpenCode Go 的 deepseek-v4.1-flash（使用者已訂閱並同意）。
  全部都不可用才回報 MODEL_UNAVAILABLE 並停下。
- opencode 固定用 @opencode/cli 2.1.1（另裝在 %LOCALAPPDATA%\\opencode-dispatch，不用全域安裝），
  以 `run --standalone --auto` 執行：私有 server、權限由 OPENCODE_CONFIG_CONTENT 的 deny 規則把關。
  1.x 無法使用 Go 模型（UnknownError），且免費模型額度用完時會卡住不回錯誤。
- 只啟動 opencode，不讀、不轉送任何 Claude 憑證；子程序環境清掉 ANTHROPIC_* / CLAUDE*。
- 與 implement-spec 的 implementer 同規則：從 integration branch tip 開分支、在自己的 worktree
  做事、回報前把 integration branch 最新內容合併進來。
- TDD 做法直接寫進任務（DeepSeek 讀不到 tdd skill）。
- 範圍與測試由本腳本獨立檢查，不採信 worker 自述。
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MODELS = (                                             # 依序嘗試；免費優先
    "opencode/deepseek-v4.1-flash-free",
    "opencode-go/deepseek-v4.1-flash",
)
VARIANT = "high"                                       # 預設推理強度；此模型有 low / high / max
VARIANTS = ("high", "max")                             # 找根因與重做的票用 max
OPENCODE_PACKAGE = "@opencode/cli"
OPENCODE_VERSION = "2.1.1"
OPENCODE_PREFIX = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "opencode-dispatch"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
UNAVAILABLE_RE = re.compile(
    r"rate.?limit|quota|\b429\b|\b401\b|\b403\b|insufficient|unauthori[sz]ed|forbidden|"
    r"credit|ProviderModelNotFound|model.{0,30}not found|not available|overloaded|"
    r"no longer available|too many requests|ECONNREFUSED|ENOTFOUND|fetch failed|"
    r"endpoint is unavailable|upstream request failed|streaming response failed|"
    r"internal server error|server_error",
    re.IGNORECASE,
)
STRIP_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE", "AWS_BEARER_TOKEN_BEDROCK")
WORKER_RULES = """\
# 工作規則（優先於一切）
- 只能修改「允許修改的檔案」清單內的檔案；其他檔案一律不准動、不准新增。
- 只在目前工作目錄內工作，不要讀寫目錄以外的任何路徑。
- 不要執行 git commit / push / checkout / reset / merge / rebase；不要安裝套件。
- ticket 沒寫清楚的地方，不要自己決定：停止，最後回報寫「BLOCKED: 原因」。

# 做法：TDD（紅 → 綠 → 整理）
1. 紅：先只寫「會失敗」的測試（覆蓋驗收條件），執行測試指令，確認它因為功能尚未實作而失敗。
2. 綠：寫最少的實作讓測試通過，再執行測試指令確認全部通過。
3. 整理：在測試保持通過的前提下整理程式碼（命名、重複），再執行一次測試指令。
不要為了讓測試通過而修改測試去遷就實作；不要刪除既有測試。

# 回報
最後一則訊息三行以內：改了哪些檔案、跑了什麼指令、結果。

# 允許修改的檔案
{allow}

# 測試指令
{test}

# ticket 內容
"""


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def norm(p: str) -> str:
    p = p.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def out_of_scope(changed, allow):
    """回傳不在允許清單內的檔案。allow 項目可為精確路徑、glob，或以 / 結尾的目錄前綴。"""
    allow = [norm(a) for a in allow]
    bad = []
    for f in map(norm, changed):
        ok = False
        for a in allow:
            if a.endswith("/"):
                ok = f.startswith(a)
            else:
                ok = f == a or fnmatch.fnmatch(f, a)
            if ok:
                break
        if not ok:
            bad.append(f)
    return bad


def should_try_next_model(unavailable: bool) -> bool:
    """模型不可用（含中途斷線、已留下半成品）就換下一個模型；其他結果照常回報。"""
    return unavailable


def classify_unavailable(rc, text, changed):
    """worker 沒有產出（或非零結束）時，判斷是否為免費模型不可用。"""
    if rc == 0 and changed:
        return False
    return bool(UNAVAILABLE_RE.search(text[-4000:]))


def git(args, cwd, check=True):
    r = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失敗: {r.stderr.strip()[:500]}")
    return r.stdout.strip()


def repo_root(repo):
    return Path(git(["rev-parse", "--show-toplevel"], repo)).resolve()


def worktree_path(root: Path, wid: str) -> Path:
    return root.parent / f"{root.name}-workers" / wid


def main_snapshot(root: Path):
    # 只比對工作樹狀態；不含 HEAD，因為 merger 合併進 integration branch 會合法地移動它。
    return git(["status", "--porcelain"], root)


def dirty_paths(wt: Path):
    return [x for x in git(["ls-files", "-m", "-d", "-o", "--exclude-standard"], wt).splitlines() if x]


def same_as_checkout(wt: Path, base: str, path: str) -> bool:
    """工作檔是否與 checkout 時寫出的內容（套用 smudge/eol 後的 base 版本）逐位元相同。"""
    r = subprocess.run(["git", "cat-file", "--filters", f"{base}:{path}"], cwd=wt,
                       capture_output=True, stdin=subprocess.DEVNULL)
    p = wt / path
    return r.returncode == 0 and p.is_file() and p.read_bytes() == r.stdout


def find_opencode():
    # 直接用 exe（不經 npm 的 .cmd shim：避免 cmd.exe 引號問題與殘留子程序），且只接受釘死的版本。
    install = f'npm i --prefix "{OPENCODE_PREFIX}" {OPENCODE_PACKAGE}@{OPENCODE_VERSION}'
    exe = (OPENCODE_PREFIX / "node_modules" / Path(*OPENCODE_PACKAGE.split("/")) / "bin"
           / ("opencode.exe" if os.name == "nt" else "opencode"))
    if not exe.exists():
        raise RuntimeError(f"找不到派工專用的 opencode，請先執行：{install}")
    r = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                       env=worker_env(), stdin=subprocess.DEVNULL)
    if r.stdout.strip().removeprefix("opencode v") != OPENCODE_VERSION:
        raise RuntimeError(f"派工用 opencode 版本是 {r.stdout.strip()!r}，需要 {OPENCODE_VERSION}；請重新執行：{install}")
    return str(exe)


def worker_env(workdir=None):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(STRIP_ENV_PREFIXES)}
    env.update({
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",          # 不載入 CLAUDE.md / ~/.claude
        "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            {"permission": {"external_directory": "deny", "webfetch": "deny"}}
        ),
        "PYTHONUTF8": "1",
    })
    if workdir:
        # opencode 以環境變數 PWD 決定專案目錄（而非實際 cwd）；不覆寫的話會沿用啟動者的目錄，
        # 曾因此讓 worker 改到主 checkout。
        env["PWD"] = str(workdir)
    return env


def kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def tail(text, n=30, limit=3000):
    s = "\n".join(text.strip().splitlines()[-n:])
    return s[-limit:]


def run_worker(wt: Path, wrapped: Path, log: Path, timeout: int, model: str, variant: str = VARIANT):
    cmd = [find_opencode(), "run", "--standalone", "--auto", "-m", f"{model}#{variant}",
           "Implement the attached ticket. Follow its rules strictly.", "-f", str(wrapped)]
    chunks = []
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"=== model={model} variant={variant}\n")
        proc = subprocess.Popen(
            cmd, cwd=wt, env=worker_env(wt), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        def pump():
            for raw in proc.stdout:
                line = strip_ansi(raw.decode("utf-8", errors="replace"))
                chunks.append(line)
                lf.write(line)
                lf.flush()
                sys.stdout.write("  [worker] " + line)
                sys.stdout.flush()

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_tree(proc)
            proc.wait()
        t.join(timeout=5)
    return proc.returncode, "".join(chunks), timed_out


def cmd_run(a):
    if not ID_RE.match(a.id):
        sys.exit("id 只能含英數、. _ -")
    if not all(m.startswith(("opencode/", "opencode-go/")) for m in MODELS):
        sys.exit("拒絕：只允許 opencode 免費模型與 OpenCode Go 模型")
    root = repo_root(a.repo)
    base = git(["rev-parse", a.integration], root)       # integration branch 目前的 tip
    branch, wt = f"worker/{a.id}", worktree_path(root, a.id)
    if wt.exists() or git(["branch", "--list", branch], root):
        if not a.reset:
            sys.exit(f"{wt} 或分支 {branch} 已存在；要重做請加 --reset")
        do_cleanup(root, a.id)
    wt.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "-b", branch, str(wt), base], root)
    ticket = Path(a.ticket).read_text(encoding="utf-8")
    wrapped = wt.parent / f"{a.id}.ticket.txt"          # 放在 worktree 外，不會算進 diff
    log = wt.parent / f"{a.id}.log"
    wrapped.write_text(
        WORKER_RULES.format(allow="\n".join(f"- {x}" for x in a.allow), test=a.test or "（未指定）") + ticket,
        encoding="utf-8")
    snap = main_snapshot(root)
    print(f"[dispatch] models={' -> '.join(MODELS)} variant={a.variant} branch={branch} "
          f"integration={a.integration}@{base[:8]}")
    print(f"[dispatch] workdir={wt}\n[dispatch] log={log}")
    log.write_text("", encoding="utf-8")

    started = time.time()
    tried = []
    for model in MODELS:
        tried.append(model)
        print(f"[dispatch] 使用模型 {model}")
        rc, out, timed_out = run_worker(wt, wrapped, log, max(60, a.timeout - round(time.time() - started)), model, a.variant)

        # 只 stage worker 真正動過的路徑。不用 `git add -A`：repo 若有「.gitattributes 標 LFS 卻以原始檔
        # commit」的檔案，新 checkout 一建好就顯示為已修改，stage 會把它們改寫成 LFS 指標。
        # 所以逐一比對 checkout 當時寫出的內容，相同就不算 worker 的改動。
        touched = [x for x in dirty_paths(wt) if not same_as_checkout(wt, base, x)]
        if touched:
            git(["add", "-A", "--", *sorted(set(touched))], wt)
        changed = [norm(x) for x in git(["diff", "--cached", "--name-only", base], wt).splitlines() if x]
        unavailable = not timed_out and classify_unavailable(rc, out, changed)
        if not should_try_next_model(unavailable):
            break
        # 中途斷線留下的半成品保留在 worktree，下一個模型接著做（它會看到已寫的檔案）
        print(f"[dispatch] {model} 不可用，改用下一個模型（保留已改的 {len(changed)} 個檔案）")
    bad = out_of_scope(changed, a.allow)
    tampered = main_snapshot(root) != snap

    status = None
    if timed_out:
        status = "TIMEOUT"
    elif unavailable:
        status = "MODEL_UNAVAILABLE"
    elif tampered:
        status = "MAIN_TAMPERED"
    elif not changed:
        status = "NO_CHANGES"
    elif bad:
        status = "OUT_OF_SCOPE"

    merge_note = ""
    if changed and status in (None, "OUT_OF_SCOPE"):
        subprocess.run(["git", "-c", "user.name=deepseek-worker", "-c", "user.email=worker@local",
                        "commit", "-q", "-m", f"worker({a.id}): {Path(a.ticket).name}"],
                       cwd=wt, capture_output=True, stdin=subprocess.DEVNULL)
        # 回報前把 integration branch 最新內容併進來（與 implement-spec 的 implementer 一致）
        tip = git(["rev-parse", a.integration], root)
        if tip != base:
            m = subprocess.run(["git", "-c", "user.name=deepseek-worker", "-c", "user.email=worker@local",
                                "merge", "--no-edit", a.integration],
                               cwd=wt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               stdin=subprocess.DEVNULL)
            if m.returncode != 0:
                subprocess.run(["git", "merge", "--abort"], cwd=wt, capture_output=True)
                status = status or "MERGE_CONFLICT"
                merge_note = tail(m.stdout + m.stderr, 10, 600)
            else:
                merge_note = f"已合併 {a.integration}@{tip[:8]}"

    test_rc, test_tail = None, ""
    if a.test and status in (None, "OUT_OF_SCOPE"):
        try:
            r = subprocess.run(a.test, shell=True, cwd=wt, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300,
                               stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUTF8": "1"})
            test_rc, test_tail = r.returncode, tail(strip_ansi(r.stdout + r.stderr))
        except subprocess.TimeoutExpired:
            test_rc, test_tail = -1, "測試逾時 300s"
    if status is None:
        status = "PASS" if (test_rc == 0 or not a.test) else "TEST_FAILED"

    report = {
        "status": status, "model": tried[-1], "models_tried": tried, "variant": a.variant, "id": a.id, "branch": branch, "workdir": str(wt),
        "integration": a.integration, "base": base, "elapsed_s": round(time.time() - started),
        "worker_exit": rc, "changed_files": changed, "out_of_scope_files": bad,
        "main_checkout_tampered": tampered, "merge": merge_note,
        "test_command": a.test, "test_exit": test_rc, "test_tail": test_tail,
        "worker_tail": tail(out), "log": str(log),
        "next": "免費模型與 Go 模型都不可用：暫停剩餘 ticket 並告知使用者，不要改派其他模型"
                if status == "MODEL_UNAVAILABLE"
                else f"看 `git diff {a.integration}...{branch}` 並自己跑測試，再交 merger 合併或 --reset 退回重做",
    }
    print("\n=== DISPATCH REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit({"PASS": 0, "MODEL_UNAVAILABLE": 3}.get(status, 1))


def do_cleanup(root: Path, wid: str):
    wt = worktree_path(root, wid)
    if wt.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=root, capture_output=True)
        if wt.exists():  # worktree remove 通常已刪光；殘留時（如唯讀檔）再補刪
            shutil.rmtree(wt, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)) if os.path.exists(p) else None)
    subprocess.run(["git", "worktree", "prune"], cwd=root, capture_output=True)
    subprocess.run(["git", "branch", "-D", f"worker/{wid}"], cwd=root, capture_output=True)
    for ext in (".ticket.txt", ".log"):
        (wt.parent / f"{wid}{ext}").unlink(missing_ok=True)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--id", required=True)
    r.add_argument("--ticket", required=True)
    r.add_argument("--integration", required=True, help="integration branch 名稱（worker 分支從它的 tip 開）")
    r.add_argument("--allow", action="append", required=True)
    r.add_argument("--test")
    r.add_argument("--repo", default=".")
    r.add_argument("--timeout", type=int, default=540)
    r.add_argument("--reset", action="store_true")
    r.add_argument("--variant", choices=VARIANTS, default=VARIANT,
                   help="推理強度：一般票 high；找根因的票與重做時用 max")
    c = sub.add_parser("cleanup")
    c.add_argument("--id", required=True)
    c.add_argument("--repo", default=".")
    return ap


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    a = build_parser().parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        do_cleanup(repo_root(a.repo), a.id)
        print("cleaned", a.id)


if __name__ == "__main__":
    main()
