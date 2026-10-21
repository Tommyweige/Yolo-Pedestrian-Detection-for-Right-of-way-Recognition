# 在新的可見 PowerShell 視窗跑 dispatch.py，同時寫 log 給調度者監看。
# 用法：powershell -File run-visible.ps1 -TaskId <id> -DispatchArgs "<dispatch.py run 之後的其餘參數>"
# 任務結束後視窗 5 秒自動關閉（使用者要求；並行上限 5 個視窗）。
# 調度者監看 <repo 旁>\<repo>-logs\dispatch-<id>.log，看到 "EXIT=" 即結束（DISPATCH REPORT 在它前面）。
param([Parameter(Mandatory)][string]$TaskId, [Parameter(Mandatory)][string]$DispatchArgs)
$repo = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$logDir = Join-Path (Split-Path $repo -Parent) ((Split-Path $repo -Leaf) + "-logs")
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "dispatch-$TaskId.log"
$runner = Join-Path $logDir "dispatch-$TaskId.ps1"
@"
`$host.UI.RawUI.WindowTitle = 'DeepSeek worker: $TaskId'
`$env:PYTHONUTF8 = '1'; `$env:PYTHONUNBUFFERED = '1'
Set-Location '$repo'
"=== dispatch $TaskId started `$(Get-Date -Format s)" | Out-File -FilePath '$log' -Encoding utf8
python .claude/skills/dispatch-worker/dispatch.py run --id $TaskId $DispatchArgs 2>&1 | ForEach-Object { `$l = "`$_"; `$l | Out-File -FilePath '$log' -Append -Encoding utf8; `$l }
"EXIT=`$LASTEXITCODE" | Out-File -FilePath '$log' -Append -Encoding utf8
Write-Host "EXIT=`$LASTEXITCODE（5 秒後自動關閉視窗）"
Start-Sleep -Seconds 5
"@ | Out-File -FilePath $runner -Encoding utf8
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-File", $runner
Write-Output "launched; log: $log"
