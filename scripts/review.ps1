# 開啟標註工具:整個影片資料夾 + 預測結果,左側清單切換影片
Set-Location (Split-Path $PSScriptRoot -Parent)
Start-Process ".\rust-ui\target\release\event_labeler.exe" -ArgumentList "--folder","datasets\來源影片","--events-dir","datasets\eval-run1\events"
