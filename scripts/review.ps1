# 開啟標註工具:整個影片資料夾,左側清單切換影片;預測結果與疊圖影片自動取 datasets\*\events、overlay 裡最新的
Set-Location (Split-Path $PSScriptRoot -Parent)
Start-Process ".\rust-ui\target\release\event_labeler.exe" -ArgumentList "--folder","datasets\來源影片"
