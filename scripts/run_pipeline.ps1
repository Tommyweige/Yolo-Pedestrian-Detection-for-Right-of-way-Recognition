# 批次跑整個流程（斑馬線分割 + 追蹤 -> 讓行人規則 -> 疊圖 + 證據片段 -> 候選清單）。
# 步驟本身寫在 scripts\run_pipeline.py；這裡只設好兩個 conda 環境的 Python，再把參數轉給它，
# 所以兩個入口的行為一定相同。輸出與記錄檔見 README。
param(
    [string]$Folder = "datasets\來源影片",
    [string]$Run = "eval-run2",
    [string]$Weights,              # 以下三個沒給就不轉,預設值只寫在 yield_pipeline.py(GUI 與 run_pipeline.py 共用)
    [double]$Conf,
    [int]$ImgSz,
    [string]$Geometry = "image",   # image = 像素判斷(舊); crosswalk = 斑馬線座標判斷
    [ValidateSet("review", "strict")]
    [string]$Mode = "strict",      # strict = 只報上了斑馬線的行人(預設); review = 路緣等待者也列為候選,候選多很多
    [string[]]$Only = @(),         # 只處理檔名含這些字串的影片
    [switch]$ReuseTracks,          # 追蹤結果已存在就不重跑偵測,只重跑規則與疊圖
    [switch]$Serial,               # 一次只跑一步(顯存不夠同時放分割與追蹤時用)
    [switch]$DryRun                # 只印出要執行的指令
)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:TRAFFIC_MODERN_PYTHON = "$env:USERPROFILE\anaconda3\envs\traffic-modern\python.exe"
$env:TRAFFIC_LABELING_PYTHON = "$env:USERPROFILE\anaconda3\envs\traffic-labeling\python.exe"
$arguments = @("scripts\run_pipeline.py", "--folder", $Folder, "--run", $Run, "--geometry", $Geometry, "--mode", $Mode)
if ($PSBoundParameters.ContainsKey("Weights")) { $arguments += @("--weights", $Weights) }
if ($PSBoundParameters.ContainsKey("Conf")) { $arguments += @("--conf", $Conf.ToString([Globalization.CultureInfo]::InvariantCulture)) }
if ($PSBoundParameters.ContainsKey("ImgSz")) { $arguments += @("--imgsz", $ImgSz) }
if ($Only.Count -gt 0) { $arguments += @("--only") + $Only }
if ($ReuseTracks) { $arguments += "--reuse-tracks" }
if ($Serial) { $arguments += "--serial" }
if ($DryRun) { $arguments += "--dry-run" }
& $env:TRAFFIC_MODERN_PYTHON @arguments
exit $LASTEXITCODE
