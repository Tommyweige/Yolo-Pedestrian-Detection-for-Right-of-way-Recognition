# 批次跑整個流程:追蹤 -> 讓行人規則 -> 疊圖影片 -> 證據片段,最後彙整候選清單。每支影片一個資料夾 datasets\<Run>\<影片檔名>\
# 斑馬線遮罩 seg\masks.npz 已存在就沿用,不存在就先做分割(需要 traffic-labeling 環境與 Mask2Former 模型)。
# 輸出:datasets\<Run>\events\<影片檔名>.jsonl 與 datasets\<Run>\overlay\<影片檔名>.mp4(標註工具會自動載入),
# datasets\<Run>\evidence\<影片檔名>\(每個候選的片段 .mp4 與原始截圖 .png),以及 datasets\<Run>\candidates.csv。
param(
    [string]$Folder = "datasets\來源影片",
    [string]$Run = "eval-run2",
    [string]$Weights = "runtime\smoke-assets\yolo26x.pt",
    [double]$Conf = 0.15,
    [int]$ImgSz = 1280,
    [string]$Geometry = "image",   # image = 像素判斷(舊); crosswalk = 斑馬線座標判斷
    [ValidateSet("review", "strict")]
    [string]$Mode = "strict",      # strict = 只報上了斑馬線的行人(預設); review = 路緣等待者也列為候選,候選多很多
    [string[]]$Only = @(),         # 只處理檔名含這些字串的影片
    [switch]$ReuseTracks  # 追蹤結果已存在就不重跑偵測,只重跑規則與疊圖
)
Set-Location (Split-Path $PSScriptRoot -Parent)
$modern = "$env:USERPROFILE\anaconda3\envs\traffic-modern\python.exe"
$labeling = "$env:USERPROFILE\anaconda3\envs\traffic-labeling\python.exe"
$log = "datasets\$Run.log"
Start-Transcript -Path $log -Append | Out-Null
$videos = Get-ChildItem $Folder -File | Where-Object { $_.Extension -in ".mp4", ".avi", ".mkv", ".mov" } | Where-Object { $name = $_.Name; ($Only.Count -eq 0) -or (@($Only | Where-Object { $name -like "*$_*" }).Count -gt 0) } | Sort-Object Name
$n = 0
# @(...) 讓單一元素仍是陣列;PowerShell 5.1 會把 if 區塊的單一元素陣列拆成字串
$waiting = @(if ($Mode -eq "review") { "--include-waiting" })
function Invoke-Step([string]$Label, [scriptblock]$Body) {
    "  $Label"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Body
    if ($LASTEXITCODE -ne 0) {
        "失敗:$Label(結束碼 $LASTEXITCODE)"
        Stop-Transcript | Out-Null
        exit 1
    }
    "        $([int]$sw.Elapsed.TotalSeconds) 秒"
}
foreach ($video in $videos) {
    $n++
    $stem = $video.BaseName
    $dir = "datasets\$Run\$stem"
    $events = "datasets\$Run\events\$stem.jsonl"
    "=== [$n/$($videos.Count)] $($video.Name)  $(Get-Date -Format HH:mm:ss)"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    New-Item -ItemType Directory -Force "$dir\seg", "$dir\tracks", "datasets\$Run\events", "datasets\$Run\overlay", "datasets\$Run\evidence\$stem" | Out-Null
    if (-not (Test-Path "$dir\seg\masks.npz")) {
        Invoke-Step "[1/5] 斑馬線分割 (GPU)" { & $labeling scripts\segment_crosswalk.py --video $video.FullName --model runtime\models\mask2former-mapillary --stride 3 --output "$dir\seg" }
    } else { "  [1/5] 沿用既有斑馬線遮罩" }
    if ($ReuseTracks -and (Test-Path "$dir\tracks\raw_tracks.json")) {
        Invoke-Step "[2/5] 沿用既有追蹤結果(只重做後照鏡過濾)" { & $modern scripts\track_video.py --video $video.FullName --refilter "$dir\tracks\raw_tracks.json" }
    } else {
        Invoke-Step "[2/5] 偵測與追蹤 ($Weights, conf $Conf, GPU)" { & $modern scripts\track_video.py --video $video.FullName --weights $Weights --conf $Conf --imgsz $ImgSz --output "$dir\tracks" }
    }
    Invoke-Step "[3/5] 讓行人規則 ($Mode)" { & $modern yield_image_rule.py --video $video.FullName --masks "$dir\seg\masks.npz" --tracks "$dir\tracks\raw_tracks.json" --geometry $Geometry --output $events $waiting }
    Invoke-Step "[4/5] 疊圖影片" { & $modern scripts\render_overlay.py --video $video.FullName --masks "$dir\seg\masks.npz" --tracks "$dir\tracks\raw_tracks.json" --events $events --labels "labels\$stem.jsonl" --output "datasets\$Run\overlay\$stem.mp4" }
    Invoke-Step "[5/5] 證據片段與截圖" { & $modern scripts\export_evidence.py --video $video.FullName --events $events --output "datasets\$Run\evidence\$stem" --pad-seconds 2 }
    $speed = & $modern -c "import sys; sys.path.insert(0, 'scripts'); import run_pipeline as r; print(r.speed_text(r._video_seconds(sys.argv[1]), float(sys.argv[2])))" $video.FullName $watch.Elapsed.TotalSeconds
    "  $($video.Name):$speed"
}
Invoke-Step "候選清單" { & $modern scripts\list_candidates.py --run-dir "datasets\$Run" }
"全部完成 $(Get-Date -Format HH:mm:ss)。候選清單:datasets\$Run\candidates.csv  證據:datasets\$Run\evidence  事件:datasets\$Run\events  疊圖:datasets\$Run\overlay"
Stop-Transcript | Out-Null
