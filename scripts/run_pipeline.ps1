# 批次跑整個流程:追蹤 -> 讓行人規則 -> 疊圖影片。每支影片一個資料夾 datasets\<Run>\<影片檔名>\
# 斑馬線遮罩 seg\masks.npz 已存在就沿用,不存在就先做分割(需要 traffic-labeling 環境與 Mask2Former 模型)。
# 輸出:datasets\<Run>\events\<影片檔名>.jsonl 與 datasets\<Run>\overlay\<影片檔名>.mp4,標註工具會自動載入。
param(
    [string]$Folder = "datasets\來源影片",
    [string]$Run = "eval-run2",
    [string]$Weights = "runtime\smoke-assets\yolo26x.pt",
    [double]$Conf = 0.15,
    [int]$ImgSz = 1280,
    [switch]$ReuseTracks  # 追蹤結果已存在就不重跑偵測,只重跑規則與疊圖
)
Set-Location (Split-Path $PSScriptRoot -Parent)
$modern = "$env:USERPROFILE\anaconda3\envs\traffic-modern\python.exe"
$labeling = "$env:USERPROFILE\anaconda3\envs\traffic-labeling\python.exe"
$log = "datasets\$Run.log"
Start-Transcript -Path $log -Append | Out-Null
$videos = Get-ChildItem $Folder -File | Where-Object { $_.Extension -in ".mp4", ".avi", ".mkv", ".mov" } | Sort-Object Name
$n = 0
foreach ($video in $videos) {
    $n++
    $stem = $video.BaseName
    $dir = "datasets\$Run\$stem"
    "=== [$n/$($videos.Count)] $($video.Name)  $(Get-Date -Format HH:mm:ss)"
    New-Item -ItemType Directory -Force "$dir\seg", "$dir\tracks", "datasets\$Run\events", "datasets\$Run\overlay" | Out-Null
    if (-not (Test-Path "$dir\seg\masks.npz")) {
        "  [1/4] 斑馬線分割 (GPU)"
        & $labeling scripts\segment_crosswalk.py --video $video.FullName --model runtime\models\mask2former-mapillary --stride 3 --output "$dir\seg"
    } else { "  [1/4] 沿用既有斑馬線遮罩" }
    if ($ReuseTracks -and (Test-Path "$dir\tracks\raw_tracks.json")) {
        "  [2/4] 沿用既有追蹤結果(只重做後照鏡過濾)"
        & $modern scripts\track_video.py --video $video.FullName --refilter "$dir\tracks\raw_tracks.json"
    } else {
        "  [2/4] 偵測與追蹤 ($Weights, conf $Conf, GPU)"
        & $modern scripts\track_video.py --video $video.FullName --weights $Weights --conf $Conf --imgsz $ImgSz --output "$dir\tracks"
    }
    "  [3/4] 讓行人規則"
    & $modern yield_image_rule.py --video $video.FullName --masks "$dir\seg\masks.npz" --tracks "$dir\tracks\raw_tracks.json" --output "datasets\$Run\events\$stem.jsonl"
    "  [4/4] 疊圖影片"
    & $modern scripts\render_overlay.py --video $video.FullName --masks "$dir\seg\masks.npz" --tracks "$dir\tracks\raw_tracks.json" --events "datasets\$Run\events\$stem.jsonl" --labels "labels\$stem.jsonl" --output "datasets\$Run\overlay\$stem.mp4"
}
"全部完成 $(Get-Date -Format HH:mm:ss)。事件:datasets\$Run\events  疊圖:datasets\$Run\overlay"
Stop-Transcript | Out-Null
