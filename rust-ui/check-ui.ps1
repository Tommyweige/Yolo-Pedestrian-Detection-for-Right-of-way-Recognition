param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$binary = Join-Path $PSScriptRoot 'target\debug\traffic-desktop.exe'
if (-not (Test-Path -LiteralPath $binary)) { throw 'Run cargo build --manifest-path rust-ui/Cargo.toml first.' }
$output = Join-Path $projectRoot 'runtime\ui-check'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$video = Join-Path $output 'preview.avi'
$makeVideo = @'
import cv2, numpy as np, sys
writer = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 360))
assert writer.isOpened(), 'Cannot create the preview fixture'
for index in range(20):
    frame = np.full((360, 640, 3), (50, 40, 25), dtype=np.uint8)
    frame[:8, :8] = index * 10
    cv2.putText(frame, 'UI preview check', (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 235, 235), 2)
    writer.write(frame)
writer.release()
'@
& $Python -c $makeVideo $video
if ($LASTEXITCODE -ne 0) { throw 'Use a Python environment with OpenCV and numpy.' }
$cases = @(
    @{ Name = 'empty'; Preview = $false; Theme = 'light'; View = 'analysis'; Layout = 'normal' },
    @{ Name = 'preview'; Preview = $true; Theme = 'light'; View = 'analysis'; Layout = 'normal' },
    @{ Name = 'playback'; Preview = $true; Theme = 'light'; View = 'playback'; Layout = 'normal' },
    @{ Name = 'folder'; Preview = $true; Theme = 'light'; View = 'folder'; Layout = 'normal' },
    @{ Name = 'calibration-dark'; Preview = $true; Theme = 'dark'; View = 'calibration'; Layout = 'normal' },
    @{ Name = 'compact'; Preview = $true; Theme = 'light'; View = 'analysis'; Layout = 'compact' }
)
foreach ($case in $cases) {
    $screenshot = Join-Path $output ($case.Name + '.png')
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new($binary)
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = 'Hidden'
    $startInfo.RedirectStandardError = $true
    $startInfo.WorkingDirectory = $projectRoot
    $startInfo.Environment['TRAFFIC_PROJECT_ROOT'] = $projectRoot
    # Preview must work even when Python cannot be launched.
    $startInfo.Environment['TRAFFIC_PYTHON'] = Join-Path $output 'python-must-not-run.exe'
    $startInfo.Environment['TRAFFIC_SCREENSHOT'] = $screenshot
    $startInfo.Environment['TRAFFIC_SCREENSHOT_THEME'] = $case.Theme
    $startInfo.Environment['TRAFFIC_SCREENSHOT_VIEW'] = $case.View
    $startInfo.Environment['TRAFFIC_SCREENSHOT_LAYOUT'] = $case.Layout
    $startInfo.Environment['TRAFFIC_SCREENSHOT_READY'] = if ($case.Preview) { '1' } else { '0' }
    if ($case.Preview) { $startInfo.Environment['TRAFFIC_PREVIEW_VIDEO'] = $video }
    else { $startInfo.Environment.Remove('TRAFFIC_PREVIEW_VIDEO') | Out-Null }
    $started = [DateTime]::UtcNow
    $process = [System.Diagnostics.Process]::Start($startInfo)
    $errors = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit(25000)) { $process.Kill($true); throw "UI check timed out: $($case.Name)" }
    $stderr = $errors.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) { throw "UI check failed: $($case.Name): $stderr" }
    $file = Get-Item -LiteralPath $screenshot
    if ($file.Length -eq 0 -or $file.LastWriteTimeUtc -lt $started) { throw "Screenshot missing: $($case.Name)" }
    if ($case.Name -eq 'folder') {
        $response = Get-Content -LiteralPath (Join-Path $output 'folder.json') -Raw | ConvertFrom-Json
        if (-not $response.picker_open -or $response.updates -lt 10 -or $response.displayed_frame -lt 5) {
            throw 'Main window stopped updating while the native folder picker was open.'
        }
    }
    Write-Output "PASS: $($case.Name) -> $screenshot"
    $process.Dispose()
}
