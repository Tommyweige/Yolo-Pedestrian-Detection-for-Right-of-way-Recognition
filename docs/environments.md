# Isolated dependency environments

This implements Issue #8 only. Do not start #9–#12 or replace the legacy predictor with a newer Ultralytics internal API.

## Validated platform and profiles

Windows x64 / Python 3.12.13 / NVIDIA RTX 4060 Laptop 8188 MiB / driver 616.92.
All profiles use the verified torch 2.13.0+cu130 / torchvision 0.28.0+cu130 pair from the official PyTorch cu130 index.

| Profile | Core versions | Purpose |
|---|---|---|
| legacy | Ultralytics 8.0.3, NumPy 1.26.4, OpenCV 4.11.0.86, MoviePy 1.0.3, setuptools 80.10.2 | Existing Hydra/BasePredictor, DeepSORT, GUI bridge, optional PyQt5 |
| labeling | Autodistill 0.1.29, Grounding DINO plugin 0.1.4, rf-groundingdino 0.1.2, rf-segment-anything 1.0, transformers 4.44.2, supervision 0.25.1 | Dependency and one-image forward checks only |
| modern | Ultralytics 8.4.175, NumPy 2.2.6, OpenCV 4.12.0.88 | Verified YOLO26n construction and official-weight CUDA forward; no training or GUI adapter |

Direct requirements are under `requirements/`; complete Windows/Python 3.12 locks are under `requirements/locks/`.
Locks contain the resolved transitive dependencies, not paths into a developer's Conda install. `pip==25.0.1` is the bootstrap resolver.

## Build fresh environments

Run from the repository root. Use an existing **explicit Python 3.12** interpreter; the script refuses other Python versions, existing targets, and any target containing or nested inside `runtime/detection-env`.
On Windows, the final environment path must be at most 60 characters: the torch wheel contains very deep license paths. A long repository worktree path reproduced WinError 206.
The builder disables inherited system packages, uses pip's isolated mode to ignore user pip install destinations/configuration, and removes PYTHONPATH/PYTHONHOME and the legacy unsafe-pickle override from installation children. It uses UTF-8 mode to build rf-groundingdino's sdist on GBK-locale Windows.

```powershell
$basePython = 'C:\path\to\Python312\python.exe'
$envRoot = Join-Path $env:USERPROFILE '.venvs\traffic'
foreach ($profile in @('legacy', 'labeling', 'modern')) {
    & $basePython scripts/setup_environment.py $profile --env-root $envRoot --locked
    if ($LASTEXITCODE -ne 0) { throw "Environment setup failed: $profile" }
    $python = Join-Path $envRoot "$profile\Scripts\python.exe"
    & $python scripts/check_environment.py $profile --require-cuda --output "runtime\$profile-smoke.json"
    if ($LASTEXITCODE -ne 0) { throw "Environment check failed: $profile" }
}
```

Choose a fresh root after a failed installation; the script never silently resumes or upgrades an existing environment.
If intentionally resolving new transitive versions, omit `--locked`, run all checks, and generate a new reviewed lock with that interpreter's `-m pip freeze`.
Do not use a labeling or modern interpreter to launch the current detection worker.

```powershell
$legacyPython = Join-Path $envRoot 'legacy\Scripts\python.exe'
$env:TRAFFIC_PYTHON = $legacyPython # explicit: start.py may otherwise prefer the existing detection-env
& $legacyPython start.py
# Optional original UI, using the same legacy interpreter:
& $legacyPython start.py --legacy
```

The existing `runtime/detection-env` and its installed packages are not upgraded. Its inherited ml-dtypes/NumPy conflict is recorded separately; clean legacy has no inherited packages and passes `pip check`.

## Checkpoint provenance and model smoke

`requirements/smoke-assets.json` records immutable official URLs, sizes and SHA-256 values. Download and verify **before** invoking model loaders; files and caches stay outside Git.
The DINO checkpoint is the original IDEA-Research release; its digest also matches the LFS digest published by the official README's Hugging Face mirror at the recorded revision.
The config is executable Python, so its immutable Roboflow commit URL and digest must also be checked.
YOLO26n's digest comes from the GitHub release asset metadata. BERT uses `bert-base-uncased`; the tested Hugging Face cache revision is recorded in the manifest. No Roboflow project credentials or private dataset are accessed.

```powershell
$assets = (Get-Content requirements/smoke-assets.json -Raw | ConvertFrom-Json).artifacts
$dinoCache = Join-Path $env:USERPROFILE '.cache\autodistill\groundingdino'
New-Item -ItemType Directory -Force $dinoCache, runtime\smoke-assets | Out-Null
foreach ($asset in $assets) {
    $folder = if ($asset.name -eq 'yolo26n.pt') { 'runtime\smoke-assets' } else { $dinoCache }
    $target = Join-Path $folder $asset.name
    if (-not (Test-Path -LiteralPath $target)) { Invoke-WebRequest $asset.url -OutFile $target }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.sha256) {
        throw "Untrusted or incomplete asset: $target" # preserve it for inspection; do not load or overwrite
    }
}
$modernPython = Join-Path $envRoot 'modern\Scripts\python.exe'
& $modernPython scripts/check_environment.py modern --require-cuda --image temp.jpg --checkpoint runtime\smoke-assets\yolo26n.pt --sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef --output runtime\modern-model.json
$labelPython = Join-Path $envRoot 'labeling\Scripts\python.exe'
& $labelPython scripts/check_environment.py labeling --require-cuda --image temp.jpg --checkpoint "$dinoCache\groundingdino_swint_ogc.pth" --sha256 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799 --config-sha256 172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1 --output runtime\labeling-model.json
```

Basic checks run torch CUDA and torchvision NMS kernels, profile imports and `pip check`; without `--image`, they do not claim model inference.
Model smoke is a one-image forward, **not** labeling quality, a dataset export, YOLO26 training, or traffic-rule accuracy.
The legacy unsafe pickle environment variable is set only by the existing trusted legacy predictor/test processes. Labeling/modern smoke rejects an inherited true value and does not set it.
Third-party loaders still have their own serialization policies; a checksum is an integrity check tied to the trusted source, not a guarantee that arbitrary pickle is safe.

## Platform caveats and licenses

- rf-groundingdino 0.1.2's released source comments out C++/CUDA extension building and uses PyTorch deformable attention; the Windows CUDA forward passed without MSVC/nvcc. This does **not** describe the separate upstream native GroundingDINO build.
- Autodistill imports scikit-learn and Roboflow without declaring them correctly; the labeling profile pins both explicitly. The SDK is imported but no login/project API call is made.
- Roboflow requires OpenCV-headless while rf-groundingdino requires OpenCV; both distributions are pinned to 4.12.0.88 in labeling because they share `cv2`. Only image I/O/inference is verified there, not OpenCV GUI; legacy has only regular OpenCV 4.11.
- The verified labeling stack still emits deprecation warnings; future changes must be resolved in a new environment, not the preserved runtime.
- WSL Ubuntu exposes the GPU on this machine, but its default Python is 3.14.4 and was **not** used for this validation. Windows passed, so WSL was not required. To experiment in WSL, use Python 3.12 and fresh profiles without the Windows-only lock, then create and validate a separate Linux lock. Do not describe the current Windows locks as Linux reproducibility evidence.
- Installed Ultralytics 8.0.3 declares GPL-3.0; 8.4.175 declares AGPL-3.0. The installed Autodistill/DINO/SAM distribution LICENSE files declare Apache-2.0 (the plugin's PyPI classifier is inconsistent); vendored DeepSORT has MIT notices. Preserve these notices and review deployment/redistribution obligations separately.
- Existing private traffic weights retain their recorded Hugging Face revision/hash and third-party notices. Dataset rights, pretrained-weight reuse and deployment licensing are separate from an import or checksum check; no new blanket permission is claimed.

## Regression commands

```powershell
& $legacyPython test_desktop_bridge.py
& $legacyPython test_traffic_rules.py # requires trusted ckpt.t7
& $modernPython test_environment_setup.py
cargo check --manifest-path rust-ui/Cargo.toml
cargo test --manifest-path rust-ui/Cargo.toml
# Existing real-weight pipeline validation; create an empty output folder first:
& $legacyPython scripts/validate_detection.py --video C:\path\to\traffic.mp4 --output runtime\validation --model yolov8s --task zebra
```

No #9–#12 ingestion, auto-label export, fine-tuning or new inference adapter is included here.
