# Issue #8 environment validation — 2026-10-10

Scope: dependency isolation and legacy regression only, against main
`f706572f0d03372477168c54c61f55a8ff4908c5`. Issues #9–#12 are not implemented.
Commands ran from the repository root in Windows PowerShell. Raw JSON/logs,
checkpoints, caches and generated videos remain in ignored local directories.

## Machine and preserved baseline

Windows 11 x64 build 26300; Python 3.12.13 (Conda-forge); RTX 4060 Laptop
8188 MiB, driver 616.92; torch 2.13.0+cu130, torchvision 0.28.0+cu130,
CUDA runtime 13.0. CUDA was available and tested, not inferred from driver presence.

The original `runtime/detection-env` imports Ultralytics 8.0.3, NumPy 1.26.4,
OpenCV 4.11.0.86, MoviePy 1.0.3 and setuptools 80.10.2. Its venv inherits
system packages; PyQt5 was absent. Original `pip check` fails because inherited
ml-dtypes 0.6.0 requires NumPy >=2.0.0. This existing environment was preserved;
the before/after `pip freeze` lists compare equal. No detector, launcher,
bridge, traffic-rule or Rust implementation was changed in this ticket.

```powershell
& runtime/detection-env/Scripts/python.exe -m pip freeze
& runtime/detection-env/Scripts/python.exe -m pip check
& runtime/detection-env/Scripts/python.exe test_desktop_bridge.py
& runtime/detection-env/Scripts/python.exe test_traffic_rules.py
```

Baseline bridge checks and all 5 traffic-rule tests passed. The known dependency
conflict above is not described as a green baseline.

## Fresh builds and CUDA checks

Base interpreter used:
`C:\Users\tommy\anaconda3\envs\breakout-rl-engineering\python.exe`.
Every new venv has `include-system-site-packages = false`.

```powershell
$basePython = 'C:\Users\tommy\anaconda3\envs\breakout-rl-engineering\python.exe'
$envRoot = 'C:\Users\tommy\.venvs\traffic-i8-repro'
foreach ($profile in @('legacy', 'labeling', 'modern')) {
    & $basePython scripts/setup_environment.py $profile --env-root $envRoot --locked
}
& "$envRoot\legacy\Scripts\python.exe" scripts/check_environment.py legacy --require-cuda --output runtime/issue8-repro-legacy-smoke.json
& "$envRoot\labeling\Scripts\python.exe" scripts/check_environment.py labeling --require-cuda --output runtime/issue8-repro-labeling-smoke.json
# Additional empty build tested the final pip --isolated bootstrap command:
& $basePython scripts/setup_environment.py modern --env-root C:\Users\tommy\.venvs\traffic-i8-final --locked
& C:\Users\tommy\.venvs\traffic-i8-final\modern\Scripts\python.exe scripts/check_environment.py modern --require-cuda --output runtime/issue8-final-modern-smoke.json
```

All three locked profiles built from empty targets and passed `pip check`.
The recorded legacy, labeling and final modern smoke reports pass imports,
CUDA matrix multiplication and the compiled torchvision CUDA NMS operator.
Core versions are listed in [environments.md](environments.md) and complete
resolved dependencies in `requirements/locks/windows-py312-*.txt`.
The modern profile constructs the YOLO26n `DetectionModel`; labeling imports
rf-groundingdino 0.1.2 and rf-segment-anything 1.0. autodistill-yolov8 is absent.

Two reproducible installation problems were fixed: torch's deep wheel paths
trigger WinError 206 under the long worktree path, and rf-groundingdino's sdist
README triggers GBK decoding errors without UTF-8 mode. The builder now requires
short Windows targets and sets UTF-8 for install children. Autodistill also
imports undeclared scikit-learn/Roboflow dependencies; both are explicitly pinned.
Labeling contains matching regular/headless OpenCV distributions as required by
upstream packages; only image I/O/inference was checked there.

## Real model forward checks

SHA-256 was verified before checkpoint loading; the executable DINO config was
also hashed before construction. Sources and digests are recorded in
`requirements/smoke-assets.json`. DINO's checkpoint digest matches the LFS digest
at the official README's [Hugging Face mirror](https://huggingface.co/ShilongLiu/GroundingDINO/tree/a94c9b567a2a374598f05c584e96798a170c56fb).
YOLO26n's digest matches [GitHub release asset metadata](https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0).

```powershell
$dinoCache = Join-Path $env:USERPROFILE '.cache\autodistill\groundingdino'
& C:\Users\tommy\.venvs\traffic-i8-repro\labeling\Scripts\python.exe scripts/check_environment.py labeling --require-cuda --image temp.jpg --checkpoint "$dinoCache\groundingdino_swint_ogc.pth" --sha256 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799 --config-sha256 172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1 --output runtime/issue8-repro-labeling-model-smoke.json
& C:\Users\tommy\.venvs\traffic-issue8\modern\Scripts\python.exe scripts/check_environment.py modern --require-cuda --image temp.jpg --checkpoint runtime/issue8-assets/yolo26n.pt --sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef --output runtime/issue8-modern-model-smoke.json
```

| Check | Result | Peak torch allocated CUDA memory |
|---|---|---|
| Grounding DINO plugin constructor + one-image predict | PASS, 31 returned boxes | 1,994,957,824 bytes |
| Official YOLO26n checkpoint + one-image predict, imgsz=320 | PASS, 7 returned boxes | 22,557,696 bytes |

The released rf-groundingdino 0.1.2 uses the PyTorch attention path; its native
extension build is commented out. Windows inference succeeded without MSVC/nvcc.
WSL Ubuntu exposes the GPU but its default Python 3.14.4 was not used: no WSL ML
installation or Linux lock is claimed. SAM import passed; SAM mask inference was
not tested. Box counts and memory are smoke observations, not quality metrics.

## Legacy and Rust regressions

Clean legacy interpreter:
`C:\Users\tommy\.venvs\traffic-issue8\legacy\Scripts\python.exe`.

```powershell
$legacyPython = 'C:\Users\tommy\.venvs\traffic-issue8\legacy\Scripts\python.exe'
& $legacyPython test_desktop_bridge.py
& $legacyPython test_traffic_rules.py
& C:\Users\tommy\.venvs\traffic-issue8\modern\Scripts\python.exe test_environment_setup.py
& $legacyPython scripts/validate_detection.py --video 'C:\Users\tommy\Videos\不禮讓行人，就是台灣駕駛日常（抱歉了小黃就你的車牌錄的最清楚，只好檢舉你）_1080p60.mp4' --output runtime/e2e/issue8-clean-legacy-real --model yolov8s --task zebra
$env:RUSTUP_HOME = Join-Path $PWD 'runtime/toolchain/rustup'
$env:CARGO_HOME = Join-Path $PWD 'runtime/toolchain/cargo'
$env:PATH = (Join-Path $PWD 'runtime/toolchain/w64devkit/bin') + ';' + $env:PATH
& runtime/toolchain/cargo/bin/cargo.exe +stable-x86_64-pc-windows-gnu check --manifest-path rust-ui/Cargo.toml
$env:TRAFFIC_TEST_VIDEO = Join-Path $PWD 'runtime/native 測試/bframes.mp4'
$env:TRAFFIC_TEST_WIDTH = '640'
$env:TRAFFIC_SEEK_VIDEO = 'C:\Users\tommy\Videos\不禮讓行人，就是台灣駕駛日常（抱歉了小黃就你的車牌錄的最清楚，只好檢舉你）_1080p60.mp4'
$env:TRAFFIC_VIDEO_ACCELERATION = 'auto'
& runtime/toolchain/cargo/bin/cargo.exe +stable-x86_64-pc-windows-gnu test --manifest-path rust-ui/Cargo.toml -- --include-ignored
```

- Clean legacy bridge checks: PASS; traffic-rule suite: 5 tests / 21 subcases PASS.
- Environment CLI safety suite: 4 tests PASS (preserved existing/protected targets,
  rejected inherited unsafe pickle override, rejected incorrect checkpoint hash).
- Rust cargo check: PASS; full test suite with local fixtures: 8 PASS, 0 ignored.
- Legacy short fixture: 6 input/output frames at 6 fps, PASS.
- User traffic clip: 900 input/output frames at 60 fps, PASS; elapsed 92.127 seconds
  (9.769 pipeline fps). No violation output on this clip and no annotated gold set;
  this verifies pipeline integrity, not recognition accuracy or speed improvement.

## License and serialization boundaries

Installed license files: old Ultralytics GPL-3.0, new Ultralytics AGPL-3.0,
Autodistill/DINO/SAM Apache-2.0, vendored DeepSORT MIT. Plugin PyPI classifier
and packaged license are inconsistent; deployment/redistribution needs separate
review. Existing private traffic weights keep their recorded source revision and
hashes; no dataset-rights or blanket checkpoint-use authorization is inferred.

The existing legacy worker/test processes alone set the trusted legacy pickle
override. Setup removes it from children; modern/labeling smoke rejects an inherited
true value. Third-party loaders have their own deserialization policies: source
and hash validation does not make arbitrary pickle safe.

Not tested: labeling/export quality, SAM forward, YOLO26 training, annotated traffic
accuracy, WSL ML execution, the future YOLO26 GUI adapter, or dependency locks on
other platforms. Those remain outside Issue #8.
