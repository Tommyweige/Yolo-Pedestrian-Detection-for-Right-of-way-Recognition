"""Create a fresh, isolated Issue #8 environment; never upgrade an existing one."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("legacy", "labeling", "modern")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--locked", action="store_true", help="Use the verified Windows/Python 3.12 full lock")
    args = parser.parse_args()
    target = (args.env_root / args.profile).resolve()
    protected = (ROOT / "runtime" / "detection-env").resolve()
    if target.exists() or target.is_relative_to(protected) or protected.is_relative_to(target):
        parser.error("Refusing to overwrite an existing environment or the protected legacy runtime")
    if sys.version_info[:2] != (3, 12):
        parser.error("Use an explicit Python 3.12 interpreter; other versions are not validated")
    if os.name == "nt" and len(str(target)) > 60:
        parser.error("Choose a shorter environment path (at most 60 characters) to avoid Windows torch path errors")
    requirements = ROOT / "requirements" / f"{args.profile}.txt"
    if args.locked:
        if os.name != "nt":
            parser.error("The full locks are validated only on Windows/Python 3.12")
        requirements = ROOT / "requirements" / "locks" / f"windows-py312-{args.profile}.txt"
    if not requirements.is_file():
        parser.error(f"Missing dependency spec: {requirements}")
    environment = os.environ.copy()
    # rf-groundingdino's sdist reads a UTF-8 README with the Windows locale default.
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    for name in ("PYTHONPATH", "PYTHONHOME", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        environment.pop(name, None)
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(target)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    def run(*arguments):
        subprocess.run([str(python), "-m", "pip", "--isolated", *arguments], env=environment, check=True)
    run("install", "pip==25.0.1")
    run("install", "-r", str(ROOT / "requirements" / "torch-cu130.txt"),
        "--index-url", "https://download.pytorch.org/whl/cu130")
    run("install", "-r", str(requirements))
    run("check")
    print(f"Interpreter: {python}")
    print(f"Run: {python} {ROOT / 'scripts' / 'check_environment.py'} {args.profile}")


if __name__ == "__main__":
    main()
