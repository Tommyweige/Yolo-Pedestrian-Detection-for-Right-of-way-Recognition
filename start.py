"""Start the Rust desktop; --legacy starts the original PyQt interface."""
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    if "--legacy" in sys.argv:
        from PyQt5 import QtWidgets
        from controller import MainWindow_controller

        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow_controller()
        window.show()
        sys.exit(app.exec_())
    root = Path(__file__).resolve().parent
    executable = "traffic-desktop.exe" if os.name == "nt" else "traffic-desktop"
    candidates = [root / "rust-ui" / "target" / profile / executable
                  for profile in ("release", "debug")]
    binary = next((path for path in candidates if path.is_file()), None)
    if binary is None:
        sys.exit("請先執行 cargo build --manifest-path rust-ui/Cargo.toml，再執行 python start.py。")
    environment = os.environ.copy()
    local_python = root / "runtime" / "detection-env" / "Scripts" / "python.exe"
    environment.setdefault("TRAFFIC_PYTHON", str(local_python) if local_python.is_file() else sys.executable)
    environment.setdefault("TRAFFIC_PROJECT_ROOT", str(root))
    sys.exit(subprocess.call([str(binary)], cwd=root, env=environment))
