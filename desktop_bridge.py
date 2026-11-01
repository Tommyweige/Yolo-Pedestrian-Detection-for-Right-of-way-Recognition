"""JSON-lines IPC for the Rust desktop. stdout contains protocol events only."""
import json
import os
import sys
import threading

from detection_backend import run_detection, run_yield


def emit(event):
    print(json.dumps(event, ensure_ascii=True), flush=True)


def stream(request, cancelled):
    """Protocol events for one request: the 不禮讓 task runs the image rule, the rest the legacy predictors."""
    if request.get("task") == "yield":
        return run_yield(request["videos"], request["output"], cancelled=cancelled)
    return run_detection(request["videos"], request["output"], request["model"],
                         request["task"], request.get("angle", 0), cancelled)


def detect():
    request = json.loads(sys.stdin.readline())
    # Initialize NumPy/OpenCV before a blocking Windows CRT pipe read starts.
    # Otherwise NumPy's DLL initialization can wait on the stdin descriptor lock.
    import cv2  # noqa: F401
    cancelled = threading.Event()

    def listen():
        # A cancel command or closing the parent pipe both stop the job.
        os.read(sys.stdin.fileno(), 1)
        cancelled.set()

    threading.Thread(target=listen, daemon=True).start()
    for event in stream(request, cancelled.is_set):
        emit(event)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        if sys.argv[1:] == ["detect"]:
            detect()
        else:
            raise ValueError("Use desktop_bridge.py detect")
    except Exception as error:
        emit({"type": "error", "message": str(error)})
        sys.exit(1)
