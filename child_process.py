"""One child process whose output is read line by line, cancellable, for the GUI's two backends.

`yield_pipeline.execute` (the three yield steps) and `detection_backend.run_detection` (the tf / zebra scripts) both
need the same care: the child's merged stdout/stderr is read on a thread into a bounded queue so a chatty child never
blocks on a full pipe, `cancelled()` is polled about every 0.1 s and terminates the child, the GUI is handed the
process while it runs (`on_process(process)`, then `on_process(None)`), and on any exit, including the caller
abandoning the loop, the child is terminated (killed after 5 s) and the pipe drained and closed.

    child = Child(command, cwd=folder, cancelled=cancelled, on_process=on_process)
    for line in child.lines():
        ...                      # e.g. parse progress
    child.code, child.stopped, child.tail

`scripts/run_pipeline.py`'s `run_together` is not this: it runs several commands at once with their output going
straight to the console.
"""
from __future__ import annotations

import queue
import subprocess
import threading

TAIL_LINES = 12


class Child:
    def __init__(self, command, cwd=None, env=None, cancelled=lambda: False, on_process=lambda process: None):
        self.command = [str(part) for part in command]
        self.cwd, self.env, self.cancelled, self.on_process = cwd, env, cancelled, on_process
        self.code = None      # exit code once lines() is exhausted
        self.stopped = False  # True when cancelled() made us terminate the child
        self.tail = []        # its last TAIL_LINES output lines, stripped, for error messages

    def lines(self):
        """Start the child and yield each output line as it arrives; returns once the child has exited."""
        process = subprocess.Popen(
            self.command, cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        self.on_process(process)
        buffer = queue.Queue(maxsize=512)

        def read():
            for line in process.stdout:
                buffer.put(line)
            buffer.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            while True:
                if self.cancelled() and process.poll() is None:
                    self.stopped = True
                    process.terminate()
                try:
                    line = buffer.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        break
                    continue
                if line is None:
                    break
                self.tail = (self.tail + [line.strip()])[-TAIL_LINES:]
                yield line
            self.code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            while reader.is_alive():  # drain so the reader can close its pipe
                try:
                    buffer.get(timeout=0.1)
                except queue.Empty:
                    pass
            process.stdout.close()
            self.on_process(None)
