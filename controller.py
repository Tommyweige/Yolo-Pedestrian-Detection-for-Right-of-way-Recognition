import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from moviepy.editor import ImageClip, VideoFileClip
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from UI import Ui_MainWindow
from video_controller import video_controller
from video_controller_rotate import video_controller_rotate


# Resolve every project path from this file instead of from the process' current
# working directory.  This keeps the application portable when it is launched
# from a shortcut, an IDE, or another directory.
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_PROJECT_ROOT = PROJECT_ROOT / "YOLOv8_DeepSORT_Object_Tracking"
DETECT_DIR = YOLO_PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
PREVIEW_IMAGE_PATH = RUNTIME_DIR / "preview.jpg"

MODEL_NAMES = ("yolov8s", "yolov8l", "yolov8x6")
TASKS = {
    "tf": {"script": "predict_tf.py", "weight_suffix": "_tf.pt"},
    "zebra": {"script": "predict_zebra.py", "weight_suffix": "_zebra.pt"},
}

# These are intentionally kept as user-selectable paths.  The video files are
# selected in the GUI and model assets can be placed in the repository's
# weights/ directory or configured through environment variables.
videos_path = []
folder_path = ""
rotate_angle = 0


def _configured_path(value, base=PROJECT_ROOT):
    """Return an absolute path for a config value.

    Relative values in environment variables are interpreted relative to the
    repository root, not relative to whichever directory launched the GUI.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _weights_dir():
    return _configured_path(os.environ.get("YOLO_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR))


def resolve_model_path(model_name, task):
    """Resolve and validate the YOLO weights required by a detection task."""

    task_config = TASKS[task]
    env_name = f"YOLO_{task.upper()}_MODEL"
    configured_model = os.environ.get(env_name)
    if configured_model:
        model_path = _configured_path(configured_model)
    else:
        filename = f"{model_name}{task_config['weight_suffix']}"
        model_path = _weights_dir() / filename

    if not model_path.is_file():
        expected_name = f"{model_name}{task_config['weight_suffix']}"
        raise FileNotFoundError(
            "找不到 YOLO 模型權重："
            f"{model_path}\n"
            f"請將 {expected_name} 放到 {_weights_dir()}，"
            f"或設定環境變數 {env_name} 指向權重檔案。"
        )
    return model_path


def resolve_reid_checkpoint():
    """Resolve the DeepSORT appearance-model checkpoint."""

    candidates = []
    configured_checkpoint = os.environ.get("DEEPSORT_REID_CKPT")
    if configured_checkpoint:
        candidates.append(_configured_path(configured_checkpoint))

    candidates.extend(
        (
            _weights_dir() / "ckpt.t7",
            DETECT_DIR
            / "deep_sort_pytorch"
            / "deep_sort"
            / "deep"
            / "checkpoint"
            / "ckpt.t7",
        )
    )

    for checkpoint in candidates:
        if checkpoint.is_file():
            return checkpoint

    locations = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "找不到 DeepSORT 權重 ckpt.t7。請將檔案放到 weights/ckpt.t7、"
        "原本的 DeepSORT checkpoint 位置，或設定 DEEPSORT_REID_CKPT。"
        f"\n已檢查：\n{locations}"
    )


def _hydra_path(path):
    """Use a platform-independent path representation in Hydra overrides."""

    return path.resolve().as_posix()


def _hydra_value(value):
    """Quote a Hydra value so paths containing spaces remain one override."""

    if isinstance(value, Path):
        value = _hydra_path(value)
    value = str(value).replace('"', '\\"')
    return f'"{value}"'


def _subprocess_environment(reid_checkpoint):
    """Build the environment used by the local YOLO/DeepSORT scripts."""

    environment = os.environ.copy()
    environment.update(
        {
            "QT_DEBUG_PLUGINS": "1",
            "KMP_DUPLICATE_LIB_OK": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "DEEPSORT_REID_CKPT": str(reid_checkpoint),
        }
    )

    # The scripts live below the local ultralytics package and also import the
    # sibling deep_sort_pytorch package.  Set both entries explicitly so an
    # installed ultralytics package cannot accidentally shadow this checkout.
    python_path = [str(YOLO_PROJECT_ROOT), str(DETECT_DIR)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def _video_width(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"無法開啟影片：{video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()

    if width <= 0:
        raise ValueError(f"無法讀取影片寬度：{video_path}")
    return width


class DetectionThread(QThread):
    """Run one of the local prediction scripts without changing process cwd."""

    qthread_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, int, str)

    task = None

    def __init__(self, ui, main_window):
        super().__init__()
        self.ui = ui
        self.mainWindow = main_window

    def run(self):
        try:
            self.RedlightViolation()
        except Exception as error:  # report worker errors in the GUI thread
            print(f"Detection failed: {error}", file=sys.stderr)
            self.error_signal.emit(str(error))

    def RedlightViolation(self):
        self._run_detection()

    def _run_detection(self):
        global folder_path, rotate_angle, videos_path

        if not videos_path:
            raise ValueError("尚未選擇要偵測的影片。")
        if not folder_path:
            raise ValueError("尚未選擇輸出資料夾。")

        model_index = self.ui.comboBox.currentIndex()
        if not 0 <= model_index < len(MODEL_NAMES):
            model_index = 0
        model_path = resolve_model_path(MODEL_NAMES[model_index], self.task)
        reid_checkpoint = resolve_reid_checkpoint()

        output_dir = _configured_path(folder_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        task_config = TASKS[self.task]
        script_path = DETECT_DIR / task_config["script"]
        if not script_path.is_file():
            raise FileNotFoundError(f"找不到偵測腳本：{script_path}")

        environment = _subprocess_environment(reid_checkpoint)
        selected_videos = list(videos_path)

        for index, selected_video in enumerate(selected_videos):
            if self.mainWindow.isHidden():
                return

            video_path = _configured_path(selected_video)
            if not video_path.is_file():
                raise FileNotFoundError(f"找不到影片：{video_path}")

            width = _video_width(video_path)

            # A rotated source is temporary and is removed after the child
            # process exits.  It never pollutes the selected output directory.
            with tempfile.TemporaryDirectory(prefix="traffic_yolo_rotate_") as temporary_dir:
                source_path = video_path
                if rotate_angle:
                    source_path = Path(temporary_dir) / "rotated.mp4"
                    clip = VideoFileClip(str(video_path))
                    rotated_clip = clip.rotate(rotate_angle)
                    try:
                        rotated_clip.write_videofile(str(source_path))
                    finally:
                        rotated_clip.close()
                        clip.close()

                command = [
                    sys.executable,
                    str(script_path),
                    f"model={_hydra_value(model_path)}",
                    f"source={_hydra_value(source_path)}",
                    f"project={_hydra_value(output_dir)}",
                    f"name={_hydra_value(video_path.name)}",
                    f"imgsz={width}",
                    "conf=0.7",
                    "iou=0.3",
                    "augment=True",
                    "half=True",
                ]

                print("Running:", " ".join(command))
                process = subprocess.Popen(
                    command,
                    cwd=str(DETECT_DIR),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                self.mainWindow.process = process
                try:
                    for line in process.stdout:
                        print(line, end="")
                        progress_match = re.search(r"\((\d+)/(\d+)\)", line)
                        if progress_match:
                            current, total = progress_match.groups()
                            self.progress_signal.emit(
                                index, int(current), int(total), line.strip()
                            )
                    return_code = process.wait()
                finally:
                    self.mainWindow.process = None

                if return_code != 0:
                    raise RuntimeError(
                        f"偵測腳本執行失敗（return code {return_code}）：{video_path.name}"
                    )

        self.finished_signal.emit(len(selected_videos) - 1)


class ThreadTask_tf(DetectionThread):
    task = "tf"


class ThreadTask_zebra(DetectionThread):
    task = "zebra"


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.setWindowIcon(QtGui.QIcon(str(PROJECT_ROOT / "RedlightIcon.png")))
        self.ui.button_stop.setIcon(QtGui.QIcon(str(PROJECT_ROOT / "stop.png")))
        self.ui.button_play.setIcon(QtGui.QIcon(str(PROJECT_ROOT / "start.png")))
        self.ui.button_pause.setIcon(QtGui.QIcon(str(PROJECT_ROOT / "pause.png")))
        self.ui.ReadFileButton.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        self.ui.RedlightViolation.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        self.ui.ReadFolderButtom.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        self.ui.comboBox.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        self.ui.zebra.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        for button in (self.ui.button_stop, self.ui.button_play, self.ui.button_pause):
            button.setStyleSheet(
                "background-color: rgb(224, 173, 119);border-radius: 10px; "
                "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
            )
        self.ui.rotate_confirm.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        self.ui.rotate_screen_bottum.setStyleSheet(
            "background-color: rgb(189, 126, 60);border-radius: 10px; "
            "border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);"
        )
        self.setWindowOpacity(0.97)
        self.rotate_cnt = 0
        self.angle = 0
        self.current_second = 0
        self.process = None

    def setup_control(self):
        self.ui.ReadFolderButtom.clicked.connect(self.open_folder)
        self.ui.ReadFileButton.clicked.connect(self.open_file)
        self.ui.RedlightViolation.clicked.connect(self.RedlightViolationButtonClick)
        self.ui.zebra.clicked.connect(self.zebraButtonClick)
        self.ui.SingleVideoProgressBar.setValue(0)
        self.ui.MultiVideoProgessBar.setValue(0)
        self.ui.rotate_screen_bottum.clicked.connect(self.rotete_screen)
        self.ui.rotate_confirm.clicked.connect(self.rotate_confirm)
        self.ui.rotate_screen_slider.valueChanged.connect(self.rotate_slider_moved)

    def open_file(self):
        video_filter = "影片文件 (*.mp4 *.avi *.mkv)"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption="選擇影片檔案",
            directory=str(PROJECT_ROOT),
            filter=video_filter,
            options=options,
        )

        global videos_path
        videos_path = [str(_configured_path(path)) for path in file_names]
        if len(videos_path) == 1:
            self.video_path = videos_path[0]
            self.video_controller = video_controller(
                video_path=self.video_path, ui=self.ui
            )
            self.ui.button_play.clicked.connect(self.video_controller.play)
            self.ui.button_stop.clicked.connect(self.video_controller.stop)
            self.ui.button_pause.clicked.connect(self.video_controller.pause)
            self.ui.ShowFilePath.setText(self.video_path)
        elif len(videos_path) > 1:
            self.ui.label_videoframe.setText("選擇了多部影片，因此不顯示影片畫面")
        else:
            self.ui.label_videoframe.setText("請先選擇影片路徑")

    def open_folder(self):
        global folder_path
        selected_folder = QFileDialog.getExistingDirectory(
            self, "選擇輸出資料夾", str(PROJECT_ROOT)
        )
        folder_path = str(_configured_path(selected_folder)) if selected_folder else ""
        if folder_path:
            self.ui.ShowFilePath.setText(f"輸出的資料夾：{folder_path}")

    def show_waring_popup(self):
        popup = QMessageBox(self)
        popup.setWindowTitle("提醒")
        popup.setText("請先選擇影片並指定輸出資料夾，再開始偵測。")
        popup.exec_()

    def show_error_popup(self, message):
        QMessageBox.critical(self, "偵測失敗", message)

    def show_finish_popup(self, index):
        popup = QMessageBox(self)
        self.ui.MultiVideoProgessBar.setValue(index + 1)
        popup.setWindowTitle("完成")
        popup.setText("影片已處理完成，請至輸出資料夾查看結果。")
        popup.exec_()

    def _start_detection(self, thread_class):
        global folder_path, videos_path

        if not videos_path or not folder_path:
            self.show_waring_popup()
            return

        output_dir = _configured_path(folder_path)
        if not output_dir.is_dir():
            self.show_error_popup(f"輸出資料夾不存在：{output_dir}")
            return

        model_index = self.ui.comboBox.currentIndex()
        if not 0 <= model_index < len(MODEL_NAMES):
            model_index = 0

        try:
            resolve_model_path(MODEL_NAMES[model_index], thread_class.task)
            resolve_reid_checkpoint()
        except (FileNotFoundError, ValueError) as error:
            self.show_error_popup(str(error))
            return

        self.qthread = thread_class(self.ui, self)
        self.qthread.finished_signal.connect(self.show_finish_popup)
        self.qthread.qthread_signal.connect(self.handleThreadSignal)
        self.qthread.error_signal.connect(self.show_error_popup)
        self.qthread.progress_signal.connect(self.update_progress)
        self.qthread.start()

    def RedlightViolationButtonClick(self):
        self._start_detection(ThreadTask_tf)

    def zebraButtonClick(self):
        self._start_detection(ThreadTask_zebra)

    def handleThreadSignal(self, _signal=0):
        print("Thread task completed!")

    def closeEvent(self, event):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        event.accept()

    def update_progress(self, index, current, total, line):
        self.ui.MultiVideoProgessBar.setMaximum(len(videos_path))
        self.ui.MultiVideoProgessBar.setValue(index)
        self.ui.SingleVideoProgressBar.setMaximum(total)
        self.ui.SingleVideoProgressBar.setValue(current)
        self.ui.FileInFiles.setText(
            f"已完成 {index}/{len(videos_path)} 部影片，請耐心等待：進度：{current}/{total}"
        )

    def rotete_screen(self):
        video_filter = "影片文件 (*.mp4 *.avi *.mkv)"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption="選擇影片檔案",
            directory=str(PROJECT_ROOT),
            filter=video_filter,
            options=options,
        )
        if not file_names:
            return

        global videos_path
        file_string = str(_configured_path(file_names[0]))
        videos_path = [file_string]
        self.video = VideoFileClip(file_string)
        self.video_controller_rotate = video_controller_rotate(
            video_path=file_string, ui=self.ui
        )
        self.video_controller_rotate.play()
        self.rotate_cnt = 0

    def rotate_slider_moved(self):
        if not hasattr(self, "video_controller_rotate"):
            return
        if self.rotate_cnt == 0:
            self.current_second = self.video_controller_rotate.pause()
        self.angle = self.ui.rotate_screen_slider.value() + 45

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.video.save_frame(str(PREVIEW_IMAGE_PATH), t=self.current_second)
        image = ImageClip(str(PREVIEW_IMAGE_PATH)).rotate(self.angle)
        image.save_frame(str(PREVIEW_IMAGE_PATH))
        self.ui.rotate_screen.setPixmap(QtGui.QPixmap(str(PREVIEW_IMAGE_PATH)))
        self.rotate_cnt = 1

    def rotate_confirm(self):
        global rotate_angle
        if self.angle:
            rotate_angle = self.angle


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow_controller()
    main_window.show()
    sys.exit(app.exec_())
