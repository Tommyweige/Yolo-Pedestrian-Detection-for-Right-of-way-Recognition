import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys

from moviepy.editor import ImageClip, VideoFileClip
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from UI import Ui_MainWindow
from video_controller import video_controller
from video_controller_rotate import video_controller_rotate


from detection_backend import (
    PROJECT_ROOT, RUNTIME_DIR, PREVIEW_IMAGE_PATH, MODEL_NAMES,
    _configured_path, resolve_model_path, resolve_reid_checkpoint, run_detection,
)

videos_path = []
folder_path = ""
rotate_angle = 0


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
        model_index = self.ui.comboBox.currentIndex()
        model_index = model_index if 0 <= model_index < len(MODEL_NAMES) else 0
        for event in run_detection(
            videos_path, folder_path, MODEL_NAMES[model_index], self.task, rotate_angle,
            cancelled=self.mainWindow.isHidden,
            on_process=lambda process: setattr(self.mainWindow, "process", process),
        ):
            if event["type"] == "progress":
                self.progress_signal.emit(event["index"], event["current"], event["total"], event["message"])
            elif event["type"] == "done":
                self.finished_signal.emit(len(videos_path) - 1)


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
