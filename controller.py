import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow,QMessageBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl,QThread,pyqtSignal,QMetaObject
import cv2
import time
from video_controller import video_controller
from video_controller_rotate import video_controller_rotate
#from YOLOv8_DeepSORT_Object_Tracking.ultralytics.yolo.v8.detect.predict import predict
import sys
from UI import Ui_MainWindow
import threading
import subprocess
import re
from PyQt5.Qt import *
from moviepy.editor import *

videos_path = '' 
folder_path = ''
rotate_angle = 0
     
    
class ThreadTask_tf(QThread):
    qthread_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(int)
    progress_signal = pyqtSignal(int, int, int, str) 

    def __init__(self, ui,mainWindow):
        super().__init__()
        self.ui = ui
        self.mainWindow = mainWindow

    def run(self):
        self.RedlightViolation()

    def RedlightViolation(self):
        import os

        current_folder = os.getcwd()

        import subprocess

        dist = {"0": "yolov8s", "1": "yolov8l", "2": "yolov8x6"}

        os.environ['QT_DEBUG_PLUGINS'] = '1'
        os.environ['KMP_DUPLICATE_LIB_OK'] = '1'
        os.environ['HYDRA_FULL_ERROR'] = '1'

        os.chdir(r"D:\Learning\SelfLearning\collage-project\專研CD_第22組\GUI\YOLOv8_DeepSORT_Object_Tracking\ultralytics\yolo\v8\detect")
        global videos_path
    
        for index, file in enumerate(videos_path):
            if self.mainWindow.isHidden():
                return
            print(file)
            
            cap = cv2.VideoCapture(file)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            video_filename = os.path.basename(file)#取得檔案名稱
            
            if rotate_angle:
                clip = VideoFileClip(file)
                clip = clip.rotate(rotate_angle)
                clip.write_videofile("temp.mp4")
                command = ["python", "predict_tf.py", "model=" + str(dist[str(self.ui.comboBox.currentIndex())]) +"_tf.pt",
                           "source=temp.mp4", "project=" + folder_path,"name=" + video_filename, "imgsz="+str(width), "conf=0.7", "iou=0.3", "augment=True", "half=True"]
                
            else:
                command = ["python", "predict_tf.py", "model=" + str(dist[str(self.ui.comboBox.currentIndex())]) +"_tf.pt",
                      "source=" + file, "project=" + folder_path,"name=" + video_filename, "imgsz="+str(width), "conf=0.7", "iou=0.3", "augment=True", "half=True"]  
           
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       universal_newlines=True, encoding='utf-8', errors='ignore')
            self.mainWindow.process = process
            # 实时按行读取输出并打印
            for line in process.stdout:
                print(line, end='')
                    
                progress_str = re.search(r'\((\d+)/(\d+)\)', line)
                    
                if progress_str:
                    current, total = progress_str.groups()
                    self.progress_signal.emit(index, int(current), int(total), line.strip())

            # 等待子进程结束
        process.wait()
        self.finished_signal.emit(index)
        os.chdir(r"../../../../../")


class ThreadTask_zebra(QThread):
    qthread_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(int)
    progress_signal = pyqtSignal(int, int, int, str) 

    def __init__(self, ui,mainWindow):
        super().__init__()
        self.ui = ui
        self.mainWindow = mainWindow

    def run(self):
        self.RedlightViolation()

    def RedlightViolation(self):
        import os

        current_folder = os.getcwd()

        import subprocess

        dist = {"0": "yolov8s", "1": "yolov8l", "2": "yolov8x6"}

        os.environ['QT_DEBUG_PLUGINS'] = '1'
        os.environ['KMP_DUPLICATE_LIB_OK'] = '1'
        os.environ['HYDRA_FULL_ERROR'] = '1'
        
        
        os.chdir(r"D:\Learning\SelfLearning\collage-project\專研CD_第22組\GUI\YOLOv8_DeepSORT_Object_Tracking\ultralytics\yolo\v8\detect")
        global videos_path
    
        for index, file in enumerate(videos_path):
            if self.mainWindow.isHidden():
                return
            print(file)
            
            cap = cv2.VideoCapture(file)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            video_filename = os.path.basename(file)#取得檔案名稱
            
            if rotate_angle:
                clip = VideoFileClip(file)
                clip = clip.rotate(rotate_angle)
                clip.write_videofile("temp.mp4")
                command = ["python", "predict_zebra.py", "model=" + str(dist[str(self.ui.comboBox.currentIndex())]) +"_zebra.pt",
                           "source=temp.mp4", "project=" + folder_path,"name=" + video_filename, "imgsz="+str(width), "conf=0.7", "augment=True", "half=True"]
            
            else:
                command = ["python", "predict_zebra.py", "model=" + str(dist[str(self.ui.comboBox.currentIndex())]) +"_zebra.pt",
                       "source=" + file, "project=" + folder_path,"name=" + video_filename, "conf=0.7", "augment=True", "half=True"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       universal_newlines=True, encoding='utf-8', errors='ignore')
            self.mainWindow.process = process
            # 实时按行读取输出并打印
            for line in process.stdout:
                print(line, end='')
                    
                progress_str = re.search(r'\((\d+)/(\d+)\)', line)
                    
                if progress_str:
                    current, total = progress_str.groups()
                    self.progress_signal.emit(index, int(current), int(total), line.strip())

            # 等待子进程结束
        process.wait()
        self.finished_signal.emit(index)
        os.chdir(r"../../../../../")
        

class MainWindow_controller(QtWidgets.QMainWindow):
    
   
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.setWindowIcon(QtGui.QIcon('RedlightIcon.png'))
        self.ui.button_stop.setIcon(QtGui.QIcon('stop.png'))
        self.ui.button_play.setIcon(QtGui.QIcon('start.png'))
        self.ui.button_pause.setIcon(QtGui.QIcon('pause.png'))
        self.ui.ReadFileButton.setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.RedlightViolation.setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.ReadFolderButtom.setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.comboBox.setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.zebra .setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.button_stop.setStyleSheet('background-color: rgb(224, 173, 119);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.button_play.setStyleSheet('background-color: rgb(224, 173, 119);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.button_pause.setStyleSheet('background-color: rgb(224, 173, 119);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.rotate_confirm.setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.ui.rotate_screen_bottum.setStyleSheet('background-color: rgb(189, 126, 60);border-radius: 10px; border: 2px groove gray;border-style: outset;color: rgb(255, 255, 255);')
        self.setWindowOpacity(0.97)
        self.rotate_cnt = 0

    
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
        
        file_name, _ = QFileDialog.getOpenFileNames(
            parent=None,
            caption="选择影片文件",
            directory=".",
            filter=video_filter,
            options=options
            )
                  # start path
        print(file_name)
        global videos_path
        videos_path = file_name
        if len(videos_path) == 1:

            self.video_path = str(file_name[0])
            self.video_controller = video_controller(video_path=self.video_path,
                                         ui=self.ui)

            self.ui.button_play.clicked.connect(self.video_controller.play) # connect to function()
            self.ui.button_stop.clicked.connect(self.video_controller.stop)
            self.ui.button_pause.clicked.connect(self.video_controller.pause)
            self.ui.ShowFilePath.setText(file_name[0])
        elif len(videos_path) > 1:
            self.ui.label_videoframe.setText("由於您選擇的是多部影片，故不顯示影片畫面")
        else:
            self.ui.label_videoframe.setText("請確實輸入影片路徑")

    def open_folder(self):
        global folder_path
        folder_path = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")                 # start path
        print(folder_path)
        self.ui.ShowFilePath.setText("儲存的資料夾為："+folder_path)

    def show_waring_popup(self):
        popup = QMessageBox()
        popup.setWindowTitle("提醒")
        popup.setText("請先確定有選擇輸入影片以及保存路徑后，再按此按鈕！")
        popup.exec_()
    
    def show_finish_popup(self,index):
        popup = QMessageBox()
        self.ui.MultiVideoProgessBar.setValue(index+1)
        popup.setWindowTitle("完成")
        popup.setText("影片已輸出完成，請確認資料夾") 
        popup.exec_()  
    
    def RedlightViolationButtonClick(self):
        if videos_path != '' and folder_path != '':
            self.qthread = ThreadTask_tf(self.ui,self)
            self.qthread.finished_signal.connect(self.show_finish_popup) 
            self.qthread.qthread_signal.connect(self.handleThreadSignal)
            self.qthread.progress_signal.connect(self.update_progress)
            self.qthread.start()
        else:
           self.show_waring_popup()
           
    def zebraButtonClick(self):
        if videos_path != '' and folder_path != '':
            self.qthread = ThreadTask_zebra(self.ui,self)
            self.qthread.finished_signal.connect(self.show_finish_popup) 
            self.qthread.qthread_signal.connect(self.handleThreadSignal)
            self.qthread.progress_signal.connect(self.update_progress)
            self.qthread.start()
        else:
           self.show_waring_popup()
    
    def handleThreadSignal(self):
        # 處理自定義訊號
        print("Thread task completed!")

    def closeEvent(self, event):
        if self.process is not None:  # 如果進程存在
            self.process.terminate()
        
    def update_progress(self, index, current, total, line):
        # This method will be called in the main thread
        self.ui.MultiVideoProgessBar.setMaximum(len(videos_path))
        self.ui.MultiVideoProgessBar.setValue(index)
        self.ui.SingleVideoProgressBar.setMaximum(total)
        self.ui.SingleVideoProgressBar.setValue(current)
        self.ui.FileInFiles.setText(f"已完成 {index}/{len(videos_path)} 部影片, 請耐心等待: 进度: {current}/{total}")
        
    def rotete_screen(self):
        video_filter = "影片文件 (*.mp4 *.avi *.mkv)"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        
        file_name, _ = QFileDialog.getOpenFileNames(
            parent=None,
            caption="选择影片文件",
            directory=".",
            filter=video_filter,
            options=options
            )
                  # start path
        print(file_name)
        global videos_path
        videos_path = file_name
        file_string = ''.join(videos_path)
        self.video = VideoFileClip(file_string)
        self.video_controller_rotate = video_controller_rotate(video_path=file_string,
                                         ui=self.ui)
        self.video_controller_rotate.play()
        self.rotate_cnt = 0

    def rotate_slider_moved(self):
        if self.rotate_cnt == 0:
            self.current_second = self.video_controller_rotate.pause()
        self.angle = self.ui.rotate_screen_slider.value()
        #self.video_controller_rotate.stop()
        self.angle = self.angle+45
        self.video.save_frame("temp.jpg", t = self.current_second)
        image = ImageClip("temp.jpg")
        image = image.rotate(self.angle)
        image.save_frame("temp.jpg")
        pixmap = QPixmap("temp.jpg")
        self.ui.rotate_screen.setPixmap(pixmap)
        self.rotate_cnt = 1

            
    def rotate_confirm(self):
        if self.angle:
            global rotate_angle
            rotate_angle = self.angle


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow_controller()
    MainWindow.show()
    sys.exit(app.exec_())
