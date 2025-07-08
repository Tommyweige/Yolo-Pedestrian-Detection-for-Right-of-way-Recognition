# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 18:22:25 2023

@author: tommy
"""

from PyQt5 import QtCore 
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer 
from moviepy.editor import *
from opencv_engine import getvideoinfo

# videoplayer_state_dict = {
#  "stop":0,   
#  "play":1,
#  "pause":2     
# }

class video_controller_rotate(object):
    def __init__(self, video_path, ui):
        self.video_path = video_path
        self.ui = ui
        self.qpixmap_fix_width = 1280 # 16x9 = 1920x1080 = 1280x720 = 800x450
        self.qpixmap_fix_height = 720
        self.current_frame_no = 0
        self.videoplayer_state = "pause"
        self.init_video_info()
        self.set_video_player()

    def init_video_info(self):
        videoinfo = getvideoinfo(self.video_path)
        self.vc = videoinfo["vc"] 
        self.video_fps = videoinfo["fps"] 
        self.video_total_frame_count = videoinfo["frame_count"] 
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"] 
        self.ui.slider_videoframe.setRange(0, self.video_total_frame_count-1)
        #self.ui.slider_videoframe.valueChanged.connect(self.getslidervalue)

    def set_video_player(self):
        self.timer=QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        # self.timer.start(1000//self.video_fps) # start Timer, here we set '1000ms//Nfps' while timeout one time
        self.timer.start(1) # but if CPU can not decode as fast as fps, we set 1 (need decode time)

    def set_current_frame_no(self, frame_no):
        self.vc.set(1, frame_no) # bottleneck
        if self.video_total_frame_count == frame_no+2:
           self.stop()
           self.pause()

    def __get_next_frame(self):
        ret, frame = self.vc.read()
        #self.setslidervalue(self.current_frame_no)
        return frame

    def __update_label_frame(self, frame):       
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        if self.qpixmap.width()/16 >= self.qpixmap.height()/9: # like 1600/16 > 90/9, height is shorter, align width
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else: # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.rotate_screen.setPixmap(self.qpixmap)
        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.rotate_screen.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center
        
    def stop(self):
        self.videoplayer_state = "stop"
    def play(self):
        self.videoplayer_state = "play"
    def pause(self):
        self.videoplayer_state = "pause"
        self.current_second = self.current_frame_no / self.video_fps
        return self.current_second
        
    def timer_timeout_job(self):
        if (self.videoplayer_state == "play"):
            if self.current_frame_no >= self.video_total_frame_count-1:
            #self.videoplayer_state = "pause"
                self.current_frame_no = 0 # auto replay
                self.set_current_frame_no(self.current_frame_no)
            else:
                self.current_frame_no += 1

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 0
            self.set_current_frame_no(self.current_frame_no)
            

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
            self.set_current_frame_no(self.current_frame_no)
            return 0

        frame = self.__get_next_frame() 
        self.__update_label_frame(frame)
