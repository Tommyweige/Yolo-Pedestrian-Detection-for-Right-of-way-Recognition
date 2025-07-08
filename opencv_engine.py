# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 18:17:35 2023

@author: tommy
"""
import cv2

def getvideoinfo(video_path): 

    videoinfo = {}
    vc = cv2.VideoCapture(video_path)
    videoinfo["vc"] = vc
    videoinfo["fps"] = vc.get(cv2.CAP_PROP_FPS)
    videoinfo["frame_count"] = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    videoinfo["width"] = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoinfo["height"] = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return videoinfo
