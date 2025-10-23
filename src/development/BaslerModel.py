import numpy as np
import cv2
from pypylon import pylon
from VideoRecordingSession import VideoRecordingSession
from InputEventHandler import ConfigurationEventPrinter

class BaslerModel:
    def __init__(self, cam_num=1):
        self.cam_num = cam_num
        self.camera = []
        self.camera_connected = []
        self.capture_on = []
        self.preview_on = []
        for _ in range(self.cam_num):
            self.camera_connected.append(False)
            self.capture_on.append(False)
            self.preview_on.append(False)
            