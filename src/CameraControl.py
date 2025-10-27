import queue
import wx
import os
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from scipy.optimize import curve_fit
import numpy as np
import cv2
import datetime
import time
import threading
from pypylon import pylon
import wx.lib.agw.floatspin as FS
import csv
from VideoRecordingSession import VideoRecordingSession
from InputEventHandler import ConfigurationEventPrinter

class CameraControl(wx.Panel):
    def __init__(self, parent, cam_index, cam_details):
        wx.Panel.__init__(self, parent)
        self.parent = parent
        self.cam_index = cam_index
        self.cam_details = cam_details
        self.camera = None
        self.frame_queue = queue.Queue()
        self.is_running = False
        self.video_session = None
        self.init_camera()
        self.create_ui()