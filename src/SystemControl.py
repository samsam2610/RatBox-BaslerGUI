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
from ImagePanel import ImagePanel
from CameraController import CameraController

class SystemControl(wx.Frame):
    def __init__(self, parent, number_of_cameras=2, *args, **kwargs):
        super(SystemControl, self).__init__(*args, **kwargs)
        self.number_of_cameras = number_of_cameras
        if self.number_of_cameras > 1:
            self.is_multi_cam = True
        else:
            self.is_multi_cam = False
            
        # Initialize UI
        self.InitSystemUI()
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.Centre()
        self.Show()
 
        
    def InitSystemUI(self):
        if self.is_multi_cam is False:
            self.camera1 = CameraController(self, cam_index=0, cam_details="Camera 1", multi_cam=False, column_pos=0, row_pos=0)
            self.camera1.InitUI()

if __name__ == '__main__':
    app = wx.App()
    ex = SystemControl(None, number_of_cameras=1)
    app.MainLoop()