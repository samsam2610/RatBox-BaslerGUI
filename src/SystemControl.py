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
    def __init__(self, number_of_cameras=2, *args, **kwargs):
        super().__init__(None,*args, **kwargs)
        # Outer panel so we can attach a sizer
        self.outer_panel = wx.Panel(self)
        
        self.camera_panels = []
        
        self.number_of_cameras = number_of_cameras
        if self.number_of_cameras > 1:
            print("Multi-camera mode activated.")
            self.is_multi_cam = True
        else:
            print("Single camera mode activated.")
            self.is_multi_cam = False
            
        # Initialize UI
        self.InitSystemUI()
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.outer_panel.Layout()
        self.outer_panel.Fit()
        self.Fit()
        self.Centre()
        self.Show()
 
        
    def InitSystemUI(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        if self.is_multi_cam is False:
            print("Initializing single camera UI...")
            self.camera_panel = CameraController(self.outer_panel, cam_index=0, cam_details="Camera 1", multi_cam=False, column_pos=0, row_pos=0)
            self.camera_panel.InitUI()
            self.SetTitle("Single Camera Control")
            
            # Put each camera panel in a StaticBoxSizer for visual grouping
            box = wx.StaticBox(self.outer_panel, label="Camera 1 Controls")
            static_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            static_sizer.Add(self.camera_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

            # Add to the main horizontal layout
            hbox.Add(static_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

            self.camera_panels.append(self.camera_panel)
        else:
            for i in range(self.number_of_cameras):
                print(f"Initializing camera {i + 1} UI...")
                camera_panel = CameraController(self.outer_panel, cam_index=i, cam_details=f"Camera {i + 1}", multi_cam=True, column_pos=i, row_pos=0)
                camera_panel.InitUI()
                self.SetTitle(f"Multi-Camera Control - {self.number_of_cameras} Cameras")

                # Put each camera panel in a StaticBoxSizer for visual grouping
                box = wx.StaticBox(self.outer_panel, label=f"Camera {i + 1} Controls")
                static_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
                static_sizer.Add(camera_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

                # Add to the main horizontal layout
                hbox.Add(static_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

                self.camera_panels.append(camera_panel)

        self.outer_panel.SetSizer(hbox)
        hbox.Layout()
        
    
    def OnCloseWindow(self, event):
        print("Closing application...")
        for p in getattr(self, "camera_panels", []):
            p.Destroy()
        self.Destroy()

if __name__ == '__main__':
    app = wx.App()
    ex = SystemControl(number_of_cameras=2)
    app.MainLoop()