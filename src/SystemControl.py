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
            
            self.column_pos = self.number_of_cameras + 1
            self.row_pos = 0
            
            # Adding another StaticBoxSizer for system-wide controls
            system_box = wx.StaticBox(self.outer_panel, label="System Controls")
            system_sizer = wx.StaticBoxSizer(system_box, wx.VERTICAL)
            system_panel = wx.Panel(self.outer_panel)
            
            sizer = wx.GridBagSizer(5, 5)
            exportfile_ctrl_label = wx.StaticText(self.system_panel, label="Export file name:")
            sizer.Add(exportfile_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.exportfile_ctrl = wx.TextCtrl(self.system_panel)
            sizer.Add(self.exportfile_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.row_pos += 1 # Current row position = 1

            exportfolder_ctrl_label = wx.StaticText(self.system_panel, label="Export directory:")
            sizer.Add(exportfolder_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)

            self.select_folder_btn = wx.Button(self.system_panel, label="Select folder")
            self.select_folder_btn.Bind(wx.EVT_BUTTON, self.OnSelectFolder)
            sizer.Add(self.select_folder_btn, pos=(self.row_pos, self.column_pos + 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.row_pos += 1 # Current row position = 2

            self.exportfolder_ctrl = wx.TextCtrl(self.system_panel)
            sizer.Add(self.exportfolder_ctrl, pos=(self.row_pos, self.column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.exportfolder_ctrl.Disable()
            self.row_pos += 1 # Current row position = 3

            self.append_date = wx.CheckBox(self.system_panel, label="Append date and time")
            sizer.Add(self.append_date, pos=(self.row_pos, self.column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.append_date.SetBackgroundColour(wx.NullColour)
            self.append_date.Bind(wx.EVT_CHECKBOX, self.OnAppendDate)
            self.append_date.SetValue(True)  
            self.row_pos += 1 # Current row position = 4

            self.auto_index = wx.CheckBox(self.system_panel, label="Auto index")
            sizer.Add(self.auto_index, pos=(self.row_pos, self.column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.auto_index.SetBackgroundColour(wx.NullColour)
            self.auto_index.Bind(wx.EVT_CHECKBOX, self.OnAutoIndex)
            self.auto_index.SetValue(True)  # Set checkbox to checked by default

            self.index_ctrl = wx.TextCtrl(self.system_panel)
            self.index_ctrl.SetValue(str(1))
            sizer.Add(self.index_ctrl, pos=(20, self.column_pos + 1), flag=wx.EXPAND | wx.ALL, border=5)
            self.row_pos += 1 # Current row position = 5

            
            # Add to the main horizontal layout
            hbox.Add(self.system_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        self.outer_panel.SetSizer(hbox)
        hbox.Layout()
        
    def OnCloseWindow(self, event):
        print("Closing application...")
        for p in getattr(self, "camera_panels", []):
            p.Destroy()
        self.Destroy()
    
    def OnSelectFolder(self, event):
        dlg = wx.DirDialog(None, "Choose input directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.ShowModal()
        self.exportfolder_ctrl.SetValue(dlg.GetPath())
    
    def OnAppendDate(self, event):
        self.append_date_flag = self.append_date.GetValue()
        
    def OnAutoIndex(self, event):
        if self.camera_connected is True:
            self.auto_index_on = self.auto_index.GetValue()
            if self.auto_index_on is True:
                self.index_ctrl.Disable()
                self.current_index = int(self.index_ctrl.GetValue())
            else:
                self.index_ctrl.Enable()



if __name__ == '__main__':
    app = wx.App()
    ex = SystemControl(number_of_cameras=2)
    app.MainLoop()