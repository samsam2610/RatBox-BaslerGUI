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
            
            self.column_pos = 0
            self.row_pos = 0
            
            # Adding another StaticBoxSizer for system-wide controls
            system_box = wx.StaticBox(self.outer_panel, label="System Controls")
            system_sizer = wx.StaticBoxSizer(system_box, wx.VERTICAL)
            
            self.system_panel = wx.Panel(self.outer_panel)
            
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
            sizer.Add(self.index_ctrl, pos=(self.row_pos, self.column_pos + 1), flag=wx.EXPAND | wx.ALL, border=5)
            self.row_pos += 1 # Current row position = 5
            
            self.set_config_btn = wx.Button(self.system_panel, label="Set configuration system-wide")
            sizer.Add(self.set_config_btn, pos=(self.row_pos, self.column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.set_config_btn.Bind(wx.EVT_BUTTON, self.SetFolderAndFileConfigurationSystemWide)
            self.row_pos += 1 # Current row position = 6
           
            self.system_preview_btn = wx.Button(self.system_panel, label="Start System Preview")
            sizer.Add(self.system_preview_btn, pos=(self.row_pos, self.column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5) 
            self.system_preview_btn.Bind(wx.EVT_BUTTON, self.OnSystemPreview)
            self.row_pos += 1 # Current row position = 7
            
            self.system_capture_btn = wx.Button(self.system_panel, label="Start System Capture")
            sizer.Add(self.system_capture_btn, pos=(self.row_pos, self.column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5) 
            self.system_capture_btn.Bind(wx.EVT_BUTTON, self.OnSystemCapture)
            self.row_pos += 1 # Current row position = 8
            
            
            self.system_panel.SetSizer(sizer)
            self.system_panel.Layout()

            # Add to the main horizontal layout
            system_sizer.Add(self.system_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
            hbox.Add(system_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

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

    def SetFolderAndFileConfigurationSystemWide(self, event):
        # Check if the export name and folder is set
        export_folder = self.exportfolder_ctrl.GetValue()
        if not export_folder:
            wx.MessageBox("Please select an export folder.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        export_name = self.exportfile_ctrl.GetValue()
        if not export_name:
            wx.MessageBox("Please enter an export file name.", "Error", wx.OK | wx.ICON_ERROR)
            return

        for cam_panel in self.camera_panels:
            cam_panel.SetExportFolder(export_folder)
            cam_panel.SetAutoIndex(self.auto_index.GetValue())
            cam_panel.SetAppendDate(self.append_date.GetValue())
            cam_panel.SetFileName(export_name)

    def EnableSystemControls(self, value, preview=True):
        if value is True:
            self.exportfile_ctrl.Enable()
            self.select_folder_btn.Enable()
            self.append_date.Enable()
            self.auto_index.Enable()
            self.index_ctrl.Enable()
            self.set_config_btn.Enable()
            self.system_preview_btn.Enable()
            self.system_capture_btn.Enable()
        elif preview is True:
            self.exportfile_ctrl.Disable()
            self.select_folder_btn.Disable()
            self.append_date.Disable()
            self.auto_index.Disable()
            self.index_ctrl.Disable()
            self.set_config_btn.Disable()
            self.system_preview_btn.Enable()
            self.system_capture_btn.Disable()
        else:
            self.exportfile_ctrl.Disable()
            self.select_folder_btn.Disable()
            self.append_date.Disable()
            self.auto_index.Disable()
            self.index_ctrl.Disable()
            self.set_config_btn.Disable()
            self.system_preview_btn.Disable()
            self.system_capture_btn.Enable()
   
    def check_camera_connected_status(self):
        all_connected = all(panel.camera_connected for panel in self.camera_panels)
        self.camera_connected = all_connected
        return all_connected
    
    def check_camera_preview_status(self):
        all_preview = all(panel.preview_on for panel in self.camera_panels)
        self.preview_on = all_preview
        return all_preview
    
    def check_camera_capture_status(self):
        all_capturing = all(panel.capture_on for panel in self.camera_panels)
        self.capturing_on = all_capturing
        return all_capturing

    def OnSystemPreview(self, event):
        if self.check_camera_preview_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartPreview()
            self.system_preview_btn.SetLabel("Stop System Preview")
            self.EnableSystemControls(value=False, preview=True)
        else:
            for cam_panel in self.camera_panels:
                cam_panel.StopPreview()
            self.system_preview_btn.SetLabel("Start System Preview")
            self.EnableSystemControls(value=True, preview=False)

    def OnSystemCapture(self, event):
        if self.check_camera_capture_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartCapture()
            self.system_capture_btn.SetLabel("Stop System Capture")
            self.EnableSystemControls(value=False, preview=False)
        else:
            for cam_panel in self.camera_panels:
                cam_panel.StopCapture()
            self.system_capture_btn.SetLabel("Start System Capture")
            self.EnableSystemControls(value=True, preview=False)          

if __name__ == '__main__':
    app = wx.App()
    ex = SystemControl(number_of_cameras=2)
    app.MainLoop()