import queue
import wx
import os
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from scipy import signal
from scipy.optimize import curve_fit
import numpy as np
import cv2
import datetime
import time
import threading
from pypylon import pylon
import nidaqmx
from nidaqmx.constants import AcquisitionType
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
        
        trigger_thread_obj = None

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
            self.exportfile_ctrl.SetValue(self.get_last_filename())
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
            self.exportfolder_ctrl.SetValue(self.get_last_dir())
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
        
        self.EnableSystemControls(value=True, preview=False)
        
    def OnCloseWindow(self, event):
        print("Closing application...")
        for p in getattr(self, "camera_panels", []):
            p.Destroy()
        self.Destroy()
    
    def OnSelectFolder(self, event):
        # Pre-fill from the text box if it has a valid path; otherwise from config
        current_val = self.exportfolder_ctrl.GetValue()
        start_dir = current_val if os.path.isdir(current_val) else self.get_last_dir()

        with wx.DirDialog(
            self, "Choose input directory",
            defaultPath=start_dir,
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.exportfolder_ctrl.SetValue(path)
                self.set_last_dir(path)  # persist for next time

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
        # Check if the cameras are busy
        if self.check_camera_preview_status() is True or self.check_camera_capture_status() is True:
            wx.MessageBox("Please stop all camera previews and captures before setting system-wide configuration.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if the export name and folder is set
        export_folder = self.exportfolder_ctrl.GetValue()
        if not export_folder:
            wx.MessageBox("Please select an export folder.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        export_name = self.exportfile_ctrl.GetValue()
        if not export_name:
            wx.MessageBox("Please enter an export file name.", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.set_last_filename(export_name)
        for cam_panel in self.camera_panels:
            cam_panel.SetExportFolder(export_folder)
            cam_panel.SetAutoIndex(self.auto_index.GetValue())
            cam_panel.SetAppendDate(self.append_date.GetValue())
            cam_panel.SetFileName(export_name)

    def EnableSystemControls(self, value, preview=True, startup=False):
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

    def check_for_file_name_and_folder(self):
        # Check if all camera panels have export folder (exportfolder_ctrl) and filename set (exportfile_ctrl) are not empty
        for cam_panel in self.camera_panels:
            export_folder = cam_panel.exportfolder_ctrl.GetValue()
            export_file_name = cam_panel.exportfile_ctrl.GetValue()
            # If either is empty, return False
            if not export_folder or not export_file_name:
                return False
        return True
    
    def GenPulse(self,samp_rate):
        for cam_panel in self.camera_panels:
            freq = cam_panel.framerate
        t = np.linspace(0,1,samp_rate,endpoint=False)
        return(5 * signal.square(2 * np.pi *freq * t,duty=0.2))
    
    def OnSystemPreview(self, event):
        if not self.check_camera_connected_status():
            wx.MessageBox("Please connect all cameras before starting preview.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_preview_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartPreview()
            self.system_preview_btn.SetLabel("Stop System Preview")
            self.EnableSystemControls(value=False, preview=True)
            self.trigger_thread_obj = threading.Thread(target=self.trigger_thread)
            self.trigger_thread_obj.start()
        else:
            for cam_panel in self.camera_panels:
                cam_panel.StopPreview()
            self.system_preview_btn.SetLabel("Start System Preview")
            self.EnableSystemControls(value=True, preview=False)
    
    def OnSystemCapture(self, event):
        if not self.check_camera_connected_status():
            wx.MessageBox("Please connect all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_preview_status():
            wx.MessageBox("Please stop all camera previews before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if not self.check_for_file_name_and_folder():
            wx.MessageBox("Please set export folder and file name for all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_capture_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartCapture()
            self.system_capture_btn.SetLabel("Stop System Capture")
            self.EnableSystemControls(value=False, preview=False)
            self.trigger_thread_obj = threading.Thread(target=self.trigger_thread)
            self.trigger_thread_obj.start()
        else:
            for cam_panel in self.camera_panels:
                cam_panel.StopCapture()
            self.system_capture_btn.SetLabel("Start System Capture")
            self.EnableSystemControls(value=True, preview=False)          
    
    def trigger_thread(self):
        if any(cam_panel.trigger_mode is True for cam_panel in self.camera_panels):
            self.nidaq_samp_rate = 12000
            while self.check_camera_preview_status() or self.check_camera_capture_status():
                with nidaqmx.Task() as ao_task:
                    ao_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao1")
                    ao_task.timing.cfg_samp_clk_timing(self.nidaq_samp_rate, sample_mode=AcquisitionType.CONTINUOUS)
                    pulse = self.GenPulse(self.nidaq_samp_rate)
                    ao_task.write(pulse)
                    #print(f"{pulse}")
                    ao_task.start()
            ao_task.close()
            with nidaqmx.Task() as ao_end:
                ao_end.ao_channels.add_ao_voltage_chan("myDAQ1/ao1")
                ao_end.write(0.0)

    def get_config(self):
        APP_NAME = "BaslerCamGUI"  # any unique name
        # Stores under a per-user location (AppData on Windows, ~/.config on Linux, etc.)
        return wx.Config(APP_NAME)
    
    def set_last_dir(self, path):
        cfg = self.get_config()
        cfg.Write("last_export_dir", path)
        cfg.Flush()

    def get_last_dir(self, fallback=None):
        cfg = self.get_config()
        last_dir = cfg.Read("last_export_dir", "")
        if last_dir and os.path.isdir(last_dir):
            return last_dir
        return fallback or os.getcwd()
    
    def set_last_filename(self, filename):
        cfg = self.get_config()
        cfg.Write("last_export_filename", filename)
        cfg.Flush()
        
    def get_last_filename(self, fallback=None):
        cfg = self.get_config()
        last_filename = cfg.Read("last_export_filename", "")
        if last_filename:
            return last_filename
        return fallback or "subject_name"

if __name__ == '__main__':
    app = wx.App()
    ex = SystemControl(number_of_cameras=2)
    app.MainLoop()