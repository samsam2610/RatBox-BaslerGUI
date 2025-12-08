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
import threading, multiprocessing
from pypylon import pylon
import nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode
import wx.lib.agw.floatspin as FS
import csv
from VideoRecordingSession import VideoRecordingSession
from InputEventHandler import ConfigurationEventPrinter
from ImagePanel import ImagePanel
from CameraController import CameraController
import pickle

# Testing system
def GenPulse(sampling_rate, frequency, duration=3600):
    print(f"Gen pulse at {frequency} Hz")
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return(5 * signal.square(2 * np.pi *frequency * t, duty=0.2))

def trigger_start_process(nidaq_samp_rate=5000, frequency=200):

    print("Child: starting NI task")

    # Context manager guarantees close()
    with nidaqmx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao1")
        ao_task.timing.cfg_samp_clk_timing(
            nidaq_samp_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=3600 * nidaq_samp_rate,
        )

        pulse = GenPulse(nidaq_samp_rate, frequency)
        ao_task.write(pulse, auto_start=True)
        ao_task.wait_until_done(timeout=nidaqmx.constants.WAIT_INFINITELY)


def trigger_start_process_continuous(nidaq_samp_rate=12000, frequency=200, ao_chan="myDAQ1/ao1"):
    """
    Run a continuous AO pulse until the process is externally terminated.
    Intended to be run in its OWN PROCESS.
    """
    print("[Child] Starting continuous NI-DAQ AO task")

    # We do NOT catch terminate here; terminate() is a hard kill.
    with nidaqmx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan(ao_chan)

        # Continuous sampling
        ao_task.timing.cfg_samp_clk_timing(
            rate=nidaq_samp_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        # Allow DAQ to loop the buffer
        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

        # Small buffer that repeats forever
        pulse = GenPulse(nidaq_samp_rate, frequency=frequency, duration=0.1)

        # Start continuous output
        ao_task.write(pulse, auto_start=True)
        print("[Child] AO continuous output running (Ctrl+C/terminate parent to stop)")

        # Keep process alive so AO keeps running
        try:
            while True:
                time.sleep(0.1)
        finally:
            # This only runs if process exits normally, not on terminate()
            print("[Child] Cleaning up AO task")


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
        path = Path(os.path.realpath(__file__))
        print(f"SystemControl.py location: {path.parent}")  # Print the directory of SystemControl.py
 
        
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
            self.cam_names = []
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
                self.cam_names.append(f"cam{i}")
            

            
            # Adding another StaticBoxSizer for system-wide controls
            system_box = wx.StaticBox(self.outer_panel, label="System Controls")
            system_sizer = wx.StaticBoxSizer(system_box, wx.VERTICAL)
            
            self.system_panel = wx.Panel(self.outer_panel)
            
            sizer = wx.GridBagSizer(5, 5)
            column_pos= 0
            row_pos= 0
            exportfile_ctrl_label = wx.StaticText(self.system_panel, label="Export file name:")
            sizer.Add(exportfile_ctrl_label, pos=(row_pos, column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.exportfile_ctrl = wx.TextCtrl(self.system_panel)
            sizer.Add(self.exportfile_ctrl, pos=(row_pos, column_pos+ 1), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.exportfile_ctrl.SetValue(self.get_last_filename())
            row_pos += 1 # Current row position = 1

            exportfolder_ctrl_label = wx.StaticText(self.system_panel, label="Export directory:")
            sizer.Add(exportfolder_ctrl_label, pos=(row_pos, column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)

            self.select_folder_btn = wx.Button(self.system_panel, label="Select folder")
            self.select_folder_btn.Bind(wx.EVT_BUTTON, self.OnSelectFolder)
            sizer.Add(self.select_folder_btn, pos=(row_pos, column_pos+ 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            row_pos += 1 # Current row position = 2
            
            self.exportfolder_ctrl = wx.TextCtrl(self.system_panel)
            sizer.Add(self.exportfolder_ctrl, pos=(row_pos, column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.exportfolder_ctrl.Disable()
            self.exportfolder_ctrl.SetValue(self.get_last_dir())
            row_pos += 1 # Current row position = 3

            self.append_date = wx.CheckBox(self.system_panel, label="Append date and time")
            sizer.Add(self.append_date, pos=(row_pos, column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.append_date.SetBackgroundColour(wx.NullColour)
            self.append_date.Bind(wx.EVT_CHECKBOX, self.OnAppendDate)
            self.append_date.SetValue(True)  
            row_pos += 1 # Current row position = 4

            self.auto_index = wx.CheckBox(self.system_panel, label="Auto index")
            sizer.Add(self.auto_index, pos=(row_pos, column_pos), span=(1, 1),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.auto_index.SetBackgroundColour(wx.NullColour)
            self.auto_index.Bind(wx.EVT_CHECKBOX, self.OnAutoIndex)
            self.auto_index.SetValue(True)  # Set checkbox to checked by default

            self.index_ctrl = wx.TextCtrl(self.system_panel)
            self.index_ctrl.SetValue(str(1))
            sizer.Add(self.index_ctrl, pos=(row_pos, column_pos+ 1), flag=wx.EXPAND | wx.ALL, border=5)
            row_pos += 1 # Current row position = 5
            
            self.set_config_btn = wx.Button(self.system_panel, label="Set configuration system-wide")
            sizer.Add(self.set_config_btn, pos=(row_pos, column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.set_config_btn.Bind(wx.EVT_BUTTON, self.SetFolderAndFileConfigurationSystemWideButton)
            row_pos += 1 # Current row position = 6
           
            self.system_preview_btn = wx.Button(self.system_panel, label="Start System Preview")
            sizer.Add(self.system_preview_btn, pos=(row_pos, column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5) 
            self.system_preview_btn.Bind(wx.EVT_BUTTON, self.OnSystemPreview)
            row_pos += 1 # Current row position = 7
            
            self.system_capture_btn = wx.Button(self.system_panel, label="Start System Capture")
            sizer.Add(self.system_capture_btn, pos=(row_pos, column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5) 
            self.system_capture_btn.Bind(wx.EVT_BUTTON, self.OnSystemCapture)
            row_pos += 1 # Current row position = 8
            
            
            self.system_panel.SetSizer(sizer)
            self.system_panel.Layout()

            # Add to the main horizontal layout
            system_sizer.Add(self.system_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
            # hbox.Add(system_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

            calibration_box = wx.StaticBox(self.outer_panel, label="Calibration Controls")
            calibration_sizer = wx.StaticBoxSizer(calibration_box, wx.VERTICAL)
            self.calibration_panel = wx.Panel(self.outer_panel)
            sizer = wx.GridBagSizer(5, 5)
            column_pos= 0
            row_pos= 0

            self.calibration_status_label = wx.StaticText(self.calibration_panel, label="Calibration Status: Not started")
            sizer.Add(self.calibration_status_label, pos=(row_pos, column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5)
            row_pos += 1 # Current row position = 1
            
            self.system_capture_calibration_btn = wx.Button(self.calibration_panel, label="Start System Calibration")
            sizer.Add(self.system_capture_calibration_btn, pos=(row_pos, column_pos), span=(1, 2),
                    flag=wx.EXPAND | wx.ALL, border=5)
            self.system_capture_calibration_btn.Bind(wx.EVT_BUTTON, self.OnSystemCalibrate)
            
            self.calibration_panel.SetSizer(sizer)
            self.calibration_panel.Layout()
            # Add to below the system controls
            calibration_sizer.Add(self.calibration_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
            right_column = wx.BoxSizer(wx.VERTICAL)
            right_column.Add(system_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
            right_column.Add(calibration_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

            # Add this single right column to your main hbox
            hbox.Add(right_column, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

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

    def SetFolderAndFileConfigurationSystemWideButton(self, event):
        self.set_folder_and_file_configuration_system_wide()

    def set_folder_and_file_configuration_system_wide(self, calibration=False):
        # Check if the cameras are busy
        if self.check_camera_preview_status() is True or self.check_camera_capture_status() is True:
            wx.MessageBox("Please stop all camera previews and captures before setting system-wide configuration.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Check if the export name and folder is set
        export_folder = self.exportfolder_ctrl.GetValue()
        if not export_folder:
            wx.MessageBox("Please select an export folder.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if calibration:
            export_name = "calib"
        else:
            export_name = self.exportfile_ctrl.GetValue()
            if not export_name:
                wx.MessageBox("Please enter an export file name.", "Error", wx.OK | wx.ICON_ERROR)
                return
            
        self.set_last_filename(export_name) # persist for next time

        self.check_camera_frame_rate_status()
        self.check_camera_trigger_status()

        for cam_panel in self.camera_panels:
            cam_panel.SetExportFolder(export_folder)
            cam_panel.SetAutoIndex(self.auto_index.GetValue())
            cam_panel.SetAppendDate(self.append_date.GetValue())
            cam_panel.SetFileName(export_name)
            cam_panel.SetTriggerMode(self.trigger_on)
            cam_panel.SetFrameRate(self.common_frame_rate)
            
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
    
    def check_camera_calibration_status(self):
        all_calibrating = all(panel.calibration_on for panel in self.camera_panels)
        self.calibrating_on = all_calibrating
        return all_calibrating
    
    def check_camera_trigger_status(self):
        modes = [panel.trigger_mode for panel in self.camera_panels]

        if not any(modes):
            self.trigger_on = False
        elif all(modes):
            self.trigger_on = True
        else:
            wx.MessageBox("Not all cameras have the same trigger mode. Setting system-wide trigger mode based on first camera.", "Warning", wx.OK | wx.ICON_WARNING)
            self.trigger_on = self.camera_panels[0].trigger_mode  # Default to first camera's mode
    
        return self.trigger_on

    def check_camera_frame_rate_status(self):
        frame_rates = [panel.GetFrameRate() for panel in self.camera_panels]

        if all(rate == frame_rates[0] for rate in frame_rates):
            self.frame_rate_consistent = True
            self.common_frame_rate = frame_rates[0]
        else:
            wx.MessageBox("Not all cameras have the same frame rate. Setting system-wide frame rate based on first camera.", "Warning", wx.OK | wx.ICON_WARNING)
            self.common_frame_rate = frame_rates[0]
            self.frame_rate_consistent = False

        return self.frame_rate_consistent

    def check_for_file_name_and_folder(self):
        # Check if all camera panels have export folder (exportfolder_ctrl) and filename set (exportfile_ctrl) are not empty
        for cam_panel in self.camera_panels:
            export_folder = cam_panel.exportfolder_ctrl.GetValue()
            export_file_name = cam_panel.exportfile_ctrl.GetValue()
            # If either is empty, return False
            if not export_folder or not export_file_name:
                return False
        return True
    
    def OnSystemPreview(self, event):
        if not self.check_camera_connected_status():
            wx.MessageBox("Please connect all cameras before starting preview.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_trigger_status() is None:
            wx.MessageBox("Please set the same trigger mode for all cameras before starting preview.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_preview_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartPreview()
            self.system_preview_btn.SetLabel("Stop System Preview")
            self.EnableSystemControls(value=False, preview=True)
            if self.trigger_on is True:
                self.proc = multiprocessing.Process(
                            target=trigger_start_process_continuous,
                            kwargs={"frequency": self.common_frame_rate},
                            daemon=True,
                            )
                self.proc.start()
                print(f"Spawned trigger process with PID {self.proc.pid}")
        else:
            if self.trigger_on is True:
                if self.proc.is_alive():
                    print(f"Terminating trigger process with PID {self.proc.pid}")
                    self.proc.terminate()
                    self.proc.join()  
                else:
                    print("Trigger process already terminated.")

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
        if self.check_camera_trigger_status() is None:
            wx.MessageBox("Please set the same trigger mode for all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if not self.check_for_file_name_and_folder():
            wx.MessageBox("Please set export folder and file name for all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_capture_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartCapture()
            
            self.system_capture_btn.SetLabel("Stop System Capture")
            self.EnableSystemControls(value=False, preview=False)
            self.system_capturing_on = True
            time.sleep(0.5)  # Give some time for the cameras to start writing
            if self.trigger_on is True:
                # Check to make sure both cameras are ready to reeceive triggers
                print("Waiting for cameras to be ready for system capture...")
                while self.check_camera_capture_status() is False:
                    time.sleep(0.1)
                print("Starting trigger process for system capture...")
                self.proc = multiprocessing.Process(
                            target=trigger_start_process_continuous,
                            kwargs={"frequency": self.common_frame_rate},
                            daemon=True,
                            )
                self.proc.start()
                print(f"Spawned trigger process with PID {self.proc.pid}")
        else:
            self.system_capturing_on = False
            if self.trigger_on is True:
                if self.proc.is_alive():
                    print(f"Terminating trigger process with PID {self.proc.pid}")
                    self.proc.terminate()
                    self.proc.join()  
                else:
                    print("Trigger process already terminated.")
            time.sleep(0.5)  # Give some time for the cameras to finalize writing
            for cam_panel in self.camera_panels:
                cam_panel.StopCapture()
            self.system_capture_btn.SetLabel("Start System Capture")
            self.EnableSystemControls(value=True, preview=False)          
    
            # ao_task.ao_channels.add_ao_voltage_chan("myDAQ1/ao1")

    # ------ Calibration methods ------
    def OnSystemCalibrate(self, event):
        if not self.check_camera_connected_status():
            wx.MessageBox("Please connect all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_preview_status():
            wx.MessageBox("Please stop all camera previews before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.check_camera_trigger_status() is None:
            wx.MessageBox("Please set the same trigger mode for all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if not self.check_for_file_name_and_folder():
            wx.MessageBox("Please set export folder and file name for all cameras before starting capture.", "Error", wx.OK | wx.ICON_ERROR)
            return
        # Setting capture toggle status
        self.recording_threads_status = []
        self.calibration_capture_toggle_status = True
        
        # Setup system calibration
        self.setup_calibration()
        
        if self.check_camera_calibration_status() is False:
            for cam_panel in self.camera_panels:
                cam_panel.StartCalibrateCapture()
                self.recording_threads_status.append(True)
            
            self.system_capture_calibration_btn.SetLabel("Stop System Calibration")
            self.EnableSystemControls(value=False, preview=False)
            self.system_capturing_calibration_on = True
            time.sleep(0.5)  # Give some time for the cameras to start writing
            if self.trigger_on is True:
                # Check to make sure both cameras are ready to reeceive triggers
                print("Waiting for cameras to be ready for system calibration...")
                while self.check_camera_calibration_status() is False:
                    time.sleep(0.1)
                print("Starting trigger process for system capture...")
                self.proc = multiprocessing.Process(
                            target=trigger_start_process_continuous,
                            kwargs={"frequency": self.common_frame_rate},
                            daemon=True,
                            )
                self.proc.start()
                print(f"Spawned trigger process with PID {self.proc.pid}")
            
            thread_name = f"Marker processing thread" 
            self.process_marker_thread = threading.Thread(target=self.process_marker_on_thread, name=thread_name)
            self.process_marker_thread.daemon = True
            self.process_marker_thread.start()
        else:
            self.system_capturing_calibration_on = False
            if self.trigger_on is True:
                if self.proc.is_alive():
                    print(f"Terminating trigger process with PID {self.proc.pid}")
                    self.proc.terminate()
                    self.proc.join()  
                else:
                    print("Trigger process already terminated.")
            time.sleep(0.5)  # Give some time for the cameras to finalize writing
            for cam_panel in self.camera_panels:
                cam_panel.StopCalibrateCapture()
            if self.process_marker_thread.is_alive():
                self.process_marker_thread.join()
            self.system_capture_calibration_btn.SetLabel("Start System Calibration")
            self.EnableSystemControls(value=True, preview=False)

    def load_calibration_settings(self, draw_calibration_board=False):
        from utils import load_config, get_calibration_board
        from pathlib import Path
        
        if self.check_for_file_name_and_folder() is False:
            wx.MessageBox("Please set export folder and file name for all cameras before starting calibration.", "Error", wx.OK | wx.ICON_ERROR)
            return None
        calibration_stats_message = 'Looking for config.toml directory ...'
        self.calibration_status_label.SetLabel(calibration_stats_message)
        print(calibration_stats_message)
        
        # Get current folder of this script
        path = Path(os.path.realpath(__file__))
        config_folder_path = Path(path.parent, 'config-files')
        # Navigate to the outer parent directory and join the filename
        config_toml_path = Path(config_folder_path, 'config.toml')
        # config_toml_path = os.path.normpath(str(path.parents/ 'config-files' / 'config.toml'))
        config_anipose = load_config(config_toml_path)
        calibration_stats_message = 'Found config.toml directory. Loading config ...'
        print(calibration_stats_message)
        
        calibration_stats_message = 'Successfully found and loaded config. Determining calibration board ...'
        self.calibration_status_label.SetLabel(calibration_stats_message)
        print(calibration_stats_message)

        self.board_calibration = get_calibration_board(config=config_anipose)
        calibration_stats_message = 'Successfully determined calibration board. Initializing camera calibration objects ...'
        self.calibration_status_label.SetLabel(calibration_stats_message)
        print(calibration_stats_message)

        self.rows_fname = os.path.join(self.exportfolder_ctrl.GetValue(), 'detections.pickle')
        self.calibration_out = os.path.join(self.exportfolder_ctrl.GetValue(), 'calibration.toml')
        
        board_dir = os.path.join(config_folder_path, 'board.png')
        if draw_calibration_board:
            numx, numy = self.board_calibration.get_size()
            size = numx*200, numy*200
            board = get_calibration_board(config_anipose)
            board_image = board.board.generateImage(size)
            cv2.imwrite(board_dir, board_image)
        
        return config_anipose
    
    def setup_calibration(self, override=False):
        """
        Method: setup_calibration

        This method initializes the calibration process. It performs the following steps:

        1. Initializes the calibration process by updating the status text.
        2. Looks for the config.toml directory if debug_mode is enabled.
        3. Loads the config file and determines the calibration board.
        4. Initializes camera calibration objects.
        5. Records frame sizes and initializes camera objects.
        6. Configures calibration buttons and toggle statuses.
        7. Clears previous calibration files.
        8. Sets calibration duration parameter.
        9. Creates a shared queue to store frames.
        10. Updates the boolean flag for detection updates.
        11. Synchronizes camera capture time using threading.Barrier.
        12. Creates output file names for the calibration videos.
        13. Sets frame sizes for the cameras.
        14. Starts the calibration process by recording frames, processing markers, and calibrating.

        Parameters:
        - None

        Return Type:
        - None
        """
        self.calibration_status_label.SetLabel('Initializing calibration process...')
        config_anipose = self.load_calibration_settings()
        
        self.calibration_status_label.SetLabel('Initializing camera calibration objects ...')
        from aniposelib.cameras import CameraGroup
        import re
        
        # Get cam names from the config file
        cam_regex = config_anipose['triangulation']['cam_regex']
        cam_names = []
        for name in self.cam_names:
            match = re.match(cam_regex, name)
            if match:
                cam_names.append(match.groups()[0])
        
        self.cgroup = CameraGroup.from_names(cam_names)
        self.calibration_status_label.SetLabel('Initialized camera object.')
        self.frame_count = []
        self.all_rows = []

        self.calibration_status_label.SetLabel('Cameras found. Recording the frame sizes')
        
        self.calibration_capture_toggle_status = False
        self.calibration_toggle_status = False
        
        frame_sizes = []
        self.frame_times = []
        self.previous_frame_count = []
        self.current_frame_count = []
        self.frame_process_threshold = 2
        self.queue_frame_threshold = 1000
        
        if override:
            # Check available detection file, if file available will delete it (for now)
            self.clear_calibration_file(self.rows_fname)
            self.clear_calibration_file(self.calibration_out)
            self.rows_fname_available = False
        else:
            self.rows_fname_available = os.path.exists(self.rows_fname)
            
        # Set calibration parameter
        # result = self.set_calibration_duration()
        # if result == 0:
            # return
        
        self.error_list = []
        # Create a shared queue to store frames
        self.frame_queue = queue.Queue(maxsize=self.queue_frame_threshold)

        # Boolean for detections.pickle is updated
        self.detection_update = False

        # create output file names
        self.vid_file = []
        self.base_name = []
        self.cam_name_no_space = []
        self.current_all_rows = []
        for cam_panel in self.camera_panels:
            frame_sizes.append(cam_panel.get_image_dimensions()) # change thi
            self.frame_count.append(1)
            self.all_rows.append([])
            self.previous_frame_count.append(0)
            self.current_frame_count.append(0)
            self.frame_times.append([])
            self.current_all_rows.append([])

            cam_panel.SetupCalibration(board_calibration=self.board_calibration,
                                       all_rows=self.all_rows,
                                       current_all_rows=self.current_all_rows)

        self.set_folder_and_file_configuration_system_wide(calibration=True)             
 
        self.calibration_status_label.SetLabel('Setting the frame sizes...')
        self.cgroup.set_camera_sizes_images(frame_sizes=frame_sizes)
        self.calibration_status_label.SetLabel('Prepping done. Ready to capture calibration frames...')
        self.calibration_status_label['bg'] = 'yellow'

        self.vid_start_time = time.perf_counter()
        
        self.recording_threads = []
        self.calibrating_thread = None

    def process_marker_on_thread(self):
        """
        Process marker on a separate thread.

        This method retrieves frame information from the frame queue and processes it. The frames are grouped by thread ID
        and stored in a dictionary called frame_groups. The method continuously loops until the calibration_capture_toggle_status
        is True or the frame queue is not empty.

        Parameters:
        - self: The current instance of the class.

        Returns:
        This method does not return any value.

        Raises:
        This method may raise an exception when an error occurs during processing.

        Example usage:
        process_marker_on_thread()
        """
        from src.aniposelib.boards import extract_points, merge_rows, reverse_extract_points, reverse_merge_rows
        
        frame_groups = {}  # Dictionary to store frame groups by thread_id
        frame_counts = {}  # array to store frame counts for each thread_id
        
        try:
            while any(thread is True for thread in self.recording_threads_status):
                # Retrieve frame information from the queue
                frame, thread_id, frame_count, capture_time = self.frame_queue.get()
                if thread_id not in frame_groups:
                    frame_groups[thread_id] = []  # Create a new group for the thread_id if it doesn't exist
                    frame_counts[thread_id] = 0

                # Append frame information to the corresponding group
                frame_groups[thread_id].append((frame, frame_count, capture_time))
                frame_counts[thread_id] += 1
                self.frame_acquired_count_label[thread_id]['text'] = f'{frame_count}'
                self.vid_out[thread_id].write(frame)
                
                # Process the frame group (frames with the same thread_id)
                # dumping the mix and match rows into detections.pickle to be pickup by calibrate_on_thread
                if all(count >= self.frame_process_threshold for count in frame_counts.values()):
                    with open(self.rows_fname, 'wb') as file:
                        pickle.dump(self.all_rows, file)
                    self.rows_fname_available = True
                    # Clear the processed frames from the group
                    frame_groups = {}
                    frame_count = {}
            
            # Process the remaining frames in the queue
            while not self.frame_queue.empty():
                print('Processing remaining frames in the queue')
                frame, thread_id, frame_count, capture_time = self.frame_queue.get()
                if thread_id not in frame_groups:
                    frame_groups[thread_id] = []
                    frame_counts[thread_id] = 0
                frame_groups[thread_id].append((frame, frame_count, capture_time))
                frame_counts[thread_id] += 1
                self.frame_acquired_count_label[thread_id]['text'] = f'{frame_count}'
                self.vid_out[thread_id].write(frame)
                
                if all(count >= self.frame_process_threshold for count in frame_counts.values()):
                    with open(self.rows_fname, 'wb') as file:
                        pickle.dump(self.all_rows, file)
                    self.rows_fname_available = True
                    print('Dumped rows into detections.pickle')
                    
                    frame_groups = {}
                    frame_count = {}
            
            # Clear the frame queue
            self.frame_queue.queue.clear()
            print('Cleared frame queue')
            
            if all(thread is False for thread in self.recording_threads_status):
                print('Terminating thread')
                self.toggle_calibration_capture(termination=True)
                
        except Exception as e:
            print("Exception occurred:", type(e).__name__, "| Exception value:", e, "| Thread ID:", thread_id,
                  "| Frame count:", frame_count, "| Capture time:", capture_time, "| Traceback:",
                  ''.join(traceback.format_tb(e.__traceback__)))

    ## ------ Configuration persistence methods ------
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