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
from typing import Callable, Optional, List, Dict
from ImagePanel import ImagePanel

from dataclasses import dataclass
from aniposelib.boards import CharucoBoard

@dataclass
class CameraSettings:
    serial: str
    # core
    target_fps: Optional[float] = 200.0
    exposure_us: Optional[float] = None      # ExposureTime is in microseconds on most Baslers
    gain: Optional[float] = None
    # ROI
    width: Optional[int] = None
    height: Optional[int] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    # GPIO
    configure_line3_as_input: bool = True
    
    
class CameraController(wx.Panel):
    output_string_1 = ""
    output_string_2 = ""
    update_ratio = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    lock = threading.Lock()

    auto_index_on = True
    current_index = 0
    append_date_flag = True
    measurement_interval = 2
    sequence_length = 1
    current_step = 0
    last_capture_time = 0
    frames_to_capture = 360
    selected_mode = 2
    encoding_mode = 0
    time_to_next = 0

    
    auto_exposure_on = False
    auto_gain_on = False
    preview_on = False
    capture_on = False
    framerate = 120
    exposure = 7
    
    gain = 0
    cameras_list = []
    capture_sequence_timer = None
    capture_status_timer = None
    camera_connected = False
    camera = []

    frame_width = 1440 
    frame_height = 1088
    offset_x = 16
    offset_y = 0
    
    max_frame_width = 1456
    max_frame_height = 1088

    roi_on = False
    roi_x = 0
    roi_y = 0
    roi_width = 10
    roi_height = 10
    min_gray_val = 5
    preview_thread_obj = None
    capture_thread_obj = None
    process_thread_obj = None
    max_contrast = 0.8
    
    video_session = VideoRecordingSession(cam_num=0)

    def __init__(self, parent, cam_index, cam_details, multi_cam=False, column_pos=0, row_pos=0, trigger_mode: bool=True, *args, **kwargs):

        self.cam_index = cam_index
        self.selected_camera = cam_index
        self.cam_details = cam_details
        self.column_pos = column_pos
        self.row_pos = row_pos
        self.is_multi_cam = multi_cam
        self.parent = parent
        self.trigger_mode = trigger_mode
        self.calibration_on = False
        self.SetTriggerModeLabel()
        super().__init__(parent)

    def InitUI(self):
        print("Initializing CameraController UI...")
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        cameras = []
        for device in devices:
            cameras.append(device.GetModelName() + "_" + device.GetSerialNumber())
            self.cameras_list.append({"name": device.GetModelName(),
                                      "serial": device.GetSerialNumber()})

        panel = self
        # self.SetTitle('Basler CAM GUI')
        sizer = wx.GridBagSizer(5, 5)

        selected_ctrl_label = wx.StaticText(panel, label="Selected camera:")
        sizer.Add(selected_ctrl_label, pos=(self.row_pos, self.column_pos),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 1
        
        self.cam_combo = wx.ComboBox(panel, choices=cameras)
        sizer.Add(self.cam_combo, pos=(self.row_pos, self.column_pos),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.cam_combo.Bind(wx.EVT_COMBOBOX, self.OnCamCombo)
        self.cam_combo.SetSelection(self.selected_camera)

        self.connect_btn = wx.Button(panel, label="Connect")
        self.connect_btn.Bind(wx.EVT_BUTTON, self.OnConnect)
        sizer.Add(self.connect_btn, pos=(self.row_pos, self.column_pos + 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 2

        self.refresh_btn = wx.Button(panel, label="Refresh list")
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.OnRefreshList)
        sizer.Add(self.refresh_btn, pos=(self.row_pos, self.column_pos),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.preview_btn = wx.Button(panel, label="Preview START")
        self.preview_btn.Bind(wx.EVT_BUTTON, self.OnPreview)
        sizer.Add(self.preview_btn, pos=(self.row_pos, self.column_pos + 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 3

        self.Window = ImagePanel(panel)
        # self.Window.SetSize((480, 360))
        # self.Window.Fit()
        row_span = 4
        sizer.Add(self.Window, pos=(self.row_pos, self.column_pos), span=(row_span, 3),
                  flag=wx.LEFT | wx.TOP | wx.EXPAND, border=5)
        self.row_pos += (row_span+1)  # Current row position = 8

        self.trigger_ctrl_label = wx.StaticText(panel, label="Trigger mode:")
        sizer.Add(self.trigger_ctrl_label, pos=(self.row_pos, self.column_pos),
                  flag=wx.EXPAND | wx.ALL, border=5)
        trigger_mode_selection = ["On", "Off"]
        self.trigger_mode_combo = wx.ComboBox(panel, choices=trigger_mode_selection)
        sizer.Add(self.trigger_mode_combo, pos=(self.row_pos, self.column_pos + 1), flag=wx.ALL, border=5)
        self.trigger_mode_combo.Bind(wx.EVT_COMBOBOX, self.OnTriggerModeCombo)
        self.trigger_mode_combo.SetSelection(self.trigger_mode_label)
        self.row_pos += 1 # Current row position = 22
        
        self.framerate_ctrl_label = wx.StaticText(panel, label="Framerate (Hz):")
        sizer.Add(self.framerate_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.framerate_slider = FS.FloatSpin(panel, -1,  min_val=100, max_val=500,
                                             size=(140, -1), increment=1.0, digits=0,
                                             value=0.1, agwStyle=FS.FS_LEFT)
        self.framerate_slider.Bind(FS.EVT_FLOATSPIN, self.FramerteSliderScroll)
        sizer.Add(self.framerate_slider, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 9

        self.exposure_ctrl_label = wx.StaticText(panel, label="Exposure (us):")
        sizer.Add(self.exposure_ctrl_label, pos=(self.row_pos, self.column_pos),  span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.exposure_slider = FS.FloatSpin(panel, -1,  min_val=1000, max_val=5000,
                                            size=(140, -1), increment=100, digits=0,
                                            value=1000, agwStyle=FS.FS_LEFT)
        self.exposure_slider.Bind(FS.EVT_FLOATSPIN, self.ExposureSliderScroll)
        sizer.Add(self.exposure_slider, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 10

        self.gain_ctrl_label = wx.StaticText(panel, label="Gain (dB):")
        sizer.Add(self.gain_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.gain_slider = FS.FloatSpin(panel, -1,  min_val=0, max_val=1,
                                        size=(140, -1), increment=0.01, value=0.1, digits=0,
                                        agwStyle=FS.FS_LEFT)
        self.gain_slider.Bind(FS.EVT_FLOATSPIN, self.GainSliderScroll)
        self.gain_slider.SetFormat("%f")
        self.gain_slider.SetDigits(3)
        sizer.Add(self.gain_slider, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 11

        self.set_auto_exposure = wx.CheckBox(panel, label="Auto Exposure")
        sizer.Add(self.set_auto_exposure, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.set_auto_exposure.SetBackgroundColour(wx.NullColour)
        self.set_auto_exposure.Bind(wx.EVT_CHECKBOX, self.OnEnableAutoExposure)

        self.set_auto_gain = wx.CheckBox(panel, label="Auto Gain")
        sizer.Add(self.set_auto_gain, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.set_auto_gain.SetBackgroundColour(wx.NullColour)
        self.set_auto_gain.Bind(wx.EVT_CHECKBOX, self.OnEnableAutoGain)
        self.row_pos += 1 # Current row position = 12

        self.offset_x_ctrl_label = wx.StaticText(panel, label="Offset X:")
        sizer.Add(self.offset_x_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.offset_x_ctrl = FS.FloatSpin(panel, -1,  min_val=1, max_val=self.max_frame_width - self.frame_width,
                                          size=(140, -1), increment=4, value=1, digits=0,
                                          agwStyle=FS.FS_LEFT)
        sizer.Add(self.offset_x_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.offset_x_ctrl.Bind(FS.EVT_FLOATSPIN, self.OnSetOffsetX)
        self.offset_x_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnSetOffsetX)
        self.row_pos += 1 # Current row position = 13

        # self.offset_x_ctrl_slider = wx.Slider(panel, value=1, minValue=1, maxValue=self.max_frame_width - self.frame_width,
        #                                      style=wx.SL_HORIZONTAL)
        # sizer.Add(self.offset_x_ctrl_slider, pos=(self.row_pos, self.column_pos), span=(1, 2),
        #           flag=wx.EXPAND | wx.ALL, border=5)
        # self.offset_x_ctrl_slider.Bind(wx.EVT_SLIDER, self.OnSetOffsetX)
        # self.row_pos += 1 # Current row position = 14

        self.offset_y_ctrl_label = wx.StaticText(panel, label="Offset Y:")
        sizer.Add(self.offset_y_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.offset_y_ctrl = FS.FloatSpin(panel, -1,  min_val=1, max_val=self.max_frame_height - self.frame_height,
                                    size=(140, -1), increment=4, value=1, digits=0,
                                    agwStyle=FS.FS_LEFT)
        sizer.Add(self.offset_y_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.offset_y_ctrl.Bind(FS.EVT_FLOATSPIN, self.OnSetOffsetY)
        self.offset_y_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnSetOffsetY)
        self.row_pos += 1 # Current row position = 14

        self.width_ctrl_label = wx.StaticText(panel, label="Width:")
        sizer.Add(self.width_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.width_ctrl = FS.FloatSpin(panel, -1,  min_val=128, max_val=self.max_frame_width,
                                        size=(140, -1), increment=4, value=128, digits=0,
                                        agwStyle=FS.FS_LEFT)
        sizer.Add(self.width_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.width_ctrl.Bind(FS.EVT_FLOATSPIN, self.OnSetWidth)
        self.row_pos += 1 # Current row position = 15

        self.height_ctrl_label = wx.StaticText(panel, label="Height:")
        sizer.Add(self.height_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.height_ctrl = FS.FloatSpin(panel, -1,  min_val=128, max_val=self.max_frame_height,
                                        size=(140, -1), increment=4, value=128, digits=0,
                                        agwStyle=FS.FS_LEFT)
        sizer.Add(self.height_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.height_ctrl.Bind(FS.EVT_FLOATSPIN, self.OnSetHeight)
        self.row_pos += 1 # Current row position = 16
        
        exportfile_ctrl_label = wx.StaticText(panel, label="Export file name:")
        sizer.Add(exportfile_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.exportfile_ctrl = wx.TextCtrl(panel)
        sizer.Add(self.exportfile_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 17

        exportfolder_ctrl_label = wx.StaticText(panel, label="Export directory:")
        sizer.Add(exportfolder_ctrl_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.select_folder_btn = wx.Button(panel, label="Select folder")
        self.select_folder_btn.Bind(wx.EVT_BUTTON, self.OnSelectFolder)
        sizer.Add(self.select_folder_btn, pos=(self.row_pos, self.column_pos + 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 18

        self.exportfolder_ctrl = wx.TextCtrl(panel)
        sizer.Add(self.exportfolder_ctrl, pos=(self.row_pos, self.column_pos), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.exportfolder_ctrl.Disable()
        self.row_pos += 1 # Current row position = 19   

        self.append_date = wx.CheckBox(panel, label="Append date and time")
        sizer.Add(self.append_date, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.append_date.SetBackgroundColour(wx.NullColour)
        self.append_date.Bind(wx.EVT_CHECKBOX, self.OnAppendDate)
        self.append_date.SetValue(True)  
        self.row_pos += 1 # Current row position = 20

        self.auto_index = wx.CheckBox(panel, label="Auto index")
        sizer.Add(self.auto_index, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.auto_index.SetBackgroundColour(wx.NullColour)
        self.auto_index.Bind(wx.EVT_CHECKBOX, self.OnAutoIndex)
        self.auto_index.SetValue(True)  # Set checkbox to checked by default

        self.index_ctrl = wx.TextCtrl(panel)
        self.index_ctrl.SetValue(str(1))
        sizer.Add(self.index_ctrl, pos=(20, self.column_pos + 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 21
        
        fourcc_label = wx.StaticText(panel, label="Fourcc Code:")
        sizer.Add(fourcc_label, pos=(self.row_pos, self.column_pos), flag=wx.EXPAND | wx.ALL, border=5)
        fourcc_modes = ["MJPG", "DIVX", "XVID"]
        self.encoding_mode_combo = wx.ComboBox(panel, choices=fourcc_modes)
        sizer.Add(self.encoding_mode_combo, pos=(self.row_pos, self.column_pos + 1), flag=wx.ALL, border=5)
        self.encoding_mode_combo.Bind(wx.EVT_COMBOBOX, self.OnCapModeCombo)
        self.encoding_mode_combo.SetSelection(self.encoding_mode)
        self.row_pos += 1 # Current row position = 22   

        # interval_ctrl_label = wx.StaticText(panel, label="Measurement interval (sec):")
        # sizer.Add(interval_ctrl_label, pos=(self.row_pos, self.column_pos),
        #           flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
        # self.interval_ctrl = wx.TextCtrl(panel)
        # self.interval_ctrl.SetValue(str(self.measurement_interval))
        # sizer.Add(self.interval_ctrl, pos=(self.row_pos, self.column_pos + 1),
        #           flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
        # self.row_pos += 1 # Current row position = 23

        # sequence_ctrl_label = wx.StaticText(panel, label="Sequence length (num):")
        # sizer.Add(sequence_ctrl_label, pos=(self.row_pos, self.column_pos),
        #           flag=wx.EXPAND | wx.ALL, border=5)
        # self.sequence_ctrl = wx.TextCtrl(panel)
        # self.sequence_ctrl.SetValue(str(self.sequence_length))
        # sizer.Add(self.sequence_ctrl, pos=(self.row_pos, self.column_pos + 1),
        #           flag=wx.EXPAND | wx.ALL, border=5)
        # self.row_pos += 1 # Current row position = 24
        
        # framescap_ctrl_label = wx.StaticText(panel, label="Video length (sec):")
        # sizer.Add(framescap_ctrl_label, pos=(self.row_pos, self.column_pos),
        #           flag=wx.EXPAND | wx.ALL, border=5)
        # self.framescap_ctrl = wx.TextCtrl(panel)
        # self.framescap_ctrl.SetValue(str(self.frames_to_capture))
        # sizer.Add(self.framescap_ctrl, pos=(self.row_pos, self.column_pos + 1),
        #           flag=wx.EXPAND | wx.ALL, border=5)
        # self.row_pos += 1 # Current row position = 25

        self.capture_btn = wx.Button(panel, label="Capture START")
        self.capture_btn.Bind(wx.EVT_BUTTON, self.OnCapture)
        sizer.Add(self.capture_btn, pos=(self.row_pos, self.column_pos), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 26

        self.current_state = wx.StaticText(panel, label="Current state: idle")
        sizer.Add(self.current_state, pos=(self.row_pos, self.column_pos), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1 # Current row position = 27

        self.offset_x_ctrl.Disable()
        self.offset_y_ctrl.Disable()
        self.width_ctrl.Disable()
        self.height_ctrl.Disable()
        
        # Text field to enter the queue
        self.note_label = wx.StaticText(panel, label="Notes (Press Enter to send):")
        sizer.Add(self.note_label, pos=(self.row_pos, self.column_pos), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.note_ctrl = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER)
        sizer.Add(self.note_ctrl, pos=(self.row_pos, self.column_pos + 1), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.row_pos += 1

        self.next_note_q = queue.Queue(maxsize=1)
        self.note_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnNoteEnter)

        self.frame = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)
        self.frame[:] = 255

        self.display_frame = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)
        self.display_frame[:] = 255

        self.preview_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.Draw, self.preview_timer)

        # self.capture_status_timer = wx.Timer(self)
        # self.Bind(wx.EVT_TIMER, self.capture_status, self.capture_status_timer)

        # self.capture_sequence_timer = wx.Timer(self)
        # self.Bind(wx.EVT_TIMER, self.count_elapsed, self.capture_sequence_timer)

        self.border = wx.BoxSizer()
        self.border.Add(sizer, 1, wx.ALL | wx.EXPAND, 20)

        # panel.SetSizerAndFit(self.border)
        panel.SetSizer(sizer)
        panel.Layout()
        self.Fit()
        self.preview_thread_obj = threading.Thread(target=self.preview_thread)
        self.capture_thread_obj = threading.Thread(target=self.capture_thread)
        self.EnableGUI(False)

    def Draw(self, evt):

        self.lock.acquire()
        if (self.selected_mode == 0):
            print("RAW")
        if (self.selected_mode == 1):
            print("LASCA")

        if (self.selected_mode == 2):
            w, h = self.Window.GetClientSize()
            w = max(1, w); h = max(1, h)
            rgb = self.DrawHistogram(self.frame, (w, h),
                                     (255, 255, 255), (250, 155, 0))
            # Fast, correct width/height order (w = cols, h = rows):
            self.bitmap = wx.Bitmap.FromBuffer(w, h, np.ascontiguousarray(rgb))
            self.Window.update_bitmap(self.bitmap)

        # self.Window.update(self.bitmap)

        if self.preview_on is True:
            self.preview_timer.Start(50, oneShot=True)

        self.lock.release()

    def EnableGUI(self, value, preview=False):
        if value is True:
            # self.interval_ctrl.Enable()
            # self.sequence_ctrl.Enable()
            self.trigger_mode_combo.Enable()
            self.framerate_slider.Enable()
            self.exposure_slider.Enable()
            self.gain_slider.Enable()
            self.exportfile_ctrl.Enable()
            self.cam_combo.Disable()
            self.encoding_mode_combo.Enable()
            self.preview_btn.Enable()
            self.select_folder_btn.Enable()
            self.capture_btn.Enable()
            self.append_date.Enable()
            self.set_auto_gain.Enable()
            self.set_auto_exposure.Enable()
            self.auto_index.Enable()
            self.index_ctrl.Enable()

            self.offset_x_ctrl.Enable()
            self.offset_y_ctrl.Enable()
            self.width_ctrl.Enable()
            self.height_ctrl.Enable()

            if self.auto_exposure_on is True:
                self.set_auto_exposure.Enable()
                self.exposure_slider.Disable()
            else:
                self.exposure_slider.Enable()

            if self.auto_gain_on is True:
                self.set_auto_exposure.Enable()
                self.gain_slider.Disable()
            else:
                self.gain_slider.Enable()

            if self.auto_index_on is True:
                self.auto_index.Enable()
                self.index_ctrl.Disable()
            else:
                self.index_ctrl.Enable()

            return
        elif preview is True:
            # self.interval_ctrl.Disable()
            # self.sequence_ctrl.Disable()
            self.trigger_mode_combo.Disable()
            self.framerate_slider.Disable()
            self.exposure_slider.Disable()
            self.gain_slider.Disable()
            self.exportfile_ctrl.Disable()
            self.encoding_mode_combo.Disable()
            self.preview_btn.Enable()
            self.offset_x_ctrl.Enable()
            self.offset_y_ctrl.Enable()
            self.width_ctrl.Disable()
            self.height_ctrl.Disable()
            self.select_folder_btn.Disable()
            self.capture_btn.Disable()
            self.append_date.Enable()
            self.set_auto_gain.Disable()
            self.set_auto_exposure.Disable()
            self.auto_index.Enable()
            self.index_ctrl.Disable()
        else:
            # self.interval_ctrl.Disable()
            # self.sequence_ctrl.Disable()
            self.trigger_mode_combo.Disable()
            self.framerate_slider.Disable()
            self.exposure_slider.Disable()
            self.gain_slider.Disable()
            self.exportfile_ctrl.Disable()
            self.encoding_mode_combo.Disable()
            self.preview_btn.Disable()
            self.offset_x_ctrl.Disable()
            self.offset_y_ctrl.Disable()
            self.width_ctrl.Disable()
            self.height_ctrl.Disable()
            self.select_folder_btn.Disable()
            self.capture_btn.Disable()
            self.append_date.Enable()
            self.set_auto_gain.Disable()
            self.set_auto_exposure.Disable()
            self.auto_index.Enable()
            self.index_ctrl.Disable()
            return

    def BlockGUI(self, value):
        if value is True:
            # self.interval_ctrl.Enable()
            # self.sequence_ctrl.Enable()
            self.trigger_mode_combo.Enable()
            self.framerate_slider.Enable()
            self.exportfile_ctrl.Enable()
            self.mode_combo.Enable()
            self.capmode_combo.Enable()
            self.preview_btn.Enable()
            self.select_folder_btn.Enable()
            self.append_date.Enable()
            self.connect_btn.Enable()

            if self.roi_on is True:
                self.offset_x_ctrl.Enable()
                self.offset_y_ctrl.Enable()
                self.width_ctrl.Enable()
                self.height_ctrl.Enable()

            if self.auto_exposure_on is True:
                self.set_auto_exposure.Enable()
                self.exposure_slider.Disable()
            else:
                self.exposure_slider.Enable()

            if self.auto_gain_on is True:
                self.set_auto_exposure.Enable()
                self.gain_slider.Disable()
            else:
                self.gain_slider.Enable()

            if self.auto_index_on is True:
                self.auto_index.Enable()
                self.index_ctrl.Disable()
            else:
                self.index_ctrl.Enable()

            return
        else:
            # self.interval_ctrl.Disable()
            # self.sequence_ctrl.Disable()
            self.trigger_mode_combo.Disable()
            self.framerate_slider.Disable()
            self.exposure_slider.Disable()
            self.gain_slider.Disable()
            self.exportfile_ctrl.Disable()
            self.mode_combo.Disable()
            self.capmode_combo.Disable()
            self.preview_btn.Disable()
            self.offset_x_ctrl.Disable()
            self.offset_y_ctrl.Disable()
            self.width_ctrl.Disable()
            self.height_ctrl.Disable()
            self.select_folder_btn.Disable()
            self.append_date.Disable()
            self.connect_btn.Disable()
            self.set_auto_gain.Disable()
            self.set_auto_exposure.Disable()
            self.auto_index.Disable()
            self.index_ctrl.Disable()
        return

    def OnConnect(self, event):
        if self.camera_connected is False:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            device_name = self.cameras_list[self.selected_camera]["name"]
            device_serial = self.cameras_list[self.selected_camera]["serial"]
            for i, device in enumerate(devices):

                if device.GetModelName() == device_name:
                    if device.GetSerialNumber() == device_serial:


                        self.camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[i]))
                        self.camera.Open()
                    
                        # Configure GPIO Pin 3 (Line3) as Input and Enable Event
                        self.camera.LineSelector.Value = "Line3"  # Select GPIO Pin 3
                        self.camera.LineMode.Value = "Input"  # Configure as Input
                        # self.camera.LineEventSource.Value = "RisingEdge" # Trigger on Rising Edge
                        # self.camera.LineEventEnable.Value = True  # Enable event generation     
                            
                        # # Register the standard event handler for configuring input detected events.
                        self.camera.RegisterConfiguration(ConfigurationEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
                        
                        self.camera.MaxNumBuffer = 2
                        self.camera.OutputQueueSize.Value = 2
                        # Setting trigger mode
                        if self.trigger_mode is True:
                            self.camera.TriggerMode.Value = "On"
                            self.camera.TriggerSelector.Value = "FrameStart"
                            self.camera.TriggerSource.Value = "Line4"
                            self.camera.TriggerActivation.Value = "RisingEdge"
                            self.camera.AcquisitionFrameRateEnable.SetValue(False)
                            # resulting_framerate = self.camera.ResultingFrameRate.GetValue()
                            # self.camera.AcquisitionFrameRate.SetValue(resulting_framerate)
                        else:
                            self.camera.TriggerMode.Value = "Off"
                            self.camera.AcquisitionFrameRateEnable.SetValue(True)
                            self.camera.AcquisitionFrameRate.SetValue(200.0)
                            resulting_framerate = self.camera.ResultingFrameRate.GetValue()
                            if (resulting_framerate != self.framerate):
                                self.framerate = resulting_framerate
                                self.framerate_slider.SetValue(self.framerate)

                        self.camera.GainAuto.SetValue("Off")
                        self.camera.ExposureAuto.SetValue("Off")

                        self.exposure_slider.SetMax(self.camera.ExposureTime.Max)
                        self.exposure_slider.SetMin(self.camera.ExposureTime.Min)
                        self.exposure_slider.SetValue(self.camera.ExposureTime.Value)
                        self.exposure = self.camera.ExposureTime.Value

                        self.gain_slider.SetMax(self.camera.Gain.Max)
                        self.gain_slider.SetMin(self.camera.Gain.Min)
                        self.gain_slider.SetValue(self.camera.Gain.Value)
                        self.gain = self.camera.Gain.Value

                        self.framerate_slider.SetMax(self.camera.AcquisitionFrameRate.Max)
                        self.framerate_slider.SetMin(self.camera.AcquisitionFrameRate.Min)
                        self.framerate_slider.SetValue(self.camera.AcquisitionFrameRate.Value)
                        self.framerate = self.camera.AcquisitionFrameRate.Value
                        
                        # Get the current frame width, height and offset
                        self.frame_width = self.camera.Width.GetValue()
                        self.frame_height = self.camera.Height.GetValue()
                        self.offset_x = self.camera.OffsetX.GetValue()
                        self.offset_y = self.camera.OffsetY.GetValue()
                        
                        # Set the frame width, height and offset
                        self.width_ctrl.SetMax(self.max_frame_width)
                        self.width_ctrl.SetValue(self.frame_width)
                        
                        self.height_ctrl.SetMax(self.max_frame_height)
                        self.height_ctrl.SetValue(self.frame_height)
                        
                        self.offset_x_ctrl.SetMax(self.max_frame_width - self.frame_width)
                        self.offset_x_ctrl.SetValue(self.offset_x)
                        
                        self.offset_y_ctrl.SetMax(self.max_frame_height - self.frame_height)
                        self.offset_y_ctrl.SetValue(self.offset_y)
                
                        self.connect_btn.SetLabel("Disconnect")
                        self.refresh_btn.Disable()
                        self.cam_combo.Disable()
                        self.camera_connected = True

                        self.EnableGUI(True)
                        return

        else:
            self.StopPreview()
            self.camera.Close()
            self.connect_btn.SetLabel("Connect")
            self.refresh_btn.Enable()
            self.cam_combo.Enable()
            self.camera_connected = False
            self.EnableGUI(False)
            return

    def OnCloseWindow(self, event):

        self.StopPreview()
        self.capture_on = False
        self.StopCapture()

        if self.camera_connected is True:
            self.camera.Close()

        self.Destroy()
        print("Closing BaslerGUI")
        return

    def OnEnableAutoExposure(self, event):
        if self.camera_connected is True:
            self.auto_exposure_on = self.set_auto_exposure.GetValue()
            if self.auto_exposure_on is True:
                self.camera.ExposureAuto.SetValue("Continuous")
                self.exposure_slider.Disable()
            else:
                self.camera.ExposureAuto.SetValue("Off")
                self.exposure_slider.SetValue(self.camera.ExposureTime.Value)
                self.exposure_slider.Enable()

    def OnEnableAutoGain(self, event):
        if self.camera_connected is True:
            self.auto_gain_on = self.set_auto_gain.GetValue()
            if self.auto_gain_on is True:
                self.camera.GainAuto.SetValue("Continuous")
                self.gain_slider.Disable()
            else:
                self.camera.GainAuto.SetValue("Off")
                self.gain_slider.SetValue(self.camera.Gain.Value)
                self.gain_slider.Enable()

    def OnRefreshList(self, event):

        if self.camera_connected is False:
            self.selected_camera = 0
            self.cam_combo.Clear()
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")

            self.cameras_list.clear()
            for device in devices:
                self.cam_combo.Append(device.GetModelName() +
                                      "_" + device.GetSerialNumber())
                self.cameras_list.append({"name": device.GetModelName(),
                                          "serial": device.GetSerialNumber()})
            self.cam_combo.SetSelection(self.selected_camera)

    def OnPreview(self, event):
        if self.camera_connected is True:
            if self.preview_on is True:
                self.StopPreview()
            else:
                self.StartPreview()

    def OnCapture(self, event):
        if self.current_step == 0:
            if self.capture_on is False:
                self.StartCapture()
            else:
                self.capture_on = False
                self.current_step = 0
                self.StopCapture()
                self.StartPreview()
        else:
            self.StopCapture()

    def FramerteSliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.SetFrameRate(val)

    def SetFrameRate(self, value: int):
        if self.camera_connected is True:
            if self.trigger_mode is True:
                self.camera.AcquisitionFrameRateEnable.SetValue(False)
                self.framerate = value
                self.framerate_slider.SetValue(self.framerate) 
            else:
                self.camera.AcquisitionFrameRate.SetValue(value)
                resulting_framerate = self.camera.ResultingFrameRate.GetValue()
                if (resulting_framerate != value):
                    self.framerate = int(resulting_framerate)
                    self.framerate_slider.SetValue(self.framerate)
    
    def GetFrameRate(self):
        return self.framerate

    def ExposureSliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.exposure = val
        if self.camera_connected is True:
            self.camera.ExposureTime.SetValue(self.exposure)
            resulting_framerate = self.camera.ResultingFrameRate.GetValue()
            if (resulting_framerate != self.framerate):
                self.framerate = int(resulting_framerate)
                self.framerate_slider.SetValue(self.framerate)

    def OnAutoIndex(self, event):
        if self.camera_connected is True:
            self.auto_index_on = self.auto_index.GetValue()
            if self.auto_index_on is True:
                self.index_ctrl.Disable()
                self.current_index = int(self.index_ctrl.GetValue())
            else:
                self.index_ctrl.Enable()

    def SetAutoIndex(self, value: bool):
        self.auto_index.SetValue(value)
        self.auto_index_on = value
        if value is True:
            self.index_ctrl.Disable()
        else:
            self.index_ctrl.Enable()

    def ContrastSliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.max_contrast = val

    def GraySliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.min_gray_val = val

    def GainSliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.gain = val
        if self.camera_connected is True:
            self.camera.Gain.SetValue(self.gain)

    def OnNoteEnter(self, event):
        text = self.note_ctrl.GetValue().strip()
        if not text:
            return
        # Keep only the most recent pending note
        try:
            self.next_note_q.put_nowait(text)
        except queue.Full:
            try:
                _ = self.next_note_q.get_nowait()
            except queue.Empty:
                pass
            self.next_note_q.put_nowait(text)

        self.note_ctrl.Clear()
        # (optional) give quick UI feedback
        # wx.LogMessage(f"Queued note for next frame: {text!r}")
        
    def OnCamCombo(self, event):
        self.selected_camera = self.cam_combo.GetSelection()

    def OnModeCombo(self, event):
        self.selected_mode = self.mode_combo.GetSelection()

    def OnCapModeCombo(self, event):
        self.capture_mode = self.capmode_combo.GetSelection()

    def OnSelectFolder(self, event):
        dlg = wx.DirDialog(None, "Choose input directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.ShowModal()
        self.exportfolder_ctrl.SetValue(dlg.GetPath())
    
    def SetExportFolder(self, folder_path):
        self.exportfolder_ctrl.SetValue(folder_path)

    def GetHistogram(self, image):
        hist_full = cv2.calcHist([image], [0], None, [256], [0, 256])
        max_val = np.max(hist_full)
        if max_val > 0:
            hist_full = (hist_full / np.max(hist_full))*100
        else:
            hist_full = np.zeros((256, 1))
        return hist_full

    def DrawHistogram(self, image, size, bcg_color, bin_color):
        histogram_data = self.GetHistogram(image)

        # draw base 256x256 histogram (OpenCV draws in BGR)
        hist = np.full((256, 256, 3), 240, np.uint8)
        R, G, B = bcg_color
        hist[:, :, 0] = B; hist[:, :, 1] = G; hist[:, :, 2] = R

        for x in range(256):
            col_pct = float(np.ravel(histogram_data[x])[0])
            h = int(np.floor(col_pct * 2.56))
            h = 0 if h < 0 else (255 if h > 255 else h)
            if h > 1:
                r, g, b = bin_color
                cv2.line(hist, (x, 255), (x, 255 - h), (b, g, r), 1)

        w, h = size  # NOTE: OpenCV wants (width, height)
        hist = cv2.resize(hist, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
        return rgb

    def OnAppendDate(self, event):
        self.append_date_flag = self.append_date.GetValue()
   
    def OnTriggerModeCombo(self, event):
        selection = self.trigger_mode_combo.GetSelection()
        if selection == 0:
            self.trigger_mode = True
        else:
            self.trigger_mode = False
        self.SetTriggerMode(self.trigger_mode)
         
    def SetTriggerModeLabel(self):
        if self.trigger_mode is True:
            self.trigger_mode_label = 0                 
        else:
            self.trigger_mode_label = 1
        # Check if UI element exists before setting value
        if hasattr(self, 'trigger_mode_combo'):
            self.trigger_mode_combo.SetSelection(self.trigger_mode_label)
    
    def SetTriggerMode(self, trigger_mode: bool):
        self.trigger_mode = trigger_mode
        self.SetTriggerModeLabel()
        if trigger_mode is True:
            self.trigger_mode = True
            self.camera.TriggerMode.Value = "On"
            self.camera.TriggerSelector.Value = "FrameStart"
            self.camera.TriggerSource.Value = "Line4"
            self.camera.TriggerActivation.Value = "RisingEdge"
            self.camera.AcquisitionFrameRateEnable.SetValue(False)
        else:
            self.trigger_mode = False
            self.camera.TriggerMode.Value = "Off"
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.SetFrameRate(self.framerate)
    
        

    def SetAppendDate(self, value):
        self.append_date.SetValue(value)
        self.append_date_flag = value

    def SetFileName(self, filename):
        self.exportfile_ctrl.SetValue(filename)

    def OnSetOffsetX(self, event):
        new_offset_x = self.offset_x_ctrl.GetValue()
        if new_offset_x == self.offset_x:
            return
        # Check if the new offset + width is divisible by 2
        new_width = new_offset_x + self.frame_width
        new_width = int(4 * round(new_width / 4)) if new_width % 4 != 0 else new_width
        
        new_offset_x = new_width - self.frame_width
        
        if (new_offset_x + self.frame_width) < self.max_frame_width:
            self.offset_x = int(new_offset_x)
            if self.preview_on is True:
                self.StopPreview()
                self.camera.OffsetX.SetValue(self.offset_x)
                self.StartPreview()
            else:
                self.camera.OffsetX.SetValue(self.offset_x)
            self.offset_x_ctrl_label.SetLabel("Offset X (max {}):".format(self.max_frame_width - self.frame_width))
        
        self.offset_x_ctrl.SetValue(self.offset_x)

    def OnSetOffsetY(self, event):
        new_offset_y = self.offset_y_ctrl.GetValue()
        if new_offset_y == self.offset_y:
            return
        # Check if the new offset + height is divisible by 2
        new_height = new_offset_y + self.frame_height
        new_height = int(4 * round(new_height / 4)) if new_height % 4 != 0 else new_height
        
        new_offset_y = new_height - self.frame_height
        
        if (new_offset_y + self.frame_height) < self.max_frame_height:
            self.offset_y = int(new_offset_y)
            if self.preview_on is True:
                self.StopPreview()
                self.camera.OffsetY.SetValue(self.offset_y)
                self.StartPreview()
            else:
                self.camera.OffsetY.SetValue(self.offset_y)
            self.offset_y_ctrl_label.SetLabel("Offset Y (max {}):".format(self.max_frame_height - self.frame_height))

        self.offset_y_ctrl.SetValue(self.offset_y)

    def OnSetWidth(self, event):
        new_width = self.width_ctrl.GetValue()
        if new_width == self.frame_width:
            return
        # Check if the new width is divisible by 2
        new_width = int(16 * round(new_width / 16)) if new_width % 16 != 0 else new_width
            
        if (self.offset_x + new_width) < self.max_frame_width:
            self.frame_width = int(new_width)
            self.camera.Width.SetValue(self.frame_width)
            self.offset_x_ctrl.SetMax(self.max_frame_width - self.frame_width)
            self.offset_x_ctrl_label.SetLabel("Offset X (max {}):".format(self.max_frame_width - self.frame_width))
        
        self.width_ctrl.SetValue(self.frame_width)

    def OnSetHeight(self, event):  
        new_height = self.height_ctrl.GetValue()
        if new_height == self.frame_height:
            return
        # Check if the new height is divisible by 2
        new_height = int(4 * round(new_height / 4)) if new_height % 4 != 0 else new_height
            
        if (self.offset_y + new_height) < self.max_frame_height:
            self.frame_height = int(new_height)
            self.camera.Height.SetValue(self.frame_height)
            self.offset_y_ctrl.SetMax(self.max_frame_height - self.frame_height)
            self.offset_y_ctrl_label.SetLabel("Offset Y (max {}):".format(self.max_frame_height - self.frame_height))
            
        self.height_ctrl.SetValue(self.frame_height)
    
    def get_image_dimensions(self):
        height = self.frame_height
        width = self.frame_width
        return (width, height) 
    
    def StartPreview(self):
        self.preview_on = True
        self.EnableGUI(False, preview=True)
        self.preview_thread_obj = threading.Thread(target=self.preview_thread)
        self.preview_thread_obj.start()
        self.preview_timer.Start(100, oneShot=True)
        self.preview_btn.SetLabel("Preview STOP")

    def StopPreview(self):
        self.preview_on = False
        self.EnableGUI(True)
        if self.preview_thread_obj.is_alive() is True:
            self.preview_thread_obj.join()
        self.preview_timer.Stop()
        self.preview_btn.SetLabel("Preview START")

    def preview_thread(self):
        import ctypes, time

        user32 = ctypes.windll.user32
        IsWindowVisible = user32.IsWindowVisible
        # Create Image Windows to display live video while capturing
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(self.cam_index, 0, 0, self.frame_width, self.frame_height)
        hwnd = imageWindow.GetWindowHandle()  # HWND
        
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        self.previous_time = int(round(time.time() * 1000))

        current_date_and_time = str(datetime.datetime.now())
        last_display_time = time.time()
        display_interval = 1/30  # Update display every 1/60 seconds (to match 60Hz refresh rate)


        while self.camera.IsGrabbing() and self.preview_on is True:
            try:
                grabResult = self.camera.RetrieveResult(100,
                                                    pylon.TimeoutHandling_ThrowException)
            except pylon.TimeoutException:
                #print("Timeout occurred while waiting for image.")
                continue
            if grabResult.GrabSucceeded():
                current_time = int(round(time.time() * 1000))
                if ((current_time - self.previous_time) > 20):
                    self.lock.acquire()
                    self.frame = grabResult.GetArray()
                    
                    timestamp = time.time() 
                    if (timestamp - last_display_time) > display_interval:
                        imageWindow.SetImage(grabResult)
                        imageWindow.Show()
                        last_display_time = time.time()
                        
                    self.lock.release()
                    self.previous_time = current_time
            else:
                print("Error: ", grabResult.ErrorCode)
            grabResult.Release()
        while self.camera.NumReadyBuffers.GetValue() > 0:
            self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
        imageWindow.Close()        
        self.camera.StopGrabbing()

    def StartCapture(self):
        self.StopPreview()
        self.SetupCapture()
        
        # Start the capture and display threads
        self.capture_thread_obj = threading.Thread(target=self.capture_thread)
        self.capture_thread_obj.start()
        self.EnableGUI(False)
        self.capture_btn.SetLabel("Capture STOP")
        self.connect_btn.Disable()
        self.capture_btn.Enable()
        # self.capture_status_timer.Start(10000, oneShot=True)
    
    def SetupCapture(self):
        # Prepare data output file before starting capture
        # sequence_length = int(self.sequence_ctrl.GetValue())
        # video_length = float(self.framescap_ctrl.GetValue())
        # frames_to_capture = int(video_length * self.framerate)
        # interval_length = float(self.interval_ctrl.GetValue())
        
        fourcc_code = str(self.encoding_mode_combo.GetValue())

        output_path = []
        output_file_name = self.exportfile_ctrl.GetValue()
        if len(output_file_name) <= 1:
            output_file_name = "output"
        # Adding cam index to the file name
        output_file_name = output_file_name + "_cam" + str(self.cam_index)
            
        output_folder_name = self.exportfolder_ctrl.GetValue()
        if len(output_folder_name) <= 1:
            output_folder_name = "C:\\"
        if len(output_folder_name) > 0:
            output_path = output_folder_name + "\\" + output_file_name
        else:
            output_path = output_file_name

        if len(output_file_name) <= 1:
            wx.MessageBox('Please provide output file name!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            # self.current_step = sequence_length
            return

        if len(output_folder_name) <= 1:
            wx.MessageBox('Please provide output folder!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            # self.current_step = sequence_length
            return

        if self.append_date_flag is True:
            output_path = output_path + "_" + time.strftime("%Y%m%d_%H%M%S")

        if self.auto_index_on is True:
            output_path = output_path + "_" + str(self.current_index)
            self.current_index += 1

        if self.auto_index_on is False and self.append_date_flag is False:
            wx.MessageBox('Turn on auto indexing or append date to' +
                            ' file name when capturing sequence!',
                            'Warning', wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            # self.current_step = sequence_length
            return

        if len(output_path) <= 4:
            wx.MessageBox('Invalid name for data output file!',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            # self.current_step = sequence_length
            return

        # Making sure the output file is .avi
        if not output_path.endswith('.avi'):
            output_path += '.avi'

        # Configure session
        # Prepare data output file and buffer
        self.video_session = VideoRecordingSession(cam_num=self.cam_index)
        print(f"Frame width: {self.frame_width}, Frame height: {self.frame_height}")
        
        # TODO: add more options for output file
        self.video_session.set_params(
            video_file=output_path,
            fourcc=fourcc_code,
            fps=200,
            dim=(self.frame_width, self.frame_height)
        )
 
    def StopCapture(self):
        self.capture_on = False
        self.current_step = 0
        if self.capture_thread_obj.is_alive() is True:
            self.capture_thread_obj.join()
        self.EnableGUI(True)
        self.capture_btn.SetLabel("Capture START")
        self.current_state.SetLabel("Current state: idle")
        self.connect_btn.Enable()

    def capture_thread(self):

        
        # Create Image Windows to display live video while capturing
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(self.cam_index, 0, 0, 640, 480)
        
        # Enable chunks in general.
        self.camera.ChunkModeActive.Value = True
        
        # Enable time stamp chunks.
        self.camera.ChunkSelector.Value = "Timestamp"
        self.camera.ChunkEnable.Value = True
        
        # Enable line status chunks.
        self.camera.ChunkSelector.Value = "LineStatusAll"
        self.camera.ChunkEnable.Value = True
        self.camera.MaxNumBuffer = 2
        self.camera.OutputQueueSize.Value = 2
        
        # Start the video recording session
        self.video_session.start_recording()
        
        # Start the camera grabbing
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

        current_date_and_time = str(datetime.datetime.now())
        last_display_time = time.time()
        display_interval = 1/30  # Update display every 1/60 seconds (to match 60Hz refresh rate)
        
        print(f'Capturing video started at: {current_date_and_time}')
        
        captured_frames = 0

        # Indefinite capture mode
        self.capture_on = True
        
        while (self.capture_on is True) or (self.camera.NumReadyBuffers.GetValue() > 0):
            try:
                grabResult = self.camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
            except pylon.TimeoutException:
                # print("Timeout occurred while waiting for a frame.")
                continue
            
            if self.camera.NumReadyBuffers.GetValue() > 0:
                print(f"Frames in buffer: {self.camera.NumReadyBuffers.GetValue()}")

            if grabResult.GrabSucceeded():
                frame = grabResult.GetArray()
                timestamp = time.time()
                frame_timestamp = grabResult.ChunkTimestamp.Value
                frame_line_status = grabResult.ChunkLineStatusAll.Value
                captured_frames += 1

                # Pull one pending note if any (non-blocking, single-shot)
                note = None
                try:
                    note = self.next_note_q.get_nowait()
                except queue.Empty:
                    pass

                self.video_session.acquire_frame(frame, frame_timestamp, captured_frames, frame_line_status, note)
                
                if (timestamp - last_display_time) > display_interval:
                    line_status = self.camera.LineStatus.GetValue()  # Retrieve line status
                    imageWindow.SetImage(grabResult)
                    imageWindow.Show()
                    last_display_time = time.time()
                
                # time.sleep(0.00001)
            elif not grabResult.isValid():
                break
            else:
                print("Error: ", grabResult.ErrorCode)
            
            grabResult.Release()

        self.camera.StopGrabbing()
        imageWindow.Close()
        self.video_session.stop_recording()

        print(f'Capturing finished after grabbing {captured_frames} frames')

    def check_buffer_status(self):
        if self.camera_connected:
            num_buffers = int(self.camera.NumQueuedBuffers.Value)
            print(f"Number of frames in buffer: {num_buffers}")
            return num_buffers
        else:
            print("Camera is not connected.")
            return 0
    
    # ############### Calibration functions
    def SetupCalibration(self,
                         board_calibration: CharucoBoard,
                         frame_queue: queue.Queue,
                         frame_count_sync,
                         all_rows,
                         current_all_rows,
                         barrier: threading.Barrier = None):
        ### This function should only be called by the SystemController

        self.board_calibration = board_calibration
        self.frame_queue = frame_queue
        self.barrier = barrier
        self.frame_count_sync = frame_count_sync
        self.all_rows = all_rows
        self.current_all_rows = current_all_rows

        # Prepare data output file before starting capture
        # sequence_length = int(self.sequence_ctrl.GetValue())
        # video_length = float(self.framescap_ctrl.GetValue())
        # frames_to_capture = int(video_length * self.framerate)
        # interval_length = float(self.interval_ctrl.GetValue())
        
        fourcc_code = str(self.encoding_mode_combo.GetValue())

        output_path = []

        output_file_name = "calib"
        # Adding cam index to the file name
        output_file_name = output_file_name + "_cam" + str(self.cam_index)
            
        output_folder_name = self.exportfolder_ctrl.GetValue()
        if len(output_folder_name) <= 1:
            output_folder_name = "C:\\"
        if len(output_folder_name) > 0:
            output_path = output_folder_name + "\\" + output_file_name
        else:
            output_path = output_file_name

        if len(output_file_name) <= 1:
            wx.MessageBox('Please provide output file name!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.calibration_on = False
            # self.current_step = sequence_length
            return

        if len(output_folder_name) <= 1:
            wx.MessageBox('Please provide output folder!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.calibration_on = False
            # self.current_step = sequence_length
            return

        if self.append_date_flag is True:
            output_path = output_path + "_" + time.strftime("%Y%m%d_%H%M%S")

        if self.auto_index_on is True:
            output_path = output_path + "_" + str(self.current_index)
            self.current_index += 1

        if self.auto_index_on is False and self.append_date_flag is False:
            wx.MessageBox('Turn on auto indexing or append date to' +
                            ' file name when capturing sequence!',
                            'Warning', wx.OK | wx.ICON_WARNING)
            self.calibration_on = False
            # self.current_step = sequence_length
            return

        if len(output_path) <= 4:
            wx.MessageBox('Invalid name for data output file!',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.calibration_on = False
            # self.current_step = sequence_length
            return

        # Making sure the output file is .avi
        if not output_path.endswith('.avi'):
            output_path += '.avi'

        # Configure session
        # Prepare data output file and buffer
        self.video_session = VideoRecordingSession(cam_num=self.cam_index)
        print(f"Frame width: {self.frame_width}, Frame height: {self.frame_height}")
        
        # TODO: add more options for output file
        self.video_session.set_params(
            video_file=output_path,
            fourcc=fourcc_code,
            fps=200,
            dim=(self.frame_width, self.frame_height)
        )

    def StartCalibrateCapture(self):
        self.StopPreview()

        # Start the capture and display threads
        self.calibration_on = True
        self.calibrate_thread_obj = threading.Thread(target=self.record_calibrate_thread)
        self.calibrate_thread_obj.start()
        self.EnableGUI(False)
        self.capture_btn.SetLabel("Calibrate Capture STOP")
        self.connect_btn.Disable()
        self.capture_btn.Enable()
    
    def StopCalibrateCapture(self):
        self.calibration_on = False
        if self.calibrate_thread_obj.is_alive() is True:
            self.calibrate_thread_obj.join()
        self.EnableGUI(True)
        self.capture_btn.SetLabel("Capture START")
        self.current_state.SetLabel("Current state: idle")
        self.connect_btn.Enable()
        
    def record_calibrate_thread(self):
        # Create Image Windows to display live video while capturing
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(self.cam_index, 0, 0, 640, 480)
        
        # Enable chunks in general.
        self.camera.ChunkModeActive.Value = True
        
        # Enable time stamp chunks.
        self.camera.ChunkSelector.Value = "Timestamp"
        self.camera.ChunkEnable.Value = True
        
        # Enable line status chunks.
        self.camera.ChunkSelector.Value = "LineStatusAll"
        self.camera.ChunkEnable.Value = True
        self.camera.MaxNumBuffer = 1
        self.camera.OutputQueueSize.Value = 1
        
        # Start the video recording session
        self.video_session.start_recording()
        
        # Start the camera grabbing
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        current_date_and_time = str(datetime.datetime.now())
        last_display_time = time.time()
        display_interval = 1/30  # Update display every 1/60 seconds (to match 60Hz refresh rate)
        
        print(f'Capturing video started at: {current_date_and_time}')
        
        captured_frames = 0

        # Calibration mode
        self.calibration_on = True
        num = self.cam_index
        
        while (self.calibration_on is True) or (self.camera.NumReadyBuffers.GetValue() > 0):
            try:
                grabResult = self.camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
            except pylon.TimeoutException:
                # print("Timeout occurred while waiting for a frame.")
                continue
            
            if self.camera.NumReadyBuffers.GetValue() > 0:
                print(f"Frames in buffer: {self.camera.NumReadyBuffers.GetValue()}")

            if grabResult.GrabSucceeded():
                frame = grabResult.GetArray()
                timestamp = time.time()
                frame_timestamp = grabResult.ChunkTimestamp.Value
                frame_line_status = grabResult.ChunkLineStatusAll.Value
                captured_frames += 1
                # detect the marker as the frame is acquired
                corners, ids = self.board_calibration.detect_image(frame)
                if corners is not None and len(corners) > 0:
                    key = captured_frames
                    row = {
                        'framenum': key,
                        'corners': corners,
                        'ids': ids
                    }

                    row = self.board_calibration.fill_points_rows([row])
                    self.all_rows[num].extend(row)
                    self.current_all_rows[num].extend(row)
                    # print(f'Camera {num}: Detected marker at frame {key} with {len(corners)} corners.')
                    # self.board_detected_count_label[num]['text'] = f'{len(self.all_rows[num])}; {len(corners)}'
                    # if num == 0:
                        # self.calibration_current_duration_value.set(f'{time.perf_counter()-start_time:.2f}')
                # else:
                #     print(f'No marker detected on cam {num} at frame {captured_frames}')
                # putting frame into the frame queue along with following information
                self.frame_queue.put((frame,  # the frame itself
                                      num,  # the id of the capturing camera
                                      captured_frames,  # the current frame count
                                      frame_timestamp))  # captured time
                note = None
                self.video_session.acquire_frame(frame, frame_timestamp, captured_frames, frame_line_status, note)
                if (timestamp - last_display_time) > display_interval:
                    line_status = self.camera.LineStatus.GetValue()  # Retrieve line status
                    imageWindow.SetImage(grabResult)
                    imageWindow.Show()
                    last_display_time = time.time()
                
                self.frame_count_sync[num] = captured_frames
                
                # Check frame_count_sync to see if all the other cameras have captured the same number of frames, if not, wait at the barrier
                if self.barrier is not None:
                    # If other cameras are behind, wait at the barrier:
                    while captured_frames > min(self.frame_count_sync) and self.calibration_on is True:
                        try:
                            time.sleep(0.0001)  # small sleep to allow other threads to catch up
                        except threading.BrokenBarrierError:
                            print(f'Barrier broken for cam {num}. Proceeding...')
                
            grabResult.Release()
                 
        self.camera.StopGrabbing()
        imageWindow.Close()
        self.video_session.stop_recording()
        self.barrier.abort()

        print(f'Capturing and calibration finished after grabbing {captured_frames} frames')
        
    # def record_calibrate_on_thread(self, num, barrier):
    #     """
    #     Records frames from a camera on a separate thread for calibration purposes.

    #     :param num: The ID of the capturing camera.
    #     :param barrier: A threading.barrier object used to synchronize the start of frame capturing.

    #     :return: None

    #     """
    #     fps = int(self.fps.get())
    #     start_time = time.perf_counter()
    #     next_frame = start_time
    #     try:
    #         while self.calibration_capture_toggle_status and (time.perf_counter()-start_time < self.calibration_duration):
    #             if time.perf_counter() >= next_frame:
    #                 try:
    #                     barrier.wait(timeout=1)
    #                 except threading.BrokenBarrierError:
    #                     print(f'Barrier broken for cam {num}. Proceeding...')
    #                     break
                        
    #                 self.frame_times[num].append(time.perf_counter())
    #                 self.frame_count[num] += 1
    #                 frame_current = self.cam[num].get_image()
    #                 # detect the marker as the frame is acquired
    #                 corners, ids = self.board_calibration.detect_image(frame_current)
    #                 if corners is not None:
    #                     key = self.frame_count[num]
    #                     row = {
    #                         'framenum': key,
    #                         'corners': corners,
    #                         'ids': ids
    #                     }

    #                     row = self.board_calibration.fill_points_rows([row])
    #                     self.all_rows[num].extend(row)
    #                     self.current_all_rows[num].extend(row)
    #                     self.board_detected_count_label[num]['text'] = f'{len(self.all_rows[num])}; {len(corners)}'
    #                     if num == 0:
    #                         self.calibration_current_duration_value.set(f'{time.perf_counter()-start_time:.2f}')
    #                 else:
    #                     print(f'No marker detected on cam {num} at frame {self.frame_count[num]}')
                    
    #                 # putting frame into the frame queue along with following information
    #                 self.frame_queue.put((frame_current,  # the frame itself
    #                                       num,  # the id of the capturing camera
    #                                       self.frame_count[num],  # the current frame count
    #                                       self.frame_times[num][-1]))  # captured time

    #                 next_frame = max(next_frame + 1.0/fps, self.frame_times[num][-1] + 0.5/fps)
                    
    #         barrier.abort()
    #         if (time.perf_counter() - start_time) > self.calibration_duration or self.calibration_capture_toggle_status:
    #             print(f"Calibration capture on cam {num}: duration exceeded or toggle status is True")
    #             self.recording_threads_status[num] = False
    #             # self.toggle_calibration_capture(termination=True)
                
    #     except Exception as e:
    #         print("Exception occurred:", type(e).__name__, "| Exception value:", e,
    #               ''.join(traceback.format_tb(e.__traceback__)))

    

    # def recalibrate(self):
    #     """
    #     Recalibrates the device.

    #     Recalibrates the device by updating the necessary calibration statuses. This method should be called when the calibration toggle status is "False".

    #     Parameters:
    #         None

    #     Returns:
    #         None
    #     """
    #     if self.calibration_toggle_status is False:
    #         self.recalibrate_status = True
    #         self.update_calibration_status = False
    #         self.calibration_toggle_status = True
    #         print(f'Recalibration status: {self.recalibrate_status}, Update calibration status: {self.update_calibration_status}, Calibration toggle status: {self.calibration_toggle_status}')
            
    #         if self.calibrating_thread is not None and self.calibrating_thread.is_alive():
    #             self.calibrating_thread.join()
    #         else:
    #             self.calibrating_thread = threading.Thread(target=self.calibrate_on_thread)
    #             self.calibrating_thread.daemon = True
    #             self.calibrating_thread.start()
 
    # def update_calibration(self):
    #     """
    #      Updates the calibration status.

    #     If the calibration toggle status is False, it sets the update calibration status to True, recalibrate status to False,
    #     and calibration toggle status to True.

    #     Parameters:
    #         self (object): The instance of the class.

    #     Returns:
    #         None
    #     """
    #     if self.calibration_toggle_status is False:
    #         self.update_calibration_status = True
    #         self.recalibrate_status = False
    #         self.calibration_toggle_status = True
           
    #         if self.calibrating_thread is not None and self.calibrating_thread.is_alive():
    #             self.calibrating_thread.join()
    #         else:
    #             self.calibrating_thread = threading.Thread(target=self.calibrate_on_thread)
    #             self.calibrating_thread.daemon = True
    #             self.calibrating_thread.start()
    
    
    # def capture_status(self, evt):
    #     if self.capture_on is True:
    #         self.capture_status_timer.Start(200, oneShot=True)
    #         self.current_state.SetLabel("Current status: capturing data!")
    #         return
    #     else:
    #         sequence_length = int(self.sequence_ctrl.GetValue())
    #         if sequence_length == 1:
    #             self.current_state.SetLabel("Current state: idle")
    #             self.EnableGUI(True)
    #             self.connect_btn.Enable()
    #             self.capture_btn.SetLabel("Capture START")
    #             self.StartPreview()
    #             return
    #         else:
    #             self.current_step += 1
    #             if sequence_length > self.current_step:
    #                 self.capture_btn.SetLabel("Capture STOP")
    #                 correction = int(time.time() - self.last_capture_time)
    #                 self.time_to_next = int(self.interval_ctrl.GetValue()) - correction
    #                 time_to_next_str = str(datetime.timedelta(seconds=self.time_to_next))
    #                 self.current_state.SetLabel(
    #                     f"Current status: step {self.current_step} out of" +
    #                     f"{sequence_length}" +
    #                     "time to next: " + time_to_next_str)
    #                 self.capture_sequence_timer.Start(1000, oneShot=True)
    #                 self.capture_btn.Enable()
    #                 self.preview_btn.Enable()
    #                 self.StartPreview()
    #                 return
    #             else:
    #                 if self.index_ctrl.GetValue() == '':
    #                     self.index_ctrl.SetValue(str(1))
    #                     self.current_index = 1
    #                 else:
    #                     self.current_index = int(self.index_ctrl.GetValue())
    #                 self.current_step = 0
    #                 self.capture_btn.SetLabel("Capture START")
    #                 self.current_state.SetLabel("Current state: idle")
    #                 self.EnableGUI(True)
    #                 self.connect_btn.Enable()
    #                 self.StartPreview()
    #                 return
    #     return

    # def count_elapsed(self, evt):
    #     self.time_to_next -= 1
    #     time_to_next_str = str(datetime.timedelta(seconds=self.time_to_next))
    #     sequence_length = int(self.sequence_ctrl.GetValue())
    #     self.current_state.SetLabel(
    #         f"Current status: step {self.current_step} out of" +
    #         f"{sequence_length}" +
    #         "time to next: " + time_to_next_str)
    #     if self.time_to_next > 0:
    #         self.capture_sequence_timer.Start(1000, oneShot=True)
    #     else:
    #         self.StartCapture()
    
    @staticmethod
    def precise_sleep(duration):
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < duration:
            pass  # Busy-waiting until the time has elapsed
        
        