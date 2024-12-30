# import pandas as pd

import wx
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

class ImagePanel(wx.Panel):

    def __init__(self, parent, frame_height=480, frame_width=640):
        wx.Panel.__init__(self, parent)
        h, w = frame_height, frame_width
        src = (255 * np.random.rand(h, w)).astype(np.uint8)
        buf = src.repeat(3, 1).tobytes()
        self.bitmap = wx.Image(w, h, buf).ConvertToBitmap()
        self.SetDoubleBuffered(True)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Size = (frame_height, frame_width)
        self.Fit()

    def OnPaint(self, evt):
        wx.BufferedPaintDC(self, self.bitmap)

    def update(self, input_image):
        self.bitmap = input_image
        wx.BufferedDC(wx.ClientDC(self), self.bitmap)


class BaslerGuiWindow(wx.Frame):

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

    selected_camera = 0
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

    # TODO: reorganize these variables to make it customizable
    current_frame = np.zeros((frame_height, frame_width, 1), np.float32)
    gray = np.zeros((frame_height, frame_width, 1), np.uint8)
    mean_img_sq = np.zeros((frame_height, frame_width, 1), np.float32)
    sq = np.zeros((frame_height, frame_width, 1), np.float32)
    img = np.zeros((frame_height, frame_width, 1), np.float32)
    mean_img = np.zeros((frame_height, frame_width, 1), np.float32)
    sq_img_mean = np.zeros((frame_height, frame_width, 1), np.float32)
    std = np.zeros((frame_height, frame_width, 1), np.float32)
    LASCA = np.zeros((frame_height, frame_width, 1), np.uint8)
    im_color = np.zeros((frame_height, frame_width, 3), np.uint8)
    mask = np.zeros((frame_height, frame_width, 1), bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)).astype(np.float32)
    kernel /= np.sum(kernel)
    
    video_session = VideoRecordingSession(cam_num=0)

    def __init__(self, *args, **kwargs):
        super(BaslerGuiWindow, self).__init__(*args, **kwargs)
        self.InitUI()
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.Centre()
        self.Show()

    def InitUI(self):
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        cameras = []
        for device in devices:
            cameras.append(device.GetModelName() + "_" + device.GetSerialNumber())
            self.cameras_list.append({"name": device.GetModelName(),
                                      "serial": device.GetSerialNumber()})

        panel = wx.Panel(self)
        self.SetTitle('Basler CAM GUI')
        sizer = wx.GridBagSizer(0, 0)

        selected_ctrl_label = wx.StaticText(panel, label="Selected camera:")
        sizer.Add(selected_ctrl_label, pos=(0, 0),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.cam_combo = wx.ComboBox(panel, choices=cameras)
        sizer.Add(self.cam_combo, pos=(1, 0),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.cam_combo.Bind(wx.EVT_COMBOBOX, self.OnCamCombo)
        self.cam_combo.SetSelection(self.selected_camera)

        self.connect_btn = wx.Button(panel, label="Connect")
        self.connect_btn.Bind(wx.EVT_BUTTON, self.OnConnect)
        sizer.Add(self.connect_btn, pos=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.refresh_btn = wx.Button(panel, label="Refresh list")
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.OnRefreshList)
        sizer.Add(self.refresh_btn, pos=(2, 0),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.preview_btn = wx.Button(panel, label="Preview START")
        self.preview_btn.Bind(wx.EVT_BUTTON, self.OnPreview)
        sizer.Add(self.preview_btn, pos=(2, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        fourcc_label = wx.StaticText(panel, label="Fourcc Code:")
        sizer.Add(fourcc_label, pos=(13, 0), flag=wx.EXPAND | wx.ALL, border=5)
        fourcc_modes = ["MJPG", "DIVX", "XVID"]
        self.encoding_mode_combo = wx.ComboBox(panel, choices=fourcc_modes)
        sizer.Add(self.encoding_mode_combo, pos=(13, 1), flag=wx.ALL, border=5)
        self.encoding_mode_combo.Bind(wx.EVT_COMBOBOX, self.OnCapModeCombo)
        self.encoding_mode_combo.SetSelection(self.encoding_mode)

        interval_ctrl_label = wx.StaticText(panel, label="Measurement interval (sec):")
        sizer.Add(interval_ctrl_label, pos=(14, 0),
                  flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)
        self.interval_ctrl = wx.TextCtrl(panel)
        self.interval_ctrl.SetValue(str(self.measurement_interval))
        sizer.Add(self.interval_ctrl, pos=(14, 1),
                  flag=wx.ALIGN_CENTER_VERTICAL | wx.ALL, border=5)

        sequence_ctrl_label = wx.StaticText(panel, label="Sequence length (num):")
        sizer.Add(sequence_ctrl_label, pos=(15, 0),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.sequence_ctrl = wx.TextCtrl(panel)
        self.sequence_ctrl.SetValue(str(self.sequence_length))
        sizer.Add(self.sequence_ctrl, pos=(15, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        mode_ctrl_label = wx.StaticText(panel, label="Preview mode:")
        sizer.Add(mode_ctrl_label, pos=(3, 0), flag=wx.EXPAND | wx.ALL, border=5)
        modes = ['RAW', 'LASCA', 'HISTOGRAM']
        self.mode_combo = wx.ComboBox(panel, choices=modes)
        sizer.Add(self.mode_combo, pos=(3, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.mode_combo.Bind(wx.EVT_COMBOBOX, self.OnModeCombo)
        self.mode_combo.SetSelection(2)

        framescap_ctrl_label = wx.StaticText(panel, label="Video length (sec):")
        sizer.Add(framescap_ctrl_label, pos=(16, 0),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.framescap_ctrl = wx.TextCtrl(panel)
        self.framescap_ctrl.SetValue(str(self.frames_to_capture))
        sizer.Add(self.framescap_ctrl, pos=(16, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.capture_btn = wx.Button(panel, label="Capture START")
        self.capture_btn.Bind(wx.EVT_BUTTON, self.OnCapture)
        sizer.Add(self.capture_btn, pos=(17, 0), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.framerate_ctrl_label = wx.StaticText(panel, label="Framerate (Hz):")
        sizer.Add(self.framerate_ctrl_label, pos=(4, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.framerate_slider = FS.FloatSpin(panel, -1,  min_val=0, max_val=1,
                                             size=(140, -1), increment=1.0,
                                             value=0.1, agwStyle=FS.FS_LEFT)
        self.framerate_slider.SetFormat("%f")
        self.framerate_slider.SetDigits(2)
        self.framerate_slider.Bind(FS.EVT_FLOATSPIN, self.FramerteSliderScroll)
        sizer.Add(self.framerate_slider, pos=(4, 1), span=(1, 1),
                  flag=wx.ALL, border=5)

        self.exposure_ctrl_label = wx.StaticText(panel, label="Exposure (us):")
        sizer.Add(self.exposure_ctrl_label, pos=(5, 0),  span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.exposure_slider = FS.FloatSpin(panel, -1,  min_val=0, max_val=1,
                                            size=(140, -1), increment=1.0,
                                            value=0.1, agwStyle=FS.FS_LEFT)
        self.exposure_slider.SetFormat("%f")
        self.exposure_slider.SetDigits(2)
        self.exposure_slider.Bind(FS.EVT_FLOATSPIN, self.ExposureSliderScroll)
        sizer.Add(self.exposure_slider, pos=(5, 1), span=(1, 1),
                  flag=wx.ALL, border=5)

        self.gain_ctrl_label = wx.StaticText(panel, label="Gain (dB):")
        sizer.Add(self.gain_ctrl_label, pos=(6, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.gain_slider = FS.FloatSpin(panel, -1,  min_val=0, max_val=1,
                                        size=(140, -1), increment=0.01, value=0.1,
                                        agwStyle=FS.FS_LEFT)
        self.gain_slider.Bind(FS.EVT_FLOATSPIN, self.GainSliderScroll)
        self.gain_slider.SetFormat("%f")
        self.gain_slider.SetDigits(3)
        sizer.Add(self.gain_slider, pos=(6, 1), span=(1, 1),
                  flag=wx.ALL, border=5)

        self.set_auto_exposure = wx.CheckBox(panel, label="Auto Exposure")
        sizer.Add(self.set_auto_exposure, pos=(7, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.set_auto_exposure.SetBackgroundColour(wx.NullColour)
        self.set_auto_exposure.Bind(wx.EVT_CHECKBOX, self.OnEnableAutoExposure)

        self.set_auto_gain = wx.CheckBox(panel, label="Auto Gain")
        sizer.Add(self.set_auto_gain, pos=(7, 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.set_auto_gain.SetBackgroundColour(wx.NullColour)
        self.set_auto_gain.Bind(wx.EVT_CHECKBOX, self.OnEnableAutoGain)

        self.set_roi = wx.CheckBox(panel, label="Set ROI")
        sizer.Add(self.set_roi, pos=(15, 3), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.set_roi.SetBackgroundColour(wx.NullColour)
        self.set_roi.Bind(wx.EVT_CHECKBOX, self.OnEnableRoi)

        offset_x_ctrl_label = wx.StaticText(panel, label="Offset X:")
        sizer.Add(offset_x_ctrl_label, pos=(16, 3), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_x_ctrl = wx.Slider(panel, value=0, minValue=0, maxValue=self.frame_width,
                                    size=(220, -1),
                                    style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.offset_x_ctrl, pos=(17, 3), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_x_ctrl.Bind(wx.EVT_SCROLL, self.OnSetOffsetX)

        offset_y_ctrl_label = wx.StaticText(panel, label="Offset Y:")
        sizer.Add(offset_y_ctrl_label, pos=(18, 3), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_y_ctrl = wx.Slider(panel, value=0, minValue=0, maxValue=self.frame_height,
                                    size=(220, 20),
                                    style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.offset_y_ctrl, pos=(19, 3), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_y_ctrl.Bind(wx.EVT_SCROLL, self.OnSetOffsetY)

        width_ctrl_label = wx.StaticText(panel, label="Width:")
        sizer.Add(width_ctrl_label, pos=(16, 4), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.width_ctrl = wx.Slider(panel, value=10, minValue=10,
                                        maxValue=self.frame_width, size=(220, -1),
                                        style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.width_ctrl, pos=(17, 4), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.width_ctrl.Bind(wx.EVT_SCROLL, self.OnSetWidth)

        height_ctrl_label = wx.StaticText(panel, label="Height:")
        sizer.Add(height_ctrl_label, pos=(18, 4), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.height_ctrl = wx.Slider(panel, value=10, minValue=10,
                                         maxValue=self.frame_height, size=(220, 20),
                                         style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.height_ctrl, pos=(19, 4), span=(1, 1),
                  flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.height_ctrl.Bind(wx.EVT_SCROLL, self.OnSetHeight)

        self.offset_x_ctrl.Disable()
        self.offset_y_ctrl.Disable()
        self.width_ctrl.Disable()
        self.height_ctrl.Disable()

        exportfile_ctrl_label = wx.StaticText(panel, label="Export file name:")
        sizer.Add(exportfile_ctrl_label, pos=(8, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.exportfile_ctrl = wx.TextCtrl(panel)
        sizer.Add(self.exportfile_ctrl, pos=(8, 1), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        exportfolder_ctrl_label = wx.StaticText(panel, label="Export directory:")
        sizer.Add(exportfolder_ctrl_label, pos=(9, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.select_folder_btn = wx.Button(panel, label="Select folder")
        self.select_folder_btn.Bind(wx.EVT_BUTTON, self.OnSelectFolder)
        sizer.Add(self.select_folder_btn, pos=(9, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.exportfolder_ctrl = wx.TextCtrl(panel)
        sizer.Add(self.exportfolder_ctrl, pos=(10, 0), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.exportfolder_ctrl.Disable()

        self.append_date = wx.CheckBox(panel, label="Append date and time")
        sizer.Add(self.append_date, pos=(11, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.append_date.SetBackgroundColour(wx.NullColour)
        self.append_date.Bind(wx.EVT_CHECKBOX, self.OnAppendDate)
        self.append_date.SetValue(True)  

        self.auto_index = wx.CheckBox(panel, label="Auto index")
        sizer.Add(self.auto_index, pos=(12, 0), span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.auto_index.SetBackgroundColour(wx.NullColour)
        self.auto_index.Bind(wx.EVT_CHECKBOX, self.OnAutoIndex)
        self.auto_index.SetValue(True)  # Set checkbox to checked by default

        self.index_ctrl = wx.TextCtrl(panel)
        self.index_ctrl.SetValue(str(1))
        sizer.Add(self.index_ctrl, pos=(12, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.current_state = wx.StaticText(panel, label="Cuttent state: idle")
        sizer.Add(self.current_state, pos=(18, 0), span=(1, 2),
                  flag=wx.EXPAND | wx.ALL, border=5)

        self.frame = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)
        self.frame[:] = 255

        self.display_frame = np.zeros([self.frame_height, self.frame_width, 3], dtype=np.uint8)
        self.display_frame[:] = 255

        self.Window = ImagePanel(panel)
        self.Window.SetSize((640, 480))
        self.Window.Fit()
        sizer.Add(self.Window, pos=(0, 3), span=(15, 3),
                  flag=wx.LEFT | wx.TOP | wx.EXPAND, border=5)

        lasca_filter_label = wx.StaticText(panel, label="LASCA filter size:")
        sizer.Add(lasca_filter_label, pos=(16, 5),
                  flag=wx.EXPAND | wx.ALL, border=5)
        modes = ['3x3', '5x5', '7x7', '9x9', '11x11']
        self.lasca_combo = wx.ComboBox(panel, choices=modes)
        sizer.Add(self.lasca_combo, pos=(17, 5), flag=wx.ALL, border=5)
        self.lasca_combo.Bind(wx.EVT_COMBOBOX, self.OnLascaCombo)
        self.lasca_combo.SetSelection(2)

        self.max_contrast_label = wx.StaticText(panel, label="Max contrast:")
        sizer.Add(self.max_contrast_label, pos=(18, 5),  span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.contrast_slider = FS.FloatSpin(panel, -1,  min_val=0.01,
                                            max_val=1, size=(140, -1),
                                            increment=0.01, value=0.8,
                                            agwStyle=FS.FS_LEFT)
        self.contrast_slider.SetFormat("%f")
        self.contrast_slider.SetDigits(2)
        self.contrast_slider.Bind(FS.EVT_FLOATSPIN, self.ContrastSliderScroll)
        sizer.Add(self.contrast_slider, pos=(19, 5), span=(1, 1),
                  flag=wx.ALL, border=5)

        self.min_gray_label = wx.StaticText(panel, label="Min gray:")
        sizer.Add(self.min_gray_label, pos=(16, 6),  span=(1, 1),
                  flag=wx.EXPAND | wx.ALL, border=5)
        self.gray_slider = FS.FloatSpin(panel, -1,  min_val=0,
                                        max_val=255, size=(140, -1),
                                        increment=1, value=5,
                                        agwStyle=FS.FS_LEFT)
        self.gray_slider.SetFormat("%f")
        self.gray_slider.SetDigits(2)
        self.gray_slider.Bind(FS.EVT_FLOATSPIN, self.GraySliderScroll)
        sizer.Add(self.gray_slider, pos=(17, 6), span=(1, 1), flag=wx.ALL,  border=5)

        self.preview_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.Draw, self.preview_timer)

        self.capture_status_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.capture_status, self.capture_status_timer)

        self.capture_sequence_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.count_elapsed, self.capture_sequence_timer)

        self.border = wx.BoxSizer()
        self.border.Add(sizer, 1, wx.ALL | wx.EXPAND, 20)

        panel.SetSizerAndFit(self.border)
        self.Fit()
        self.preview_thread_obj = threading.Thread(target=self.preview_thread)
        self.capture_thread_obj = threading.Thread(target=self.capture_thread)
        self.EnableGUI(False)

    def AllocateMemory(self):
        self.gray = np.zeros((self.frame_height, self.frame_width), np.uint8)
        self.mean_img_sq = np.zeros((self.frame_height, self.frame_width), np.float32)
        self.sq = np.zeros((self.frame_height, self.frame_width), np.float32)
        self.img = np.zeros((self.frame_height, self.frame_width), np.float32)
        self.mean_img = np.zeros((self.frame_height, self.frame_width), np.float32)
        self.sq_img_mean = np.zeros((self.frame_height, self.frame_width), np.float32)
        self.std = np.zeros((self.frame_height, self.frame_width), np.float32)
        self.LASCA = np.zeros((self.frame_height, self.frame_width), np.uint8)
        self.im_color = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self.mask = np.zeros((self.frame_height, self.frame_width), bool)

    def CalculateLASCA(self):
        self.img = self.frame.astype(np.float32, copy=False)
        cv2.filter2D(self.img, dst=self.mean_img, ddepth=cv2.CV_32F,
                     kernel=self.kernel)
        np.multiply(self.mean_img, self.mean_img, out=self.mean_img_sq)
        np.multiply(self.img, self.img, out=self.sq)
        cv2.filter2D(self.sq, dst=self.sq_img_mean, ddepth=cv2.CV_32F,
                     kernel=self.kernel)
        cv2.subtract(self.sq_img_mean, self.mean_img_sq, dst=self.std)
        cv2.sqrt(self.std, dst=self.std)
        self.mask = self.mean_img < self.min_gray_val
        cv2.pow(self.mean_img, power=-1.0, dst=self.mean_img)
        cv2.multiply(self.std, self.mean_img, dst=self.mean_img,
                     scale=255.0/self.max_contrast, dtype=cv2.CV_32F)
        self.mean_img[self.mean_img > 255.0] = 255.0
        self.LASCA = self.mean_img.astype(np.uint8)
        self.LASCA = 255 - self.LASCA
        self.LASCA[self.mask] = 0
        cv2.filter2D(self.LASCA, dst=self.LASCA, ddepth=cv2.CV_8U, kernel=self.kernel)

    def Draw(self, evt):

        self.lock.acquire()
        if (self.selected_mode == 0):
            print("RAW")
        if (self.selected_mode == 1):
            print("LASCA")

        if (self.selected_mode == 2):
            self.im_color = self.DrawHistogram(self.frame,
                                                   (640, 480),
                                                   (255, 255, 255),
                                                   (250, 155, 0))

            self.bitmap = wx.Image(640, 480,
                                   self.im_color.tobytes()).ConvertToBitmap()

        self.Window.update(self.bitmap)

        if self.preview_on is True:
            self.preview_timer.Start(50, oneShot=True)

        self.lock.release()

    def EnableGUI(self, value, preview=False):
        if value is True:
            self.interval_ctrl.Enable()
            self.sequence_ctrl.Enable()
            self.framescap_ctrl.Enable()
            self.framerate_slider.Enable()
            self.exposure_slider.Enable()
            self.gain_slider.Enable()
            self.exportfile_ctrl.Enable()
            self.cam_combo.Disable()
            self.encoding_mode_combo.Enable()
            self.preview_btn.Enable()
            self.set_roi.Enable()
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
            self.interval_ctrl.Disable()
            self.sequence_ctrl.Disable()
            self.framescap_ctrl.Disable()
            self.framerate_slider.Disable()
            self.exposure_slider.Disable()
            self.gain_slider.Disable()
            self.exportfile_ctrl.Disable()
            self.encoding_mode_combo.Disable()
            self.preview_btn.Enable()
            self.offset_x_ctrl.Disable()
            self.offset_y_ctrl.Disable()
            self.width_ctrl.Disable()
            self.height_ctrl.Disable()
            self.set_roi.Disable()
            self.select_folder_btn.Disable()
            self.capture_btn.Disable()
            self.append_date.Enable()
            self.set_auto_gain.Disable()
            self.set_auto_exposure.Disable()
            self.auto_index.Enable()
            self.index_ctrl.Disable()
        else:
            self.interval_ctrl.Disable()
            self.sequence_ctrl.Disable()
            self.framescap_ctrl.Disable()
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
            self.set_roi.Disable()
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
            self.interval_ctrl.Enable()
            self.sequence_ctrl.Enable()
            self.framescap_ctrl.Enable()
            self.framerate_slider.Enable()
            self.exportfile_ctrl.Enable()
            self.mode_combo.Enable()
            self.capmode_combo.Enable()
            self.preview_btn.Enable()
            self.set_roi.Enable()
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
            self.interval_ctrl.Disable()
            self.sequence_ctrl.Disable()
            self.framescap_ctrl.Disable()
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
            self.set_roi.Disable()
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
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")

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
                        
                        self.camera.MaxNumBuffer = 180
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

                        self.AllocateMemory()
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
                self.capture_btn.SetLabel("Capture STOP")
            else:
                self.capture_on = False
                self.current_step = 0
                self.StopCapture()
                self.capture_btn.SetLabel("Capture START")
                self.current_state.SetLabel("Current state: idle")
                self.connect_btn.Enable()
                self.StartPreview()
        else:
            self.current_step = 0
            self.StopCapture()
            self.capture_btn.SetLabel("Capture START")
            self.current_state.SetLabel("Current state: idle")
            self.connect_btn.Enable()

    def FramerteSliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.framerate = val
        if self.camera_connected is True:
            self.camera.AcquisitionFrameRate.SetValue(self.framerate)
            resulting_framerate = self.camera.ResultingFrameRate.GetValue()
            if (resulting_framerate != self.framerate):
                self.framerate = resulting_framerate
                self.framerate_slider.SetValue(self.framerate)

    def ExposureSliderScroll(self, event):
        obj = event.GetEventObject()
        val = obj.GetValue()
        self.exposure = val
        if self.camera_connected is True:
            self.camera.ExposureTime.SetValue(self.exposure)
            resulting_framerate = self.camera.ResultingFrameRate.GetValue()
            if (resulting_framerate != self.framerate):
                self.framerate = resulting_framerate
                self.framerate_slider.SetValue(self.framerate)

    def OnAutoIndex(self, event):
        if self.camera_connected is True:
            self.auto_index_on = self.auto_index.GetValue()
            if self.auto_index_on is True:
                self.index_ctrl.Disable()
                self.current_index = int(self.index_ctrl.GetValue())
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

    def OnLascaCombo(self, event):
        current_selection = self.lasca_combo.GetSelection()
        filter_size = int(2*(current_selection+2) - 1)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (filter_size, filter_size)).astype(np.float32)
        self.kernel /= np.sum(self.kernel)

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
        histogram_image = np.ones((256, 256, 3), np.uint8)*240
        R, G, B = bcg_color
        histogram_image[:, :, 0] = B
        histogram_image[:, :, 1] = G
        histogram_image[:, :, 2] = R

        for column in range(0, len(histogram_data)):
            column_height = int(np.floor((histogram_data[column]/100)*256))
            if column_height > 1:
                R, G, B = bin_color
                color = (B, G, R)
                cv2.line(histogram_image, (column, 255),
                         (column, 255-column_height), color, 1)

        resized = cv2.resize(histogram_image, size, interpolation=cv2.INTER_AREA)
        return resized

    def OnAppendDate(self, event):
        self.append_date_flag = self.append_date.GetValue()

    def OnEnableRoi(self, event):
        self.roi_on = self.set_roi.GetValue()
        if self.roi_on is True:
            self.offset_x_ctrl.Enable()
            self.offset_y_ctrl.Enable()
            self.width_ctrl.Enable()
            self.height_ctrl.Enable()
        else:
            self.offset_x_ctrl.Disable()
            self.offset_y_ctrl.Disable()
            self.width_ctrl.Disable()
            self.height_ctrl.Disable()

    def OnSetOffsetX(self, event):
        new_offset_x = self.offset_x_ctrl.GetValue()
        # Check if the new offset + width is divisible by 2
        new_width = new_offset_x + self.frame_width
        new_width = int(16 * round(new_width / 16)) if new_width % 16 != 0 else new_width
        
        new_offset_x = new_width - self.frame_width
        
        if (new_offset_x + self.frame_width) < self.max_frame_width:
            self.offset_x = new_offset_x
            self.camera.OffsetX.SetValue(self.offset_x)
        
        self.offset_x_ctrl.SetValue(self.offset_x)

    def OnSetOffsetY(self, event):
        new_offset_y = self.offset_y_ctrl.GetValue()
        # Check if the new offset + height is divisible by 2
        new_height = new_offset_y + self.frame_height
        new_height = int(4 * round(new_height / 4)) if new_height % 4 != 0 else new_height
        
        new_offset_y = new_height - self.frame_height
        
        if (new_offset_y + self.frame_height) < self.max_frame_height:
            self.offset_y = new_offset_y
            self.camera.OffsetY.SetValue(self.offset_y)
        
        self.offset_y_ctrl.SetValue(self.offset_y)

    def OnSetWidth(self, event):
        new_width = self.width_ctrl.GetValue()
        # Check if the new width is divisible by 2
        new_width = int(16 * round(new_width / 16)) if new_width % 16 != 0 else new_width
            
        if (self.offset_x + new_width) < self.max_frame_width:
            self.frame_width = new_width
            self.camera.Width.SetValue(self.frame_width)
            self.offset_x_ctrl.SetMax(self.max_frame_width - self.frame_width)
        
        self.width_ctrl.SetValue(self.frame_width)

    def OnSetHeight(self, event):  
        new_height = self.height_ctrl.GetValue()
        # Check if the new height is divisible by 2
        new_height = int(4 * round(new_height / 4)) if new_height % 4 != 0 else new_height
            
        if (self.offset_y + new_height) < self.max_frame_height:
            self.frame_height = new_height
            self.camera.Height.SetValue(self.frame_height)
            self.offset_y_ctrl.SetMax(self.max_frame_height - self.frame_height)
        
        self.height_ctrl.SetValue(self.frame_height)

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
        # Create Image Windows to display live video while capturing
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1, 0, 0, 640, 480)
        
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        self.previous_time = int(round(time.time() * 1000))
       
        current_date_and_time = str(datetime.datetime.now())
        last_display_time = time.time()
        display_interval = 1/30  # Update display every 1/60 seconds (to match 60Hz refresh rate)
         
        while self.preview_on is True:
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000,
                                                        pylon.TimeoutHandling_ThrowException)
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

        imageWindow.Close()
        self.camera.StopGrabbing()

    def StartCapture(self):
        self.StopPreview()
        self.SetupCapture()
        
        # Start the capture and display threads
        self.capture_thread_obj = threading.Thread(target=self.capture_thread)
        self.capture_thread_obj.start()
        
        self.EnableGUI(False)
        self.connect_btn.Disable()
        self.capture_btn.Enable()
        self.capture_status_timer.Start(10000, oneShot=True)
    
    def SetupCapture(self):
        # Prepare data output file before starting capture
        sequence_length = int(self.sequence_ctrl.GetValue())
        video_length = float(self.framescap_ctrl.GetValue())
        frames_to_capture = int(video_length * self.framerate)
        interval_length = float(self.interval_ctrl.GetValue())
        
        fourcc_code = str(self.encoding_mode_combo.GetValue())

        output_path = []
        output_file_name = self.exportfile_ctrl.GetValue()
        if len(output_file_name) <= 1:
            output_file_name = "output"
            
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
            self.current_step = sequence_length
            return

        if len(output_folder_name) <= 1:
            wx.MessageBox('Please provide output folder!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            self.current_step = sequence_length
            return

        if self.append_date_flag is True:
            output_path = output_path + "_" + time.strftime("%Y%m%d_%H%M%S")

        if self.auto_index_on is True:
            output_path = output_path + "_" + str(self.current_index)
            self.current_index += 1

        if self.auto_index_on is False and self.append_date_flag is False:
            if sequence_length > 1:
                wx.MessageBox('Turn on auto indexing or append date to' +
                                ' file name when capturing sequence!',
                                'Warning', wx.OK | wx.ICON_WARNING)
                self.capture_on = False
                self.current_step = sequence_length
                return

        if len(output_path) <= 4:
            wx.MessageBox('Invalid name for data output file!',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            self.current_step = sequence_length
            return

        if sequence_length < 1:
            wx.MessageBox('Invalid length of measurement sequence! Minimum' +
                          ' required value is 1.',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            return
        
        if sequence_length > 1:
            if(video_length > interval_length):
                wx.MessageBox('Interval length should be greater than video length',
                              'Warning', wx.OK | wx.ICON_WARNING)
                self.capture_on = False
                return

        if frames_to_capture < 1:
            wx.MessageBox('Invalid number of frames to capture! Minimum' +
                          ' required value is 5 frames.',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.capture_on = False
            self.current_step = sequence_length
            return
        
        # Making sure the output file is .avi
        if not output_path.endswith('.avi'):
            output_path += '.avi'

        # Configure session
        # Prepare data output file and buffer
        self.video_session = VideoRecordingSession(cam_num=0)
        print(f"Frame width: {self.frame_width}, Frame height: {self.frame_height}")
        
        # TODO: add more options for output file
        self.video_session.set_params(
            video_file=output_path,
            fourcc=fourcc_code,
            fps=200,
            dim=(self.frame_width, self.frame_height)
        )
 
    def StopCapture(self):
        if self.capture_thread_obj.is_alive() is True:
            self.capture_thread_obj.join()
        # if self.display_thread_thread_obj.is_alive() is True:
        #     self.display_thread_thread_obj.join()
        self.EnableGUI(True)
        self.capture_status_timer.Stop()
        self.capture_sequence_timer.Stop()

    def capture_thread(self):
        # Indefinite capture mode
        self.capture_on = True
        
        # Create Image Windows to display live video while capturing
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1)
        
        # Enable chunks in general.
        self.camera.ChunkModeActive.Value = True
        
        # Enable time stamp chunks.
        self.camera.ChunkSelector.Value = "Timestamp"
        self.camera.ChunkEnable.Value = True
        # Start the video recording session
        self.video_session.start_recording()
        
        # Start the camera grabbing
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

        current_date_and_time = str(datetime.datetime.now())
        last_display_time = time.time()
        display_interval = 1/30  # Update display every 1/60 seconds (to match 60Hz refresh rate)
        
        print(f'Capturing video started at: {current_date_and_time}')
        
        captured_frames = 0
        while self.camera.IsGrabbing() and self.capture_on is True:
            grabResult = self.camera.RetrieveResult(500,
                                                    pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                frame = grabResult.GetArray()
                timestamp = time.time()
                frame_number = grabResult.BlockID
                print("TimeStamp (Result): ", grabResult.ChunkTimestamp.Value)
                captured_frames += 1

                self.video_session.acquire_frame(frame, timestamp, frame_number)
                
                if (timestamp - last_display_time) > display_interval:
                    line_status = self.camera.LineStatus.GetValue()  # Retrieve line status
                    imageWindow.SetImage(grabResult)
                    imageWindow.Show()
                    last_display_time = time.time()
                
                time.sleep(0.00001)
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

    def capture_status(self, evt):
        if self.capture_on is True:
            self.capture_status_timer.Start(200, oneShot=True)
            self.current_state.SetLabel("Current status: capturing data!")
            return
        else:
            sequence_length = int(self.sequence_ctrl.GetValue())
            if sequence_length == 1:
                self.current_state.SetLabel("Current state: idle")
                self.EnableGUI(True)
                self.connect_btn.Enable()
                self.capture_btn.SetLabel("Capture START")
                self.StartPreview()
                return
            else:
                self.current_step += 1
                if sequence_length > self.current_step:
                    self.capture_btn.SetLabel("Capture STOP")
                    correction = int(time.time() - self.last_capture_time)
                    self.time_to_next = int(self.interval_ctrl.GetValue()) - correction
                    time_to_next_str = str(datetime.timedelta(seconds=self.time_to_next))
                    self.current_state.SetLabel(
                        f"Current status: step {self.current_step} out of" +
                        f"{sequence_length}" +
                        "time to next: " + time_to_next_str)
                    self.capture_sequence_timer.Start(1000, oneShot=True)
                    self.capture_btn.Enable()
                    self.preview_btn.Enable()
                    self.StartPreview()
                    return
                else:
                    if self.index_ctrl.GetValue() == '':
                        self.index_ctrl.SetValue(str(1))
                        self.current_index = 1
                    else:
                        self.current_index = int(self.index_ctrl.GetValue())
                    self.current_step = 0
                    self.capture_btn.SetLabel("Capture START")
                    self.current_state.SetLabel("Current state: idle")
                    self.EnableGUI(True)
                    self.connect_btn.Enable()
                    self.StartPreview()
                    return
        return

    def count_elapsed(self, evt):
        self.time_to_next -= 1
        time_to_next_str = str(datetime.timedelta(seconds=self.time_to_next))
        sequence_length = int(self.sequence_ctrl.GetValue())
        self.current_state.SetLabel(
            f"Current status: step {self.current_step} out of" +
            f"{sequence_length}" +
            "time to next: " + time_to_next_str)
        if self.time_to_next > 0:
            self.capture_sequence_timer.Start(1000, oneShot=True)
        else:
            self.StartCapture()
    
    @staticmethod
    def precise_sleep(duration):
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < duration:
            pass  # Busy-waiting until the time has elapsed


if __name__ == '__main__':
    app = wx.App()
    ex = BaslerGuiWindow(None)
    app.MainLoop()
