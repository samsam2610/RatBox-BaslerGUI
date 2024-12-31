import wx
import wx.lib.agw.floatspin as FS
import numpy as np

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
    def __init__(self, controller, *args, **kwargs):
        super(BaslerGuiWindow, self).__init__(*args, **kwargs)
        self.controller = controller
        self.InitUI()
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.Centre()
        self.Show()

    def InitUI(self):
        panel = wx.Panel(self)
        self.SetTitle('Basler CAM GUI')
        sizer = wx.GridBagSizer(0, 0)

        selected_ctrl_label = wx.StaticText(panel, label="Selected camera:")
        sizer.Add(selected_ctrl_label, pos=(0, 0), flag=wx.EXPAND | wx.ALL, border=5)
        self.cam_combo = wx.ComboBox(panel, choices=self.controller.get_camera_list())
        sizer.Add(self.cam_combo, pos=(1, 0), flag=wx.EXPAND | wx.ALL, border=5)
        self.cam_combo.Bind(wx.EVT_COMBOBOX, self.controller.OnCamCombo)
        self.cam_combo.SetSelection(self.controller.selected_camera)

        self.connect_btn = wx.Button(panel, label="Connect")
        self.connect_btn.Bind(wx.EVT_BUTTON, self.controller.OnConnect)
        sizer.Add(self.connect_btn, pos=(1, 1), flag=wx.EXPAND | wx.ALL, border=5)

        self.refresh_btn = wx.Button(panel, label="Refresh list")
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.controller.OnRefreshList)
        sizer.Add(self.refresh_btn, pos=(2, 0), flag=wx.EXPAND | wx.ALL, border=5)

        self.preview_btn = wx.Button(panel, label="Preview START")
        self.preview_btn.Bind(wx.EVT_BUTTON, self.controller.OnPreview)
        sizer.Add(self.preview_btn, pos=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)

        # ...existing code for other UI elements...

        offset_x_ctrl_label = wx.StaticText(panel, label="Offset X:")
        sizer.Add(offset_x_ctrl_label, pos=(16, 3), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_x_ctrl = wx.Slider(panel, value=0, minValue=0, maxValue=self.controller.model.frame_width,
                                       size=(220, -1), style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.offset_x_ctrl, pos=(17, 3), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_x_ctrl.Bind(wx.EVT_SCROLL, self.controller.OnSetOffsetX)

        offset_y_ctrl_label = wx.StaticText(panel, label="Offset Y:")
        sizer.Add(offset_y_ctrl_label, pos=(18, 3), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_y_ctrl = wx.Slider(panel, value=0, minValue=0, maxValue=self.controller.model.frame_height,
                                       size=(220, 20), style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.offset_y_ctrl, pos=(19, 3), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.offset_y_ctrl.Bind(wx.EVT_SCROLL, self.controller.OnSetOffsetY)

        width_ctrl_label = wx.StaticText(panel, label="Width:")
        sizer.Add(width_ctrl_label, pos=(16, 4), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.width_ctrl = wx.Slider(panel, value=10, minValue=10, maxValue=self.controller.model.frame_width,
                                    size=(220, -1), style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.width_ctrl, pos=(17, 4), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.width_ctrl.Bind(wx.EVT_SCROLL, self.controller.OnSetWidth)

        height_ctrl_label = wx.StaticText(panel, label="Height:")
        sizer.Add(height_ctrl_label, pos=(18, 4), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.height_ctrl = wx.Slider(panel, value=10, minValue=10, maxValue=self.controller.model.frame_height,
                                     size=(220, 20), style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sizer.Add(self.height_ctrl, pos=(19, 4), span=(1, 1), flag=wx.ALL | wx.ALIGN_CENTER, border=0)
        self.height_ctrl.Bind(wx.EVT_SCROLL, self.controller.OnSetHeight)

        self.offset_x_ctrl.Disable()
        self.offset_y_ctrl.Disable()
        self.width_ctrl.Disable()
        self.height_ctrl.Disable()

        self.Window = ImagePanel(panel)
        self.Window.SetSize((640, 480))
        self.Window.Fit()
        sizer.Add(self.Window, pos=(0, 3), span=(15, 3), flag=wx.LEFT | wx.TOP | wx.EXPAND, border=5)

        self.border = wx.BoxSizer()
        self.border.Add(sizer, 1, wx.ALL | wx.EXPAND, 20)
        panel.SetSizerAndFit(self.border)
        self.Fit()

    def EnableGUI(self, value, preview=False):
        if value:
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

            if self.controller.model.auto_exposure_on:
                self.set_auto_exposure.Enable()
                self.exposure_slider.Disable()
            else:
                self.exposure_slider.Enable()

            if self.controller.model.auto_gain_on:
                self.set_auto_exposure.Enable()
                self.gain_slider.Disable()
            else:
                self.gain_slider.Enable()

            if self.controller.model.auto_index_on:
                self.auto_index.Enable()
                self.index_ctrl.Disable()
            else:
                self.index_ctrl.Enable()
        elif preview:
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

    def BlockGUI(self, value):
        if value:
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

            if self.controller.model.roi_on:
                self.offset_x_ctrl.Enable()
                self.offset_y_ctrl.Enable()
                self.width_ctrl.Enable()
                self.height_ctrl.Enable()

            if self.controller.model.auto_exposure_on:
                self.set_auto_exposure.Enable()
                self.exposure_slider.Disable()
            else:
                self.exposure_slider.Enable()

            if self.controller.model.auto_gain_on:
                self.set_auto_exposure.Enable()
                self.gain_slider.Disable()
            else:
                self.gain_slider.Enable()

            if self.controller.model.auto_index_on:
                self.auto_index.Enable()
                self.index_ctrl.Disable()
            else:
                self.index_ctrl.Enable()
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

    def OnCloseWindow(self, event):
        self.controller.OnCloseWindow(event)
        self.Destroy()
