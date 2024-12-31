import threading
import time
import datetime
from BaslerModel import BaslerModel
from BaslerView import BaslerGuiWindow

class BaslerController:
    def __init__(self):
        self.model = BaslerModel()
        self.view = BaslerGuiWindow(self, None)
        self.selected_camera = 0
        self.cameras_list = []

    def get_camera_list(self):
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        cameras = []
        for device in devices:
            cameras.append(device.GetModelName() + "_" + device.GetSerialNumber())
            self.cameras_list.append({"name": device.GetModelName(), "serial": device.GetSerialNumber()})
        return cameras

    def OnConnect(self, event):
        if self.model.connect_camera(self.selected_camera, self.cameras_list):
            self.view.connect_btn.SetLabel("Disconnect")
            self.view.refresh_btn.Disable()
            self.view.cam_combo.Disable()

            # Get the current frame width, height, and offset
            self.model.frame_width = self.model.camera.Width.GetValue()
            self.model.frame_height = self.model.camera.Height.GetValue()
            self.model.offset_x = self.model.camera.OffsetX.GetValue()
            self.model.offset_y = self.model.camera.OffsetY.GetValue()

            # Set the frame width, height, and offset
            self.view.width_ctrl.SetMax(self.model.max_frame_width)
            self.view.width_ctrl.SetValue(self.model.frame_width)

            self.view.height_ctrl.SetMax(self.model.max_frame_height)
            self.view.height_ctrl.SetValue(self.model.frame_height)

            self.view.offset_x_ctrl.SetMax(self.model.max_frame_width - self.model.frame_width)
            self.view.offset_x_ctrl.SetValue(self.model.offset_x)

            self.view.offset_y_ctrl.SetMax(self.model.max_frame_height - self.model.frame_height)
            self.view.offset_y_ctrl.SetValue(self.model.offset_y)

            self.model.camera_connected = True
            self.model.allocate_memory()
            self.view.EnableGUI(True)
        else:
            self.view.connect_btn.SetLabel("Connect")
            self.view.refresh_btn.Enable()
            self.view.cam_combo.Enable()

    def OnCloseWindow(self, event):
        self.model.preview_on = False
        self.model.capture_on = False
        if self.model.camera_connected:
            self.model.camera.Close()
        self.view.Destroy()

    def OnRefreshList(self, event):
        self.selected_camera = 0
        self.view.cam_combo.Clear()
        self.view.cam_combo.AppendItems(self.get_camera_list())
        self.view.cam_combo.SetSelection(self.selected_camera)

    def OnPreview(self, event):
        if self.model.camera_connected:
            if self.model.preview_on:
                self.StopPreview()
            else:
                self.StartPreview()

    def StartPreview(self):
        self.model.preview_on = True
        self.view.EnableGUI(False, preview=True)
        self.model.preview_thread_obj = threading.Thread(target=self.preview_thread)
        self.model.preview_thread_obj.start()
        self.view.preview_timer.Start(100, oneShot=True)
        self.view.preview_btn.SetLabel("Preview STOP")

    def StopPreview(self):
        self.model.preview_on = False
        self.view.EnableGUI(True)
        if self.model.preview_thread_obj.is_alive():
            self.model.preview_thread_obj.join()
        self.view.preview_timer.Stop()
        self.view.preview_btn.SetLabel("Preview START")

    def preview_thread(self):
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1, 0, 0, 640, 480)
        self.model.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        last_display_time = time.time()
        display_interval = 1/30

        while self.model.preview_on:
            if self.model.camera.IsGrabbing():
                grabResult = self.model.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    self.model.frame = grabResult.GetArray()
                    timestamp = time.time()
                    if (timestamp - last_display_time) > display_interval:
                        imageWindow.SetImage(grabResult)
                        imageWindow.Show()
                        last_display_time = time.time()
                grabResult.Release()

        imageWindow.Close()
        self.model.camera.StopGrabbing()

    def capture_thread(self):
        # Indefinite capture mode
        self.model.capture_on = True
        
        # Create Image Windows to display live video while capturing
        imageWindow = pylon.PylonImageWindow()
        imageWindow.Create(1)
        
        # Enable chunks in general.
        self.model.camera.ChunkModeActive.Value = True
        
        # Enable time stamp chunks.
        self.model.camera.ChunkSelector.Value = "Timestamp"
        self.model.camera.ChunkEnable.Value = True
        
        # Enable line status chunks.
        self.model.camera.ChunkSelector.Value = "LineStatusAll"
        self.model.camera.ChunkEnable.Value = True
        
        # Start the video recording session
        self.model.video_session.start_recording()
        
        # Start the camera grabbing
        self.model.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

        current_date_and_time = str(datetime.datetime.now())
        last_display_time = time.time()
        display_interval = 1/30  # Update display every 1/30 seconds (to match 30Hz refresh rate)
        
        print(f'Capturing video started at: {current_date_and_time}')
        
        captured_frames = 0
        while self.model.camera.IsGrabbing() and self.model.capture_on:
            grabResult = self.model.camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                frame = grabResult.GetArray()
                timestamp = time.time()
                frame_timestamp = grabResult.ChunkTimestamp.Value
                frame_line_status = grabResult.ChunkLineStatusAll.Value
                captured_frames += 1

                self.model.video_session.acquire_frame(frame, frame_timestamp, captured_frames, frame_line_status)
                
                if (timestamp - last_display_time) > display_interval:
                    line_status = self.model.camera.LineStatus.GetValue()  # Retrieve line status
                    imageWindow.SetImage(grabResult)
                    imageWindow.Show()
                    last_display_time = time.time()
                
                time.sleep(0.00001)
            else:
                print("Error: ", grabResult.ErrorCode)
            
            grabResult.Release()

        self.model.camera.StopGrabbing()
        imageWindow.Close()
        self.model.video_session.stop_recording()

        print(f'Capturing finished after grabbing {captured_frames} frames')

    def OnCapture(self, event):
        if self.model.current_step == 0:
            if not self.model.capture_on:
                self.StartCapture()
                self.view.capture_btn.SetLabel("Capture STOP")
            else:
                self.model.capture_on = False
                self.model.current_step = 0
                self.StopCapture()
                self.view.capture_btn.SetLabel("Capture START")
                self.view.current_state.SetLabel("Current state: idle")
                self.view.connect_btn.Enable()
                self.StartPreview()
        else:
            self.model.current_step = 0
            self.StopCapture()
            self.view.capture_btn.SetLabel("Capture START")
            self.view.current_state.SetLabel("Current state: idle")
            self.view.connect_btn.Enable()

    def StartCapture(self):
        self.StopPreview()
        self.SetupCapture()
        
        # Start the capture and display threads
        self.model.capture_thread_obj = threading.Thread(target=self.capture_thread)
        self.model.capture_thread_obj.start()
        
        self.view.EnableGUI(False)
        self.view.connect_btn.Disable()
        self.view.capture_btn.Enable()
        self.view.capture_status_timer.Start(10000, oneShot=True)
    
    def StopCapture(self):
        if self.model.capture_thread_obj.is_alive():
            self.model.capture_thread_obj.join()
        self.view.EnableGUI(True)
        self.view.capture_status_timer.Stop()
        self.view.capture_sequence_timer.Stop()

    def SetupCapture(self):
        # Prepare data output file before starting capture
        sequence_length = int(self.view.sequence_ctrl.GetValue())
        video_length = float(self.view.framescap_ctrl.GetValue())
        frames_to_capture = int(video_length * self.model.framerate)
        interval_length = float(self.view.interval_ctrl.GetValue())
        
        fourcc_code = str(self.view.encoding_mode_combo.GetValue())

        output_path = []
        output_file_name = self.view.exportfile_ctrl.GetValue()
        if len(output_file_name) <= 1:
            output_file_name = "output"
            
        output_folder_name = self.view.exportfolder_ctrl.GetValue()
        if len(output_folder_name) <= 1:
            output_folder_name = "C:\\"
        if len(output_folder_name) > 0:
            output_path = output_folder_name + "\\" + output_file_name
        else:
            output_path = output_file_name

        if len(output_file_name) <= 1:
            wx.MessageBox('Please provide output file name!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.model.capture_on = False
            self.model.current_step = sequence_length
            return

        if len(output_folder_name) <= 1:
            wx.MessageBox('Please provide output folder!', 'Warning',
                          wx.OK | wx.ICON_WARNING)
            self.model.capture_on = False
            self.model.current_step = sequence_length
            return

        if self.model.append_date_flag:
            output_path = output_path + "_" + time.strftime("%Y%m%d_%H%M%S")

        if self.model.auto_index_on:
            output_path = output_path + "_" + str(self.model.current_index)
            self.model.current_index += 1

        if not self.model.auto_index_on and not self.model.append_date_flag:
            if sequence_length > 1:
                wx.MessageBox('Turn on auto indexing or append date to' +
                                ' file name when capturing sequence!',
                                'Warning', wx.OK | wx.ICON_WARNING)
                self.model.capture_on = False
                self.model.current_step = sequence_length
                return

        if len(output_path) <= 4:
            wx.MessageBox('Invalid name for data output file!',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.model.capture_on = False
            self.model.current_step = sequence_length
            return

        if sequence_length < 1:
            wx.MessageBox('Invalid length of measurement sequence! Minimum' +
                          ' required value is 1.',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.model.capture_on = False
            return
        
        if sequence_length > 1:
            if(video_length > interval_length):
                wx.MessageBox('Interval length should be greater than video length',
                              'Warning', wx.OK | wx.ICON_WARNING)
                self.model.capture_on = False
                return

        if frames_to_capture < 1:
            wx.MessageBox('Invalid number of frames to capture! Minimum' +
                          ' required value is 5 frames.',
                          'Warning', wx.OK | wx.ICON_WARNING)
            self.model.capture_on = False
            self.model.current_step = sequence_length
            return
        
        # Making sure the output file is .avi
        if not output_path.endswith('.avi'):
            output_path += '.avi'

        # Configure session
        # Prepare data output file and buffer
        self.model.video_session = VideoRecordingSession(cam_num=0)
        print(f"Frame width: {self.model.frame_width}, Frame height: {self.model.frame_height}")
        
        # TODO: add more options for output file
        self.model.video_session.set_params(
            video_file=output_path,
            fourcc=fourcc_code,
            fps=200,
            dim=(self.model.frame_width, self.model.frame_height)
        )

    def OnSetOffsetX(self, event):
        new_offset_x = self.view.offset_x_ctrl.GetValue()
        new_width = new_offset_x + self.model.frame_width
        new_width = int(16 * round(new_width / 16)) if new_width % 16 != 0 else new_width
        new_offset_x = new_width - self.model.frame_width
        
        if (new_offset_x + self.model.frame_width) < self.model.max_frame_width:
            self.model.offset_x = new_offset_x
            self.model.camera.OffsetX.SetValue(self.model.offset_x)
        
        self.view.offset_x_ctrl.SetValue(self.model.offset_x)

    def OnSetOffsetY(self, event):
        new_offset_y = self.view.offset_y_ctrl.GetValue()
        new_height = new_offset_y + self.model.frame_height
        new_height = int(4 * round(new_height / 4)) if new_height % 4 != 0 else new_height
        new_offset_y = new_height - self.model.frame_height
        
        if (new_offset_y + self.model.frame_height) < self.model.max_frame_height:
            self.model.offset_y = new_offset_y
            self.model.camera.OffsetY.SetValue(self.model.offset_y)
        
        self.view.offset_y_ctrl.SetValue(self.model.offset_y)

    def OnSetWidth(self, event):
        new_width = self.view.width_ctrl.GetValue()
        new_width = int(16 * round(new_width / 16)) if new_width % 16 != 0 else new_width
            
        if (self.model.offset_x + new_width) < self.model.max_frame_width:
            self.model.frame_width = new_width
            self.model.camera.Width.SetValue(self.model.frame_width)
            self.view.offset_x_ctrl.SetMax(self.model.max_frame_width - self.model.frame_width)
        
        self.view.width_ctrl.SetValue(self.model.frame_width)

    def OnSetHeight(self, event):
        new_height = self.view.height_ctrl.GetValue()
        new_height = int(4 * round(new_height / 4)) if new_height % 4 != 0 else new_height
            
        if (self.model.offset_y + new_height) < self.model.max_frame_height:
            self.model.frame_height = new_height
            self.model.camera.Height.SetValue(self.model.frame_height)
            self.view.offset_y_ctrl.SetMax(self.model.max_frame_height - self.model.frame_height)
        
        self.view.height_ctrl.SetValue(self.model.frame_height)
