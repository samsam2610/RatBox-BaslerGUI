import numpy as np
import cv2
from pypylon import pylon
from VideoRecordingSession import VideoRecordingSession
from InputEventHandler import ConfigurationEventPrinter

class BaslerModel:
    def __init__(self):
        self.camera_connected = False
        self.camera = None
        self.frame_width = 1440
        self.frame_height = 1088
        self.offset_x = 16
        self.offset_y = 0
        self.max_frame_width = 1456
        self.max_frame_height = 1088
        self.framerate = 120
        self.exposure = 7
        self.gain = 0
        self.auto_exposure_on = False
        self.auto_gain_on = False
        self.preview_on = False
        self.capture_on = False
        self.video_session = VideoRecordingSession(cam_num=0)
        self.current_frame = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.gray = np.zeros((self.frame_height, self.frame_width, 1), np.uint8)
        self.mean_img_sq = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.sq = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.img = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.mean_img = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.sq_img_mean = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.std = np.zeros((self.frame_height, self.frame_width, 1), np.float32)
        self.LASCA = np.zeros((self.frame_height, self.frame_width, 1), np.uint8)
        self.im_color = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self.mask = np.zeros((self.frame_height, self.frame_width, 1), bool)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)).astype(np.float32)
        self.kernel /= np.sum(self.kernel)

    def connect_camera(self, selected_camera, cameras_list):
        if not self.camera_connected:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")

            device_name = cameras_list[selected_camera]["name"]
            device_serial = cameras_list[selected_camera]["serial"]

            for i, device in enumerate(devices):
                if device.GetModelName() == device_name and device.GetSerialNumber() == device_serial:
                    self.camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[i]))
                    self.camera.Open()

                    # Setup line 3 to register front sensor trigger
                    self.camera.LineSelector.Value = "Line3"
                    self.camera.LineMode.Value = "Input"
                    self.camera.RegisterConfiguration(ConfigurationEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
                    self.camera.MaxNumBuffer = 180
                    self.camera.AcquisitionFrameRateEnable.SetValue(True)
                    self.camera.AcquisitionFrameRate.SetValue(200.0)
                    self.camera.GainAuto.SetValue("Off")
                    self.camera.ExposureAuto.SetValue("Off")
                    self.camera_connected = True
                    self.allocate_memory()
                    return True
        else:
            self.camera.Close()
            self.camera_connected = False
            return False

    def allocate_memory(self):
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

    def calculate_lasca(self):
        self.img = self.frame.astype(np.float32, copy=False)
        cv2.filter2D(self.img, dst=self.mean_img, ddepth=cv2.CV_32F, kernel=self.kernel)
        np.multiply(self.mean_img, self.mean_img, out=self.mean_img_sq)
        np.multiply(self.img, self.img, out=self.sq)
        cv2.filter2D(self.sq, dst=self.sq_img_mean, ddepth=cv2.CV_32F, kernel=self.kernel)
        cv2.subtract(self.sq_img_mean, self.mean_img_sq, dst=self.std)
        cv2.sqrt(self.std, dst=self.std)
        self.mask = self.mean_img < self.min_gray_val
        cv2.pow(self.mean_img, power=-1.0, dst=self.mean_img)
        cv2.multiply(self.std, self.mean_img, dst=self.mean_img, scale=255.0/self.max_contrast, dtype=cv2.CV_32F)
        self.mean_img[self.mean_img > 255.0] = 255.0
        self.LASCA = self.mean_img.astype(np.uint8)
        self.LASCA = 255 - self.LASCA
        self.LASCA[self.mask] = 0
        cv2.filter2D(self.LASCA, dst=self.LASCA, ddepth=cv2.CV_8U, kernel=self.kernel)
