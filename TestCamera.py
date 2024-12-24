import cv2
import threading
import time
from collections import deque
from pypylon import pylon


class VideoRecordingSession:
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.recording_status = False
        self.vid_out = None
        self.frame_buffer = deque(maxlen=5000)
        self.buffer_lock = threading.Lock()
        self.frame_count = 0
        self.frame_times = []
        self.frame_num = []
        self.dim = None
        self.fourcc = None
        self.fps = None

    def set_params(self, video_file, fourcc, fps, dim):
        self.video_file = video_file
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.fps = fps
        self.dim = dim
        self.vid_out = cv2.VideoWriter(video_file, self.fourcc, fps, dim, False)
        print(f"Cam {self.cam_num}: Video writer set up with {video_file}")
        if not self.vid_out.isOpened():
            print(f"Error: Failed to initialize video writer for {video_file}")
        else:
            print(f"Cam {self.cam_num}: Video writer initialized successfully")

    def acquire_frame(self, frame, timestamp, frame_number):
        with self.buffer_lock:
            self.frame_buffer.append((frame, timestamp, frame_number))

    def start_recording(self):
        self.recording_status = True
        processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        processing_thread.start()
        print(f"Cam {self.cam_num}: Recording started.")

    def stop_recording(self):
        self.recording_status = False
        time.sleep(0.1)  # Allow buffer processing to finish
        self.write_remaining_frames()
        if self.vid_out:
            print(f"Cam {self.cam_num}: Releasing video writer")
            self.vid_out.release()
            self.vid_out = None
        print(f"Cam {self.cam_num}: Recording stopped.")

    def write_remaining_frames(self):
        with self.buffer_lock:
            while self.frame_buffer:
                self._write_frame()

    def _process_frames(self):
        print(f"Cam {self.cam_num}: Starting frame processing")
        while self.recording_status:
            with self.buffer_lock:
                buffer_len = len(self.frame_buffer)
                if buffer_len:
                    self._write_frame()
                    if self.frame_count % 1000 == 0:  # Print every 100 frames
                        print(f"Cam {self.cam_num}: Written {self.frame_count} frames. Current buffer size: {len(self.frame_buffer)}")
            # time.sleep(0.01)
        
        print(f"Cam {self.cam_num}: Frame processing stopped")

    def _write_frame(self):
        frame, timestamp, frame_number = self.frame_buffer.popleft()
        try:
            if frame is None:
                print(f"Cam {self.cam_num}: Error - Empty frame detected")
                return
            
            self.vid_out.write(frame)
            self.frame_times.append(timestamp)
            self.frame_num.append(frame_number)
            self.frame_count += 1
            
            if self.frame_count % 1000 == 0:  # Print every 100 frames
                print(f"Cam {self.cam_num}: Written {self.frame_count} frames. Current buffer size: {len(self.frame_buffer)}")
        except Exception as e:
            print(f"Cam {self.cam_num}: Error writing frame: {str(e)}")

def main():
    # Create camera instance
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    
    session = VideoRecordingSession(cam_num=0)

    try:
        camera.Open()
        camera.MaxNumBuffer = 5000
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(200.0)
        camera.ExposureTime.SetValue(1000)
        camera.OffsetX.SetValue(16)
        camera.OffsetY.SetValue(0)
        camera.Width.SetValue(1440)
        camera.Height.SetValue(1088)
        camera.PixelFormat.SetValue("Mono8")
        # Configure session
        session.set_params(
            video_file="output.avi",
            fourcc="DIVX",
            fps=200,
            dim=(1440, 1088)
        )
        session.start_recording()

        time.sleep(2)
        # Start grabbing frames
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        print("Press Ctrl+C to stop.")

        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = grab_result.GetArray()  # NumPy array of the image
                timestamp = time.time()
                frame_number = grab_result.BlockID

                session.acquire_frame(frame, timestamp, frame_number)
                # time.sleep(0.001)
            grab_result.Release()

    except KeyboardInterrupt:
        print("Stopping recording...")

    finally:
        session.stop_recording()
        camera.StopGrabbing()
        camera.Close()

if __name__ == "__main__":
    main()
