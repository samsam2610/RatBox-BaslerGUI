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
        while self.frame_buffer:
            self._write_frame()

    def _process_frames(self):
        print(f"Cam {self.cam_num}: Starting frame processing")
        while self.recording_status:
            self._write_frame()
            buffer_len = len(self.frame_buffer)
            if buffer_len == 0:
                time.sleep(0.001)
        
        print(f"Cam {self.cam_num}: Frame processing stopped")
        
    def _write_frame(self):
        while len(self.frame_buffer) > 0:
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
    
    @staticmethod
    def precise_sleep(duration):
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < duration:
            pass  # Busy-waiting until the time has elapsed