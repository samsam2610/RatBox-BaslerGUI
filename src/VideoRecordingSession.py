import cv2
import threading
import time
from collections import deque
from pypylon import pylon
import datetime
import csv
import typing
# import keyboard


class VideoRecordingSession:
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.recording_status = False
        self.vid_out = None
        self.frame_buffer = deque(maxlen=500)
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
        
        # Initialize CSV writer
        self.csv_file = video_file.replace('.avi', '.csv')
        self.csv_file_handle = open(self.csv_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow(['timestamp', 'frame_number', 'frame_line_status', 'note'])
        print(f"Cam {self.cam_num}: CSV writer initialized with {self.csv_file}")

    def acquire_frame(self, frame, timestamp, frame_number, frame_line_status, note: typing.Optional[str]=None):
        if self.recording_status:
            with self.buffer_lock:
                self.frame_buffer.append((frame, timestamp, frame_number, frame_line_status, note))

    def start_recording(self):
        self.recording_status = True
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        print(f"Cam {self.cam_num}: Recording started.")

    def stop_recording(self):
        self.recording_status = False
        if self.processing_thread.is_alive():
            current_date_and_time = str(datetime.datetime.now())
            print(f"Cam {self.cam_num}: Stopping frame processing at {current_date_and_time}")
            self.processing_thread.join()
        time.sleep(0.1)  # Allow buffer processing to finish
        self.write_remaining_frames()
        if self.vid_out:
            print(f"Cam {self.cam_num}: Releasing video writer")
            self.vid_out.release()
            self.vid_out = None
        if hasattr(self, 'csv_file_handle'):
            print(f"Cam {self.cam_num}: Closing CSV file")
            self.csv_file_handle.close()
        print(f"Cam {self.cam_num}: Recording stopped.")

    def write_remaining_frames(self):
        print(f"Remaining frames: {len(self.frame_buffer)}")
        while self.frame_buffer:
            with self.buffer_lock:
                if len(self.frame_buffer) > 0:
                    self._write_frame()

    def _process_frames(self):
        print(f"Cam {self.cam_num}: Starting frame processing")
        while self.recording_status:
            if len(self.frame_buffer) > 0:
                self._write_frame()
            # else:
            #     time.sleep(0.001)
        
        print(f"Cam {self.cam_num}: Frame processing stopped")
    
    def _write_frame(self):
        frame, timestamp, frame_number, frame_line_status, note = self.frame_buffer.popleft()    
        self.vid_out.write(frame)
        self.frame_count += 1
        
        # Write to CSV
        self.csv_writer.writerow([timestamp, frame_number, frame_line_status, note])
        
        if self.frame_count % 1000 == 0:
            print(f"Cam {self.cam_num}: Recorded {self.frame_count} frames. Remaining: {len(self.frame_buffer)}")
    
    @staticmethod
    def precise_sleep(duration):
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < duration:
            pass  # Busy-waiting until the time has elapsed