import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time
from VideoRecordingSession import VideoRecordingSession

class TestVideoRecordingSession(unittest.TestCase):
    def setUp(self):
        self.session = VideoRecordingSession(cam_num=0)
        # Use the same dimensions as in main()
        self.test_width = 1440
        self.test_height = 1088
        self.test_frame = np.zeros((self.test_height, self.test_width), dtype=np.uint8)
        
    @patch('cv2.VideoWriter')
    def test_basler_camera_configuration(self, mock_video_writer):
        # Setup mock video writer
        mock_instance = Mock()
        mock_instance.isOpened.return_value = True
        mock_video_writer.return_value = mock_instance

        # Test the exact parameters used in main()
        self.session.set_params(
            video_file="output.avi",
            fourcc="DIVX",
            fps=200,
            dim=(self.test_width, self.test_height)
        )

        # Verify video writer parameters
        self.assertEqual(self.session.fps, 200)
        self.assertEqual(self.session.dim, (self.test_width, self.test_height))
        self.assertEqual(self.session.video_file, "output.avi")

    @patch('pypylon.pylon.InstantCamera')
    def test_camera_acquisition(self, mock_camera):
        # Setup mock camera and grab result
        mock_grab_result = MagicMock()
        mock_grab_result.GrabSucceeded.return_value = True
        mock_grab_result.GetArray.return_value = self.test_frame
        mock_grab_result.BlockID = 1

        mock_camera_instance = Mock()
        mock_camera_instance.RetrieveResult.return_value = mock_grab_result
        mock_camera.return_value = mock_camera_instance

        # Simulate frame acquisition
        frame = mock_grab_result.GetArray()
        timestamp = time.time()
        frame_number = mock_grab_result.BlockID

        self.session.acquire_frame(frame, timestamp, frame_number)

        # Verify frame was added to buffer
        self.assertEqual(len(self.session.frame_buffer), 1)
        buffered_frame, _, _ = self.session.frame_buffer[0]
        self.assertTrue(np.array_equal(buffered_frame, self.test_frame))

    @patch('cv2.VideoWriter')
    def test_high_speed_recording(self, mock_video_writer):
        # Setup mock
        mock_instance = Mock()
        mock_instance.isOpened.return_value = True
        mock_video_writer.return_value = mock_instance

        # Configure for 200 FPS recording as in main()
        self.session.set_params(
            video_file="output.avi",
            fourcc="DIVX",
            fps=200,
            dim=(self.test_width, self.test_height)
        )

        # Start recording
        self.session.start_recording()

        # Simulate high-speed frame acquisition
        num_frames = 6000
        for i in range(num_frames):
            self.session.acquire_frame(self.test_frame, time.time(), i)
            time.sleep(0.001)

        # Stop recording
        self.session.stop_recording()

        # Verify all frames were processed
        self.assertEqual(self.session.frame_count, num_frames)
        self.assertEqual(len(self.session.frame_buffer), 0)

    def test_buffer_overflow_protection(self):
        # Test buffer with main()'s configuration of 5000 frames
        num_frames = 6000  # Exceed buffer size
        
        for i in range(num_frames):
            self.session.acquire_frame(self.test_frame, time.time(), i)
            
        # Verify buffer doesn't exceed 5000 frames
        self.assertEqual(len(self.session.frame_buffer), 5000)
        
        # Verify the buffer contains the most recent frames
        _, _, last_frame_num = self.session.frame_buffer[-1]
        self.assertEqual(last_frame_num, num_frames - 1)

if __name__ == '__main__':
    unittest.main()
