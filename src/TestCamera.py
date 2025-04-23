import cv2
import threading
import time
from collections import deque
from pypylon import pylon
from VideoRecordingSession import VideoRecordingSession

def main():
    # Create camera instance
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    
    session = VideoRecordingSession(cam_num=0)

    try:
        camera.Open()
        camera.MaxNumBuffer = 5000
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(200.0)
        camera.ExposureTime.SetValue(1200)
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
            time.sleep(0.001)
            grab_result.Release()

    except KeyboardInterrupt:
        print("Stopping recording...")

    finally:
        session.stop_recording()
        camera.StopGrabbing()
        camera.Close()

if __name__ == "__main__":
    main()
