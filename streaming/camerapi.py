import time
import io
import threading
import picamera


class VideoCamera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    stop_camera = False

    def initialize(self):
        if VideoCamera.thread is None:
            # start background frame thread
            VideoCamera.thread = threading.Thread(target=self._thread)
            VideoCamera.thread.start()

            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        VideoCamera.last_access = time.time()
        self.initialize()
        return self.frame

    def StopPreview():
        VideoCamera.stop_camera = True

    @classmethod
    def _thread(cls):
        with picamera.PiCamera() as camera:
            # camera setup
            camera.resolution = (640, 480)

            # let camera warm up
            camera.start_preview()
            time.sleep(2)

            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=True):
                # store frame
                stream.seek(0)
                cls.frame = stream.read()

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()

                # if there hasn't been any clients asking for frames in
                # the last 10 seconds stop the thread
                if time.time() - cls.last_access > 10:
                    break
                elif VideoCamera.stop_camera is True:
                    break
        cls.thread = None