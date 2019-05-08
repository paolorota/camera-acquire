import os
import cv2
import threading
import time

# addr = 'http://192.168.180.161:5000/video_feed'
addr = 'http://192.168.180.226:5000/video_feed'
outvideo = '/home/prota/Downloads/out.avi'


class ImageAcquisitionFromStream(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    stop = False
    last_access = 0

    def initialize(self):
        if ImageAcquisitionFromStream.thread is None:
            # start background frame thread
            ImageAcquisitionFromStream.thread = threading.Thread(target=self._thread_acquire)
            ImageAcquisitionFromStream.thread.start()

            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def start_acquisition(self):
        ImageAcquisitionFromStream.last_access = time.time()
        self.initialize()
        return self.frame

    def stop_acquisition(self):
        ImageAcquisitionFromStream.stop = True

    def _thread_acquire(cls):

        video = cv2.VideoCapture(addr)
        print('[{}] Warming up stream...'.format(time.time()))
        time.sleep(1)
        print('[{}] Stream ready.'.format(time.time()))

        out = cv2.VideoWriter(outvideo, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800, 600), isColor=False)
        idx = 0
        while not cls.stop:
            idx += 1
            print('[{}] Frame {}'.format(time.time(), idx))
            success, cls.frame = video.read()
            cls.last_access = time.time()
            if not success:
                continue
            gray = cv2.cvtColor(cls.frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (800, 600)) 
            # cv2.imwrite('/home/prota/Downloads/image_{0:04}.jpg'.format(idx), gray)
            out.write(gray)
            cv2.imshow('frame', gray)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cls.stop_acquisition()
                
            if time.time() - cls.last_access > 20:
                out.release()
                break
            
        video.release()
        cv2.destroyAllWindows()


iafs = ImageAcquisitionFromStream()
iafs.start_acquisition()
print('[{}] shall print this while threading.'.format(time.time()))


