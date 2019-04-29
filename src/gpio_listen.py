import RPi.GPIO as GPIO  
from time import sleep 
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import cv2 

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)

savedir = '/home/pi/Downloads'

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


# allow the camera to warmup
print('warming up the camera...')
time.sleep(1)
print('camera ready!')


idx = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # ret, frame = cap.read()
    image = frame.array

    cv2.imshow('frame', image)

    key = cv2.waitKey(1) 
    # sleep(0.5)

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    if GPIO.input(26):
        continue
    else:
        fileout = os.path.join(savedir, 'image{0:04}.png'.format(idx))
        idx += 1
        cv2.imwrite(fileout, image)
        print('writing: {}'.format(fileout))


    if key == ord('q'):
        break

# When everything done, release the capture
sys.stdout.write(fileout + '\n')