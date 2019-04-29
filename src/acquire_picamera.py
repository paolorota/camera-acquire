import os
import cv2
import sys
import configparser
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

config = configparser.ConfigParser()
config.read('config.ini')

savedir = config['general']['save_path']
image_name = config['general']['image_name']

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.2)

# cap = cv2.VideoCapture(camera_n)
if not os.path.exists(savedir):
    os.makedirs(savedir)

fileout = 'No fucking fileout!'
idx = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # ret, frame = cap.read()
    image = frame.array

    cv2.imshow('frame', image)

    i = cv2.waitKey(1) 

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    if i == ord('q'):
        break
    elif i == ord('a'):
        fileout = os.path.join(savedir, 'image{0:04}.png'.format(idx))
        idx += 1
        cv2.imwrite(fileout, image)


# When everything done, release the capture
sys.stdout.write(fileout + '\n')









# # import the necessary packages
# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time
# import cv2
 
# # initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# rawCapture = PiRGBArray(camera)
 
# # allow the camera to warmup
# time.sleep(0.1)
 
# # grab an image from the camera
# camera.capture(rawCapture, format="bgr")
# image = rawCapture.array
 
# # display the image on screen and wait for a keypress
# cv2.imshow("Image", image)
# cv2.waitKey(0)