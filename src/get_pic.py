# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import configparser
import os
import argparse


t0 = time.time()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default="img.png",
                    help='file to store the tmp image')
# parser.add_argument('--config', type=str, default="../config.ini",
#                     help='path to config.ini')

args = parser.parse_args()

# config = configparser.ConfigParser()
# config.read('../config.ini')

# savedir = config['general']['save_path']
# image_name = config['general']['image_name']


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)
 
# allow the camera to warmup
time.sleep(0.1)
 
# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array
 
# display the image on screen and wait for a keypress
# fileout = os.path.join(savedir, '{0}.png'.format(image_name))
fileout = args.file
cv2.imwrite(fileout, image)
print('file written in: {} -> time: {} sec'.format(fileout, time.time() - t0))
