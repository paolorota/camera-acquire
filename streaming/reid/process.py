import os
import cv2
import threading
import time

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from os.path import join

# addr = 'http://192.168.180.161:5000/video_feed'


def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


class ImageAcquisitionFromStream(object):
    thread_acquisition = None  # background thread that reads frames from camera
    thread_process = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    img_browsed = None
    stop = False
    last_access = 0
    addr = 'http://192.168.180.226:5000/video_feed'
    fn = 0

    def __init__(self):
        self.video = cv2.VideoCapture(self.addr)
        self.initialize()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        self.fn += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'frame: {}'.format(self.fn), (10,30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        if success:
            self.frame = image
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

    def initialize(self):
        if ImageAcquisitionFromStream.thread_process is None:
            ImageAcquisitionFromStream.thread_process = threading.Thread(target=self._thread_process)
            ImageAcquisitionFromStream.thread_process.start()

            while self.img_browsed is None:
                time.sleep(0)

    def _thread_process(cls):
        color = (30, 240, 230)
        weights_file = 'model/vgg_face_weights.h5'
        face_haar_file = 'model/haarcascade_frontalface_default.xml'
        db_dir = 'face_db/'

        face_cascade = cv2.CascadeClassifier(face_haar_file)
        model = loadVggFaceModel()

        # GET DICTIONARY
        employee_pictures = db_dir
        employees = dict()
        for file in listdir(employee_pictures):
	        employee, extension = file.split(".")
	        employees[employee] = model.predict(preprocess_image(join(employee_pictures, file)))
        print("employee representations retrieved successfully")

        def findCosineSimilarity(source_representation, test_representation):
	        source_representation = np.reshape(source_representation, [source_representation.shape[1], ])
	        a = np.matmul(np.transpose(source_representation), test_representation)
	        b = np.sum(np.multiply(source_representation, source_representation))
	        c = np.sum(np.multiply(test_representation, test_representation))
	        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        width = 640
        height = 360
        img_size = 224

        while(True):
            img = cls.frame
            if img is None:
                print('[{}] Missing input image (frame)'.format(time.time()))
            img = cv2.resize(img, (640, 360))
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            
            for (x, y, w, h) in faces:
                if w > 130: 
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
                    
                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                    detected_face = cv2.resize(detected_face, (img_size, img_size)) #resize to 224x224
                    
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    img_pixels /= 255
                    
                    captured_representation = model.predict(img_pixels)[0,:]
                    
                    found = 0
                    for i in employees:
                        employee_name = i
                        representation = employees[i]
                        
                        similarity = findCosineSimilarity(representation, captured_representation)
                        if(similarity < 0.30):
                            cv2.putText(img, employee_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            found = 1
                            break
                            
                    #connect face and text
                    cv2.line(img, (int((x+x+w)/2), y+15), (x+w, y-20), color, 1)
                    cv2.line(img, (x+w, y-20), (x+w+10, y-20), color, 1)
                
                    if(found == 0): #if found image is not in employee database
                        cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cls.img_browsed = img


# iafs = ImageAcquisitionFromStream()
# iafs.start_acquisition()
# print('[{}] shall print this while threading.'.format(time.time()))


