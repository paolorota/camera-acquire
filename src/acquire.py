import os
import cv2
import sys
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

savedir = config['general']['save_path']
camera_n = int(config['general']['camera'])
image_name = config['general']['image_name']

cap = cv2.VideoCapture(camera_n)
if not os.path.exists(savedir):
    os.makedirs(savedir)

fileout = 'No fucking fileout!'
idx = 0
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    i = cv2.waitKey(1) 
    if i == ord('q'):
        break
    elif i == ord('a'):
        fileout = os.path.join(savedir, 'image{0:04}.png'.format(idx))
        idx += 1
        cv2.imwrite(fileout, frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

sys.stdout.write(fileout + '\n')

