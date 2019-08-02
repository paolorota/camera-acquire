import time
import socketio
import sys
import json
from random import randint
import cv2
import numpy as np
import base64


sio = socketio.Client()
sio.connect('http://0.0.0.0:8080')


@sio.on('connect')
def on_connect():
	print('I\'m connected!')


@sio.on('message')
def on_message(data):
	print('I received a message!')


@sio.on('my message')
def on_message(data):
	print('I received a custom message!')


@sio.on('disconnect')
def on_disconnect():
	print('I\'m disconnected!')



cap = cv2.VideoCapture(0)
fn = 1


while True:
	_, frame = cap.read()
	ret, jpeg = cv2.imencode('.jpg', frame)
	enc = base64.b64encode(jpeg)
	enc_str = enc.decode('utf-8')
	## Decoding
	b = bytes(enc_str, 'utf-8')
	r = base64.decodebytes(b)
	q = np.frombuffer(r, dtype=np.uint8)
	d = cv2.imdecode(q, cv2.IMREAD_COLOR)

	imagedata = {
		'image': enc_str,
		'success': ret,
		'fn': fn
	}

	sio.emit('image', json.dumps(imagedata))
	time.sleep(0.03)
	fn += 1
