import socketio
from aiohttp import web
import cv2
import numpy as np
import json
import base64

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)


@sio.on('connect')
async def connect(sid, environ):
    print('I\'m connected!', sid)


@sio.on('disconnect')
async def disconnect(sid):
    print('disconnect ', sid)


@sio.on('event')
async def event(sid, data):
    print('mex ->', data)


@sio.on('image')
async def get_image(sid, data):
    # img = np.fromstring(data, dtype=np.uint8)
    # print('got image: {} \ndata shape: {}'.format(img, img.shape))
    j = json.loads(data)
    b = bytes(j['image'], 'utf-8')
    r = base64.decodebytes(b)
    q = np.frombuffer(r, dtype=np.uint8)
    d = cv2.imdecode(q, cv2.IMREAD_COLOR)
    cv2.imshow('image', d)
    cv2.waitKey(1)
    # print(data)


web.run_app(app)
