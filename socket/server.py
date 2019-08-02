import socketio
from aiohttp import web
import cv2
import numpy as np

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
    print(data)


web.run_app(app)
