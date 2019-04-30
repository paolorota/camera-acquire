import socketio
from aiohttp import web

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
	print('mex', data)

web.run_app(app)