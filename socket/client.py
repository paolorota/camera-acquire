import time
import socketio
import sys
import json
from random import randint

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

while True:
	sensordata = {
		"lm-temperature-01": {"sid":"lm-temperature-01", "value": str(randint(20, 21)), "setpoint": "30"}, 
		"lm-current-01": {"sid":"lm-current-01", "value": str(randint(30, 50)), "setpoint": "40"},
		"shg-temperature-01": {"sid":"lm-current-01", "value": str(randint(14, 15)), "setpoint": "15"},
		"shg-current-01": {"sid":"lm-current-01", "value": str(randint(10, 11)), "setpoint": "10"},
		"shg-power-01": {"sid":"lm-current-01", "value": str(randint(21, 31)), "setpoint": "30"},
	}
	sio.emit('event', json.dumps(sensordata))
	time.sleep(1)