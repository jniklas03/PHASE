from picamera2 import Picamera2
from libcamera import controls, Transform

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)

timestr = time.strftime("%d.%m.%Y-%H.%M.%S")

picam2 = Picamera2()

config = picam2.create_still_configuration(
	main={"size": (4056, 3040)},
	transform=Transform(vflip=True, hflip=True) #incubator image "upside down"
)

picam2.configure(config)
picam2.start()

GPIO.output(26, GPIO.LOW)
time.sleep(3)

picam2.set_controls({
	"AfMode": controls.AfModeEnum.Continuous,
	"AfSpeed": controls.AfSpeedEnum.Fast
})

picam2.capture_file(f"/home/rubusidaeus/camera_test/{timestr}.jpg")

time.sleep(3)
GPIO.output(26, GPIO.HIGH)
picam2.stop()
