import sensor, image, time, os, tf, time, pyb, gif
from pyb import LED, Pin
from motors import DCMotors

MAX_SPEED = 1500
MAX_DIRECTION = 3
CAPTURE_FRAME_TIME = 300.0

class Robot:
    def __init__(self):
        self.speed = 0.0
        self.direction = 0.0
        self.recording = False
        self.recording_id = 0
        self.recording_gif = None
        self.last_frame_time = 0
        self.labels = None
        self.direction_history = []

    def setup(self):
        self.motor = DCMotors()

        sensor.reset()
        sensor.set_pixformat(sensor.GRAYSCALE)
        sensor.set_framesize(sensor.QQVGA)
        sensor.skip_frames(time = 2000)
        sensor.set_vflip(True)

        self.clock = time.clock()
        self.usb = pyb.USB_VCP()

        self.set_speed(0)

        print('Vehicle is ready!')

    def update_motors(self):
        self.motor.speed(0, int(self.speed * (1.0 + self.direction / 5.5))) # back left
        self.motor.speed(3, int(self.speed * (1.0 + self.direction / 5.5))) # front left
        self.motor.speed(1, int(self.speed * (1.0 - self.direction / 5.5))) # front right
        self.motor.speed(2, int(self.speed * (1.0 - self.direction / 5.5))) # back right

    def set_speed(self, speed, log=True):
        self.speed = max(0, min(speed, MAX_SPEED))
        if log:
            print('Speed: ' + str(self.speed))
        self.update_motors()

    def set_direction(self, direction, log=True):
        self.direction = max(-MAX_DIRECTION, min(direction, MAX_DIRECTION))
        if log:
            print('Direction: ' + str(self.direction))
        self.update_motors()

    def start_recording(self):
        if self.recording == True:
            return

        self.set_speed(500)
        self.recording = True
        self.last_frame_time = pyb.millis()
        self.recording_id = pyb.millis()
        self.direction_history = []

        self.labels = open(str(self.recording_id) + ".txt", "w")
        self.recording_gif = gif.Gif(str(self.recording_id) + ".gif")
        time.sleep(100)
        print('Starting recording ' + str(self.recording_id))

    def stop_recording(self):
        if self.recording == False:
            return

        self.recording = False
        self.recording_gif.close()
        self.labels.close()
        self.set_speed(0)
        print('Stop recording ' + str(self.recording_id))


    def capture_frame(self):
        print('Capturing frame')
        self.led.on()
        self.recording_gif.add_frame(sensor.snapshot())
        avg_direction = float(sum(self.direction_history)) / float(len(self.direction_history))
        self.labels.write(str(avg_direction) + '\n')
        self.direction_history = []
        self.last_frame_time = pyb.millis()
        self.led.off()

    def autopilot(self):
        print('Entering autopilot mode...')
        self.led = pyb.LED(3)

        while(True):
            img = sensor.snapshot()
            self.led.on()
            pred_dir = 0
            t0 = pyb.millis()
            res = tf.classify('trained.tflite', img)[0].output()
            tt = pyb.millis() - t0
            print('Inference time: ' + str(tt))
            pred_dir = res[0] * -3.0 + res[1] * -2.0 + res[2] * -1.0 + res[4] * 1.0 + res[5] * 2.0 + res[6] * 3.0
            pred_dir = pred_dir * 1.1
            self.led.off()
            if (self.speed == 0):
                self.set_speed(1500, False)
            self.set_direction(pred_dir, False)

    def manual(self):
        print('Entering manual mode...')
        self.led = pyb.LED(2)
        usb = pyb.USB_VCP()

        while(True):
            inp = usb.read(1)
            if (inp == b'w'):
                self.set_speed(self.speed + 500)
            elif (inp == b's'):
                self.set_speed(self.speed - 500)
            elif (inp == b'a'):
                self.set_direction(self.direction - 1)
            elif (inp == b'd'):
                self.set_direction(self.direction + 1)
            elif (inp == b'o'):
                self.start_recording()
            elif (inp == b'p'):
                self.stop_recording()
            if (self.recording == True):
                self.direction_history.append(self.direction)
                if ((pyb.millis() - self.last_frame_time) > CAPTURE_FRAME_TIME):
                    self.capture_frame()
            time.sleep(10)

robot = Robot()
robot.setup()

time.sleep(2000)

usb = pyb.USB_VCP()
if(usb.isconnected()):
    print('Type <a> for autopilot, <m> for manual control')
    while(True):
        inp = usb.read(1)
        if (inp == b'a'):
            robot.autopilot()
            break
        elif (inp == b'm'):
            robot.manual()
            break
else:
    robot.autopilot()
