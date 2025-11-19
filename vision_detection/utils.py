from collections import deque   #Deaqu will be really important because we don't want to have to many frames (we want to delete old ones)
from typing import Tuple
import numpy as np  #For numerical operations
import time

#Here we are going to use 2 classes to smother FPS and eyes
#First one is used to check how many frames we actualy have
class FPSMeter:
    #We need to konw time of last frame as well as current fps (in the begining both will be 0)
    def __init__(self):
        self.prev_time = None
        self.current_fps = 0.0

    #This is the main metode of class and it starts every tick or frame
    def tick(self) -> float:
        now = time.time()
        if self.prev_time is None:  #This happens when we start the program. It will just start time and give us starting FPS, which is 0
            self.prev_time = now
            self.current_fps = 0.0
            return self.current_fps
        dt = now - self.prev_time   #We are getting the diff between current time and last frame in seconds
        self.prev_time = now    #Ofcourse last time needs to become currnet time because we are executing this every second
        if dt > 0:
            self.current_fps = 1.0 / dt     #We are just calculating, depending on dt how many frames could we fit in one second
        return self.current_fps

#Second one is used to smother eyes beacues of the constant blinking
#The point is to stabilize camera
class EyeSmoother:
    #We need to have 2 lists of frames for each eye
    def __init__(self, window: int = 5):
        self.window = window
        self.left_hist = deque(maxlen=window)
        self.right_hist = deque(maxlen=window)

    #Now we are taking centers of eyes and for every frame we are going to apend new center pixels
    #Since we don't want to eat all of our memeory we use special list deque which will start deleting oldest elemetn when we reach limit (we set limit to 5)
    def update(self, left_xy: Tuple[float, float], right_xy: Tuple[float, float]):
        self.left_hist.append(left_xy)
        self.right_hist.append(right_xy)

    def get_smoothed(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        def _avg(dq):
            arr = np.array(dq, dtype=np.float32)        #We are creating matrix from deque list in order to extract x and y and calculate their average
            return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))   #First part returns mean of x and second mean of y
        return _avg(self.left_hist), _avg(self.right_hist)
