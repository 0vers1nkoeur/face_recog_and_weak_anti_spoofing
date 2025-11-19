import os
import cv2

#Just to make sure that folder data where we store images already exists
def ensure_data_dir(path: str = "data") -> None:
    os.makedirs(path, exist_ok=True)

#We are just initializing camera and we are returning camera object
#Index shows us which camera device is open
#Width shows us how many pixels are we capturing in width
#Height shows us how many pixels are we capturing in height
def init_camera(index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

#From that camera above we are taking out one frame
def get_frame(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    return frame if ok else None

#After we have finished with the camera we need to release it!
def release_camera(cap: cv2.VideoCapture) -> None:
    try:
        cap.release()
    except Exception:
        pass
