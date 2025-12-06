from typing import List, Tuple, Optional    
import cv2
import numpy as np      #For matrixes
import math
import os

#From this 4 dots we can calculate angle of eyes
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

#We use it for calculating mean value of cords in eyes 
#We will give it Left eye and right eye cords in next function
#I tried without this function but it was just too messy and anoing so I did it this way!
def mean_point(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    arr = np.array(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())

#We give it two dots for each eye and it will calcualte the center of each eye by using previous function
def compute_eye_centers(coords: List[Tuple[int, int]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    left_eye = mean_point([coords[i] for i in LEFT_EYE_IDX])
    right_eye = mean_point([coords[i] for i in RIGHT_EYE_IDX])
    return left_eye, right_eye

#Rotates whole face over specific dot and specific angle
#Dot will represent a center between two eyes and angle will be calucated depending on horizontal and vertical diff of two centers of eyes 
def rotate_image_about_point(image, center_xy: Tuple[float, float], angle_deg: float):
    M = cv2.getRotationMatrix2D(center_xy, angle_deg, 1.0)      #Creating matrix for rotation

    #Here we are actualy using rotation matrix, that we have created to perform rotation!
    #We give it image, rotation matrix, height, widht (in order to preserve image) and flag that is like a quality of a rotation (we need to guess what is the color of some pixels when we rotate it)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC) 
    return rotated

#We need to provide frame we are using
#All dots from mesh face (we are going to pick thoes we are interested in)
#Size of cropped face in pixels
#And where are we going to save this picture (ofcourse if we don't find any face we won't save anything)
def align_and_crop(
    frame_bgr,
    coords: List[Tuple[int, int]],
    crop_size: int = 320,
    save_debug_path: Optional[str] = None,
    bbox_scale: float = 1.6,
    align: bool = True,
):
    #Finding centers of eyes
    (lx, ly), (rx, ry) = compute_eye_centers(coords)

    rotated = frame_bgr
    if align:
        #We are calculating diference in height and in horizontal pose between two centers
        #Than we need to calcuate angle between eyes and we do it by using their diferences
        dy, dx = (ry - ly), (rx - lx)
        angle_deg = math.degrees(math.atan2(dy, dx)) 

        #Now we are trying to find center between two eyes
        #And then we roatet aorund it
        cx, cy = (lx + rx) * 0.5, (ly + ry) * 0.5
        rotated = rotate_image_about_point(frame_bgr, (cx, cy), angle_deg)

    # Use the face bbox as the crop center so we keep the whole head, not only the eyes.
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    face_size = max(bbox_w, bbox_h)

    # Expand the box to keep forehead/chin and a bit of background for stability.
    side = max(crop_size, int(face_size * bbox_scale))
    half = side // 2
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    x1, y1 = int(center_x) - half, int(center_y) - half
    x2, y2 = x1 + side, y1 + side

    # If the crop falls outside the frame, pad the image instead of failing.
    h, w = rotated.shape[:2]        #We are taking only height and width of rotated picture (if we have put :3 we would also take color)
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        rotated = cv2.copyMakeBorder(
            rotated,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    #Finaly we are taking cropped picture
    #We used .copy in order to not lose original picture (just in case Lorenzo and Konstantions want to use it later)
    cropped = rotated[y1:y2, x1:x2].copy()

    #We need to save cropped face in direcotry that we choose
    if save_debug_path:
        os.makedirs(os.path.dirname(save_debug_path), exist_ok=True)
        cv2.imwrite(save_debug_path, cropped)

    return cropped
