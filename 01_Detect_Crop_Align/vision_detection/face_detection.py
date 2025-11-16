from typing import List, Tuple, Optional    #Importing new type for easier function returns
import cv2
import mediapipe as mp

#Because mediapipe is used for everything not just face we need to specifie that
# Some IDEs need the # type: ignore comment to avoid type checking issues with mediapipe even if it's properly installed
mp_face_mesh = mp.solutions.face_mesh # type: ignore

class FaceMeshDetector:
    #We are defining constructor to add our values
    def __init__(
        self, #This represents a instant of a mediapipe facemash model
        max_num_faces: int = 1,
        refine_landmarks: bool = True, #Not needed but why not
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, #If we don't use it it will work! Problem is it's going to be slow because it's going detect every frame independetly (again and again)
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    #Here we are using new return types!
    #Parametars object for mash and picture frame that was caught
    def process(self, frame_bgr) -> Optional[List[Tuple[int, int]]]:
        h, w = frame_bgr.shape[:2]          #Taking the dimension of caught picture
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)       #Change of picture color space because mediapipe works with RBG and openCV with BRG
        result = self.mesh.process(frame_rgb)       #Trying to put mash on a face on picture

        #If we don't find face ofcourse we won't return noting
        if not result.multi_face_landmarks:
            return None

        #Take the first detected face
        #Every cordinate is turend in pixel (we have 468 of them)
        landmarks = result.multi_face_landmarks[0].landmark
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        return coords

    #Finding smallest and biggest coordinate in order to make a box where face is
    @staticmethod
    def compute_bbox(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [x for x, _ in coords]     #Taking just x coordinates from landmarks
        ys = [y for _, y in coords]     #Taking just y coordinates from landmarks
        return min(xs), min(ys), max(xs), max(ys)