import cv2
import numpy as np

from vision_detection.video_capture import init_camera, get_frame, release_camera, ensure_data_dir
from vision_detection.face_detection import FaceMeshDetector
from vision_detection.face_alignment import align_and_crop, compute_eye_centers
from vision_detection.utils import FPSMeter, EyeSmoother

#Just for estetics we are going to draw a box around face
#Helps user see how his face is being captured
def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

#Again just for estetics we are going to write FPS on screen
def put_fps(frame, fps_value: float):
    cv2.putText(frame, f"FPS: {int(fps_value)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA) #Just definig a style of text


def main():
    ensure_data_dir("data")
    cap = init_camera(index=0, width=640, height=480)
    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
    
    #I belive we could go without these but I was scared how will it affect Konstantinos and Lorenzo
    fps = FPSMeter()
    smoother = EyeSmoother(window=5)

    #We need to write something if we can't open camera
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    #While program is actrive we are catching every possible frame
    #Ofcourse if we can't catch any we write a warrnign (usualyn starts in beggining)
    while True:
        frame = get_frame(cap)
        if frame is None:
            print("Warning: empty frame from camera, stopping.")
            break
        #We want to get coords for every frame (for every frame we build a mash)
        coords = detector.process(frame)
        if coords is not None:
            
            #After face is detected we but box on it (we do it with smallest and biggest coord)
            bbox = detector.compute_bbox(coords)
            draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2)

            #We find eye ceneters (used for aligment)
            left_xy, right_xy = compute_eye_centers(coords)

            #We are using smoother to smoothe eyes
            #We smoothe them by suing mean of 5 frames instead of each frame
            smoother.update(left_xy, right_xy)
            sm_left_xy, sm_right_xy = smoother.get_smoothed()

            _ = align_and_crop(
                frame_bgr=frame,
                coords=coords,          
                crop_size=224,
                save_debug_path="data/aligned_face.jpg",
            )

            #This is again just for estetics (it will just show how system converst eyes to smoother eyes)
            for (ex, ey) in (sm_left_xy, sm_right_xy):
                cv2.circle(frame, (int(ex), int(ey)), 3, (0, 255, 255), -1)

        #Just calling function to write fps on screen
        put_fps(frame, fps.tick())

        #Writes above frame what it is
        #Every ms we check if key ESC is pressed, in order to stop and exit
        cv2.imshow("Vision & Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  #27 is ESC in ASCII 
            break

    release_camera(cap)
    cv2.destroyAllWindows()     #Even thoe we have closed camera that dosen't mean all OpenCV windows are closed


if __name__ == "__main__":
    main()
    