import cv2
import numpy as np
from collections import deque
import os                       

from vision_detection.video_capture import init_camera, get_frame, release_camera, ensure_data_dir
from vision_detection.face_detection import FaceMeshDetector
from vision_detection.face_alignment import align_and_crop, compute_eye_centers
from vision_detection.utils import FPSMeter, EyeSmoother

from vision_detection.face_capture import save_latest_aligned_face


#Just for aesthetics we are going to draw a box around face
#Helps user see how his face is being captured
def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

#Again just for aesthetics we are going to write FPS on screen
def put_fps(frame, fps_value: float):
    cv2.putText(frame, f"FPS: {int(fps_value)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA) #Just definig a style of text


def main():
    ensure_data_dir("data")
    ensure_data_dir("data/aligned_preview")  #JPEG preview buffer

    cap = init_camera(index=0, width=640, height=480)
    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
    
    #I belive we could go without these but I was scared how will it affect Konstantinos and Lorenzo
    fps = FPSMeter()
    smoother = EyeSmoother(window=5)

    frame_counter = 0   #Counter for aligned face saving
    saving_aligned_faces = True

    #THIS BUFFER IS FOR MY FRIEND LORENZZO!------------------------------------------------------------------------------------------------------------------------------------------------
    #Anti-spoofing requires 15-30 seconds of video
    #Assuming camera runs approximately at 30 FPS
    FPS_TARGET = 30
    BUFFER_SECONDS = 15  # store ~15 seconds
    BUFFER_SIZE = FPS_TARGET * BUFFER_SECONDS  # = 450 frames

    #RAM buffer
    frame_buffer = deque(maxlen=BUFFER_SIZE)

    #JPEG buffer
    MAX_JPEG_IMAGES = 15  #We are limiting images saved on disk to 15 (once we get to 15 we are deleting older buffer)
    preview_dir = "data/aligned_preview"

    #We need to write something if we can't open camera
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    #While program is actrive we are catching every possible frame
    #Ofcourse if we can't catch any we write a warning
    while True:
        frame = get_frame(cap)
        if frame is None:
            print("Warning: empty frame from camera, stopping.")
            break

        #Storing raw frames for anti-spoofing model
        frame_buffer.append(frame.copy())

        #Indicator when buffer is fully filled and we can preform anti-spoof
        if len(frame_buffer) == BUFFER_SIZE:
            cv2.putText(frame, "BUFFER READY", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #We want to get coords for every frame (for every frame we build a mash)
        coords = detector.process(frame)
        if coords is not None:
            
            #After face is detected we but box on it (we do it with smallest and biggest coord)
            bbox = detector.compute_bbox(coords)
            draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2)

            #We find eye ceneters (used for aligment)
            left_xy, right_xy = compute_eye_centers(coords)

            #We are using smoother to smoothe eyes
            #We smoothe them by using mean of 5 frames instead of each frame
            smoother.update(left_xy, right_xy)
            sm_left_xy, sm_right_xy = smoother.get_smoothed()

            if saving_aligned_faces:
                if (frame_counter == 0):
                    print("Started saving aligned faces to disk.")
                    
                aligned_path = f"{preview_dir}/frame_{frame_counter:06d}.jpg"

                _ = align_and_crop(
                    frame_bgr=frame,
                    coords=coords,
                    crop_size=224,
                    save_debug_path=aligned_path,
                )

                frame_counter += 1

                files = sorted(os.listdir(preview_dir))
                if len(files) > MAX_JPEG_IMAGES:
                    oldest = files[0]
                    os.remove(os.path.join(preview_dir, oldest))

            #This is again just for aesthetics (it will just show how system converst eyes to smoother eyes)
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

        if key == ord(' '):
            print(f"{'Stopping' if saving_aligned_faces else 'Starting'} aligned face saving.")
            saving_aligned_faces = not saving_aligned_faces

        if key == ord('s') or key == ord('S'):  
            save_latest_aligned_face()          #THIS FUNCTION IS FOR MY FRIEND KONSTANTINOS!---------------------------------------------------------------------------------------------------------------------


    release_camera(cap)
    cv2.destroyAllWindows()     #Even thoe we have closed camera that dosen't mean all OpenCV windows are closed


if __name__ == "__main__":
    main()
