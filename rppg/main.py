import cv2
from vision_detection.main import put_fps
from vision_detection.face_detection import FaceMeshDetector
from vision_detection.utils import FPSMeter
from rppg.rppg_class import RPPG
from vision_detection.video_capture import get_frame, init_camera, release_camera

def main():
    debug = True                                                            # Set to True to enable debug visualisations
    cap = init_camera(index=0, width=640, height=480)
    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
    rppg = RPPG(fps=30, roi_landmark_sets=RPPG.CHEEKS.LEFT_CHEEK)                                                     # Initialize rPPG with desired FPS, TODO make fps dynamic according to actual camera fps
    fps = FPSMeter()
    frame_id = 0

    #We need to write something if we can't open camera
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    #While program is active we are catching every possible frame
    #Ofcourse if we can't catch any we write a warrnign (usualyn starts in beggining)
    while True:
        frame = get_frame(cap)
        if frame is None:
            print("Warning: empty frame from camera, stopping.")
            break

        #------------------------------------------------------------
        # rPPG processing
        # rPPG update with the current frame and landmarks
        coords = detector.process(frame)

        if coords is not None and len(coords) > 0:
            rppg.update_buffer(frame)

        # Debug / visualisation: show the last extracted ROIs 1 frame every 5 frames
        if debug:
            frame_id += 1
            if frame_id % 5 == 0:
                for name, roi in rppg.last_rois.items():
                    if roi is not None:
                        cv2.imshow(f"{name} ROI", roi)
        #-----------------------------------------------------------
        #-----------------------------------------------------------
        if frame_id % 30 == 0:  # par ex. toutes les ~1s
            bpm, snr, is_live = rppg.compute_liveness()
            # afficher bpm/snr sur la frame ou logger
        #-----------------------------------------------------------


        # 👉 Fusion with face recognition liveness score

        #    face_score = face_recognition.update(frame)
        #    decision = fusion(face_score, rppg.liveness)

        #Just calling function to write fps on screen
        put_fps(frame, fps.tick())

        #Writes above frame what it is
        #Every ms we check if key ESC is pressed, in order to stop and exit
        cv2.imshow("Vision & Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                                       #27 is ESC in ASCII 
            break

    release_camera(cap)
    cv2.destroyAllWindows()                                                 #Even thoe we have closed camera that dosen't mean all OpenCV windows are closed

if __name__ == "__main__":
    main()
