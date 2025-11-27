import time
import threading
import cv2
from rppg.roisets_class import ROISet
from rppg.rppg_class import RPPG
from vision_detection.vision_thread import VisionThread

def main():
    debug = True                                                            # Set to True to enable debug visualisations
    vision_thread = VisionThread()                                          # Initialize the vision thread
    rppg = RPPG(fps=30, roi_landmark_sets=RPPG.CHEEKS)                  # Initialize the rPPG processor with the CHEEKS ROI set
    frame_id = 0

    vision_thread.start()                                                  # Start the vision thread
    while not vision_thread.running and vision_thread.is_alive():
        time.sleep(0.01)  # laisser le thread initialiser la caméra

    while True:
        #------------------------------------------------------------
        # rPPG processing
        # rPPG update with the current frame and landmarks
        coords = vision_thread.last_coords

        if coords is not None and len(coords) > 0:
            rppg.update_buffer(vision_thread.last_frame, coords)

        # Debug / visualisation: show the last extracted ROIs 1 frame every 5 frames
        if debug:
            if frame_id % 5 == 0:
                for name, roi in rppg.last_rois.items():
                    if roi is not None:
                        cv2.imshow(f"{name} ROI", roi)
        #-----------------------------------------------------------
        #-----------------------------------------------------------
        if frame_id % 30 == 0:  # par ex. toutes les ~1s
            bpm, snr, is_live = rppg.compute_liveness()
            # afficher bpm/snr sur la frame ou logger
            print(f"Estimated BPM: {bpm}, SNR: {snr} dB, Liveness: {is_live}")
        #-----------------------------------------------------------


        # 👉 Fusion with face recognition liveness score

        #    face_score = face_recognition.update(frame)
        #    decision = fusion(face_score, rppg.liveness)

        frame_id += 1

    vision_thread.stop()
    vision_thread.join()

if __name__ == "__main__":
    main()
