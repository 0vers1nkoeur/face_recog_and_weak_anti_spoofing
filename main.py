from vision_detection.vision_thread import VisionThread
from rppg.rppg_class import RPPG
import time
import cv2


def main():

    vt = VisionThread()
    vt.start()
    rppg = RPPG(fps=30, window_seconds=15, roi_landmark_sets=RPPG.CHEEKS)
    debug = False
    frame_id = 0

    print("VisionThread started.")

    while True:

        #print("Buffer len:", len(vt.frame_buffer))
        #time.sleep(1)

        while not vt.running and vt.is_alive():
            time.sleep(0.01)  # let the thread initialize the camera

        #---------------------- LORENZO (ANTI-SPOOF) ----------------------
        frame_id += 1

        if debug:
            print(vt.last_frame is None, vt.last_coords is None)
        
        rppg.update_buffer(vt.last_frame, vt.last_coords)
        if debug:
            if frame_id % 5 == 0:
                for name, roi in rppg.last_rois.items():
                    if roi is not None:
                        # TODO Draw the box on the ROI for better visualisation
                        cv2.imshow(f"{name} ROI", roi)
        if frame_id % 30 == 0:  # every 1 second at 30 FPS
            bpm, snr, is_live = rppg.compute_liveness()
            print(f"Estimated BPM: {bpm}, SNR: {snr} dB, Liveness: {is_live}")
        #------------------------------------------------------------------


        #---------------------- KONSTANTINOS (VERIFICATION) ---------------
        #aligned = vt.last_aligned_face
        

        #------------------------------------------------------------------

        #ESC stops EVERYTHING immediately
        if cv2.waitKey(1) == 27:
            vt.stop()       # stop Thread
            break           # EXIT MAIN LOOP IMMEDIATELY

        #If thread died for any reason
        if not vt.running:
            break
        
        # If sigint (CTRL+C), stop thread
        if not vt.is_alive():
            break

    vt.join()
    print("VisionThread stopped")

    cv2.destroyAllWindows()  #Close all windows immediately


if __name__ == "__main__":
    main()
