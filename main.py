from vision_detection.vision_thread import VisionThread
from rppg.rppg_class import RPPG
import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from vision_detection.verification import get_embedding_from_aligned_face, compare_embeddings
from rppg.utils import plot_signal_buffer, plot_filtered_signal

def main():

    # Konstantinos: load reference embedding once
    USER_ID = "user1"
    ref_path = os.path.join("data", "embeddings", f"{USER_ID}.npy")
    if not os.path.exists(ref_path):
        print(f"[Konst] ❌ Reference embedding not found: {ref_path}")
        return

    emb_ref = np.load(ref_path)
    print(f"[Konst] 📂 Loaded reference embedding for {USER_ID} from {ref_path}")

    vt = VisionThread()
    vt.start()
    rppg = RPPG(fps=30, window_seconds=15, snr_thresh_db=6.0, roi_landmark_sets=RPPG.CHEEKS)
    debug = False
    frame_id = 0

    print("VisionThread started.")

    while True:

        #print("Buffer len:", len(vt.frame_buffer))
        #time.sleep(1)

        while not vt.running and vt.is_alive():
            time.sleep(0.01)  # let the thread initialize the camera
        

        frame_id += 1

        is_live = False  # default liveness state

        #---------------------- LORENZO (ANTI-SPOOF) ----------------------
        if frame_id == 0 : print('RPPG processing...\n')

        if vt.last_frame is not None and vt.last_coords is not None:
            rppg.update_buffer(vt.last_frame, vt.last_coords)

            if debug and (frame_id % 5 == 0):
                    for name, roi in rppg.last_rois.items():
                        if roi is not None:
                            # TODO Draw the box on the ROI for better visualisation
                            cv2.imshow(f"{name} ROI", roi)
            if len(rppg.signal_buffer) == rppg.buffer_size:
                plt.close()  # close previous plot
                plot_signal_buffer(rppg)
            bpm, snr, is_live = rppg.compute_liveness()
            if rppg.last_filtered_signal is not None:
                plt.close()  # close previous plot
                plot_filtered_signal(rppg)
            print(f"Estimated BPM: {bpm}, SNR: {snr} dB, Liveness: {is_live}")

        elif vt.last_coords is None:
            if frame_id % 30 == 0 : 
                print('No face detected, clearing rPPG buffer...\n')
            rppg.signal_buffer.clear()  # reset buffer if no face detected
            continue
        else :
            if frame_id % 30 == 0 : 
                print('No frame available...\n')
            continue
        #------------------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        #---------------------- KONSTANTINOS (VERIFICATION) ---------------
        if is_live:
            aligned = vt.last_aligned_face
            # Only try to verify when we actually have an aligned face
            if aligned is not None:
                # Press 'v' to run verification on the current aligned face
                if key == ord('v'):
                    print("[Konst] ▶ Running verification on last_aligned_face")

                    emb_live = get_embedding_from_aligned_face(aligned)
                    if emb_live is None:
                        print("[Konst] ❌ Could not compute embedding (no face detected).")
                    else:
                        distance, is_match = compare_embeddings(emb_live, emb_ref, threshold=0.7)

                        print("\n[Konst] 🔍 Verification result")
                        print("---------------------------")
                        print(f"Distance: {distance:.4f}")
                        print(f"Match:    {is_match}")

                        if is_match:
                            print(f"✅ ACCEPT: face matches {USER_ID}")
                        else:
                            print(f"❌ REJECT: face does NOT match {USER_ID}")
            else:
                # nothing aligned yet – optional debug print
                pass
            #-----------------------------------------------------------------

        #ESC stops EVERYTHING immediately
        if key == 27:
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
