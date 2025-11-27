from datetime import datetime
from signal import pause
from vision_detection.vision_thread import VisionThread
from rppg.rppg_class import RPPG
import time
import cv2
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from vision_detection.verification import get_embedding_from_aligned_face, compare_embeddings
from vision_detection.face_alignment import align_and_crop
from rppg.utils import SignalPlotter

SIZELIST = 10  # size of the list of liveness for the final choice to accept/reject
PREVIEW_DIR = "data/verification"

def ensure_mediapipe_env():
    """Relance le script avec l'interpréteur du venv mediapipe si on n'est pas déjà dedans."""
    venv_python = Path(__file__).parent / "mediapipe_env" / "bin" / "python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(venv_python, [str(venv_python)] + sys.argv)


ensure_mediapipe_env()

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
    rppg = RPPG(fps=30, window_seconds=15, snr_thresh_db=10.0, roi_landmark_sets=RPPG.CHEEKS)
    plotter = SignalPlotter(rppg, stop_callback=vt.stop)
    debug = True
    debug_rois_enabled = False  # toggled via touche clavier
    counter = 0
    is_live = False  # default liveness state
    liveness_list = []
    phase = 1  # phase
    aligned = None  # last aligned face for verification

    print("VisionThread started.")

    while True:

        #print("Buffer len:", len(vt.frame_buffer))
        #time.sleep(1)

        while not vt.running and vt.is_alive():
            time.sleep(0.01)  # let the thread initialize the camera
        

        counter += 1

        #---------------------- LORENZO (ANTI-SPOOF) ----------------------
        if phase == 1 :  # rPPG processing phase
            if vt.last_frame is not None and vt.last_coords is not None:
                rppg.update_buffer(vt.last_frame, vt.last_coords)

                if False :  # Set to True to enable debug plots every 5 frames
                        for idx, (name, roi) in enumerate(rppg.last_rois.items()):
                            if roi is not None:
                                win_name = f"{name} ROI"
                                cv2.imshow(win_name, roi)
                                # Décale chaque fenêtre pour éviter la superposition
                                cv2.moveWindow(win_name, 40 + idx * (roi.shape[1] + 40), 40)
                
                # Every 30 frames...
                if counter % 30 == 0:
                    # Compute liveness
                    bpm, snr, is_live = rppg.compute_liveness()
                    if len(liveness_list) >= SIZELIST :
                        liveness_list.pop(0)    # keep list size manageable by removing oldest entry
                    liveness_list.append(is_live)

                    if debug: print ("\n[rPPG]--------------------------------------------------------\n"
                                    "liveness_list:", liveness_list, 
                                    f"\nCurrent BPM: {bpm}, SNR: {snr} dB, Liveness: {is_live}\n"
                                    "-----------------------------------------------------------------\n")
                    
                    if debug:
                        plotter.plot_signals()
                        plt.pause(0.001)
                    
                    if len(liveness_list) == SIZELIST and liveness_list.count(True) > SIZELIST / 2:
                        print("\n[rPPG] ✅ Liveness confirmed by rPPG.\n")

                        # Save last aligned face for verification. The file will be named with the datetime
                        aligned_path = f"{PREVIEW_DIR}/{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.jpg"

                        print("[rPPG] ▶ Saving last aligned face for verification to:", aligned_path)

                        aligned = align_and_crop(
                            frame_bgr=vt.last_frame,
                            coords=vt.last_coords,
                            crop_size=224,
                            save_debug_path=aligned_path,
                        )

                        phase = 2  # go to verification phase

            elif vt.last_coords is None:
                if counter % 30 == 0 : 
                    print('No face detected, clearing rPPG buffers...\n')
                rppg.signal_buffer = []  # reset buffer if no face detected
                rppg.filtered_signal_buffer = [] # reset filtered buffer if no face detected
                plt.close()
                plt.pause(0.001)
                continue
            else :
                if counter % 30 == 0 : 
                    print('No frame available...\n')
                continue
        #------------------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            debug_rois_enabled = not debug_rois_enabled
            state = "ON" if debug_rois_enabled else "OFF"
            print(f"[Debug ROI] {state}")
            if not debug_rois_enabled:
                # Hide all ROI debug windows
                for name in rppg.last_rois:
                    cv2.destroyWindow(f"{name} ROI")

        #---------------------- KONSTANTINOS (VERIFICATION) ---------------
        
        if phase == 2 :  # verification phase

            print("[Konst] ✅ Liveness confirmed by rPPG, proceeding with verification.")
            
            # Only try to verify when we actually have an aligned face
            if aligned is not None:
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
                        # Pause rPPG and verification processing
                        phase = 0
                    else:
                        print(f"❌ REJECT: face does NOT match {USER_ID}")                            # pause for 30 seconds to avoid immediate retries
                        print("[Konst] ⏳ Pausing verification for 30 seconds to prevent immediate retries.")
                        time.sleep(30)
                        phase = 1  # go back to rPPG processing phase
            else:
                # nothing aligned yet – optional debug print
                pass
            #-----------------------------------------------------------------

        #ESC stops EVERYTHING immediately
        if key == 27: # Escape key
            print("ESC pressed. Exiting...")
            vt.stop()       # stop Thread
            break           # EXIT MAIN LOOP IMMEDIATELY

        if phase == 0 :
            print("[rPPG & Konst] ✅ Verification successful. System will close...)")
            time.sleep(5)  # wait a bit to let user see the message
            vt.stop()       # stop Thread
            break           # EXIT MAIN LOOP IMMEDIATELY

        #If thread died for any reason
        if not vt.running:
            break

    vt.join()
    print("VisionThread stopped")
    cv2.destroyAllWindows()  #Close all windows immediately


if __name__ == "__main__":
    main()
