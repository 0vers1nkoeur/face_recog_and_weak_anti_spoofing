from datetime import datetime
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

SIZELIST = 10                     # size of the list of liveness for the final choice to accept/reject
PREVIEW_DIR = "data/verification" # live captures (what rPPG saves)
REF_DIR = "data/Enrolled"         # enrolled users (reference faces)
MODE = "phone"                    # "phone" or "laptop"
THRESHOLD = 0.12                  # distance threshold for accept / reject


def ensure_mediapipe_env():
    """
    Relaunches the script with the mediapipe venv interpreter
    if we are not already inside it.
    """
    venv_python = Path(__file__).parent / "mediapipe_env" / "bin" / "python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(venv_python, [str(venv_python)] + sys.argv)


ensure_mediapipe_env()


#For easy close of camera, GUI **AND** thread (IF YOU ARE GOING TO STOP CAMERA OR PROGRAM IN ANY PLEASE USE THIS!!!)
def force_stop(vt, cap):
    try:
        cap.release()
    except:
        pass

    try:
        cv2.destroyAllWindows()
    except:
        pass

    try:
        vt.stop()
    except:
        pass

    try:
        if hasattr(vt, "frame_queue"):  #Unblocks thread
            vt.frame_queue.put(None) 
    except:
        pass

    try:
        vt.join()
    except:
        pass

def main():
    # ===================== KONSTANTINOS: LOAD ENROLLED GALLERY =====================
    # We support multiple enrolled users. Each image file in REF_DIR is one template.
    # Example filenames:
    #   data/Enrolled/aleksa_01.jpg -> user_id = "aleksa_01"
    #   data/Enrolled/konst_01.jpg  -> user_id = "konst_01"

    enrolled_embeddings = {}  # user_id -> embedding

    if not os.path.exists(REF_DIR):
        print(f"[Konst] ❌ Enrolled directory not found: {REF_DIR}")
        return

    for fname in os.listdir(REF_DIR):
        fpath = os.path.join(REF_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        user_id = os.path.splitext(fname)[0]  # e.g. "aleksa_01"
        ref_bgr = cv2.imread(fpath)
        if ref_bgr is None:
            print(f"[Konst] ⚠️ Could not read enrolled image: {fpath}")
            continue

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        emb_ref = get_embedding_from_aligned_face(ref_rgb)
        if emb_ref is None:
            print(f"[Konst] ⚠️ Could not compute embedding for enrolled image: {fpath}")
            continue

        enrolled_embeddings[user_id] = emb_ref

    if not enrolled_embeddings:
        print("[Konst] ❌ No valid enrolled faces found in Enrolled/.")
        return

    print(f"[Konst] 📸 Loaded {len(enrolled_embeddings)} enrolled face(s) from {REF_DIR}:")
    print("       ", ", ".join(enrolled_embeddings.keys()))

    # -------- VisionThread (processing only) --------
    vt = VisionThread()
    vt.start()
    # ------------------------------------------------

    # -------- rPPG ----------------------------------
    rppg = RPPG(
        fps=30,
        window_seconds=15,
        snr_thresh_db=10.0,
        roi_landmark_sets=RPPG.CHEEKS + RPPG.FOREHEAD
    )
    plotter = SignalPlotter(rppg)

    debug_rois_enabled = False  # toggled via 'r' key
    counter = 0
    is_live = False             # default liveness state
    liveness_list = []
    phase = 1                   # 1 = rPPG, 2 = verification, 0 = finished
    aligned = None              # last aligned face for verification
    counter_before_stop = 15    # attempts before blocking after several failed tries

    final_distance = None
    final_match = None
    final_user_id = None

    print("VisionThread started.")

    # ------------------------------------------------

    # -------- CAMERA OPENED IN MAIN  ----------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        force_stop(vt, cap)
        return
    
    cv2.namedWindow("Vision & Detection", cv2.WINDOW_NORMAL)

    # ------------------------------------------------

    # -------- CAMERA OPENED IN MAIN  ----------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        force_stop(vt, cap)
        return
    
    cv2.namedWindow("Vision & Detection", cv2.WINDOW_NORMAL)

    # ===================== MAIN LOOP =======================================
    while True:

        # Key for interactions
        key = cv2.waitKey(1) & 0xFF

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            continue

        # Send frame to background processing thread
        vt.submit_frame(frame)

        # -------------------------------- ALEKSA (GUI) ----------------------------------

        # Checking if face is detected
        face_detected = vt.last_coords is not None

        # Semi-transparent header 
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], 60),
            (0, 0, 0),
            -1
        )
        alpha = 0.35
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Status tekst: Detected / Not detected
        status_text = "Detected" if face_detected else "Not detected"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
        )

        #Just adding read/green dot on header if face is detected or not
        dot_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 9, dot_color, -1)

        #Angles in borrders instead a big green box
        if vt.last_bbox is not None:
            x1, y1, x2, y2 = vt.last_bbox
            COLOR = (0, 200, 255)
            THICK = 2
            corner_len = 25

            # top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), COLOR, THICK)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), COLOR, THICK)
            # top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), COLOR, THICK)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), COLOR, THICK)
            # bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), COLOR, THICK)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), COLOR, THICK)
            # bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), COLOR, THICK)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), COLOR, THICK)

        # FPS from VisionThread
        cv2.putText(
            frame,
            f"FPS: {vt.last_fps}",
            (20, 85),  # pomjereno ispod headera
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # rPPG Buffer size
        buffer_size = len(rppg.signal_buffer)
        buffer_max = rppg.buffer_size

        bar_x = 20
        bar_y = frame.shape[0] - 60
        bar_w = 250   # width of progress bar
        bar_h = 15    # height of progress bar

        #Background of buffer line
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)

        #Filled part of buffer line
        if buffer_max > 0:
            filled_w = int((buffer_size / buffer_max) * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 180, 255), -1)
            if buffer_size/buffer_max == 1:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        #Frame
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

        #Buffer size in text
        cv2.putText(
            frame,
            f"Buffer: {buffer_size}/{buffer_max}",
            (bar_x, bar_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # GUI Display
        cv2.imshow("Vision & Detection", frame)

        #---------------------------------------------------------------------------

        counter += 1

        #---------------------- LORENZO (ANTI-SPOOF) -------------------------------
        if phase == 1:  # rPPG processing phase
            if vt.last_frame is not None and vt.last_coords is not None:
                rppg.update_buffer(vt.last_frame, vt.last_coords)

                # Every frames per second...
                if counter % rppg.fps == 0:

                    if debug_rois_enabled :
                        for idx, (name, roi) in enumerate(rppg.last_rois.items()):
                            if roi is not None:
                                win_name = f"{name} ROI"
                                cv2.imshow(win_name, roi)
                    
                    if plotter.hidden == False : 
                        plotter.plot_signals()

                    if len(rppg.signal_buffer) == rppg.buffer_size:
                        # Compute liveness
                        bpm, snr, is_live = rppg.compute_liveness()

                        # Handle case of not enough signal
                        if bpm is None and snr is None and is_live is False:
                            print("[rPPG] Not enough signal for liveness computation.\n")
                            continue
                        # Update liveness list
                        elif len(liveness_list) >= SIZELIST:
                            liveness_list.pop(0)    # keep list size manageable by removing oldest entry
                        liveness_list.append(is_live)

                        # Print current status in stdout
                        print("\n[rPPG]--------------------------------------------------------\n"
                              "liveness_list:", liveness_list,
                              f"\nCurrent BPM: {bpm}, SNR: {snr} dB, Liveness: {is_live}\n"
                              "-----------------------------------------------------------------\n")

                        # Final decision after collecting enough liveness results
                        if len(liveness_list) == SIZELIST:
                            print("[rPPG] Liveness results collected. Making final decision... Number of attempts left before stop:",
                                  counter_before_stop)
                            counter_before_stop -= 1

                            # Final decision based on majority in liveness_list or timeout

                            # --- REJECTED AS SPOOF ---
                            if counter_before_stop == 0:
                                print("\n[rPPG] ❌ Too many failed liveness attempts. Stopping the process...\n")

                                force_stop(vt, cap)
                                break
                            # --- ACCEPTED AS LIVE ---
                            elif liveness_list.count(True) > SIZELIST / 2:
                                print("\n[rPPG] ✅ Liveness confirmed by rPPG.\n")

                                # Save last aligned face for verification. File named with datetime
                                aligned_path = os.path.join(
                                    PREVIEW_DIR,
                                    f"{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.jpg"
                                )
                                print("[rPPG] ▶ Saving last aligned face for verification to:", aligned_path)

                                aligned = align_and_crop(
                                    frame_bgr=vt.last_frame,
                                    coords=vt.last_coords,
                                    crop_size=224,
                                    save_debug_path=aligned_path,
                                    frame_bgr=vt.last_frame,
                                    coords=vt.last_coords,
                                    crop_size=224,
                                    save_debug_path=aligned_path,
                                )

                                phase = 2  # go to verification phase

            elif vt.last_coords is None:
                if counter % rppg.fps == 0:
                    print('No face detected, clearing rPPG buffers...\n')
                rppg.signal_buffer = []  # reset buffer if no face detected
                rppg.filtered_signal_buffer = []  # reset filtered buffer if no face detected
                continue
            else:
                if counter % rppg.fps == 0:
                    print('No frame available...\n')
                continue
        # -------------------------------------------------------------------

        if key == ord('r'):
            debug_rois_enabled = not debug_rois_enabled
            state = "ON" if debug_rois_enabled else "OFF"
            print(f"[Debug ROI] {state}")
            if not debug_rois_enabled:
                # Hide all ROI debug windows
                for name in rppg.last_rois:
                    cv2.destroyWindow(f"{name} ROI")

        # Hide/show rPPG plotter with 'h' key
        if key == ord('h'):
            if plotter.hidden:
                plotter.show()
                print("[rPPG] Signal plotter shown.")
            else:
                plotter.hide()
                print("[rPPG] Signal plotter hidden.")

        #---------------------- KONSTANTINOS (VERIFICATION) ---------------

        if phase == 2:  # verification phase
        # ---------------------- KONSTANTINOS (VERIFICATION) ----------------
        if phase == 2:  # verification phase

            print("[Konst] ✅ Liveness confirmed by rPPG, proceeding with verification.")


            # Only try to verify when we actually have an aligned face
            if aligned is not None:
                print("[Konst] ▶ Running verification on last_aligned_face")

                emb_live = get_embedding_from_aligned_face(aligned)

                if emb_live is None:
                    print("[Konst] ❌ Could not compute embedding (no face detected).")
                else:
                    # Compare against all enrolled users and pick best match
                    best_user_id = None
                    best_distance = None

                    for user_id, emb_ref in enrolled_embeddings.items():
                        distance, _ = compare_embeddings(emb_live, emb_ref, threshold=THRESHOLD)
                        if best_distance is None or distance < best_distance:
                            best_distance = distance
                            best_user_id = user_id

                    final_distance = best_distance
                    final_user_id = best_user_id
                    final_match = best_distance is not None and best_distance < THRESHOLD

                    print("\n[Konst] 🔍 Verification result (gallery)")
                    print("---------------------------")
                    print(f"Best match user: {best_user_id}")
                    print(f"Distance: {best_distance:.4f}" if best_distance is not None else "Distance: None")
                    print(f"Match:    {final_match}")

                    if final_match:
                        print("---------------------------")
                        print(f"✅ ACCEPT: face matches {best_user_id}")
                    else:
                        print("---------------------------")
                        print(f"❌ REJECT: face does NOT match any enrolled user")

                    # In both cases, end the process after one decision
                    phase = 0
            else:
                # nothing aligned yet
                pass
        #-----------------------------------------------------------------

        # ESC stops EVERYTHING immediately
        if key == 27:  # Escape key
            print("ESC pressed. Exiting...")
            force_stop(vt, cap)
            break

        if phase == 0:
            print("\n[rPPG & Konst] 🧾 Process finished. System will close...")
            if final_distance is not None and final_match is not None:
                print(f"  • Final distance: {final_distance:.4f}")
                print(f"  • Final decision: {'ACCEPT' if final_match else 'REJECT'}")
                if final_user_id is not None:
                    print(f"  • Matched user: {final_user_id}")
            vt.stop()  # stop Thread
            break  # EXIT MAIN LOOP IMMEDIATELY

        # If thread died for any reason
        if not vt.running:
            force_stop(vt, cap)
            break

    force_stop(vt, cap)
    print("VisionThread stopped")


if __name__ == "__main__":
    main()