# Standard imports
import argparse
from datetime import datetime
import cv2
import os
import numpy as np

# Custom imports
from vision_detection.vision_thread import VisionThread, force_stop
from vision_detection.face_alignment import align_and_crop
from rppg.rppg_class import RPPG
from rppg.utils import SignalPlotter
from identificationCS_evaluation.verification import get_embedding_from_aligned_face, compare_embeddings
from identificationCS_evaluation.identification_eval import load_enrolled_gallery, reliability_srr

# ------------------------ CONSTANTS ------------------------------
SIZELIST = 10                               # size of the list of liveness for the final choice to accept/reject
PREVIEW_DIR = "data/verification"           # live captures (what rPPG saves)
REF_DIR = "data/Enrolled"                   # enrolled users (reference faces)
REF_ALIGNED_DIR = "data/Enrolled_aligned"   # debug: aligned crops of enrolled faces
THRESHOLD = 0.15                            # distance threshold for accept / reject (tight: your ~0.16 passes, others ~0.17+ fail)
LIVE_EMB_COUNT = 10                         # number of live embeddings to collect for a stable decision
SECOND_GAP = 0.010                          # require best user to beat 2nd-best by this margin (your gap ~0.012)
RPPG_SKIP = False                           # default: run rPPG; set via CLI flag to skip
SRR_MIN = 0.01                              # minimum reliability score to accept (very permissive, distance is primary)
ALIGN_CROP_SIZE = 320                       # aligned face crop size (pixels)
ALIGN_BBOX_SCALE = 2.0                      # expand around bbox to keep forehead/chin/ears
ALIGN_ROTATE = False                        # disable rotation to keep full face (enroll/live consistent)


def parse_args():
    '''This function parses the command line arguments.
    Returns:
        argparse.Namespace: The parsed arguments.'''
    parser = argparse.ArgumentParser(description="Facial recognition + rPPG verification")
    parser.add_argument(
        "--rppg-skip",
        action="store_true",
        help="Skip rPPG and go directly to verification (default runs full flow)",
    )
    return parser.parse_args()
# ------------------------ MAIN FUNCTION ------------------------------
def main():
    enrolled_embeddings = load_enrolled_gallery(
        ref_dir=REF_DIR,
        ref_aligned_dir=REF_ALIGNED_DIR,
        align_crop_size=ALIGN_CROP_SIZE,
        align_bbox_scale=ALIGN_BBOX_SCALE,
        align_rotate=ALIGN_ROTATE,
        threshold=THRESHOLD,
    )
    if not enrolled_embeddings:
        return

    # -------- VisionThread (processing only) --------
    vt = VisionThread()
    vt.start()
    # ------------------------------------------------

    # -------- rPPG ----------------------------------
    rppg = RPPG(
        fps=30,
        window_seconds=15,
        snr_thresh_db=10.0,
        roi_landmark_sets=RPPG.NEW_CHEEKS + RPPG.FOREHEAD
    )
    plotter = SignalPlotter(rppg)

    debug_rois_enabled = False  # toggled via 'r' key
    counter = 0
    is_live = False             # default liveness state
    liveness_list = []
    phase = 1                   # 1 = rPPG, 2 = verification, 0 = finished
    live_embeddings = []        # collect multiple live embeddings for averaging
    counter_before_stop = 15    # attempts before blocking after several failed tries

    final_distance = None
    final_match = None
    final_user_id = None
    rejection_reasons = []
    first_round_phase2 = True

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

    # ===================== MAIN LOOP =======================================
    while True:

        # Key for interactions
        key = cv2.waitKey(1) & 0xFF

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            continue

        # Mirror the frame horizontally so preview and processing are consistent with user view
        frame = cv2.flip(frame, 1)

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
            # Down right corner
            (frame.shape[1] - 120, frame.shape[0] - 20),
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

        #-------------------------------------------------------------------------------
        if vt.last_frame is not None:
            counter += 1
            #---------------------- LORENZO (ANTI-SPOOF) -------------------------------
            if phase == 1 and not RPPG_SKIP :  # rPPG processing phase
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

                                    _ = align_and_crop(
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

                # Keyboard interactions for phase 1
                # Toggle ROI debug windows with 'r' key
                if key == ord('r') :
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
            # -------------------------------------------------------------------

            #---------------------- KONSTANTINOS (VERIFICATION) ---------------

            if phase == 2 or RPPG_SKIP:  # verification phase
                if first_round_phase2:
                    print("[Konst] ✅ Liveness confirmed by rPPG, proceeding with verification.")

                if vt.last_frame is not None and vt.last_coords is not None and len(live_embeddings) < LIVE_EMB_COUNT:
                    if first_round_phase2 : print('[Konst] ▶ Collecting live embeddings for verification...')
                    aligned_live = align_and_crop(
                        frame_bgr=vt.last_frame,
                        coords=vt.last_coords,
                        crop_size=ALIGN_CROP_SIZE,
                        bbox_scale=ALIGN_BBOX_SCALE,
                        align=ALIGN_ROTATE,
                    )

                    if aligned_live is not None:
                        emb_live = get_embedding_from_aligned_face(aligned_live)
                        if emb_live is not None:
                            live_embeddings.append(emb_live)
                            print(f"[Konst] ▶ Collected live embedding {len(live_embeddings)}/{LIVE_EMB_COUNT}")
                    
                    first_round_phase2 = False
    
                if len(live_embeddings) >= LIVE_EMB_COUNT:
                    print("[Konst] ▶ Collected sufficient live embeddings, performing verification against enrolled gallery...")
                    # For each user, compute per-sample min distance across their templates, then take median across samples
                    distances = []
                    for user_id, ref_emb_list in enrolled_embeddings.items():
                        per_sample_distances = []
                        for emb_live in live_embeddings:
                            dists = [
                                compare_embeddings(emb_live, emb_ref, threshold=THRESHOLD)[0]
                                for emb_ref in ref_emb_list
                            ]
                            per_sample_distances.append(min(dists))
                        median_distance = float(np.median(per_sample_distances))
                        distances.append((median_distance, user_id))

                    distances.sort(key=lambda x: x[0])

                    best_distance, best_user_id = distances[0]
                    second_distance = distances[1][0] if len(distances) > 1 else None
                    srr = reliability_srr(best_distance, second_distance)

                    final_distance = best_distance
                    final_user_id = best_user_id
                    # If SECOND_GAP is 0, skip the gap check entirely
                    gap_ok = SECOND_GAP == 0.0 or second_distance is None or (second_distance - best_distance) > SECOND_GAP
                    final_match = (
                        best_distance is not None
                        and best_distance < THRESHOLD
                        and gap_ok
                        and srr >= SRR_MIN
                    )

                    print("\n[Konst] 🔍 Verification result (gallery)")
                    print("---------------------------")
                    print(f"Best match user: {best_user_id}")
                    print(f"Distance: {best_distance:.4f}" if best_distance is not None else "Distance: None")
                    if second_distance is not None:
                        print(f"2nd-best distance: {second_distance:.4f} (needs gap > {SECOND_GAP:.4f})")
                    print(f"SRR: {srr:.3f} (needs >= {SRR_MIN:.3f})")
                    print(f"Match:    {final_match}")

                    if final_match:
                        print("---------------------------")
                        print(f"✅ ACCEPT: face matches {best_user_id}")
                    else:
                        # Record rejection reasons
                        rejection_reasons = []
                        if best_distance is None or best_distance >= THRESHOLD:
                            rejection_reasons.append(
                                f"Distance too high: {best_distance:.4f} >= {THRESHOLD}"
                            )
                        if not gap_ok:
                            gap = second_distance - best_distance if second_distance is not None else 0
                            rejection_reasons.append(
                                f"Gap between best and 2nd-best too small: {gap:.4f} <= {SECOND_GAP}"
                            )
                        if srr < SRR_MIN:
                            rejection_reasons.append(
                                f"Reliability score (SRR) too low: {srr:.3f} < {SRR_MIN:.3f}"
                            )
                        print("---------------------------")
                        print(f"❌ REJECT: face does NOT match any enrolled user")

                    # In both cases, end the process after one decision
                    phase = 0
            # -------------------------------------------------------------------

            if phase == 0:
                print("\n[rPPG & Konst] 🧾 Process finished. System will close...")
                if final_distance is not None and final_match is not None:
                    print(f"  • Final distance: {final_distance:.4f}")
                    print(f"  • Final decision: {'ACCEPT' if final_match else 'REJECT'}")
                    if final_user_id is not None:
                        print(f"  • Matched user: {final_user_id}")
                    if not final_match and rejection_reasons:
                        print(f"  • Rejection reasons:")
                        for reason in rejection_reasons:
                            print(f"    - {reason}")
                vt.stop()  # stop Thread
                break  # EXIT MAIN LOOP IMMEDIATELY
        
        # END OF FRAME PROCESSING --------------------------------

        # ESC stops EVERYTHING immediately
        if key == 27:  # Escape key
            print("ESC pressed. Exiting...")
            vt.stop()  # stop Thread
            break  # EXIT MAIN LOOP IMMEDIATELY

        # If thread died for any reason
        if not vt.running:
            force_stop(vt, cap)
            break

    force_stop(vt, cap)
    print("VisionThread stopped")


if __name__ == "__main__":
    args = parse_args()
    RPPG_SKIP = args.rppg_skip
    main()
