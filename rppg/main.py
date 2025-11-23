import time
import threading
import cv2
from rppg.roisets_class import ROISet
from rppg.rppg_class import RPPG
from vision_detection.vision_thread import VisionThread


def _check_correctness() -> None:
    """Fail fast if the main dependencies do not expose the expected API."""
    issues = []
    if not issubclass(VisionThread, threading.Thread):
        issues.append("VisionThread must inherit from threading.Thread.")

    for method_name in ("start", "stop", "join"):
        method = getattr(VisionThread, method_name, None)
        if not callable(method):
            issues.append(f"VisionThread.{method_name}() must be callable.")

    for method_name in ("update_buffer", "compute_liveness"):
        method = getattr(RPPG, method_name, None)
        if not callable(method):
            issues.append(f"RPPG.{method_name}() must be callable.")

    roi_set = getattr(RPPG, "CHEEKS", None)
    if roi_set is None:
        issues.append("RPPG must expose the CHEEKS ROI set.")
    elif not isinstance(roi_set, ROISet):
        issues.append("RPPG.CHEEKS must be defined as an ROISet.")

    if issues:
        problems = "\n - ".join(issues)
        raise RuntimeError(f"rPPG pipeline misconfigured:\n - {problems}")


_check_correctness()

def main():
    debug = True                                                            # Set to True to enable debug visualisations
    vision_thread = VisionThread()                                          # Initialize the vision thread
    rppg = RPPG(fps=30, roi_landmark_sets=RPPG.CHEEKS)                  # Initialize the rPPG processor with the CHEEKS ROI set
    frame_id = 0

    vision_thread.start()                                                  # Start the vision thread
    while not vision_thread.running and vision_thread.is_alive():
        time.sleep(0.01)  # laisser le thread initialiser la caméra

    last_buffer_frame = None

    while True:
        if not vision_thread.frame_buffer:
            if not vision_thread.running:
                print("Warning: vision thread stopped with an empty buffer, stopping.")
                break
            continue

        latest_frame = vision_thread.frame_buffer[-1]
        if latest_frame is last_buffer_frame:
            continue
        frame = latest_frame
        last_buffer_frame = latest_frame

        #------------------------------------------------------------
        # rPPG processing
        # rPPG update with the current frame and landmarks
        coords = vision_thread.last_coords

        if coords is not None and len(coords) > 0:
            rppg.update_buffer(frame, coords)

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
