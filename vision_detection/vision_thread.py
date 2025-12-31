import cv2
import threading
import queue
import numpy as np

# relative imports inside the vision_detection package
from .video_capture import init_camera, get_frame, release_camera
from .face_detection import FaceMeshDetector
from .face_alignment import align_and_crop, compute_eye_centers
from .utils import FPSMeter, EyeSmoother


def force_stop(vt, cap):
    """Stop the VisionThread, release the camera, and close OpenCV windows."""
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if vt is None:
        return

    try:
        vt.stop()
    except Exception:
        pass

    frame_queue = getattr(vt, "frame_queue", None)
    if frame_queue is not None:
        try:
            frame_queue.put_nowait(None)
        except queue.Full:
            pass
        except Exception:
            pass

    try:
        vt.join()
    except Exception:
        pass


class VisionThread(threading.Thread):
    """
    Background thread:
      - Opens camera
      - Reads frames
      - Detects a face
      - Aligns & crops face
      - Updates self.last_aligned_face
    It does NOT open any OpenCV window or call cv2.waitKey.
    """

    def __init__(self, fps_estimate: int = 30, camera_index: int = 0, debug: bool = False):
        super().__init__()

        # camera index (0 = default)
        self.camera_index = camera_index
        self.debug = debug

        # things other modules read
        self.last_aligned_face = None
        self.last_coords = None
        self.last_frame = None
        self.last_bbox = None

        # helpers
        self.running = False
        self.detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
        self.fps_meter = FPSMeter()
        self.eye_smoother = EyeSmoother(window=5)
        self.last_fps = 0

        # NEW: safe queue for frames
        self.frame_queue = queue.Queue(maxsize=1)  # always only latest frame

    def stop(self):
        """Ask the thread to stop."""
        self.running = False

    def submit_frame(self, frame):
        """Main thread calls this to send a frame for processing."""
        if not self.running:
            return
        # drop old frame if thread is slow
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(frame)

    def run(self):
        """Main loop that runs in background."""
        self.running = True
        print("Camera thread started (processing only).")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                continue

            # ---- PROCESS EXACTLY THE SAME AS BEFORE ----

            # store frame & fill buffer (for Lorenzo)
            self.last_frame = frame.copy()

            # detect face
            coords = self.detector.process(frame)
            self.last_coords = coords

            if coords is not None:
                # bounding box
                min_x = min(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_x = max(p[0] for p in coords)
                max_y = max(p[1] for p in coords)
                self.last_bbox = (min_x, min_y, max_x, max_y)

                # eyes
                left_xy, right_xy = compute_eye_centers(coords)
                self.eye_smoother.update(left_xy, right_xy)
                sm_left_xy, sm_right_xy = self.eye_smoother.get_smoothed()

                # align & crop face (this is what YOU need)
                aligned = align_and_crop(
                    frame_bgr=frame,
                    coords=coords,
                    crop_size=224,
                    bbox_scale=1.10,
                )
                if aligned is not None:
                    self.last_aligned_face = aligned

            # FPS counter (no GUI, just keep it updated)
            self.last_fps = int(self.fps_meter.tick())

        print("Camera thread stopped")
