import cv2
import threading
import queue
import numpy as np

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
    Background processing thread:
    - Receives frames from the main thread
    - Detects FaceMesh landmarks
    - Computes bbox + (optionally) aligned/cropped preview face

    IMPORTANT:
    We do NOT rely on vt.last_aligned_face to guarantee enrollment=identification.
    In main.py we will call align_and_crop with the SAME params for both cases.
    """

    def __init__(
        self,
        fps_estimate: int = 30,
        camera_index: int = 0,
        debug: bool = False,
        # Optional: consistent crop config for the preview face (can match identification)
        align_crop_size: int = 224,
        align_bbox_scale: float = 1.10,
        align_rotate: bool = True,
    ):
        super().__init__()

        self.camera_index = camera_index
        self.debug = debug

        # Preview/canonical config (main.py may override it)
        self.align_crop_size = align_crop_size
        self.align_bbox_scale = align_bbox_scale
        self.align_rotate = align_rotate

        # Shared outputs read by other modules
        self.last_aligned_face = None
        self.last_coords = None
        self.last_frame = None
        self.last_bbox = None

        self.running = False
        self.detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
        self.fps_meter = FPSMeter()
        self.eye_smoother = EyeSmoother(window=5)
        self.last_fps = 0

        # Keep only the latest frame (drop older frames)
        self.frame_queue = queue.Queue(maxsize=1)

    def stop(self):
        """Ask the thread to stop."""
        self.running = False

    def submit_frame(self, frame):
        """Main thread sends frames here."""
        if not self.running:
            return

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass
        self.frame_queue.put(frame)

    def run(self):
        """Main loop of the background thread."""
        self.running = True
        print("Camera thread started (processing only).")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                continue

            # Store the last frame for other modules (rPPG, main loop, etc.)
            self.last_frame = frame.copy()

            # Detect landmarks
            coords = self.detector.process(frame)
            self.last_coords = coords

            if coords is not None:
                # Compute bbox from landmarks (for GUI corners)
                min_x = min(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_x = max(p[0] for p in coords)
                max_y = max(p[1] for p in coords)
                self.last_bbox = (min_x, min_y, max_x, max_y)

                # Smooth eye centers (optional, kept for stability if you use it later)
                left_xy, right_xy = compute_eye_centers(coords)
                self.eye_smoother.update(left_xy, right_xy)
                _ = self.eye_smoother.get_smoothed()

                # Optional: compute a preview aligned face (NOT used for identity consistency)
                aligned = align_and_crop(
                    frame_bgr=frame,
                    coords=coords,
                    crop_size=self.align_crop_size,
                    bbox_scale=self.align_bbox_scale,
                    align=self.align_rotate,
                )
                if aligned is not None:
                    self.last_aligned_face = aligned

            # FPS meter
            self.last_fps = int(self.fps_meter.tick())

        print("Camera thread stopped")
