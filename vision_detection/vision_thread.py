import cv2
import threading
from collections import deque
import numpy as np

# relative imports inside the vision_detection package
from .video_capture import init_camera, get_frame, release_camera
from .face_detection import FaceMeshDetector
from .face_alignment import align_and_crop, compute_eye_centers
from .utils import FPSMeter, EyeSmoother


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

    def stop(self):
        """Ask the thread to stop."""
        self.running = False

    def run(self):
        """Main loop that runs in background."""
        cap = init_camera(self.camera_index)
        self.running = True
        print("Camera thread started")

        while self.running:
            frame = get_frame(cap)           # BGR frame from camera
            if frame is None:
                continue

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
                )
                if aligned is not None:
                    self.last_aligned_face = aligned

            # FPS counter (no GUI, just keep it updated)
            _ = int(self.fps_meter.tick())

            #GUI show & keyboard handling (guarded for headless environments)
            if self.debug:
                try:
                    cv2.imshow("Vision & Detection", frame)
                    if cv2.waitKey(1) == 27:
                        self.stop()
                except cv2.error as exc:
                    # When running headless OpenCV may lack GUI support; disable debug windows gracefully.
                    print(f"Warning: OpenCV GUI unavailable ({exc}), disabling debug view.")
                    self.debug = False
                    cv2.destroyAllWindows()

        #Clean up
        print("Stopping camera GUI")
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Camera thread stopped")