import numpy as np
from scipy.signal import detrend, butter, filtfilt, find_peaks
from vision_detection.face_detection import FaceMeshDetector
from rppg.roisets_class import ROISet

class RPPG:
    # Define preset ROI sets for easy use----------
    CHEEKS = ROISet(
        {
            "left_cheek": ROISet.LEFT_CHEEK,
            "right_cheek": ROISet.RIGHT_CHEEK,
        }
    )
    FOREHEAD = ROISet({"forehead": ROISet.FOREHEAD})
    #-----------------------------------------------

    def __init__(
        self,
        fps,
        window_seconds=10,
        low_hz=0.75,
        high_hz=3.0,
        snr_thresh_db=3.0,
        roi_landmark_sets=None,
    ):
        """
        Initialize the RPPG processor.
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.buffer_size = int(fps * window_seconds)
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.snr_thresh_db = snr_thresh_db

        # If no ROISet is chosen, use the default CHEEKS set
        if roi_landmark_sets is None:
            roi_landmark_sets = self.CHEEKS

        # Normalize roi_landmark_sets to a dict to make processing easier
        if not isinstance(roi_landmark_sets, ROISet):
            raise TypeError("roi_landmark_sets must be provided as ROISet presets.")
        
        # Convert ROISet to a dictionary for internal use
        roi_mapping = roi_landmark_sets.to_dict()

        # Assign and validate the ROI landmark sets
        self.roi_landmark_sets = roi_mapping

        # Validate that at least one ROI is defined
        if not self.roi_landmark_sets:
            raise ValueError("roi_landmark_sets must contain at least one ROI definition.")

        # Initialize last ROIs storage for debug visualisation
        self.last_rois = {name: None for name in self.roi_landmark_sets}

        # Initialize the rolling signal buffer, this will store the mean green channel values over time
        self.signal_buffer = []
    
    def update_buffer(self, frame_rgb):
        """
        Extracts an ROI from landmarks, computes the mean green channel 
        and appends it to the rolling rPPG buffer.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Current video frame in RGB format (cropped and aligned).
        Returns
        -------
        None
            Updates the internal signal buffer in-place.
        """
        # Sanity checks
        if frame_rgb is None or self.roi_landmark_sets is None or len(self.roi_landmark_sets) == 0:       # This checks if we have frame and landmarkss
            print("Error: Missing frame or ROI landmark sets.")
            return
        #----------------------------------------------------------------------
        # 1) extract ROI(s) using landmarks of the class
        # We extract the height and width of the frame to make sure we don't go out of bounds
        frame_h, frame_w = frame_rgb.shape[:2]
        # Initialize list to store mean green values from each ROI
        mean_values = []
        # We loop over each defined ROI set to extract and compute mean green channel
        for name, roi_indices in self.roi_landmark_sets.items():                    # roi_landmark_sets is a dict where key is name of roi and value is list of landmark indices. Ex : {"left_cheek": [50, 101, 102, 103], "right_cheek": [...]}
            roi_points = [self.roi_landmark_sets[i] for i in roi_indices]           # This gonna extract the ROI using the landmark indices. Ex : if roi_indices = [50, 101, 102, 103], then we take the points corresponding to these indices
            # Using compute_bbox to get the bounding box of the ROI
            x1, y1, x2, y2 = FaceMeshDetector.compute_bbox(roi_points)              # We use FaceMeshDetector's static method to compute bbox
            # Ensure the bounding box is within frame bounds---------------------
            x1 = max(0, min(frame_w - 1, x1))                                       # Ensure x1 is at least 0 and at most frame width - 1 to avoid going out of bounds
            x2 = max(x1 + 1, min(frame_w, x2))                                      # Ensure x2 is at least one pixel more than x1 and at most frame width                      
            y1 = max(0, min(frame_h - 1, y1))                                       # Ensure y1 is at least 0 and at most frame height - 1
            y2 = max(y1 + 1, min(frame_h, y2))                                      # Ensure y2 is at least one pixel more than y1 and at most frame height
            roi = frame_rgb[y1:y2, x1:x2]                                           # Extract the ROI from the frame
            # Sanity check for empty ROI
            if roi.size == 0:
                print(f"Warning: Empty ROI for {name}, skipping.")
                continue
            #--------------------------------------------------------------------
            # Store the last extracted ROIs for debug visualisation for each defined ROI
            self.last_rois[name] = roi
            # Compute mean of green channel in the ROI and store it
            mean_values.append(roi[:, :, 1].mean(dtype=np.float32))

        # Check if we have any mean values computed
        if not mean_values:
            print("Warning: No valid ROIs found, skipping frames.")
            return
        #----------------------------------------------------------------------
        # 2) average mean of green channel over the available ROIs
        mean_green = float(np.mean(mean_values, dtype=np.float64))
        #----------------------------------------------------------------------
        # 3) push into buffer (rolling window)
        self.signal_buffer.append(mean_green)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

    # TODO: implement the actual signal processing steps
    def compute_liveness(self):
        """
        Use self.signal_buffer to:
        - preprocess the signal,
        - compute BPM + SNR,
        - decide if is_live or not.
        """
        ...
        return bpm, snr_db, is_live # pyright: ignore[reportUndefinedVariable]