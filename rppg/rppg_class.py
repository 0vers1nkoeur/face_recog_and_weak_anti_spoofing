import numpy as np
import cv2
from typing import Dict, Optional
from scipy.signal import detrend, butter, filtfilt
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
    NEW_CHEEKS = ROISet(
        {
            "left_cheek": ROISet.NEW_LEFT_CHEEK,
            "right_cheek": ROISet.NEW_RIGHT_CHEEK,
        }
    )
    FOREHEAD = ROISet({"forehead": ROISet.FOREHEAD})
    #-----------------------------------------------

    def __init__(
        self,
        fps,
        window_seconds=10,
        low_hz=0.70,
        high_hz=3.0,
        snr_thresh_db=3.0,
        roi_landmark_sets=None,
        debug=False,
    ):
        """
        Initialize the RPPG processor.
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.buffer_size = int(fps * window_seconds)
        self.low_hz = low_hz                                # Lower frequency bound for bandpass filter (Hz)
        self.high_hz = high_hz                              # Upper frequency bound for bandpass filter (Hz)
        self.snr_thresh_db = snr_thresh_db                  # SNR threshold in dB for liveness decision
        self.debug = debug                                  # Toggle detailed debug

        # If no ROISet is chosen, use the default CHEEKS set
        if roi_landmark_sets is None:
            roi_landmark_sets = self.CHEEKS

        # Normalize roi_landmark_sets to a dict to make processing easier
        if not isinstance(roi_landmark_sets, ROISet):
            raise TypeError("[rPPG] roi_landmark_sets must be provided as ROISet presets.")
        
        # Convert ROISet to a dictionary for internal use
        roi_mapping = roi_landmark_sets.to_dict()

        # Assign and validate the ROI landmark sets
        self.roi_landmark_sets = roi_mapping

        # Validate that at least one ROI is defined
        if not self.roi_landmark_sets:
            raise ValueError("[rPPG] roi_landmark_sets must contain at least one ROI definition.")

        # Initialize last ROIs storage for debug visualisation
        self.last_rois: Dict[str, Optional[np.ndarray]] = {name: None for name in self.roi_landmark_sets}

        # Initialize the rolling signal buffer, this will store the mean green channel values over time
        self.signal_buffer = []
        self.filtered_signal_buffer = []
    
    def update_buffer(self, frame_rgb, landmarks):
        """
        Extracts an ROI from landmarks, computes the mean green channel 
        and appends it to the rolling rPPG buffer.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Current video frame in RGB format (cropped and aligned).
        landmarks : Sequence[Tuple[int, int]]
            Facial landmarks detected on the current frame.
        Returns
        -------
        None
            Updates the internal signal buffer in-place.
        """
        # Sanity checks
        if frame_rgb is None or landmarks is None or len(landmarks) == 0:
            print("[rPPG] Error: Missing frame or landmarks.")
            return
        if self.roi_landmark_sets is None or len(self.roi_landmark_sets) == 0:       # This checks if we have frame and landmarkss
            print("[rPPG] Error: Missing ROI landmark sets.")
            return
        #----------------------------------------------------------------------
        # 1) extract ROI(s) using landmarks of the class
        # We extract the height and width of the frame to make sure we don't go out of bounds
        frame_h, frame_w = frame_rgb.shape[:2]
        # Initialize list to store mean green values from each ROI
        mean_values = []
        # We loop over each defined ROI set to extract and compute mean green channel
        for name, roi_indices in self.roi_landmark_sets.items():                    # roi_landmark_sets is a dict where key is name of roi and value is list of landmark indices. Ex : {"left_cheek": [50, 101, 102, 103], "right_cheek": [...]}
            try:
                roi_points = [landmarks[i] for i in roi_indices]
            except IndexError:
                print(f"[rPPG] Warning: ROI {name} references invalid landmark indices, skipping.")
                continue
            # Use polygonal ROI to avoid including background pixels
            poly = FaceMeshDetector.compute_polygone(roi_points)
            if poly.shape[0] < 3:
                print(f"[rPPG] Warning: ROI {name} has too few points for a polygon, skipping.")
                continue
            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], (1.0,))
            masked = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
            green_pixels = masked[:, :, 1][mask == 1]
            if green_pixels.size == 0:
                print(f"[rPPG] Warning: Empty ROI for {name}, skipping.")
                continue
            roi = masked
            #--------------------------------------------------------------------
            # Store the last extracted ROIs for debug visualisation for each defined ROI
            self.last_rois[name] = roi
            # Compute mean of green channel in the ROI and store it
            mean_values.append(green_pixels.mean(dtype=np.float32))

        # Check if we have any mean values computed
        if not mean_values:
            print("[rPPG] Warning: No valid ROIs found, skipping frames.")
            return
        #----------------------------------------------------------------------
        # 2) average mean of green channel over the available ROIs
        mean_green = float(np.mean(mean_values, dtype=np.float64))
        #----------------------------------------------------------------------
        # 3) push into buffer (rolling window)
        self.signal_buffer.append(mean_green)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

    # TODO Check the correctness of the code 
    def compute_liveness(self):
        """
        Use self.signal_buffer to:
        - preprocess the signal,
        - compute BPM + SNR,
        - decide if is_live or not.
        """
        # Convert signal buffer to numpy array for processing
        signal = np.array(self.signal_buffer, dtype=np.float32)     # float32 used to save memory
        if len(signal) < self.buffer_size:                          # Ensure we have enough data in the buffer
            print("[rPPG] Warning: Not enough data in signal buffer to compute liveness. Size of buffer:", len(signal), "Required:", self.buffer_size)
            return None, None, False
        # Detrend the signal to remove linear trends : 
        # remove slow variations in the signal that are not related to the heart rate. 
        # Ex of slow variations : gradual changes in lighting or movement artifacts.
        signal = detrend(signal)
        # Bandpass filter design, we use Butterworth filter to isolate the heart rate frequency band------------------------
        nyquist = 0.5 * self.fps                                    # Nyquist frequency is half the sampling rate (fps)
        low = self.low_hz / nyquist                                 # Normalized low cutoff frequency             
        high = self.high_hz / nyquist                               # Normalized high cutoff frequency
        # 3rd order Butterworth bandpass filter
        # The filter is deterministic, meaning it will produce the same output for the same input every time.
        # Here, we only put into parameters the parameters of the filter
        # 3 is the order of the filter, [low, high] are the cutoff frequencies and btype is bandpass
        # How it works ? We use the known Butterworth
        b,a = butter(4, [low, high], btype='bandpass')              # type: ignore
        #------------------------------------------------------------------------------------------------------------------
        # Apply the bandpass filter
        filtered_signal_buffer = filtfilt(b, a, signal)
        # Store as a plain list so we can clear/extend it easily elsewhere
        self.filtered_signal_buffer = filtered_signal_buffer
        # Rejet si quasi pas de dynamique
        if np.std(filtered_signal_buffer) < 1e-3:  # valeur à ajuster empiriquement
            return None, None, False
        
        # Compute the power spectrum using FFT
        n_samples = filtered_signal_buffer.shape[0]
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fps)
        spectrum = np.fft.rfft(filtered_signal_buffer)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag

        hr_mask = (freqs >= self.low_hz) & (freqs <= self.high_hz) # The mask is used to filter the frequency components within the heart rate range
        if not np.any(hr_mask):
            print("[rPPG] Warning: No frequency components found in heart rate range.")
            return None, None, False

        # Extract peak frequency and compute BPM and SNR
        band_freqs = freqs[hr_mask]
        band_power = power[hr_mask]
        peak_idx = int(np.argmax(band_power))
        peak_freq = band_freqs[peak_idx]
        bpm = float(peak_freq * 60.0)

        if bpm < 40 or bpm > 180:  # human reasonable range
            return bpm, None, False
        
        noise_power = np.median(band_power)
        snr = band_power[peak_idx] / (noise_power + 1e-12)  # avoid division by zero
        snr_db = float(10.0 * np.log10(snr))
        
        # Determine liveness based on SNR threshold
        is_live = snr_db >= self.snr_thresh_db
        return bpm, snr_db, is_live
