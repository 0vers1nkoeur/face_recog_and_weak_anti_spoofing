import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rppg.rppg_class import RPPG
from rppg.roisets_class import ROISet
from vision_detection.face_detection import FaceMeshDetector


def _load_gt_bpm(gt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times_s: List[float] = []
    bpm_values: List[float] = []
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                times_s.append(float(parts[0]) / 1000.0)
                bpm_values.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(times_s, dtype=np.float32), np.asarray(bpm_values, dtype=np.float32)


def _interpolate_bpm(times_s: np.ndarray, bpm_values: np.ndarray, t_s: float) -> Optional[float]:
    if times_s.size == 0 or bpm_values.size == 0:
        return None
    if t_s < float(times_s[0]) or t_s > float(times_s[-1]):
        return None
    return float(np.interp(t_s, times_s, bpm_values))


def _find_video_path(subject_dir: Path) -> Optional[Path]:
    for ext in (".avi", ".mp4", ".mov"):
        matches = sorted(subject_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def evaluate_ubfc_simple(
    dataset_dir: Path,
    roi_landmark_sets: Optional[ROISet] = None,
    window_seconds: int = 10,
    low_hz: float = 0.70,
    high_hz: float = 3.0,
    snr_thresh_db: float = 3.0,
    step_seconds: float = 1.0,
    max_subjects: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate rPPG on UBFC Simple dataset folders.

    Returns a dict with per-subject metrics and a 'global' aggregate.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    results: Dict[str, Dict[str, float]] = {}
    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)

    for subject_dir in subject_dirs:
        video_path = _find_video_path(subject_dir)
        gt_path = subject_dir / "gtdump.xmp"
        if video_path is None or not gt_path.exists():
            if debug:
                print(f"[rPPG] Skipping {subject_dir.name}: missing video or gtdump.xmp")
            continue

        gt_times_s, gt_bpm = _load_gt_bpm(gt_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        rppg = RPPG(
            fps=fps,
            window_seconds=window_seconds,
            low_hz=low_hz,
            high_hz=high_hz,
            snr_thresh_db=snr_thresh_db,
            roi_landmark_sets=roi_landmark_sets or RPPG.NEW_CHEEKS + RPPG.FOREHEAD,
            debug=debug,
        )

        step_frames = max(1, int(round(fps * step_seconds)))
        frame_idx = 0
        total_windows = 0
        valid_windows = 0
        errors: List[float] = []

        start_t = time.perf_counter()
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            coords = detector.process(frame_bgr)
            if coords is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rppg.update_buffer(frame_rgb, coords)

            if len(rppg.signal_buffer) == rppg.buffer_size and (frame_idx % step_frames == 0):
                bpm, _, _ = rppg.compute_liveness()
                total_windows += 1
                if bpm is not None:
                    valid_windows += 1
                    t_s = frame_idx / fps
                    gt_bpm_at_t = _interpolate_bpm(gt_times_s, gt_bpm, t_s)
                    if gt_bpm_at_t is not None:
                        errors.append(abs(bpm - gt_bpm_at_t))

            frame_idx += 1

        elapsed = max(1e-9, time.perf_counter() - start_t)
        cap.release()

        processed_fps = frame_idx / elapsed
        real_time_factor = processed_fps / fps if fps > 0 else 0.0
        coverage = valid_windows / total_windows if total_windows else 0.0
        mae = float(np.mean(errors)) if errors else float("nan")
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")

        results[subject_dir.name] = {
            "mae": mae,
            "rmse": rmse,
            "coverage": coverage,
            "real_time_factor": real_time_factor,
            "processed_fps": processed_fps,
            "windows": float(total_windows),
            "valid_windows": float(valid_windows),
        }

        if debug:
            print(
                f"[rPPG] {subject_dir.name} | MAE={mae:.2f} | RMSE={rmse:.2f} | "
                f"coverage={coverage:.2f} | xRT={real_time_factor:.2f}"
            )

    if results:
        maes = [v["mae"] for v in results.values() if not np.isnan(v["mae"])]
        rmses = [v["rmse"] for v in results.values() if not np.isnan(v["rmse"])]
        coverages = [v["coverage"] for v in results.values()]
        rtf = [v["real_time_factor"] for v in results.values()]

        results["global"] = {
            "mae": float(np.mean(maes)) if maes else float("nan"),
            "rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "coverage": float(np.mean(coverages)) if coverages else float("nan"),
            "real_time_factor": float(np.mean(rtf)) if rtf else float("nan"),
        }

    return results


if __name__ == "__main__":
    base_dir = Path("evaluation_rppg") / "datasets" / "ubfc_simple"
    metrics = evaluate_ubfc_simple(base_dir, debug=True)
    if "global" in metrics:
        print("[rPPG] Global:", metrics["global"])
