# ============================================================
# LiveNES (rPPG-based) Liveness Evaluation
#
# This script performs OFFLINE liveness evaluation using
# rPPG (remote photoplethysmography) on pre-recorded sequences.
#
# The evaluation logic is intentionally aligned with the LIVE
# verification pipeline used in main.py:
#  - Multiple rPPG decisions per sequence
#  - Majority voting for final liveness decision
#  - Median SNR used as the confidence score
#
# Metrics reported:
#  - Accuracy
#  - FAR (False Acceptance Rate)
#  - FRR (False Rejection Rate)
#  - HTER
#  - ROC curve (illustrative, not threshold-driven)
# ============================================================

import os
import re
import cv2
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -------------------------------------------------
# Ensure project root is available in PYTHONPATH
# so that relative imports behave exactly as in main.py
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------------------------------

from vision_detection.face_detection import FaceMeshDetector
from rppg.rppg_class import RPPG

# ===================== CONFIG =====================
# Root directory of the liveness dataset
# Expected structure:
#   data/liveness_frames/
#       ├── live/
#       └── spoof/
DATA_ROOT = "data/liveness_frames"

# Frame rate used during rPPG processing
FPS = 30

# Temporal window length (seconds) for rPPG analysis
WINDOW_SECONDS = 15

# SNR threshold used inside rPPG.compute_liveness()
# This value is IDENTICAL to the live verification pipeline
SNR_THRESH_DB = 16

# Minimum number of frames required for a valid sequence
MIN_FRAMES = 30

# Maximum number of buffer re-iterations per sequence
# (used to allow looping over short sequences)
MAX_LOOP_FACTOR = 2
# =================================================

# Expected filename format: s<subject>v<version>f<frame>.png
FNAME_PATTERN = re.compile(
    r"^s(\d+)v(\d+)f(\d+)\.png$", re.IGNORECASE
)

# -------------------------------------------------
def group_sequences(folder):
    """
    Groups individual frame files into valid temporal sequences.

    Only files matching the expected naming convention are used.
    Sequences shorter than MIN_FRAMES are discarded.
    """
    sequences = defaultdict(list)

    for fname in os.listdir(folder):
        m = FNAME_PATTERN.match(fname)
        if not m:
            continue

        subject, version, frame_idx = m.groups()
        key = f"s{subject}v{version}"
        sequences[key].append((int(frame_idx), fname))

    cleaned = {}
    for key, frames in sequences.items():
        if len(frames) < MIN_FRAMES:
            continue

        # Sort frames by frame index
        cleaned[key] = [f for _, f in sorted(frames)]

    return cleaned

# -------------------------------------------------
def evaluate_sequence(seq_files, folder, gt_live):
    """
    Evaluates a single liveness sequence using rPPG.

    Steps:
      1. Initialize FaceMesh detector and rPPG processor
      2. Iterate through frames (with looping if necessary)
      3. Collect multiple rPPG decisions
      4. Apply majority voting (same as live system)
      5. Return median SNR as confidence score

    Returns:
        gt_live   : ground-truth label
        pred_live : predicted liveness (True / False)
        median_snr: median SNR value for the sequence
    """
    detector = FaceMeshDetector(max_num_faces=1)

    rppg = RPPG(
        fps=FPS,
        window_seconds=WINDOW_SECONDS,
        snr_thresh_db=SNR_THRESH_DB,
        roi_landmark_sets=RPPG.NEW_CHEEKS + RPPG.FOREHEAD
    )

    idx = 0
    max_iters = rppg.buffer_size * MAX_LOOP_FACTOR

    liveness_votes = []
    snr_values = []

    for _ in range(max_iters):
        fname = seq_files[idx]
        idx = (idx + 1) % len(seq_files)

        frame = cv2.imread(os.path.join(folder, fname))
        if frame is None:
            continue

        coords = detector.process(frame)
        if coords is None:
            continue

        rppg.update_buffer(frame, coords)

        # Once rPPG buffer is full, compute liveness
        if len(rppg.signal_buffer) == rppg.buffer_size:
            _, snr_db, is_live = rppg.compute_liveness()
            liveness_votes.append(is_live)
            snr_values.append(snr_db)

            # Same stopping rule as live system
            if len(liveness_votes) >= 5:
                break

    if not liveness_votes:
        return gt_live, False, 0.0

    # MAJORITY VOTING (live-consistent decision rule)
    pred_live = liveness_votes.count(True) > len(liveness_votes) / 2

    # Median SNR used as sequence-level confidence score
    median_snr = float(np.median(snr_values))

    return gt_live, pred_live, median_snr


# -------------------------------------------------
def main():
    """
    Main evaluation routine.
    Iterates over live and spoof sequences, computes predictions,
    reports performance metrics, and plots ROC curve.
    """
    results = []

    for label, gt in [("live", True), ("spoof", False)]:
        folder = os.path.join(DATA_ROOT, label)
        if not os.path.isdir(folder):
            continue

        sequences = group_sequences(folder)

        for _, frames in sequences.items():
            gt_label, pred, snr = evaluate_sequence(frames, folder, gt)
            results.append((gt_label, pred, snr))

    if not results:
        print("❌ No valid sequences found.")
        return

    # ----------------- METRICS -----------------
    TP = sum(1 for gt, p, _ in results if gt and p)
    TN = sum(1 for gt, p, _ in results if not gt and not p)
    FP = sum(1 for gt, p, _ in results if not gt and p)
    FN = sum(1 for gt, p, _ in results if gt and not p)

    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (FN + TP + 1e-8)
    HTER = (FAR + FRR) / 2
    ACC = (TP + TN) / len(results)

    print("\n[rPPG LIVENESS] VERIFICATION RESULTS")
    print("-------------------------------------------")
    print(f"FAR             : {FAR:.3f}")
    print(f"FRR             : {FRR:.3f}")
    print(f"HTER            : {HTER:.3f}")
    print(f"SNR Threshold   : {SNR_THRESH_DB:.2f} dB")
    print("-------------------------------------------")

    # ----------------- ROC -----------------
    # ROC is computed using median SNR as confidence score.
    # This curve is used only for visualization and analysis,
    # not for operational threshold selection.
    y_true = np.array([1 if gt else 0 for gt, _, _ in results])
    scores = np.array([snr for _, _, snr in results])

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"LiveNES ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("LiveNES (rPPG) Liveness ROC")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    out_path = "liveness_roc.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[LiveNES] ✅ ROC saved to: {out_path}")

# -------------------------------------------------
if __name__ == "__main__":
    main()
