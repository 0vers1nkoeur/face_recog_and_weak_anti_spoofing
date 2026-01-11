# identificationCS_evaluation/lfw_eval_csv_clean.py
# ============================================================
# LFW Face Verification Evaluation (CSV protocol)
# Manual distance threshold (same logic as main.py)
# ============================================================

import csv
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ------------------------------------------------------------------
# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# ------------------------------------------------------------------

from vision_detection.face_detection import FaceMeshDetector
from vision_detection.face_alignment import align_and_crop
from identificationCS_evaluation.verification import get_embedding_from_aligned_face

# ============================= CONFIG ==============================
LFW_ROOT = Path("data/lfw/lfw")
PAIRS_CSV = Path("data/lfw/pairs.csv")
CROP_SIZE = 224

# 🔴 MANUAL DISTANCE THRESHOLD
DIST_THRESHOLD = 0.4
MAX_DISTANCE = 2.0
# ==================================================================


# ------------------------ DISTANCE ------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32").ravel()
    b = b.astype("float32").ravel()
    return 1.0 - float(
        np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)
    )


# ------------------------ IO ------------------------
def load_image_bgr(path: Path):
    return cv2.imread(str(path))


def compute_embedding(img_path: Path, detector: FaceMeshDetector):
    img = load_image_bgr(img_path)
    if img is None:
        return None

    coords = detector.process(img)
    if coords is None:
        return None

    aligned = align_and_crop(img, coords, crop_size=CROP_SIZE)
    if aligned is None:
        return None

    return get_embedding_from_aligned_face(aligned)


# ------------------------ CSV LOADER ------------------------
def load_lfw_pairs(csv_path: Path):
    pairs = []
    labels = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header if exists

        for row in reader:
            row = [r.strip() for r in row if r.strip()]

            # Genuine
            if len(row) == 3:
                name, i1, i2 = row
                try:
                    i1, i2 = int(i1), int(i2)
                except ValueError:
                    continue

                p1 = LFW_ROOT / name / f"{name}_{i1:04d}.jpg"
                p2 = LFW_ROOT / name / f"{name}_{i2:04d}.jpg"
                label = 1

            # Impostor
            elif len(row) == 4:
                name1, i1, name2, i2 = row
                try:
                    i1, i2 = int(i1), int(i2)
                except ValueError:
                    continue

                p1 = LFW_ROOT / name1 / f"{name1}_{i1:04d}.jpg"
                p2 = LFW_ROOT / name2 / f"{name2}_{i2:04d}.jpg"
                label = 0

            else:
                continue

            if p1.exists() and p2.exists():
                pairs.append((p1, p2))
                labels.append(label)

    return pairs, np.array(labels, dtype=np.int32)


# ============================= MAIN ==============================
def main():
    print("[LFW] Loading pairs from CSV...")
    pairs, labels = load_lfw_pairs(PAIRS_CSV)

    if not pairs:
        print("[LFW] ❌ No valid pairs loaded.")
        return

    print(f"[LFW] Total pairs     : {len(pairs)}")
    print(f"[LFW] Genuine pairs  : {labels.sum()}")
    print(f"[LFW] Impostor pairs : {len(labels) - labels.sum()}")

    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)

    distances = []
    valid_labels = []

    print("[LFW] Computing embeddings and distances...")
    for (img1, img2), label in zip(pairs, labels):
        e1 = compute_embedding(img1, detector)
        e2 = compute_embedding(img2, detector)

        if e1 is None or e2 is None:
            dist = MAX_DISTANCE
        else:
            v1 = e1["data"] if isinstance(e1, dict) else e1
            v2 = e2["data"] if isinstance(e2, dict) else e2
            dist = cosine_distance(v1, v2)

        distances.append(dist)
        valid_labels.append(label)

    distances = np.array(distances, dtype=np.float32)
    valid_labels = np.array(valid_labels, dtype=np.int32)

    print(f"[LFW] Successfully evaluated pairs: {len(distances)}")

    # ================= MANUAL THRESHOLD METRICS =================
    TP = np.sum((distances < DIST_THRESHOLD) & (valid_labels == 1))
    FN = np.sum((distances >= DIST_THRESHOLD) & (valid_labels == 1))
    FP = np.sum((distances < DIST_THRESHOLD) & (valid_labels == 0))
    TN = np.sum((distances >= DIST_THRESHOLD) & (valid_labels == 0))

    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (FN + TP + 1e-8)
    HTER = (FAR + FRR) / 2

    print("\n[LFW] IDENTIFICATION RESULTS")
    print("------------------------------------------------")
    print(f"Distance threshold : {DIST_THRESHOLD:.4f}")
    print(f"FAR                : {FAR:.4f}")
    print(f"FRR                : {FRR:.4f}")
    print("------------------------------------------------")

    # ================= ROC (ILLUSTRATIVE ONLY) =================
    # ROC needs higher = more genuine → invert distance
    scores = -distances
    fpr, tpr, _ = roc_curve(valid_labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("LFW Identification ROC (Cosine Distance)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    out_path = "lfw_roc.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[LFW] ✅ ROC saved to: {out_path}")


# ================================================================
if __name__ == "__main__":
    main()
