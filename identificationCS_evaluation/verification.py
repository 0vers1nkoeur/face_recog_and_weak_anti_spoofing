import os
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# 🔴 MAIN SWITCH (THIS IS WHAT YOU CHANGE)
USE_SFACE = True        # True → use SFace (CNN), False → use HOG only

# --- SFace (CNN, primary method) ---
SFACE_MODEL_PATH = os.path.join(
    "models", "face_recognition_sface_2021dec.onnx"
)
SFACE_THRESHOLD = 0.40   # cosine distance (lower = stricter)

# --- HOG (fallback / alternative) ---
HOG_SIZE = (64, 64)
HOG_THRESHOLD = 0.15

# ============================================================
# SFace INITIALIZATION (CONTROLLED BY FLAG)
# ============================================================

SFACE_AVAILABLE = False
sface = None

if USE_SFACE:
    try:
        if os.path.exists(SFACE_MODEL_PATH):
            sface = cv2.FaceRecognizerSF.create(
                SFACE_MODEL_PATH, ""
            )
            SFACE_AVAILABLE = True
            print("[Verification] SFace ENABLED (CNN)")
        else:
            print(
                f"[Verification] SFace model not found → HOG fallback"
            )
    except Exception as e:
        print("[Verification] Failed to init SFace → HOG fallback:", e)
else:
    print("[Verification] SFace DISABLED by configuration → using HOG only")

# ============================================================
# HOG INITIALIZATION (ALWAYS AVAILABLE)
# ============================================================

_HOG = cv2.HOGDescriptor(
    _winSize=HOG_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def _hog_embedding(face_bgr):
    """
    Compute HOG embedding from an aligned face image.
    Used either as fallback or as primary method when SFace is disabled.
    """
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, HOG_SIZE)
        vec = _HOG.compute(gray)
        if vec is None:
            return None
        vec = vec.reshape(-1).astype("float32")
        return vec / (np.linalg.norm(vec) + 1e-8)
    except Exception:
        return None


def get_embedding_from_aligned_face(face_bgr):
    """
    Extracts an embedding from an aligned face image.

    Returns:
        {
            "method": "SFACE" or "HOG",
            "data": 1D numpy array (float32)
        }
    """
    if face_bgr is None:
        return None

    # ---------- SFace (CNN) ----------
    if SFACE_AVAILABLE:
        try:
            emb = sface.feature(face_bgr)
            emb = emb.reshape(-1).astype("float32")
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return {
                "method": "SFACE",
                "data": emb
            }
        except Exception:
            # Silent fallback to HOG
            pass

    # ---------- HOG ----------
    hog = _hog_embedding(face_bgr)
    if hog is not None:
        return {
            "method": "HOG",
            "data": hog
        }

    return None

# ============================================================
# EMBEDDING COMPARISON
# ============================================================

def compare_embeddings(e_live, e_ref, threshold=None):
    """
    Compare two embeddings using cosine distance.

    Returns:
        (distance, is_match)
    """

    if e_live is None or e_ref is None:
        return float("inf"), False

    # Do not compare embeddings extracted with different methods
    if e_live["method"] != e_ref["method"]:
        return float("inf"), False

    v1 = e_live["data"]
    v2 = e_ref["data"]

    # ---------------- SFace ----------------
    if e_live["method"] == "SFACE":
        cos = float(np.dot(v1, v2))
        dist = 1.0 - cos

        th = SFACE_THRESHOLD if threshold is None else threshold
        return dist, dist < th

    # ---------------- HOG ------------------
    if e_live["method"] == "HOG":
        cos = float(
            np.dot(v1, v2) /
            ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8)
        )
        dist = 1.0 - cos

        th = HOG_THRESHOLD if threshold is None else threshold
        return dist, dist < th

    return float("inf"), False
