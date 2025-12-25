import cv2
import numpy as np

# HOG descriptor configured for square 64x64 inputs (fallback if deepface fails)
_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

# We intentionally avoid DeepFace to keep the project cross-platform and lightweight.
# Embeddings are computed using OpenCV HOG features.
EMBEDDING_MODEL = "HOG"
print("[Embedding] Using HOG embeddings (DeepFace disabled)")


def get_embedding_from_aligned_face(aligned_face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Get face embedding using ArcFace (via deepface) if available, otherwise HOG.
    
    ArcFace embeddings provide strong discrimination for distinguishing identities.

    Returns:
        1D float32 numpy array, or None if image is invalid.
    """

    if aligned_face_bgr is None:
        return None

    # HOG-only embedding (DeepFace intentionally removed)
    try:
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        hog_vec = _HOG.compute(resized)
        if hog_vec is None:
            return None
        vec = hog_vec.reshape(-1).astype("float32")
        # L2 normalize so cosine distance is meaningful
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
    except Exception as e:
        print(f"[get_embedding_from_aligned_face] Error: {e}")
        return None


def compare_embeddings(
    embedding_live: np.ndarray,
    embedding_ref: np.ndarray,
    threshold: float = 0.16,
) -> tuple[float, bool]:
    """
    Compare two embeddings using cosine similarity turned into a 'distance'.

    We do:
        distance = 1 - cosine_similarity

    So:
        distance ≈ 0   -> VERY similar
        distance ≈ 2   -> very different

    Returns:
        (distance, is_match)
    """

    # Ensure 1D float vectors
    e1 = embedding_live.astype("float32").ravel()
    e2 = embedding_ref.astype("float32").ravel()

    if e1.shape != e2.shape:
        print(
            f"[compare_embeddings] ⚠️ Embedding shape mismatch: {e1.shape} vs {e2.shape}. "
            "Regenerate stored embeddings with the current pipeline."
        )
        return float("inf"), False

    # Cosine similarity
    dot = float(np.dot(e1, e2))
    n1 = float(np.linalg.norm(e1)) + 1e-8
    n2 = float(np.linalg.norm(e2)) + 1e-8
    cos_sim = dot / (n1 * n2)

    distance = 1.0 - cos_sim
    is_match = distance < threshold

    return distance, is_match
