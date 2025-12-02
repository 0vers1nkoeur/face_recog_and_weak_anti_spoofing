import cv2
import numpy as np

# HOG descriptor configured for square 64x64 inputs. This is far more robust
# than raw pixels for distinguishing people.
_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def get_embedding_from_aligned_face(aligned_face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Hand-crafted HOG embedding (no deep learning) that is more stable than
    raw pixels and less sensitive to background.

    Steps:
      1. Convert to grayscale
      2. Detect and crop the largest face region (fallback to full image)
      3. Resize to 64x64
      4. Compute HOG descriptor
      5. L2-normalize feature vector

    Returns:
        1D float32 numpy array, or None if image is invalid.
    """

    if aligned_face_bgr is None:
        return None

    try:
        # 1. BGR -> Gray
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Resize to fixed size (use the aligned crop directly to avoid double-cropping jitter)
        resized = cv2.resize(gray, (64, 64))  # shape (64, 64)

        # 3. HOG descriptor -> vector
        hog_vec = _HOG.compute(resized)
        if hog_vec is None:
            return None

        vec = hog_vec.reshape(-1).astype("float32")

        # 4. Normalize (so brightness changes don't explode distances)
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm

        return vec

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
