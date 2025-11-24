import cv2
import numpy as np


def get_embedding_from_aligned_face(aligned_face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Simple, robust 'embedding' without deep learning.

    Steps:
      1. Convert to grayscale
      2. Resize to 64x64
      3. Flatten to 1D vector
      4. L2-normalize

    Returns:
        1D float32 numpy array of length 4096, or None if image is invalid.
    """

    if aligned_face_bgr is None:
        return None

    try:
        # 1. BGR -> Gray
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Resize to fixed size
        resized = cv2.resize(gray, (64, 64))  # shape (64, 64)

        # 3. Flatten
        vec = resized.astype("float32").reshape(-1)  # shape (4096,)

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
    threshold: float = 0.2,
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

    # Cosine similarity
    dot = float(np.dot(e1, e2))
    n1 = float(np.linalg.norm(e1)) + 1e-8
    n2 = float(np.linalg.norm(e2)) + 1e-8
    cos_sim = dot / (n1 * n2)

    distance = 1.0 - cos_sim
    is_match = distance < threshold

    return distance, is_match
