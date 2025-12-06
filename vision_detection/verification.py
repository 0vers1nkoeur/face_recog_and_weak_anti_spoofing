import cv2
import numpy as np
import tempfile
import os

# HOG descriptor configured for square 64x64 inputs (fallback if deepface fails)
_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

# Try to use deepface (ArcFace) for better discrimination
# Fallback to HOG if deepface is not available
try:
    from deepface import DeepFace
    EMBEDDING_MODEL = "ArcFace"
    print("[Embedding] Using ArcFace (DeepFace) for embeddings - best discrimination")
except ImportError:
    print("[Embedding] ⚠️ deepface not installed, falling back to HOG")
    EMBEDDING_MODEL = "HOG"


def get_embedding_from_aligned_face(aligned_face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Get face embedding using FaceNet (via deepface) if available, otherwise HOG.
    
    FaceNet provides 128-dimensional embeddings that are highly discriminative
    for distinguishing between different people.

    Returns:
        1D float32 numpy array, or None if image is invalid.
    """

    if aligned_face_bgr is None:
        return None

    try:
        if EMBEDDING_MODEL == "FaceNet":
            # Use deepface with FaceNet backend
            # deepface.represent() requires a file path, so write to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, aligned_face_bgr)
            
            try:
                embedding_obj = DeepFace.represent(
                    img_path=tmp_path,
                    model_name="ArcFace",
                    enforce_detection=False,
                )
                if embedding_obj and len(embedding_obj) > 0:
                    vec = np.array(embedding_obj[0]["embedding"], dtype="float32")
                    # Normalize for consistency
                    norm = np.linalg.norm(vec) + 1e-8
                    vec = vec / norm
                    return vec
                return None
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            # Fallback to HOG
            gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            hog_vec = _HOG.compute(resized)
            if hog_vec is None:
                return None
            vec = hog_vec.reshape(-1).astype("float32")
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
