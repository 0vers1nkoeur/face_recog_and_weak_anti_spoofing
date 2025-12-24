import cv2
import numpy as np
import os

# HOG descriptor configured for square 64x64 inputs (fallback if deepface fails)
_HOG = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)

# Try to use ArcFace ONNX for better discrimination (cross-platform)
# Fallback to HOG if onnxruntime/model is not available
_ARCFACE_MODEL = os.path.join(os.path.dirname(__file__), "models", "arcface_r100.onnx")
_ARCFACE_SIZE = (112, 112)
_arcface_session = None

try:
    import onnxruntime as ort

    if os.path.exists(_ARCFACE_MODEL):
        _arcface_session = ort.InferenceSession(
            _ARCFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        EMBEDDING_MODEL = "ArcFace_ONNX"
        print(f"[Embedding] Using ArcFace ONNX model: {_ARCFACE_MODEL}")
    else:
        EMBEDDING_MODEL = "HOG"
        print(f"[Embedding] ⚠️ ArcFace ONNX model not found at {_ARCFACE_MODEL}. Falling back to HOG.")
except ImportError:
    EMBEDDING_MODEL = "HOG"
    print("[Embedding] ⚠️ onnxruntime not installed, falling back to HOG")


def _preprocess_arcface(img_bgr: np.ndarray) -> np.ndarray:
    """Preprocess BGR face crop into ArcFace ONNX input tensor."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, _ARCFACE_SIZE)
    arr = resized.astype("float32")
    arr = (arr - 127.5) / 128.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, 0)        # add batch
    return arr


def get_embedding_from_aligned_face(aligned_face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Get face embedding using ArcFace ONNX if available, otherwise HOG.
    
    ArcFace embeddings provide strong discrimination for distinguishing identities.

    Returns:
        1D float32 numpy array, or None if image is invalid.
    """

    if aligned_face_bgr is None:
        return None

    try:
        if _arcface_session is not None:
            blob = _preprocess_arcface(aligned_face_bgr)
            output = _arcface_session.run(
                None, {_arcface_session.get_inputs()[0].name: blob}
            )[0]
            vec = output.squeeze().astype("float32")
            norm = np.linalg.norm(vec) + 1e-8
            vec = vec / norm
            return vec
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
