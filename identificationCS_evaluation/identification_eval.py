"""
Offline identification evaluation for the face-recognition pipeline.

Features:
- Loads gallery/probe images from directory structure.
- Builds embeddings with the same alignment/detector stack as main.py.
- Supports multiple templates per user; min distance per user.
- Closed-set and open-set evaluation, Top-1 accuracy, simple CMC.
- Reliability score (SRR) from gap between best and second-best distances.
- Euclidean vs Cosine distance comparison.

Expected folder layout (can be changed via CLI args):
  data/eval/gallery/<user_id>/*.jpg
  data/eval/probe_closed/<user_id>/*.jpg
  data/eval/probe_open/<user_id>/*.jpg
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from vision_detection.face_detection import FaceMeshDetector
from vision_detection.face_alignment import align_and_crop
from identificationCS_evaluation.verification import (
    compare_embeddings,
    get_embedding_from_aligned_face,
)


def summarize_enrollment_distances(enrolled_embeddings: Dict[str, List[np.ndarray]], threshold: float):
    """
    Debug helper: show how far same-user and different-user templates are.
    Use this to pick a sensible threshold.
    """
    intra = []
    inter = []
    user_ids = list(enrolled_embeddings.keys())

    # same user pairs
    for uid, embs in enrolled_embeddings.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                intra.append(compare_embeddings(embs[i], embs[j], threshold=threshold)[0])

    # different user pairs
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            for emb_a in enrolled_embeddings[user_ids[i]]:
                for emb_b in enrolled_embeddings[user_ids[j]]:
                    inter.append(compare_embeddings(emb_a, emb_b, threshold=threshold)[0])

    def stats(arr):
        if not arr:
            return "None"
        a = np.array(arr, dtype=np.float32)
        return f"min={a.min():.3f}, median={np.median(a):.3f}, 95p={np.percentile(a,95):.3f}"

    if intra or inter:
        print("[Eval] 📈 Enrollment distance stats (use to tune threshold):")
        print(f"       same user: {stats(intra)}")
        print(f"       diff user: {stats(inter)}")
        if intra and inter:
            # a conservative suggestion: halfway between hard same-user (95p) and easy diff-user (5p)
            suggest = 0.5 * (np.percentile(intra, 95) + np.percentile(inter, 5))
            print(f"       suggested threshold ~ {suggest:.3f}")

    return {"intra": intra, "inter": inter}


def canonical_user_id(fname: str) -> str:
    """Drop a trailing numeric suffix to group multiple samples of the same user."""
    base = Path(fname).stem
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def load_image_bgr(path: str):
    """
    Load image correcting EXIF orientation to avoid upside-down/sideways faces.
    Returns BGR numpy array or None.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.array(im)[:, :, ::-1]  # RGB -> BGR for OpenCV
            return arr
    except Exception as e:
        print(f"[Eval] ⚠️ PIL load failed for {path}: {e}. Falling back to cv2.")
    bgr = cv2.imread(path)
    return bgr


def load_enrolled_gallery(
    ref_dir: Union[str, Path],
    ref_aligned_dir: Union[str, Path],
    align_crop_size: int,
    align_bbox_scale: float,
    align_rotate: bool,
    threshold: float,
):
    """
    Load enrolled face templates into memory and compute embeddings.

    Args:
        ref_dir: Directory containing enrolled face images (one file per template).
        ref_aligned_dir: Directory to save aligned reference crops for inspection.
        align_crop_size: Output crop size for alignment.
        align_bbox_scale: Scale factor for expanding bbox during alignment.
        align_rotate: Whether to rotate during alignment.
        threshold: Distance threshold used when summarizing enrollment distances.

    Returns:
        Dict mapping user_id -> list of embeddings.
    """
    ref_dir = Path(ref_dir)
    ref_aligned_dir = Path(ref_aligned_dir)

    enrolled_embeddings: Dict[str, List[np.ndarray]] = {}
    #enrollment_detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)

    if not ref_dir.exists():
        print(f"[Konst] ❌ Enrolled directory not found: {ref_dir}")
        return enrolled_embeddings

    for img_path in ref_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        raw_id = img_path.stem  # e.g. "aleksa_01"
        user_id = canonical_user_id(img_path.name)   # drop trailing numeric suffix

        # Keep images in BGR here so they match the live `aligned` image
        ref_bgr = load_image_bgr(str(img_path))
        if ref_bgr is None:
            print(f"[Konst] ⚠️ Could not read enrolled image: {img_path}")
            continue
        aligned_ref = ref_bgr

        #coords = enrollment_detector.process(ref_bgr)
        #if coords is None:
            #print(f"[Konst] ⚠️ No face detected in enrolled image: {img_path}")
            #continue

        #aligned_ref = align_and_crop(
            #ref_bgr,
            #coords,
            #crop_size=align_crop_size,
            #bbox_scale=align_bbox_scale,
            #align=align_rotate,
        #)
        #if aligned_ref is None:
            #print(f"[Konst] ⚠️ Could not align enrolled image: {img_path}")
            #continue

        #ref_aligned_dir.mkdir(parents=True, exist_ok=True)
        #aligned_save_path = ref_aligned_dir / f"{raw_id}.jpg"
        #cv2.imwrite(str(aligned_save_path), aligned_ref)

        emb_ref = get_embedding_from_aligned_face(aligned_ref)
        if emb_ref is None:
            print(f"[Konst] ⚠️ Could not compute embedding for enrolled image: {img_path}")
            continue

        enrolled_embeddings.setdefault(user_id, []).append(emb_ref)

    for user_id, embs in enrolled_embeddings.items():
        if len(embs) > 1:
            print(f"[Konst] 📦 Loaded {len(embs)} templates for {user_id}")

    if not enrolled_embeddings:
        print("[Konst] ❌ No valid enrolled faces found in Enrolled/.")
        return enrolled_embeddings

    print(f"[Konst] 📸 Loaded {len(enrolled_embeddings)} enrolled face(s) from {ref_dir}:")
    print("       ", ", ".join(enrolled_embeddings.keys()))
    summarize_enrollment_distances(enrolled_embeddings, threshold)

    return enrolled_embeddings


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def reliability_srr(d1: Optional[float], d2: Optional[float]) -> float:
    """
    Relative gap between best and second-best distances. Higher => more reliable.
    If only one identity exists (no second distance), treat as fully reliable.
    """
    if d2 is None:
        return 1.0
    if d1 is None or d2 <= d1:
        return 0.0
    return max(0.0, min(1.0, (d2 - d1) / (d2 + d1 + 1e-8)))


def iter_image_paths(root: Path) -> Sequence[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in sorted(root.glob("**/*")) if p.suffix.lower() in exts and p.is_file()]


def embed_image(img_path: Path, detector: FaceMeshDetector, crop_size: int = 224):
    bgr = load_image_bgr(str(img_path))
    if bgr is None:
        print(f"[Eval] ⚠️ Could not read image: {img_path}")
        return None
    coords = detector.process(bgr)
    if coords is None:
        print(f"[Eval] ⚠️ No face detected: {img_path}")
        return None
    aligned = align_and_crop(bgr, coords, crop_size=crop_size)
    if aligned is None:
        print(f"[Eval] ⚠️ Could not align face: {img_path}")
        return None
    emb = get_embedding_from_aligned_face(aligned)
    if emb is None:
        print(f"[Eval] ⚠️ Could not compute embedding: {img_path}")
        return None
    return emb


def load_embeddings_by_user(root: Path) -> Dict[str, List[np.ndarray]]:
    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
    gallery: Dict[str, List[np.ndarray]] = {}
    for user_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img_path in iter_image_paths(user_dir):
            emb = embed_image(img_path, detector)
            if emb is None:
                continue
            gallery.setdefault(user_dir.name, []).append(emb)
    return gallery


def load_probe_list(root: Path) -> List[Tuple[str, np.ndarray]]:
    detector = FaceMeshDetector(max_num_faces=1, refine_landmarks=True)
    probes: List[Tuple[str, np.ndarray]] = []
    for user_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img_path in iter_image_paths(user_dir):
            emb = embed_image(img_path, detector)
            if emb is None:
                continue
            probes.append((user_dir.name, emb))
    return probes


def user_distance(probe_emb: np.ndarray, gallery: Dict[str, List[np.ndarray]], metric: str) -> Dict[str, float]:
    dfunc = euclidean_distance if metric == "euclidean" else cosine_distance
    dists = {}
    for user, embs in gallery.items():
        per_template = [dfunc(probe_emb, g) for g in embs]
        dists[user] = min(per_template)
    return dists


def closed_set_eval(
    probes: List[Tuple[str, np.ndarray]],
    gallery: Dict[str, List[np.ndarray]],
    metric: str,
    ranks: Sequence[int] = (1, 5, 10),
):
    max_rank = max(ranks)
    cmc_hits = np.zeros(max_rank, dtype=int)
    top1 = 0
    logs = []

    for true_id, emb in probes:
        dists = user_distance(emb, gallery, metric)
        ordered = sorted(dists.items(), key=lambda x: x[1])
        best_id, best_d = ordered[0]
        second_d = ordered[1][1] if len(ordered) > 1 else None

        rank = next((i for i, (uid, _) in enumerate(ordered) if uid == true_id), None)
        if rank == 0:
            top1 += 1
        if rank is not None:
            for r in ranks:
                if rank < r:
                    cmc_hits[r - 1] += 1

        srr = reliability_srr(best_d, second_d)
        logs.append((true_id, best_id, best_d, second_d, srr))

    n = len(probes) if probes else 1
    cmc = [cmc_hits[r - 1] / n for r in ranks]
    return top1 / n, cmc, logs


def open_set_eval(
    probes: List[Tuple[str, np.ndarray]],
    gallery: Dict[str, List[np.ndarray]],
    metric: str,
    threshold: float,
):
    correct = 0
    logs = []

    for true_id, emb in probes:
        dists = user_distance(emb, gallery, metric)
        ordered = sorted(dists.items(), key=lambda x: x[1])
        best_id, best_d = ordered[0]
        second_d = ordered[1][1] if len(ordered) > 1 else None

        pred = best_id if best_d < threshold else "Unknown"
        if (true_id not in gallery and pred == "Unknown") or (true_id in gallery and pred == true_id):
            correct += 1

        srr = reliability_srr(best_d, second_d)
        logs.append((true_id, pred, best_d, second_d, srr))

    n = len(probes) if probes else 1
    return correct / n, logs

def main():
    parser = argparse.ArgumentParser(description="Closed-set and open-set identification evaluation.")
    parser.add_argument("--gallery", type=Path, required=True, help="Path to gallery root (user folders).")
    parser.add_argument("--probe-closed", type=Path, help="Path to closed-set probes (all users in gallery).")
    parser.add_argument("--probe-open", type=Path, help="Path to open-set probes (may include unknown users).")
    parser.add_argument("--threshold", type=float, default=0.195, help="Distance threshold for open-set (Euclidean).")
    args = parser.parse_args()

    if not args.gallery.exists():
        raise FileNotFoundError(f"Gallery path not found: {args.gallery}")

    gallery = load_embeddings_by_user(args.gallery)
    if not gallery:
        raise RuntimeError("No gallery embeddings loaded.")
    print(f"[Eval] Loaded gallery users: {list(gallery.keys())}")

    if args.probe_closed:
        probes_closed = load_probe_list(args.probe_closed)
        if probes_closed:
            logs: List[Tuple[str, str, float, Optional[float], float]] = []
            for metric in ["euclidean", "cosine"]:
                top1, cmc, logs = closed_set_eval(probes_closed, gallery, metric)
                cmc_str = ", ".join(f"@{r}:{v:.3f}" for r, v in zip((1, 5, 10), cmc))
                print(f"[Closed-set][{metric}] Top-1: {top1:.3f}, CMC {cmc_str}")
            print("[Closed-set] Example logs: true_id, pred_id, d1, d2, SRR")
            for row in logs[:5]:
                print(row)
        else:
            print("[Closed-set] No probe images found.")

    if args.probe_open:
        probes_open = load_probe_list(args.probe_open)
        if probes_open:
            acc, logs = open_set_eval(probes_open, gallery, "euclidean", args.threshold)
            print(f"[Open-set][euclidean] Accuracy (ID or Unknown): {acc:.3f} (threshold={args.threshold})")
            print("[Open-set] Example logs: true_id, pred_id, d1, d2, SRR")
            for row in logs[:5]:
                print(row)
        else:
            print("[Open-set] No probe images found.")


if __name__ == "__main__":
    main()
