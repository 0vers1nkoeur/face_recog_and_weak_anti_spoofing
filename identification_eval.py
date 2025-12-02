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
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from vision_detection.face_detection import FaceMeshDetector
from vision_detection.face_alignment import align_and_crop
from vision_detection.verification import get_embedding_from_aligned_face


def canonical_user_id(fname: str) -> str:
    """Drop a trailing numeric suffix to group multiple samples of the same user."""
    base = Path(fname).stem
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def reliability_srr(d1: float, d2: float) -> float:
    """
    Relative distance-based reliability score in [0,1].
    Higher when the gap between best and second-best is larger.
    """
    if d1 is None or d2 is None or d2 <= d1:
        return 0.0
    return max(0.0, min(1.0, (d2 - d1) / (d2 + d1 + 1e-8)))


def iter_image_paths(root: Path) -> Sequence[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in sorted(root.glob("**/*")) if p.suffix.lower() in exts and p.is_file()]


def embed_image(img_path: Path, detector: FaceMeshDetector, crop_size: int = 224):
    bgr = cv2.imread(str(img_path))
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
