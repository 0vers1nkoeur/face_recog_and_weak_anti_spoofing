from typing import List, Tuple, Optional
import cv2
import numpy as np
import math
import os

# MediaPipe FaceMesh landmark indices for eyes
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]


def mean_point(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Return the average (x, y) of a list of pixel points."""
    arr = np.array(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def compute_eye_centers(coords: List[Tuple[int, int]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute left/right eye centers using two landmarks per eye."""
    left_eye = mean_point([coords[i] for i in LEFT_EYE_IDX])
    right_eye = mean_point([coords[i] for i in RIGHT_EYE_IDX])
    return left_eye, right_eye


def rotate_image_about_point(image, center_xy: Tuple[float, float], angle_deg: float):
    """
    Rotate the whole image around a given point by angle_deg degrees.
    Used to align the face so that eyes are horizontal.
    """
    M = cv2.getRotationMatrix2D(center_xy, angle_deg, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)


def align_and_crop(
    frame_bgr,
    coords: List[Tuple[int, int]],
    crop_size: int = 224,
    save_debug_path: Optional[str] = None,
    bbox_scale: float = 1.10,
    align: bool = True,
):
    if frame_bgr is None or coords is None or len(coords) == 0:
        return None

    rotated = frame_bgr
    coords_used = coords  # by default, use original coords

    if align:
        (lx, ly), (rx, ry) = compute_eye_centers(coords)
        dy, dx = (ry - ly), (rx - lx)
        angle_deg = math.degrees(math.atan2(dy, dx))
        cx, cy = (lx + rx) * 0.5, (ly + ry) * 0.5

        # Rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # Rotate image
        rotated = cv2.warpAffine(
            frame_bgr, M, (frame_bgr.shape[1], frame_bgr.shape[0]),
            flags=cv2.INTER_CUBIC
        )

        # Rotate landmarks with the SAME matrix (important for large head tilt)
        coords_used = []
        for (x, y) in coords:
            xr = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            yr = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            coords_used.append((int(round(xr)), int(round(yr))))

    # bbox on rotated coords
    xs = [x for x, _ in coords_used]
    ys = [y for _, y in coords_used]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    face_size = max(bbox_w, bbox_h)

    # dynamic tight crop
    side = int(face_size * bbox_scale)
    side = max(96, side)
    half = side // 2

    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    x1, y1 = int(round(center_x - half)), int(round(center_y - half))
    x2, y2 = x1 + side, y1 + side

    # pad if out of bounds
    h, w = rotated.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        rotated = cv2.copyMakeBorder(
            rotated, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    cropped = rotated[y1:y2, x1:x2].copy()

    # resize for consistent embedding input
    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        cropped = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

    if save_debug_path:
        os.makedirs(os.path.dirname(save_debug_path), exist_ok=True)
        cv2.imwrite(save_debug_path, cropped)

    return cropped

