"""Geometric + Hu-moment + HOG feature vector for a segmented hand."""
import cv2
import numpy as np
from skimage.feature import hog

from segmentation import hand_roi


def shoelace_area(contour: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def finger_count(contour: np.ndarray) -> int:
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    count = 0
    for s, e, f, d in defects[:, 0]:
        a, b, c = (np.linalg.norm(contour[e][0] - contour[s][0]),
                   np.linalg.norm(contour[f][0] - contour[s][0]),
                   np.linalg.norm(contour[e][0] - contour[f][0]))
        angle = np.arccos(np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-8), -1, 1))
        if angle < np.pi / 2 and d > 10000:      # deep, narrow valley between two fingers
            count += 1
    return min(count + 1, 5) if count else 0


def geometric_features(contour: np.ndarray) -> np.ndarray:
    area = shoelace_area(contour)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour)) + 1e-8
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([
        np.log1p(area), np.log1p(perimeter), area / hull_area,          # solidity
        w / (h + 1e-8), area / (w * h + 1e-8),                           # aspect, extent
        4 * np.pi * area / (perimeter ** 2 + 1e-8),                      # circularity
        finger_count(contour),
    ])


def hu_features(mask: np.ndarray) -> np.ndarray:
    hu = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)                 # log-scale for stability


def hog_features(mask: np.ndarray) -> np.ndarray:
    return hog(mask, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True)


def extract(bgr: np.ndarray):
    """Full feature vector F = [geometry, Hu1..7, HOG], or None if no hand is found."""
    roi = hand_roi(bgr)
    if roi is None:
        return None
    mask, contour, _ = roi
    return np.concatenate([geometric_features(contour), hu_features(mask), hog_features(mask)])
