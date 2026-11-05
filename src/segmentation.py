"""Deterministic skin segmentation in HSV space followed by morphological cleaning."""
import cv2
import numpy as np

# Default skin bounds (OpenCV HSV: H in [0,180], S,V in [0,255]); tune for your lighting.
SKIN_LOWER = np.array([0, 40, 50], dtype=np.uint8)
SKIN_UPPER = np.array([25, 180, 255], dtype=np.uint8)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def skin_mask(bgr: np.ndarray, lower=SKIN_LOWER, upper=SKIN_UPPER) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)                         # logical threshold matrix
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)   # remove speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)  # fill holes
    return mask


def largest_contour(mask: np.ndarray, min_area: int = 2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def hand_roi(bgr: np.ndarray, size: int = 128):
    """Returns (cropped mask, contour, bounding box) of the dominant skin region, or None."""
    mask = skin_mask(bgr)
    c = largest_contour(mask)
    if c is None:
        return None
    x, y, w, h = cv2.boundingRect(c)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)
    crop = cv2.resize(clean[y:y + h, x:x + w], (size, size), interpolation=cv2.INTER_NEAREST)
    return crop, c, (x, y, w, h)
