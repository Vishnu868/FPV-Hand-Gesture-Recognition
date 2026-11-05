"""Checkerboard camera calibration -> intrinsic matrix K and distortion coefficients."""
import argparse
import glob

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="calib/*.jpg")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--square", type=float, default=25.0, help="square size in mm")
    ap.add_argument("--out", default="camera.npz")
    args = ap.parse_args()

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
    obj_pts, img_pts, size = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for f in glob.glob(args.images):
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))
        if ok:
            obj_pts.append(objp)
            img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

    if not obj_pts:
        raise SystemExit("No checkerboard detected in the given images.")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    np.savez(args.out, K=K, dist=dist)
    print(f"Calibrated from {len(obj_pts)} images | RMS reprojection error {rms:.3f}px\nK =\n{K}")


if __name__ == "__main__":
    main()
