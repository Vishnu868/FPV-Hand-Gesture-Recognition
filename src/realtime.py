"""Live FPV gesture recognition dashboard.  Q = quit."""
import argparse
import os
import time

import cv2
import joblib
import numpy as np

from features import extract
from segmentation import hand_roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/svm.joblib")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", default="camera.npz")
    ap.add_argument("--min-conf", type=float, default=0.6)
    args = ap.parse_args()

    model = joblib.load(args.model)
    calib = np.load(args.calib) if os.path.exists(args.calib) else None
    cap = cv2.VideoCapture(args.camera)
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if calib is not None:
            frame = cv2.undistort(frame, calib["K"], calib["dist"])
        label = "NO HAND"
        roi = hand_roi(frame)
        if roi:
            x, y, w, h = roi[2]
            feat = extract(frame)
            if feat is not None:
                probs = model.predict_proba([feat])[0]
                i = int(np.argmax(probs))
                label = (f"{model.classes_[i].upper()} {probs[i]:.0%}"
                         if probs[i] >= args.min_conf else "UNCERTAIN")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        now = time.time()
        fps, prev = 1 / max(now - prev, 1e-6), now
        cv2.putText(frame, f"[ GESTURE: {label} ]", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("FPV Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
