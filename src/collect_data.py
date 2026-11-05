"""Capture labelled FPV gesture images from a webcam.  SPACE = save, A = toggle auto, Q = quit."""
import argparse
import os
import time

import cv2

from segmentation import hand_roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gesture", required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--out", default="data")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    folder = os.path.join(args.out, args.gesture)
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    saved, auto, last = len(os.listdir(folder)), False, 0.0

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        roi = hand_roi(frame)
        if roi:
            x, y, w, h = roi[2]
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(view, f"{args.gesture}: {saved}/{args.count} {'[AUTO]' if auto else ''}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("collect", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        if roi and (key == ord(" ") or (auto and time.time() - last > 0.5)):
            cv2.imwrite(os.path.join(folder, f"{args.gesture}_{saved:04d}.jpg"), frame)
            saved, last = saved + 1, time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} samples in {folder}")


if __name__ == "__main__":
    main()
