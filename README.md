# Robust FPV-Based Hand Gesture Recognition

**Real-time hand gesture recognition from a first-person-view (wearable) camera. A deterministic classical computer-vision pipeline (camera calibration, HSV segmentation, morphology, Canny edges, contours, Hu moments, HOG) is combined with lightweight statistical classifiers.**

---

## The Egocentric Problem

| Static camera (3rd person) | Egocentric camera (1st person) |
|---|---|
| Origin is fixed and the background is static | Origin is **chaotic** and the background is **dynamic** |
| Gesture motion is absolute | Hand kinematics must be **decoupled from ego-motion** |

**FPV constraints:** unconstrained illumination · dynamic background clutter · self-occlusion of fingers · high-frequency head motion (blur and scale changes)

**Approach:** instead of a heavy end-to-end network, raw pixels are distilled into a compact, illumination-invariant geometric signature that a fast classifier can label within a single frame's time budget.

## Pipeline

```mermaid
flowchart LR
    A[Camera calibration<br/>K·R·t via DLT + SVD] --> B[Preprocessing<br/>RGB → HSV]
    B --> C[Skin segmentation<br/>H/S thresholds]
    C --> D[Morphology<br/>opening + closing]
    D --> E[Canny edges<br/>+ contours]
    E --> F[Feature vector<br/>geometry · Hu · HOG]
    F --> G[Classifier<br/>KNN · SVM · RF · MLP]
    G --> H[Gesture label<br/>Peace · Stop · Fist · ...]
```

### Stages

1. **Pinhole camera model and calibration:** `s·[u v 1]ᵀ = K [R | t] [X Y Z 1]ᵀ`. A checkerboard target is used, the Direct Linear Transformation linearises the projection, and the 12 unknowns come from the SVD null space.
2. **HSV decoupling:** separates chromaticity (Hue) from luminance (Value = max(R, G, B)), so FPV shadows don't destroy the skin profile.
3. **Deterministic skin mask:** `Mask(x, y) = 1 if H_min ≤ H ≤ H_max and S_min ≤ S ≤ S_max`.
4. **Morphological noise reduction:** erosion `A ⊖ B` and dilation `A ⊕ B`, combined as opening and closing, give a smooth, solid binary mask.
5. **Edge detection:** Sobel gradients (G = √(Gx² + Gy²), θ = atan(Gy/Gx)), then Canny (Gaussian smoothing, gradients, non-maximum suppression, hysteresis).
6. **Topological contours:** closed polygons, convex hull and defects (finger count), centroid, and area by the Shoelace formula `A = ½ |Σ(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)|`.
7. **HOG:** gradient orientation histograms with block normalisation, for contrast invariance.
8. **Final feature vector:** `F = [area, perimeter, solidity, finger_count, Hu₁ … Hu₇, HOG₁ … HOGₙ]`.

## Results

80/20 train/test split. The test set is isolated **before** feature normalisation to prevent data leakage.

| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| K-Nearest Neighbours (k = 5) | 84.5% | 82.1% | 86.0% | 84.0% |
| **Support Vector Machine (RBF)** | **91.2%** | **93.5%** | 90.1% | 91.8% |
| Random Forest | 89.6% | 90.0% | 89.2% | 89.6% |
| Multi-Layer Perceptron | 93.1% | 92.0% | 94.3% | 93.1% |

**Key insight:** the MLP reaches the highest accuracy, but the **SVM** gives the best balance of F1-score and real-time compute cost, so it was chosen for low-latency FPV deployment. The live dashboard runs at **60 FPS**.

## Project Structure

```
FPV-Hand-Gesture-Recognition/
├── src/
│   ├── calibrate.py         # checkerboard camera calibration
│   ├── segmentation.py      # HSV skin mask + morphology
│   ├── features.py          # contours, convexity, Hu moments, HOG
│   ├── collect_data.py      # capture labelled FPV samples from webcam
│   ├── train.py             # KNN / SVM / RF / MLP comparison
│   └── realtime.py          # live FPV gesture recognition
├── data/                    # data/<gesture_name>/*.jpg
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt
python src/collect_data.py --gesture open_palm --count 300     # repeat per gesture
python src/train.py --data data
python src/realtime.py --model models/svm.joblib
```

Optional: `python src/calibrate.py --images calib/*.jpg` writes `camera.npz`, which `realtime.py` uses to undistort frames.

## Future Work

- **Dynamic gestures:** move from static shape classification to temporal trajectories (swiping, waving)
- **3D pose estimation:** depth mapping or stereo disparity to recover full skeletal joints
- **Temporal tracking:** LSTMs or optical flow for robustness to occlusion and motion blur

## Tech Stack

Python · OpenCV · scikit-image · scikit-learn · NumPy

## Team

Team 12 · Amrita Vishwa Vidyapeetham, Coimbatore

- Dodla Nithya Sai
- M. Vishnu Vardhan
- P. Sreekar Reddy
- S. Beena Reddy
