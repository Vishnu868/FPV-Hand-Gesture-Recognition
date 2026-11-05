"""Extract features from data/<gesture>/*.jpg and compare KNN, SVM, Random Forest and MLP."""
import argparse
import glob
import os

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract


def load(data_dir: str):
    X, y = [], []
    for label in sorted(os.listdir(data_dir)):
        for f in glob.glob(os.path.join(data_dir, label, "*.jpg")):
            feat = extract(cv2.imread(f))
            if feat is not None:
                X.append(feat)
                y.append(label)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    X, y = load(args.data)
    print(f"{len(X)} samples, {X.shape[1]} features, {len(set(y))} gestures")
    # split first -> scaler is fit on training data only (no leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True,
                             random_state=42),
    }
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name, clf in models.items():
        pipe = make_pipeline(StandardScaler(), clf).fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        rows.append({"model": name.upper(), "accuracy": accuracy_score(y_te, p),
                     "precision": precision_score(y_te, p, average="macro", zero_division=0),
                     "recall": recall_score(y_te, p, average="macro", zero_division=0),
                     "f1": f1_score(y_te, p, average="macro")})
        joblib.dump(pipe, os.path.join(args.out, f"{name}.joblib"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nSVM report:\n", classification_report(y_te, joblib.load(
        os.path.join(args.out, "svm.joblib")).predict(X_te)))


if __name__ == "__main__":
    main()
