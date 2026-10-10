#!/usr/bin/env python3
"""
Train a logistic regression that labels every pixel as
background, White, Yellow or Red.

    python3 train_lr.py

Reads the labeled pixels in lab_dataset.npz and writes the learned weights
to model_weights.npz. Then run test_lr.py to see what the model does.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from ml_filter import CLASS_NAMES  # ["background", "White", "Yellow", "Red"]

# ----------------------------------------------------------------------
# KNOB: change it, run again, and compare the results
# ----------------------------------------------------------------------
# C sets how tightly the model fits the training pixels.
#   small C (for example 0.01): smoother, more cautious boundaries (may underfit)
#   large C (for example 100):  fits the training pixels closely (may overfit)
C = 1.0
# ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

# 1. Load the labeled pixels.
#    Each row of X is one pixel: [L, a, b, r_norm, g_norm].
#    y is the class of that pixel: 0 background, 1 White, 2 Yellow, 3 Red.
#    The train pixels teach the model. The val pixels are kept aside to check it.
data = np.load(HERE / "lab_dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
print(f"train pixels: {len(X_train)}, val pixels: {len(X_val)}")

# 2. Put every feature on the same scale (mean 0, spread 1).
#    Without this, a feature with big numbers (L goes up to 255)
#    would dominate one with small numbers (r_norm is between 0 and 1).
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 3. Train. This finds one weight per feature for each class, plus a bias.
#    class_weight="balanced" stops a class with fewer pixels from being ignored.
model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
model.fit(X_train_scaled, y_train)

# 4. Check it on pixels it has not trained on.
y_pred = model.predict(X_val_scaled)
print(f"\nvalidation accuracy: {accuracy_score(y_val, y_pred):.3f}\n")
print(classification_report(y_val, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES))

# 5. Save the learned numbers. ml_filter.py reads this file at test time and on the robot.
np.savez(
    HERE / "model_weights.npz",
    feature_mean=scaler.mean_.astype(np.float32),
    feature_scale=scaler.scale_.astype(np.float32),
    coef=model.coef_.astype(np.float32),
    intercept=model.intercept_.astype(np.float32),
    classes=model.classes_.astype(np.int64),
)
print("saved model_weights.npz")
