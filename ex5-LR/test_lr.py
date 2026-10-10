#!/usr/bin/env python3
"""
Try the trained model on the sample images.

    python3 test_lr.py

For each image it saves a picture in outputs/ with four panels side by side:
the cropped image, then only the White, Yellow and Red pixels the model found.
"""
from pathlib import Path

import cv2
import numpy as np

from ml_filter import ColorClassifier

# ----------------------------------------------------------------------
# KNOBS: change them, run again, and compare the pictures.
# The ROS node has the same three settings as launch parameters.
# ----------------------------------------------------------------------
# Ignore a pixel unless the model is at least this sure (0 to 1), per class.
# 0 keeps every prediction for that class. Higher values remove noise but can
# also erase thin lines. Classes often need different thresholds -- e.g. White
# being much less confident than Yellow on a given camera.
MIN_CONFIDENCE = {
    "White": 0,
    "Yellow": 0,
    "Red": 0.,
}

# Size in pixels of the clean-up filter that removes tiny specks
# (erode/dilate, like Ex5d). 0 turns it off.
MORPH_KERNEL = 0

# Shrink the image to (width, height) before classifying, which is faster.
# Use None to classify at full size.
RESIZE_TO = (160, 120)
# ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "sample_images"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

model = ColorClassifier(HERE / "model_weights.npz")

for path in sorted(IMAGES_DIR.glob("*.png")):
    image = cv2.imread(str(path))

    # Same crop as the ROS crop node: keep the bottom 70%, drop the top 30%.
    image = image[int(image.shape[0] * 0.3):, :]

    masks = model.segment(image, min_confidence=MIN_CONFIDENCE, morph_kernel=MORPH_KERNEL, resize_to=RESIZE_TO)

    # Each mask is 255 where the model found that color and 0 elsewhere.
    white = cv2.bitwise_and(image, image, mask=masks["White"])
    yellow = cv2.bitwise_and(image, image, mask=masks["Yellow"])
    red = cv2.bitwise_and(image, image, mask=masks["Red"])

    result = np.hstack([image, white, yellow, red])
    cv2.imwrite(str(OUT_DIR / f"{path.stem}_result.png"), result)
    print("saved", OUT_DIR / f"{path.stem}_result.png")
