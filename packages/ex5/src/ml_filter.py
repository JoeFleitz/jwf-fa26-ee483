#!/usr/bin/env python3
"""
Per-pixel color classifier for White/Yellow (and Red) lane-marking
segmentation. Trained offline with scikit-learn (train_color_classifier.py),
but this module only needs numpy + cv2 so it can run inside the ROS
container without a scikit-learn/joblib dependency.

Usage mirrors white_filter.py's cv2.inRange call, but instead of a fixed
HSV box, each pixel is classified from [L, a, b, r_norm, g_norm] features
(CIELAB decouples chroma from brightness, so this holds up better than
HSV thresholds when lighting changes).
"""
import cv2
import numpy as np

CLASS_IDS = {"background": 0, "Red": 3, "White": 1, "Yellow": 2}
CLASS_NAMES = sorted(CLASS_IDS, key=CLASS_IDS.get)  # index == label id: background, White, Yellow, Red


def extract_features(bgr_img):
    """bgr_img: HxWx3 uint8 -> HxWx5 float32 feature map [L,a,b,r_norm,g_norm]."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bgr = bgr_img.astype(np.float32)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    s = r + g + b + 1e-6
    r_norm = r / s
    g_norm = g / s
    return np.stack([lab[..., 0], lab[..., 1], lab[..., 2], r_norm, g_norm], axis=-1)


class ColorClassifier:
    def __init__(self, weights_path):
        w = np.load(weights_path, allow_pickle=True)
        self.feature_mean = w["feature_mean"]
        self.feature_scale = w["feature_scale"]
        self.coef = w["coef"]          # (n_classes, n_features)
        self.intercept = w["intercept"]  # (n_classes,)
        self.classes = w["classes"]     # (n_classes,) label ids matching CLASS_IDS values

    def predict_label_map(self, bgr_img, min_confidence=0.0, resize_to=None):
        """Returns an HxW int array of predicted class labels (0=background,1=White,2=Yellow,3=Red),
        H,W matching bgr_img's original size regardless of resize_to.

        min_confidence: if the winning class's softmax probability is below this
        (0-1), the pixel falls back to background instead. Raise it to trade
        recall for precision (fewer, more confident detections). Either a single
        number applied to White/Yellow/Red alike, or a dict like
        {"White": 0.5, "Yellow": 0.85, "Red": 0.85} to tune each class separately
        -- useful since classes can need very different thresholds (e.g. White
        being much less confident than Yellow on a given camera). A class left
        out of the dict is not thresholded (equivalent to 0.0 for it).

        resize_to: optional (width, height) to downscale bgr_img to before running
        extract_features/inference -- classification cost scales with pixel count,
        so this trades detection of very thin/small regions for speed. The
        resulting label map is scaled back up (nearest-neighbor, since labels
        are categorical) to bgr_img's original size, so callers don't need to
        know this happened."""
        orig_h, orig_w = bgr_img.shape[:2]
        small_img = cv2.resize(bgr_img, resize_to, interpolation=cv2.INTER_NEAREST) if resize_to else bgr_img
        h, w = small_img.shape[:2]
        feats = extract_features(small_img).reshape(-1, 5)
        feats_s = (feats - self.feature_mean) / self.feature_scale
        logits = feats_s @ self.coef.T + self.intercept  # (N, n_classes)
        pred_idx = np.argmax(logits, axis=1)
        pred_label = self.classes[pred_idx]

        if (isinstance(min_confidence, dict) and any(v > 0.0 for v in min_confidence.values())) or \
           (not isinstance(min_confidence, dict) and min_confidence > 0.0):
            # thresholds[class_id] = required confidence for that class; background is never thresholded
            thresholds = np.zeros(len(CLASS_NAMES))
            if isinstance(min_confidence, dict):
                for name, value in min_confidence.items():
                    thresholds[CLASS_IDS[name]] = value
            else:
                thresholds[:] = min_confidence
                thresholds[CLASS_IDS["background"]] = 0.0

            probs = softmax(logits)
            confidence = probs[np.arange(len(probs)), pred_idx]
            required = thresholds[pred_label]
            pred_label = np.where(confidence >= required, pred_label, CLASS_IDS["background"])

        label_map = pred_label.reshape(h, w)
        if resize_to:
            label_map = cv2.resize(label_map.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return label_map

    def segment(self, bgr_img, min_confidence=0.0, morph_kernel=0, resize_to=None):
        """Returns dict {"White": mask, "Yellow": mask, "Red": mask}, each HxW uint8 0/255,
        matching bgr_img's original size (see predict_label_map's resize_to).
        min_confidence: see predict_label_map -- a single number or a per-class dict.

        morph_kernel: if > 0, apply an opening (erode+dilate) then closing
        (dilate+erode) with a square kernel of this size to each mask, to
        clean up isolated speckle noise without needing a confidence change."""
        label_map = self.predict_label_map(bgr_img, min_confidence=min_confidence, resize_to=resize_to)
        masks = {}
        for name in ("White", "Yellow", "Red"):
            mask = np.where(label_map == CLASS_IDS[name], 255, 0).astype(np.uint8)
            if morph_kernel > 0:
                kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masks[name] = mask
        return masks


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
