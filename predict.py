#!/usr/bin/env python3
"""
predict.py - Real photo vs Photo-of-a-screen (recapture) detector.

Usage:
    python predict.py some_image.jpg
    -> prints a single float in [0, 1]: 0 = real photo, 1 = photo of a screen.

Approach: classic computer-vision feature engineering (no deep net) fed into
a small RandomForest classifier trained on ~170 self-collected photos. See
features.py for the full feature list and NOTE.md for accuracy / discussion.
"""

import sys
import joblib
import numpy as np

from features import extract_features

MODEL_PATH = "model.joblib"


def predict(image_path):
    bundle = joblib.load(MODEL_PATH)
    scaler, clf = bundle["scaler"], bundle["clf"]

    feats = extract_features(image_path).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    score = clf.predict_proba(feats_scaled)[0, 1]  # P(screen)
    return float(score)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    score = predict(sys.argv[1])
    print(f"{score:.4f}")
