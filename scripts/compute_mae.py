#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Mean Angular Error (MAE) for line-based pattern evaluation.

Usage:
    python compute_mae.py --image path/to/image.png
"""

import argparse
import cv2
import numpy as np


def compute_mae(image: np.ndarray, reference_axes=(0, 45, 90, 135)) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # 1. Otsu binarization
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Zhang-Suen skeletonization
    skeleton = cv2.ximgproc.thinning(binary)

    # 3. HoughLinesP line detection
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180,
                            threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0.0

    # 4. Minimum angular deviation from nearest reference axis
    errors = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        err = min(min(abs(angle - a % 180),
                      180 - abs(angle - a % 180)) for a in reference_axes)
        errors.append(err)

    return float(np.mean(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--axes", type=float, nargs="+",
                        default=[0, 45, 90, 135])
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    print(f"MAE = {compute_mae(img, args.axes):.2f}°")
