#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute PCA intrinsic dimensionality (90% variance, 100 bootstrap rounds).

Usage:
    python compute_pca.py --image_dir dataset/train
"""

import argparse
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_dimensionality(image_dir, variance=0.90,
                               n_bootstrap=100, size=(128, 128)):
    images = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, size).flatten())

    if not images:
        raise ValueError("No valid images found.")

    X = StandardScaler().fit_transform(np.array(images))

    n_list = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), len(X), replace=True)
        pca = PCA().fit(X[idx])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_list.append(int(np.argmax(cumsum >= variance) + 1))

    return float(np.mean(n_list)), float(np.std(n_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--variance", type=float, default=0.90)
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()

    mean_n, std_n = compute_pca_dimensionality(
        args.image_dir, args.variance, args.bootstrap)
    print(f"Intrinsic dimensionality: {mean_n:.1f} ± {std_n:.1f}")
