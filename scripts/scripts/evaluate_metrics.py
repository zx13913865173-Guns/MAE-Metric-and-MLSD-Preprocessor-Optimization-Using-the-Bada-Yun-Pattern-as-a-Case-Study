#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute SSIM, PSNR, LPIPS, FID between generated and reference images.

Usage:
    python evaluate_metrics.py --gen_dir generated/E --ref_dir dataset/test
"""

import argparse
import os
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths


def list_images(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')))


def compute_ssim_psnr(gen_dir, ref_dir):
    gen_files, ref_files = list_images(gen_dir), list_images(ref_dir)
    n = min(len(gen_files), len(ref_files))
    ssims, psnrs = [], []
    for i in range(n):
        g = cv2.imread(os.path.join(gen_dir, gen_files[i]), 0)
        r = cv2.imread(os.path.join(ref_dir, ref_files[i]), 0)
        if g is None or r is None:
            continue
        if g.shape != r.shape:
            g = cv2.resize(g, (r.shape[1], r.shape[0]))
        ssims.append(ssim(r, g, data_range=255))
        psnrs.append(psnr(r, g, data_range=255))
    return float(np.mean(ssims)), float(np.mean(psnrs))


def compute_lpips(gen_dir, ref_dir, device='cuda'):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    gen_files, ref_files = list_images(gen_dir), list_images(ref_dir)
    n = min(len(gen_files), len(ref_files))
    scores = []
    for i in range(n):
        g = lpips.im2tensor(lpips.load_image(
            os.path.join(gen_dir, gen_files[i]))).to(device)
        r = lpips.im2tensor(lpips.load_image(
            os.path.join(ref_dir, ref_files[i]))).to(device)
        if g.shape != r.shape:
            r = torch.nn.functional.interpolate(
                r, size=g.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            scores.append(loss_fn(g, r).item())
    return float(np.mean(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    args = parser.parse_args()

    s, p = compute_ssim_psnr(args.gen_dir, args.ref_dir)
    print(f"SSIM  = {s:.3f}")
    print(f"PSNR  = {p:.2f} dB")

    if not args.skip_lpips:
        print(f"LPIPS = {compute_lpips(args.gen_dir, args.ref_dir):.3f}")

    if not args.skip_fid:
        fid = calculate_fid_given_paths(
            [args.gen_dir, args.ref_dir], batch_size=50,
            device='cuda', dims=2048)
        print(f"FID   = {fid:.1f}")
