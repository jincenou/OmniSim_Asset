"""Frozen reference regions and balanced soft-alpha L1 objectives."""
import cv2
import numpy as np
import torch

Y, X = np.mgrid[-3:4, -3:4]
DISK = ((X * X + Y * Y) <= 9).astype(np.uint8)
REGIONS = {0: 'far', 1: 'interior', 2: 'inner', 3: 'outer'}


def regions(soft_alpha):
    mask = (soft_alpha >= .5).astype(np.uint8)
    inner = cv2.erode(mask, DISK, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    dilated = cv2.dilate(mask, DISK, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    output = np.zeros(mask.shape, dtype=np.uint8)
    output[inner] = 1
    output[mask.astype(bool) & ~inner] = 2
    output[dilated & ~mask.astype(bool)] = 3
    return output


def extra_alpha(prediction, target, partition, arm):
    assert arm in ('A', 'U', 'B')
    band_in, band_out = partition == 2, partition == 3
    valid = bool(band_in.any()) and bool(band_out.any())
    if arm == 'A' or not valid:
        return prediction.sum() * 0, valid
    error = (prediction - target).abs()
    positive = target >= .5 if arm == 'U' else band_in
    negative = target < .5 if arm == 'U' else band_out
    return .5 * (error[positive].mean() + error[negative].mean()), valid


def old_alpha(prediction, target):
    foreground = target > .5
    assert bool(foreground.any()) and bool((~foreground).any()), 'Old L0 undefined for empty class'
    error = (prediction - target).abs()
    return .5 * (error[foreground].mean() + error[~foreground].mean())
