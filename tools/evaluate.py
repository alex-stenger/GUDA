#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Evaluation of the sampled masks.

    For each ground-truth mask '<name>.png', the sampled masks '<name>_<k>.png' are averaged, the mean is thresholded
    and compared to the ground truth (foreground IoU / Dice, and their mean over foreground and background).
"""
import os
import re
import sys
import argparse
import logging

import cv2
import numpy as np

sys.path.append(os.path.dirname(sys.path[0]))
from utils.utils import list_images

logger = logging.getLogger(__name__)


def calculate_scores(pred_mask, gt_mask):
    """
    Compute the IoU and Dice of binary masks
    :param pred_mask: Binary prediction
    :param gt_mask: Binary ground truth
    :return: foreground iou, foreground dice, mean iou and mean dice over foreground and background
    """
    scores = []
    for pred, gt in [(pred_mask, gt_mask), (~pred_mask, ~gt_mask)]:
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        scores.append((iou, dice))
    (iou_fg, dice_fg), (iou_bg, dice_bg) = scores
    return iou_fg, dice_fg, (iou_fg + iou_bg) / 2, (dice_fg + dice_bg) / 2


def evaluate(pred_path, gt_path, threshold_factor=0.5, n_mask=10, verbose=False):
    """
    Evaluate the sampled masks
    :param pred_path: Folder of the sampled masks '<name>_<k>.png', the mean masks are also written there
    :param gt_path: Folder of the ground-truth masks (searched recursively)
    :param threshold_factor: Threshold applied to the mean of the sampled masks, between 0 and 1
    :param n_mask: Number of sampled masks averaged per image
    :param verbose: Print the scores of each image
    :return: Dict of the scores averaged over the images
    """
    sample_pattern = re.compile(r"^(.*)_(\d+)\.png$")
    predictions = {}
    for name in os.listdir(pred_path):
        match = sample_pattern.match(name)
        if match:
            predictions.setdefault(match.group(1), []).append((int(match.group(2)), name))

    totals = np.zeros(4)
    total_samples = 0
    missing = 0
    for gt_name, gt_file in sorted(list_images(gt_path).items()):
        if gt_name not in predictions:
            missing += 1
            continue
        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_masks = []
        for _, pred_file in sorted(predictions[gt_name])[:n_mask]:
            pred_img = cv2.imread(os.path.join(pred_path, pred_file), cv2.IMREAD_GRAYSCALE)
            # Resize the prediction to match the size of the ground truth
            pred_masks.append(cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0])))

        # Mean of the sampled masks, then threshold
        pred_mean = np.mean(pred_masks, axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(pred_path, f"{gt_name}_mean.png"), pred_mean)
        _, pred_thresholded = cv2.threshold(pred_mean, int(threshold_factor * 255), 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(pred_path, f"{gt_name}_mean_thresholded.png"), pred_thresholded)

        scores = calculate_scores(pred_thresholded > 0, gt_img > 0)
        if verbose:
            print(f"GT: {gt_name}, iou: {scores[0]:.4f}, dice: {scores[1]:.4f}")
        totals += scores
        total_samples += 1

    if missing:
        logger.warning(msg=f"{missing} ground-truth masks have no prediction in '{pred_path}'.")
    if total_samples == 0:
        raise FileNotFoundError(f"No prediction of '{pred_path}' matches a ground-truth mask of '{gt_path}'.")
    iou, dice, mean_iou, mean_dice = (float(x) for x in totals / total_samples)
    print(f"Global IoU: {iou}, Global mean IoU: {mean_iou}, Global Dice: {dice}, Global mean Dice: {mean_dice}")
    return {"iou": iou, "dice": dice, "mean_iou": mean_iou, "mean_dice": mean_dice}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU and Dice scores for predicted masks compared to "
                                                 "ground truth masks.")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to the directory containing predicted masks.")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to the directory containing ground truth masks.")
    parser.add_argument("--threshold_factor", type=float, default=0.5, help="Threshold factor between 0 and 1.")
    parser.add_argument("--n_mask", type=int, default=10,
                        help="Number of generated masks to compute the mean prediction.")
    args = parser.parse_args()

    evaluate(args.pred_path, args.gt_path, args.threshold_factor, args.n_mask, verbose=True)
