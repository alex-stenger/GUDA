#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    SAM pseudo-labels.

    Segment Anything (ViT-H, 'sam_vit_h_4b8939.pth') is run in automatic "segment everything" mode, prompted by a
    regular 32 x 32 point grid. The masks of every shape found in the image are merged into one binary pseudo-label,
    except the masks that are too large: SAM often returns masks covering (almost) the whole image (the background).
    The pseudo-labels are computed offline, for every split of a domain, and written to <split>/pred_sam/.

    Requires: pip install git+https://github.com/facebookresearch/segment-anything.git
    Weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch

from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sys.path.append(os.path.dirname(sys.path[0]))
from utils.utils import list_images


def pseudo_label(mask_generator, image, max_area_ratio):
    """
    Compute the SAM pseudo-label of an image
    :param mask_generator: SAM automatic mask generator
    :param image: RGB image, uint8 array of shape (H, W, 3)
    :param max_area_ratio: Masks covering more than this fraction of the image are discarded
    :return: Binary pseudo-label, uint8 array of shape (H, W) with values 0 and 255
    """
    label = np.zeros(image.shape[:2], dtype=bool)
    max_area = max_area_ratio * label.size
    for mask in mask_generator.generate(image):
        if mask["area"] <= max_area:
            label |= mask["segmentation"]
    return label.astype(np.uint8) * 255


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=args.points_per_side)
    for split in args.splits:
        img_dir = os.path.join(args.dataset_path, split, "img")
        if not os.path.isdir(img_dir):
            print(f"Skip '{img_dir}', not found.")
            continue
        for name, path in tqdm(sorted(list_images(img_dir).items()), desc=split):
            # Keep the sub-folder structure of img/ in pred_sam/
            sub_folder = os.path.relpath(os.path.dirname(path), img_dir)
            out_path = os.path.join(args.dataset_path, split, "pred_sam", sub_folder, f"{name}.png")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, pseudo_label(mask_generator, image, args.max_area_ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the SAM pseudo-labels of a domain.")
    # Domain root folder, the images are read from <split>/img/ and the pseudo-labels written to <split>/pred_sam/
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="vit_h")
    # Size of the regular point grid used to prompt SAM
    parser.add_argument("--points_per_side", type=int, default=32)
    # Masks covering more than this fraction of the image (typically the background) are discarded
    parser.add_argument("--max_area_ratio", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
