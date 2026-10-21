#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: paired image / label / pseudo-label dataset.
"""
import os
import time
import logging
import argparse

import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def str2bool(value):
    """
    Parse a boolean command line argument ('argparse' type=bool treats any non-empty string as True)
    :param value: Input string
    :return: bool
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def list_images(path):
    """
    Recursively list the images of a folder
    :param path: Folder path
    :return: Dict {file stem: file path}
    """
    files = {}
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(IMG_EXTENSIONS):
                files[os.path.splitext(name)[0]] = os.path.join(root, name)
    return files


def save_images_with_name(images, names, path, sample_num=0):
    """
    Save each image of a batch as '<name>_<sample_num>.png'
    :param images: Image
    :param names: Names of the images
    :param path: Save path
    :param sample_num: Index of the sample, several masks are sampled per image
    :return: None
    """
    # Imported here so that tools/sam_pseudolabels.py does not need matplotlib
    from matplotlib import pyplot as plt
    for i in range(images.shape[0]):
        img = images[i].to("cpu").numpy()
        name = f"{os.path.splitext(names[i])[0]}_{sample_num}.png"
        plt.imsave(os.path.join(path, name), img[0, :, :], cmap="gray")


class SegDataset(Dataset):
    """
    Segmentation dataset of one domain split

    The split folder contains three sub-folders, matched by file name (without extension):
        <split>/img/...        images
        <split>/lbl/...        ground-truth masks (only read if 'with_labels')
        <split>/pred_sam/...   SAM pseudo-labels
    Files can be in nested sub-folders (e.g. 'img/images/case_001.png').
    """

    def __init__(self, split_path, transform, with_labels=True):
        """
        Initialize the dataset
        :param split_path: Path of the split, e.g. 'datasets/domain_a/train'
        :param transform: Transform applied to the image, the label and the pseudo-label
        :param with_labels: Whether to load the ground-truth masks, else an empty tensor is returned instead
        """
        self.transform = transform
        self.with_labels = with_labels
        images = list_images(os.path.join(split_path, "img"))
        labels = list_images(os.path.join(split_path, "lbl")) if with_labels else {}
        pseudo_labels = list_images(os.path.join(split_path, "pred_sam"))
        if len(images) == 0:
            raise FileNotFoundError(f"No image found in '{os.path.join(split_path, 'img')}'.")
        missing = [name for name in images if name not in pseudo_labels or (with_labels and name not in labels)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images of '{split_path}' have no matching "
                                    f"{'label or ' if with_labels else ''}pseudo-label, e.g. '{missing[0]}'.")
        self.samples = [(images[name], labels.get(name), pseudo_labels[name]) for name in sorted(images)]

    def __len__(self):
        return len(self.samples)

    def load(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, index):
        img_path, lbl_path, sam_path = self.samples[index]
        lbl = self.load(lbl_path) if self.with_labels else torch.empty(0)
        return self.load(img_path), lbl, self.load(sam_path), os.path.basename(img_path)


def get_dataset(args, dataset_path, split, with_labels=True):
    """
    Get dataloader of one domain split
    :param args: Parameters
    :param dataset_path: Root folder of the domain
    :param split: 'train' or 'test'
    :param with_labels: Whether to load the ground-truth masks
    :return: dataloader yielding (image, label, pseudo-label, name)
    """
    transforms = torchvision.transforms.Compose([
        # Resize input size
        torchvision.transforms.Resize(size=int(args.image_size)),
        # To Tensor Format
        torchvision.transforms.ToTensor(),
        # For standardization, the mean and standard deviation are both 0.5
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = SegDataset(split_path=os.path.join(dataset_path, split), transform=transforms, with_labels=with_labels)
    return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                      pin_memory=True)


def setup_logging(save_path, run_name):
    """
    Set log saving path
    :param save_path: Saving path
    :param run_name: Saving name
    :return: List of file paths
    """
    results_root_dir = save_path
    results_dir = os.path.join(save_path, run_name)
    results_vis_dir = os.path.join(save_path, run_name, "vis")
    results_tb_dir = os.path.join(save_path, run_name, "tensorboard")
    # Root folder
    os.makedirs(name=results_root_dir, exist_ok=True)
    # Saving folder
    os.makedirs(name=results_dir, exist_ok=True)
    # Visualization folder
    os.makedirs(name=results_vis_dir, exist_ok=True)
    # Visualization folder for Tensorboard
    os.makedirs(name=results_tb_dir, exist_ok=True)
    return [results_root_dir, results_dir, results_vis_dir, results_tb_dir]


def save_train_logging(arg, save_path):
    """
    Save train log
    :param arg: Argparse
    :param save_path: Save path
    :return: None
    """
    with open(file=f"{save_path}/train.log", mode="a") as f:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        f.write(f"{current_time}: {arg}\n")
