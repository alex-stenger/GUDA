#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: sample several segmentation masks for each image of a split.
"""
import os
import sys
import argparse
import logging

sys.path.append(os.path.dirname(sys.path[0]))
from model.networks.unet import UNetAttention
from utils.initializer import device_initializer, sample_initializer, generate_initializer, seed_initializer
from utils.utils import save_images_with_name, get_dataset
from utils.checkpoint import load_ckpt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def generate(args):
    """
    Generating
    :param args: Input parameters
    :return: None
    """
    logger.info(msg=f"Input params: {args}")
    device = device_initializer(device_id=args.use_gpu)
    seed_initializer(seed_id=args.seed)
    # Sample type, image size, activation function and network configuration are read from the checkpoint
    sample, image_size, act, network = generate_initializer(ckpt_path=args.weight_path, args=args, device=device)
    args.image_size = image_size
    os.makedirs(args.result_path, exist_ok=True)
    diffusion = sample_initializer(sample=sample, image_size=image_size, device=device,
                                   sample_step=args.sample_step_ddim)
    model = UNetAttention(device=device, image_size=image_size, act=act, **network).to(device)
    load_ckpt(ckpt_path=args.weight_path, model=model, device=device)

    dataloader = get_dataset(args=args, dataset_path=args.dataset_path, split=args.split, with_labels=False)
    for images, _, sam, names in dataloader:
        # The images are all resized in dataloader
        images = images.to(device)
        sam = sam.to(device)
        for j in range(args.num_sample):
            sampled_images = diffusion.sample_seg(model=model, img=images, sam=sam)
            save_images_with_name(sampled_images, names, args.result_path, sample_num=j)
    logger.info(msg=f"Masks saved in '{args.result_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample segmentation masks with a trained GUDA model.")
    # Trained checkpoint, e.g. workdir/<run_name>/last.pt
    parser.add_argument("--weight_path", type=str, required=True)
    # Domain root folder
    parser.add_argument("--dataset_path", type=str, required=True)
    # Split to segment
    parser.add_argument("--split", type=str, default="test")
    # Saving path of the masks '<name>_<k>.png'
    parser.add_argument("--result_path", type=str, required=True)
    # Number of masks sampled per image (N), they are averaged by tools/evaluate.py
    parser.add_argument("--num_sample", type=int, default=10)
    # Number of DDIM sampling steps
    parser.add_argument("--sample_step_ddim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_gpu", type=int, default=0)
    # Used only if missing from the checkpoint
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--sample", type=str, default="ddim")
    parser.add_argument("--act", type=str, default="gelu")

    generate(parser.parse_args())
