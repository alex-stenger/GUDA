#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: two-stage training of the guided diffusion segmentation model.

    Stage 'source': warm-up on the labelled source domain, the denoising network learns to generate the source masks
    conditioned on the source image and its SAM pseudo-label.
    Stage 'adapt': starting from the source model, keep the supervised source loss and add a self-training loss on
    the target domain, where the target SAM pseudo-labels play the role of the masks to denoise (weighted by lambda_st).
    Both stages run for a fixed number of iterations and the final model is used, no target label is needed.
"""
import os
import sys
import argparse
import logging

import torch

from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from model.networks.unet import UNetAttention
from utils.initializer import device_initializer, seed_initializer, optimizer_initializer, sample_initializer, \
    amp_initializer
from utils.utils import get_dataset, setup_logging, save_train_logging, str2bool
from utils.checkpoint import load_ckpt, save_ckpt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def cycle(dataloader):
    """
    Iterate over a dataloader indefinitely
    :param dataloader: Dataloader
    :return: Batches
    """
    while True:
        for batch in dataloader:
            yield batch


def train(args):
    """
    Training
    :param args: Input parameters
    :return: None
    """
    logger.info(msg=f"Input params: {args}")
    seed_initializer(seed_id=args.seed)
    adapt = args.stage == "adapt"
    if adapt and (args.dataset_target_path is None or not args.pretrain_path):
        raise ValueError("Stage 'adapt' requires --dataset_target_path and --pretrain_path (the source model).")
    iterations = args.iterations if args.iterations is not None else (800000 if adapt else 200000)
    image_size = args.image_size
    device = device_initializer(device_id=args.use_gpu, is_train=True)

    # Create data logging path
    _, results_dir, _, results_tb_dir = setup_logging(save_path=args.result_path, run_name=args.run_name)

    # Dataloader, the target labels are never loaded
    batches_src = cycle(get_dataset(args=args, dataset_path=args.dataset_source_path, split="train"))
    batches_trg = cycle(get_dataset(args=args, dataset_path=args.dataset_target_path, split="train",
                                    with_labels=False)) if adapt else None

    # Initialize the denoising UNet
    network = dict(channel=args.channel, attention_resolutions=args.attention_resolutions, num_heads=args.num_heads)
    model = UNetAttention(device=device, image_size=image_size, act=args.act, **network).to(device)
    optimizer = optimizer_initializer(model=model, optim=args.optim, init_lr=args.lr, device=device)

    start_iteration = 0
    if args.resume:
        start_iteration = load_ckpt(ckpt_path=os.path.join(results_dir, "last.pt"), model=model, device=device,
                                    optimizer=optimizer, is_resume=True)
        logger.info(msg=f"[{device}]: Resume training from iteration {start_iteration}.")
    elif args.pretrain_path:
        load_ckpt(ckpt_path=args.pretrain_path, model=model, device=device)
        logger.info(msg=f"[{device}]: Successfully load pretrain model checkpoint.")

    scaler = amp_initializer(amp=args.amp, device=device)
    mse = nn.MSELoss()
    diffusion = sample_initializer(sample=args.sample, image_size=image_size, device=device)
    tb_logger = SummaryWriter(log_dir=results_tb_dir)
    save_train_logging(args, results_dir)
    ckpt_params = dict(results_dir=results_dir, image_size=image_size, sample=args.sample, network=network,
                       act=args.act)

    logger.info(msg=f"[{device}]: Start training for {iterations} iterations.")
    pbar = tqdm(range(start_iteration, iterations), initial=start_iteration, total=iterations)
    for iteration in pbar:
        # The images are all resized in dataloader
        images_src, labels_src, sam_src, _ = next(batches_src)
        images_src, labels_src, sam_src = images_src.to(device), labels_src.to(device), sam_src.to(device)
        n_src = images_src.shape[0]
        if adapt:
            images_trg, _, sam_trg, _ = next(batches_trg)
            images_trg, sam_trg = images_trg.to(device), sam_trg.to(device)
            n_trg = images_trg.shape[0]
        # Randomly sampled time steps, shared by source and target (the last batch of an epoch can be smaller)
        time = diffusion.sample_time_steps(max(n_src, n_trg) if adapt else n_src).to(device)
        # Add noise to the source ground-truth masks
        x_time_source, noise_source = diffusion.noise_images(x=labels_src, time=time[:n_src])
        if adapt:
            # Self-training: the target pseudo-labels are the masks to denoise
            x_time_target, noise_target = diffusion.noise_images(x=sam_trg, time=time[:n_trg])
        with autocast(enabled=args.amp):
            # Input of size (B, 9, H, W): noisy mask + image + pseudo-label
            concat_source = torch.cat([x_time_source, images_src, sam_src], dim=1)
            predicted_noise_source = model(concat_source, time[:n_src])
            loss_source = mse(noise_source, predicted_noise_source)
            loss = loss_source
            if adapt:
                concat_target = torch.cat([x_time_target, images_trg, sam_trg], dim=1)
                predicted_noise_target = model(concat_target, time[:n_trg])
                loss_target = mse(noise_target, predicted_noise_target)
                loss = loss_source + args.lambda_st * loss_target

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if iteration % args.log_interval == 0:
            pbar.set_postfix(MSE=loss.item())
            tb_logger.add_scalar(tag="MSE", scalar_value=loss.item(), global_step=iteration)
            tb_logger.add_scalar(tag="MSE source", scalar_value=loss_source.item(), global_step=iteration)
            if adapt:
                tb_logger.add_scalar(tag="MSE target", scalar_value=loss_target.item(), global_step=iteration)
        if (iteration + 1) % args.save_interval == 0 or iteration + 1 == iterations:
            save_ckpt(iteration=iteration, save_name="last", ckpt_model=model.state_dict(),
                      ckpt_optimizer=optimizer.state_dict(), **ckpt_params)

    logger.info(msg=f"[{device}]: Finish training, final model saved in '{os.path.join(results_dir, 'last.pt')}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GUDA guided diffusion segmentation model.")
    # =================================Base settings=================================
    # 'source': warm-up on the source domain, 'adapt': domain adaptation to the target domain
    parser.add_argument("--stage", type=str, choices=["source", "adapt"], required=True)
    # Source domain root folder, containing the train/ split
    parser.add_argument("--dataset_source_path", type=str, required=True)
    # Target domain root folder (stage 'adapt' only), containing the train/ split (no label needed)
    parser.add_argument("--dataset_target_path", type=str, default=None)
    # Checkpoint to start from, the source model for stage 'adapt'
    parser.add_argument("--pretrain_path", type=str, default="")
    # Weight of the target self-training loss (stage 'adapt' only)
    parser.add_argument("--lambda_st", type=float, default=1e-4)
    # Number of training iterations (default: 200000 for stage 'source', 800000 for stage 'adapt')
    parser.add_argument("--iterations", type=int, default=None)
    # Saving path, the results are written in <result_path>/<run_name>
    parser.add_argument("--result_path", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="guda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    # Number of sub-processes used for data loading
    parser.add_argument("--num_workers", type=int, default=0)
    # Input image size, the images, labels and pseudo-labels are resized to it
    parser.add_argument("--image_size", type=int, default=64)
    # Denoising UNet: channels of each level (the resolution is halved between levels), resolutions at which
    # self-attention is applied and number of attention heads
    parser.add_argument("--channel", type=int, nargs="+", default=[128, 256, 384, 512])
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=[16])
    parser.add_argument("--num_heads", type=int, default=1)
    # Sampler, only DDIM is supported
    parser.add_argument("--sample", type=str, default="ddim")
    # Option: adam/adamw/sgd
    parser.add_argument("--optim", type=str, default="adam")
    # Option: gelu/silu/relu/relu6/lrelu
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=1e-4)
    # Enable automatic mixed precision training
    parser.add_argument("--amp", type=str2bool, default=False)
    # Log the losses to TensorBoard every 'log_interval' iterations
    parser.add_argument("--log_interval", type=int, default=100)
    # Save <result_path>/<run_name>/last.pt every 'save_interval' iterations and at the end of the training
    parser.add_argument("--save_interval", type=int, default=10000)
    # Resume an interrupted training from <result_path>/<run_name>/last.pt
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--use_gpu", type=int, default=0)

    train(parser.parse_args())
