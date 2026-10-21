#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 19:05
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: kept the attention UNet and the DDIM sampler only.
"""
import random
import numpy as np
import torch
import logging

from torch.cuda.amp import GradScaler

from model.samples.ddim import DDIMDiffusion

logger = logging.getLogger(__name__)


def device_initializer(device_id=0, is_train=False):
    """
    This function initializes the running device information when the program runs for the first time
    :param device_id: Device id
    :param is_train: Whether to train mode
    :return: cpu or cuda
    """
    logger.info(msg="Init program, it is checking the basic device setting.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda", device_id)
        logger.info(msg=f"Using {torch.cuda.get_device_name(device=device)} (cuda:{device_id}).")
        return device
    if is_train:
        raise NotImplementedError("Training requires a CUDA GPU.")
    logger.warning(msg="Warning: The device is using cpu, the device would slow down the model running speed.")
    return torch.device(device="cpu")


def seed_initializer(seed_id=0):
    """
    Initialize the seed
    :param seed_id: The seed id
    :return: None
    """
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(msg=f"The seed is initialized, and the seed ID is {seed_id}.")


def optimizer_initializer(model, optim, init_lr, device):
    """
    Initialize optimizer
    :param model: Model
    :param optim: Optimizer name
    :param init_lr: Initialize learning rate
    :param device: GPU or CPU
    :return: optimizer
    """
    model_param = model.parameters()
    if optim == "adam":
        optimizer = torch.optim.Adam(params=model_param, lr=init_lr)
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(params=model_param, lr=init_lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(params=model_param, lr=init_lr, momentum=0.937)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")
    logger.info(msg=f"[{device}]: This base optimizer is {optim}")
    return optimizer


def sample_initializer(sample, image_size, device, sample_step=100):
    """
    Initialize sample
    :param sample: Sample function
    :param image_size: image size
    :param device: GPU or CPU
    :param sample_step: Number of DDIM sampling steps
    :return: diffusion
    """
    if sample != "ddim":
        raise ValueError(f"Unsupported sampler: {sample}, only 'ddim' is supported.")
    return DDIMDiffusion(sample_steps=sample_step, img_size=image_size, device=device)


def amp_initializer(amp, device):
    """
    Initialize automatic mixed precision
    :param amp: Enable automatic mixed precision
    :param device: GPU or CPU
    :return: scaler
    """
    if amp:
        logger.info(msg=f"[{device}]: Automatic mixed precision training.")
    else:
        logger.info(msg=f"[{device}]: Normal training.")
    # Used to scale gradients to prevent overflow
    return GradScaler(enabled=amp)


def generate_initializer(ckpt_path, args, device):
    """
    Read the training parameters stored in the checkpoint, falling back to the command line arguments
    :param ckpt_path: Checkpoint path
    :param args: Generating model parameters
    :param device: GPU or CPU
    :return: [sample, image_size, act, network configuration]
    """
    ckpt_state = torch.load(f=ckpt_path, map_location=device, weights_only=True)
    sample = ckpt_state.get("sample") or args.sample
    image_size = ckpt_state.get("image_size") or args.image_size
    act = ckpt_state.get("act") or args.act
    network = ckpt_state.get("network") if isinstance(ckpt_state.get("network"), dict) else {}
    logger.info(msg=f"[{device}]: Checkpoint parameters: sample={sample}, image_size={image_size}, act={act}, "
                    f"network={network}.")
    return sample, image_size, act, network
