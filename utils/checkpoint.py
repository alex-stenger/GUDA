#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/2 22:43
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: removed EMA and class-conditional handling.
"""
import os
import logging
import torch

from collections import OrderedDict

logger = logging.getLogger(__name__)


def load_ckpt(ckpt_path, model, device, optimizer=None, is_resume=False):
    """
    Load checkpoint weight files
    :param ckpt_path: Checkpoint path
    :param model: Network
    :param device: GPU or CPU
    :param optimizer: Optimizer, restored when resuming
    :param is_resume: Whether to resume an interrupted training (restore the optimizer and the iteration)
    :return: Next iteration if resuming, else None
    """
    ckpt_state = torch.load(f=ckpt_path, map_location=device, weights_only=True)
    logger.info(msg=f"[{device}]: Successfully load checkpoint, path is '{ckpt_path}'.")
    load_model_ckpt(model=model, model_ckpt=ckpt_state["model"])
    if is_resume:
        optimizer.load_state_dict(state_dict=ckpt_state["optimizer"])
        logger.info(msg=f"[{device}]: Successfully load optimizer checkpoint.")
        return ckpt_state["iteration"] + 1


def load_model_ckpt(model, model_ckpt):
    """
    Load the weights into the model, handling the 'module.' prefix of distributed training
    :param model: Model
    :param model_ckpt: Model checkpoint
    :return: None
    """
    is_distributed_model = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    model_weights_dict = {}
    for key, value in model_ckpt.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if is_distributed_model:
            key = "module." + key
        model_weights_dict[key] = value
    model.load_state_dict(state_dict=OrderedDict(model_weights_dict))


def save_ckpt(iteration, save_name, ckpt_model, ckpt_optimizer, results_dir, image_size=None, sample=None,
              network=None, act=None):
    """
    Save the model checkpoint weight files
    :param iteration: Current iteration
    :param save_name: Save the model's name
    :param ckpt_model: Model
    :param ckpt_optimizer: Optimizer
    :param results_dir: Results dir
    :param image_size: Default image size
    :param sample: Sample type
    :param network: Network configuration, keyword arguments of the network
    :param act: Activation function name
    :return: None
    """
    ckpt_state = {
        "iteration": iteration, "model": ckpt_model, "optimizer": ckpt_optimizer,
        "image_size": image_size, "sample": sample, "network": network, "act": act,
    }
    filename = os.path.join(results_dir, f"{save_name}.pt")
    torch.save(obj=ckpt_state, f=filename)
    logger.info(msg=f"Save the {save_name}.pt")
