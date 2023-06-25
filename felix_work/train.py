"""
A Single script with a uniform data pipeline that can handle training of different types of models, such as Unet, ResNet3D, and attention pooling

"""

# {{{ Module Imports

# Generic Imports
import pickle
import warnings
import sys
import pandas as pd
import os
import shutil
import gc
import sys
import math
import time
import random
import datetime
import importlib
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
from functools import partial
from typing import List
import copy

# ML modules
import cv2
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, fbeta_score
import cupy as xp
from einops import rearrange, reduce, repeat

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from warmup_scheduler import GradualWarmupScheduler

# TIMM
import timm
from timm.models.resnet import resnet34d

# Plotting
import matplotlib.pyplot as plt

# Monitoring
from tqdm.auto import tqdm
import wandb

# Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

# Config Manager
from dataclasses import dataclass, field
import pyrallis

# Logging
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

# Model Imports
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from resnet3d import generate_model 

# For importing pre-trained model. See https://github.com/pytorch/pytorch/issues/33288
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.extend([".",".."])
from paths import project_path, models_path, dataset_path

# }}}

# {{{ Config


@dataclass
class Config:

    # === Core Paths === 
    comp_name: str = 'vesuvius-challenge-ink-detection'
    exp_name: str = 'exp_name'

    root_path: str = project_path
    dataset_path: str = dataset_path
    models_path: str = models_path

    target_size: int = 1 # target classes

    # === Model ===
    model_name: str = 'Unet'
    #backbone: str = 'efficientnet-b0'
    #backbone: str = 'se_resnext50_32x4d'
    #backbone: str = 'se_resnext101_32x4d'
    #backbone: str = 'resnet34d'
    #backbone: str = 'mit_b3'
    backbone: str = 'resnet3d'
    model_depth: int = 18 # For 3DResnet 

    in_chans: int = 30
    z_dims: int = 20

    start_slice: int = 15
    end_slice: int = 45

    input_size: int = 224
    tile_size: int = 224

    train_stride: int = 224 // 2
    test_stride: int = (224 - 32*2) // 2

    train_batch_size: int = 16
    valid_batch_size: int = 16
    loss_fn: str = 'hybrid' # bce, dice, hybrid
    use_amp: bool = True

    # === Optimizer ===
    scheduler: str = 'GradualWarmupSchedulerV2'
    # scheduler: str = 'CosineAnnealingLR'
    # scheduler: str = 'OneCycleLR'
    epochs: int = 30
    warmup_factor: int = 10
    initial_lr: float = 1e-4 / warmup_factor
    lr: float = 1e-4

    # === OneCycleLR Config ===
    max_lr: float = 3e-3 

    # === CV Strategy ===
    nfolds: int = 4
    cv_4folds: List[int] = field(default_factory=lambda: [1,3,4,5]) # Split Fragment 2 into three smaller fragments 4,5,6 for 5fold CV
    cv_3folds: List[int] = field(default_factory=lambda: [1,2,3]) # Split Fragment 2 into three smaller fragments 4,5,6 for 5fold CV
    validation_fold: int = 1 
    metric_direction: str = 'maximize'  # maximize, 'minimize'
    select_best_model: bool = False

    # === Hyperparams ===
    pretrained: bool = True
    inf_weight: str = 'best' 

    min_lr: float = 1e-6
    weight_decay: float = 1e-6
    max_grad_norm: int = 1000

    print_freq: int = 50
    num_workers: int = 4

    seed: int = 38

    # === Attention Pooling ===
    use_attn_pooling: bool = False

    # === Channel Branch === 
    use_channel_branch: bool = False
    channel_stride: int = 2

    # Post Processing:
    use_tta: bool = True
    use_channel_tta: bool = False
    use_denoising: bool = False
    add_label_noise: bool = False
    use_inference_mask: bool = False
    edge_size: int = 32

    # === wandb ===
    use_wandb: bool = False

# }}}

# {{{ Helper Functions

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_logger(log_file):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 38

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def make_dirs(cfg):

    # Dirs
    outputs_path = f'{dataset_path}outputs/{cfg.exp_name}/'
    submission_dir = outputs_path + 'submissions/'
    model_dir = outputs_path + f'{cfg.comp_name}-models/fold{cfg.validation_fold}/'
    figures_dir = outputs_path + 'figures/'
    log_dir = outputs_path + 'logs/'
    final_models_dir = cfg.models_path + f'/{cfg.exp_name}/'

    # Files
    submission_path = submission_dir + f'submission_{cfg.exp_name}.csv'
    log_path = log_dir + f'{cfg.exp_name}.txt'

    dirs = {
        'outputs_path': outputs_path,
        'submission_dir': submission_dir,
        'submission_path': submission_path,
        'model_dir': model_dir,
        'figures_dir': figures_dir,
        'log_dir': log_dir,
        'log_path': log_path,
        'final_models_dir': final_models_dir,
    }

    for dir_name, dir_path in dirs.items():
        setattr(cfg, dir_name, dir_path)
        if dir_path[-1] == '/':
            os.makedirs(dir_path, exist_ok=True)

def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)

    if mode == 'train':
        make_dirs(cfg)
# }}}

# {{{ Dataset
def read_image_and_masks(fragment_id):

    images = []

    start = cfg.start_slice
    end = cfg.end_slice
    idxs = range(start, end)

    for i in tqdm(idxs):

        # Read the tiff from the fragment (in grayscale mode)
        image = cv2.imread(cfg.dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        # Pad the image with zeros to avoid null prediction caused by inference mask
        #if cfg.use_inference_mask:
        #    image = np.pad(image, [(cfg.edge_size, cfg.edge_size), (cfg.edge_size, cfg.edge_size)], constant_values=0)

        # Pad the image so that it is divisible by the tile size (for sliding window training/inference)
        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

    images = np.stack(images, axis=2)

    ink_mask = cv2.imread(cfg.dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    binary_mask = cv2.imread(cfg.dataset_path + f"train/{fragment_id}/mask.png", 0)

    # Pad the masks with zeros to avoid null prediction caused by inference mask
    #if cfg.use_inference_mask:
    #    ink_mask = np.pad(ink_mask, [(cfg.edge_size, cfg.edge_size), (cfg.edge_size, cfg.edge_size)], constant_values=0)
    #    binary_mask = np.pad(binary_mask, [(cfg.edge_size, cfg.edge_size), (cfg.edge_size, cfg.edge_size)], constant_values=0)

    # Pad the mask as well so that it has the same dimension as the image
    ink_mask = np.pad(ink_mask, [(0, pad0), (0, pad1)], constant_values=0)

    ink_mask = ink_mask.astype('float32')
    ink_mask /= 255.0

    binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)
    binary_mask = (binary_mask / 255).astype(int)
    
    return images, ink_mask, binary_mask

def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    if cfg.nfolds == 3:
        cv_folds = cfg.cv_3folds
    elif cfg.nfolds == 4:
        cv_folds = cfg.cv_4folds

    for fragment_id in cv_folds:

        image, ink_mask, binary_mask = read_image_and_masks(fragment_id)

        if fragment_id == cfg.validation_fold:

            # "Marker" pixel for each patch
            x1_list = list(range(0, image.shape[1]-cfg.tile_size+1, cfg.test_stride))
            y1_list = list(range(0, image.shape[0]-cfg.tile_size+1, cfg.test_stride))

            for y1 in y1_list: 
                for x1 in x1_list:

                    if binary_mask[y1:y1+cfg.tile_size, x1:x1+cfg.tile_size].max() > 0:

                        y2 = y1 + cfg.tile_size
                        x2 = x1 + cfg.tile_size

                        if np.all(image[y1:y2, x1:x2]==0):
                            continue

                        # Image Patches for Validation
                        valid_images.append(image[y1:y2, x1:x2])
                        valid_masks.append(ink_mask[y1:y2, x1:x2, None])

                        # Save pixels for validation purpose (see valid_fn)
                        valid_xyxys.append([x1, y1, x2, y2])
            
        else:

            # "Marker" pixel for each patch
            x1_list = list(range(0, image.shape[1]-cfg.tile_size+1, cfg.train_stride))
            y1_list = list(range(0, image.shape[0]-cfg.tile_size+1, cfg.train_stride))

            for y1 in y1_list: 
                for x1 in x1_list:
                    y2 = y1 + cfg.tile_size
                    x2 = x1 + cfg.tile_size

                    # Image Patches for Training
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(ink_mask[y1:y2, x1:x2, None])


    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

# Data Augmentation
def get_transforms(data, cfg):

    if data == 'train':

        aug = A.Compose([

            #A.Resize(cfg.input_size, cfg.input_size),
            A.Resize(cfg.input_size, cfg.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.75),
            A.ShiftScaleRotate(p=0.75),
            A.OneOf([
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    ], p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=int(cfg.input_size * 0.3), max_height=int(cfg.input_size * 0.3), 
                            mask_fill_value=0, p=0.5),
            # A.Cutout(max_h_size=int(size * 0.6),
            #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
            A.Normalize(
                mean= [0] * cfg.in_chans,
                std= [1] * cfg.in_chans
            ),
            ToTensorV2(transpose_mask=True),

        ])

    elif data == 'valid':

        aug = A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.Normalize(
                mean= [0] * cfg.in_chans,
                std= [1] * cfg.in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ])

    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None, mode='train'):
        self.images = images # List of 224 x 224 x 6 image patches
        self.cfg = cfg
        self.labels = labels # List of 224 x 224 x 1 mask label
        self.transform = transform 
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Add random label noise: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/407972
        if self.mode == 'train' and cfg.add_label_noise:
            if np.random.rand() < 0.3:
                k = np.random.choice(4)
                rot_label = np.rot90(label, k, axes=(0,1))
                label = rot_label.copy()

        if self.transform:
            data = self.transform(image=image, mask=label) # input is 224 x 224 x 6
            image = data['image']
            label = data['mask']

        #if self.mode == 'train':
        #    k = np.random.randint(4)
        #    image = torch.rot90(image, k=k, dims=(1,2))
        #    label = torch.rot90(label, k=k, dims=(1,2))

        # Start from a random channel for channel TTA
        if cfg.use_channel_tta:
            if self.mode == 'train':
                start_idx = np.random.randint(cfg.in_chans - cfg.z_dims)
                image = image[start_idx:start_idx + cfg.z_dims,:,:]

        return image, label
# }}}

# {{{ L1/Hessian denoising
# Reference https://www.kaggle.com/code/brettolsen/improving-performance-with-l1-hessian-denoising

import cupy as cp
xp = cp

delta_lookup = {
    "xx": xp.array([[1, -2, 1]], dtype=float),
    "yy": xp.array([[1], [-2], [1]], dtype=float),
    "xy": xp.array([[1, -1], [-1, 1]], dtype=float),
}

def operate_derivative(img_shape, pair):
    assert len(img_shape) == 2
    delta = delta_lookup[pair]
    fft = xp.fft.fftn(delta, img_shape)
    return fft * xp.conj(fft)

def soft_threshold(vector, threshold):
    return xp.sign(vector) * xp.maximum(xp.abs(vector) - threshold, 0)

def back_diff(input_image, dim):
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r+1, n+1), dtype=float)
    temp2 = xp.zeros((r+1, n+1), dtype=float)
    
    temp1[position[0]:size[0], position[1]:size[1]] = input_image
    temp2[position[0]:size[0], position[1]:size[1]] = input_image
    
    size[dim] += 1
    position[dim] += 1
    temp2[position[0]:size[0], position[1]:size[1]] = input_image
    temp1 -= temp2
    size[dim] -= 1
    return temp1[0:size[0], 0:size[1]]

def forward_diff(input_image, dim):
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r+1, n+1), dtype=float)
    temp2 = xp.zeros((r+1, n+1), dtype=float)
        
    size[dim] += 1
    position[dim] += 1

    temp1[position[0]:size[0], position[1]:size[1]] = input_image
    temp2[position[0]:size[0], position[1]:size[1]] = input_image
    
    size[dim] -= 1
    temp2[0:size[0], 0:size[1]] = input_image
    temp1 -= temp2
    size[dim] += 1
    return -temp1[position[0]:size[0], position[1]:size[1]]

def iter_deriv(input_image, b, scale, mu, dim1, dim2):
    g = back_diff(forward_diff(input_image, dim1), dim2)
    d = soft_threshold(g + b, 1 / mu)
    b = b + (g - d)
    L = scale * back_diff(forward_diff(d - b, dim2), dim1)
    return L, b

def iter_xx(*args):
    return iter_deriv(*args, dim1=1, dim2=1)

def iter_yy(*args):
    return iter_deriv(*args, dim1=0, dim2=0)

def iter_xy(*args):
    return iter_deriv(*args, dim1=0, dim2=1)

def iter_sparse(input_image, bsparse, scale, mu):
    d = soft_threshold(input_image + bsparse, 1 / mu)
    bsparse = bsparse + (input_image - d)
    Lsparse = scale * (d - bsparse)
    return Lsparse, bsparse

def denoise_image(input_image, iter_num=100, fidelity=150, sparsity_scale=10, continuity_scale=0.5, mu=1):
    image_size = xp.shape(input_image)
    #print("Initialize denoising")
    norm_array = (
        operate_derivative(image_size, "xx") + 
        operate_derivative(image_size, "yy") + 
        2 * operate_derivative(image_size, "xy")
    )
    norm_array += (fidelity / mu) + sparsity_scale ** 2
    b_arrays = {
        "xx": xp.zeros(image_size, dtype=float),
        "yy": xp.zeros(image_size, dtype=float),
        "xy": xp.zeros(image_size, dtype=float),
        "L1": xp.zeros(image_size, dtype=float),
    }
    g_update = xp.multiply(fidelity / mu, input_image)
    for i in tqdm(range(iter_num), total=iter_num):
        #print(f"Starting iteration {i+1}")
        g_update = xp.fft.fftn(g_update)
        if i == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
        g_update = xp.multiply((fidelity / mu), input_image)
        
        #print("XX update")
        L, b_arrays["xx"] = iter_xx(g, b_arrays["xx"], continuity_scale, mu)
        g_update += L
        
        #print("YY update")
        L, b_arrays["yy"] = iter_yy(g, b_arrays["yy"], continuity_scale, mu)
        g_update += L
        
        #print("XY update")
        L, b_arrays["xy"] = iter_xy(g, b_arrays["xy"], 2 * continuity_scale, mu)
        g_update += L
        
        #print("L1 update")
        L, b_arrays["L1"] = iter_sparse(g, b_arrays["L1"], sparsity_scale, mu)
        g_update += L
        
    g_update = xp.fft.fftn(g_update)
    g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
    
    g[g < 0] = 0
    g -= g.min()
    g /= g.max()
    return g
# }}}

# {{{ Inference Mask
def create_inference_mask():
    """
    It is used to mask out the edge pixels for model prediction

    If is of dimension input_size x input_size with zeros on the border with edge_size
    """
    mask = torch.zeros((cfg.input_size, cfg.input_size))
    effective_pred_size = cfg.input_size - cfg.edge_size*2 
    start = cfg.edge_size
    end = start + cfg.input_size - cfg.edge_size*2 
    mask[start:end, start:end] = 1

    return mask
# }}}

# {{{ Model Definition
"""
ResNet3D Model
- Encoder is a 3D ResNet model. The architecture has been modified to remove temporal downsampling between blocks.
- A 2D decoder is used for predicting the segmentation map.
- The encoder feature maps are average pooled over the Z dimension before passing it to the decoder -> i.e. transform from 3D to 2D
"""

class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()

        # List of Conv blocks - each block contains a 2d Conv, BN, and ReLU
        # There are 4 blocks in total
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1): # In reverse order!

            # Upsample feature_map by a factor of 2 using bilinear interpolation
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")

            # Concatenate wth feature maps of previous layers
            f = torch.cat([feature_maps[i-1], f_up], dim=1)

            # Forward pass through convolution layers 
            f_down = self.convs[i-1](f)

            # Save feature maps of previous layer
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self, model_depth=34):
        super().__init__()
        self.encoder = generate_model(model_depth=model_depth, n_input_channels=1)
        self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)
        
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None] # Add an extra dimension

        feat_maps = self.encoder(x)

        # Transform 3D to 2D by avg pooling over depth dimension
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]

        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel (since input is grayscale)
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        print(self.encoder.load_state_dict(state_dict, strict=False))

"""
Unet Model with Attention Pooling along the z (depth) dimension
"""

class SmpUnetDecoder(UnetDecoder):
    """
    Literally the same as UnetDecoder
    """

    def __init__(self, **kwargs):
        super(SmpUnetDecoder, self).__init__(**kwargs)

    def forward(self, feature):
        feature = feature[::-1]  # reverse channels to start from head of encoder
        head = feature[0] # Obtain the bottleneck feature of the encoder (e.g. Resnet)
        skip = feature[1:] + [None] # Features from encoder to be concatenated to the decoder features as skip connections
        d = self.center(head) # Identity

        decoder = []

        # Iterate through the decoder blocks
        for i, decoder_block in enumerate(self.blocks):
            s = skip[i]
            d = decoder_block(d, s) 
            decoder.append(d)

        last = d
        return last

class Unet_Attn_Pooling(nn.Module):

    def __init__(self, backbone="resnet34d"):
        super().__init__()

        self.backbone = backbone
        print(f"Choose backbone {self.backbone} for attention pooling")

        if self.backbone == "resnet34d":

            self.encoder_channels = [64, 64, 128, 256, 512] # Standard Resnet34 out_channels
            self.decoder_channels = [256, 128, 64, 32, 16] 

            #self.encoder = resnet34d(pretrained=True, in_chans=cfg.z_dims) # Timm resnet34d as the Unet Encoder
            self.encoder = timm.create_model("resnet34d", pretrained=False, in_chans=cfg.z_dims) # Timm resnet34d as the Unet Encoder
            self.z_offsets = [0,2,4]

        elif self.backbone == "se_resnext50_32x4d":

            self.encoder_channels = [64, 256, 512, 1024, 2048] 
            self.decoder_channels = [256, 128, 64, 32, 16] 

            self.z_offsets = [0,2,4]

            # Get Encoder from segmentation_models_pytorch
            self.encoder = get_encoder(
                cfg.backbone,
                in_channels=cfg.z_dims,
                depth=5,
                #weights="imagenet",
                weights=None,
            )

        elif self.backbone == "mit_b3":

            self.encoder_channels = [0, 64, 128, 320, 512] 
            self.decoder_channels = [256, 128, 64, 32, 16] 

            if cfg.in_chans == 11: # 25-36
                self.z_offsets = [0,2,4,6,8]
            elif cfg.in_chans == 15: # 22-37
                self.z_offsets = [0,2,4,6,8,10,12]

            # Get Encoder from segmentation_models_pytorch
            self.encoder = get_encoder(
                cfg.backbone,
                in_channels=cfg.z_dims, # 3
                depth=5,
                weights="imagenet",
            )

        self.decoder = SmpUnetDecoder(
            encoder_channels=[0] + self.encoder_channels, # (3, 64, 64, 128, 256, 512)
            decoder_channels=self.decoder_channels, # (256, 128, 64, 32, 16)
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.logit = nn.Conv2d(self.decoder_channels[-1], cfg.target_size, kernel_size=1)

        if self.backbone == "mit_b3":

            # Dirty hack to avoid 0 channel buy for nn.Conv2d
            dummy_encoder_channels = [1, 64, 128, 320, 512] 

            # Attention Pooling
            self.pooling_weight = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ) for channel in dummy_encoder_channels
            ])

        else:

            # Attention Pooling
            self.pooling_weight = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ) for channel in self.encoder_channels
            ])

    def forward(self, x):

        B, C, H, W = x.shape # (8, 12, 224, 224), the first element 8 being the batch size
        K = len(self.z_offsets)
        x = torch.cat([x[:,i:i+cfg.z_dims,:,:] for i in self.z_offsets], 0) # shape: [24, 6, 224, 224]

        if cfg.backbone == "resnet34d":

            # Foward pass through Resnet and obtain feature maps
            feat_maps = []

            x = self.encoder.conv1(x) # [24, 64, 112, 112]
            x = self.encoder.bn1(x)
            x = self.encoder.act1(x)
            feat_maps.append(x)

            x = F.avg_pool2d(x, kernel_size=2, stride=2) # [24, 64, 56, 56]

            x = self.encoder.layer1(x) # [24, 64, 56, 56]
            feat_maps.append(x)

            x = self.encoder.layer2(x) # [24, 128, 28, 28]
            feat_maps.append(x)

            x = self.encoder.layer3(x) # [24, 256, 14, 14]
            feat_maps.append(x)
            
            x = self.encoder.layer4(x) # [24, 512, 7, 7]
            feat_maps.append(x)

        else:

            feat_maps = self.encoder(x)

            # Exclude the first feature
            feat_maps = feat_maps[1:]


        # Attention Pooling across slices (z)
        for idx in range(len(feat_maps)):
            feat = feat_maps[idx] # [24, 64, 112, 112]
            if feat.shape[1] != 0:
                attn_map = self.pooling_weight[idx](feat)
                _, c, h, w = attn_map.shape # [24, 64, 112, 112]
                attn_map = rearrange(attn_map, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #f.reshape(B, K, c, h, w) # [8, 3, 64, 112, 112]
                feat = rearrange(feat, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #e.reshape(B, K, c, h, w) # [8, 3, 64, 112, 112]
                attn_weight = F.softmax(attn_map, 1) # [8, 3, 64, 112, 112]
                feat = (attn_weight * feat).sum(1) # [8, 64, 112, 112]
                feat_maps[idx] = feat
            else:
                _, c, h, w = feat.shape # [24, 0, 112, 112]
                feat = rearrange(feat, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #e.reshape(B, K, c, h, w) # [8, 3, 0, 112, 112]
                feat_maps[idx] = feat.sum(1) # [8, 0, 112, 112]

        top_feat = self.decoder(feat_maps) # [8, 16, 224, 224]

        top_feat = F.dropout(top_feat, p=0.5, training=True)

        logit = self.logit(top_feat) # [8, 1, 224, 224]

        return logit


class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        if cfg.use_attn_pooling:
            self.encoder = Unet_Attn_Pooling(cfg.backbone)

        elif cfg.backbone == "resnet3d":
            self.encoder = SegModel(model_depth=cfg.model_depth)
            self.encoder.load_pretrained_weights(torch.load(cfg.models_path + f"/r3d{cfg.model_depth}_K_200ep.pth")["state_dict"])

        elif cfg.backbone[:3] == "mit":
            self.encoder = smp.Unet(
                encoder_name=cfg.backbone, 
                encoder_weights=weight,
                classes=cfg.target_size,
                activation=None,
            )
        else:
            self.encoder = smp.Unet(
                encoder_name=cfg.backbone, 
                encoder_weights=weight,
                in_channels=cfg.in_chans, # 6 -> middle 6 slices out of a total of 65
                classes=cfg.target_size,
                activation=None,
            )

    def forward(self, image):

        if cfg.use_channel_branch:

            num_splits = (cfg.in_chans - cfg.z_dims) // cfg.channel_stride
            inputs_list = [image[:,cfg.channel_stride*i:cfg.channel_stride*i+cfg.z_dims,:,:] for i in range(num_splits)]

            outputs = [self.encoder(x) for x in inputs_list]
            output = sum(outputs) / len(outputs)

        else:
            output = self.encoder(image)
            #output = output.squeeze(-1)

        return output


def build_model(cfg, weight="imagenet"):
    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    model = CustomModel(cfg, weight)

    return model
# }}}

# {{{ Scheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    # Class function. Compute the learning rate at each epoch
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):

    if cfg.scheduler == 'GradualWarmupSchedulerV2':

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=1e-7)

        # Use LR warmup first -> LR increases by 10x over the warmup period which last 1 epoch. Afterwards revert to base scheduler which os CosineAnnealing
        scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    elif cfg.scheduler == 'CosineAnnealingLR':

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=1e-7)

    elif cfg.scheduler == 'OneCycleLR':

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, steps_per_epoch=10, epochs=cfg.epochs//10, pct_start=0.1)

    return scheduler

def scheduler_step(scheduler, val_loss, epoch):
    scheduler.step(epoch)
# }}}

# {{{ Metrics
def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    F0.5 Score
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def dice_coef_torch(preds, targets, beta=0.5, smooth=1e-5):

    #comment out if your model contains a sigmoid or equivalent activation layer
    preds = torch.sigmoid(preds)

    # flatten label and prediction tensors
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def dice_loss(preds, targets):
    return 1.0 - dice_coef_torch(preds, targets)

class FocalLoss(nn.Module):
    # From https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

def calc_fbeta(mask, mask_pred):
    """
    Find the best confidence threshold by F0.5 Score 
    """

    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0

    # Find best threshold
    for th in np.array(range(10, 80+1, 5)) / 100:
        
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        print(f'th: {th}, fbeta: {dice}')

        if dice > best_dice:
            best_dice = dice
            best_th = th
    
    Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th

def criterion(y_pred, y_true):

    # Only prediction on the label size region, excluding the edge pixels
    #if cfg.use_inference_mask:
    #    inference_mask = create_inference_mask().to("cuda:0")
    #    y_pred *= inference_mask
    #    y_true *= inference_mask

    if cfg.loss_fn == 'bce':
        return BCELoss(y_pred, y_true)
    elif cfg.loss_fn == 'dice':
        return dice_loss(y_pred, y_true)
    elif cfg.loss_fn == 'hybrid':
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * dice_loss(y_pred, y_true)
    elif cfg.loss_fn == 'lovasz':
        return LovaszLoss(y_pred, y_true)

# }}}

# {{{ Trainer
def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    # Perform Mixed Precision Training

    scaler = GradScaler(enabled=cfg.use_amp) # Scale Gradient to avoid underflow and overflow as a result of mixed precision training
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):

        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(cfg.use_amp): # use half-precision training
            y_preds = model(images) # Images: [8, 12, 224, 224]
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward() # scale gradients to prevent overflow/underflow

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg

def valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt):
    #mask_pred = np.zeros(valid_mask_gt.shape)
    #mask_count = np.zeros(valid_mask_gt.shape)

    mask_pred = torch.zeros(valid_mask_gt.shape, device="cuda:0")
    mask_count = torch.zeros(valid_mask_gt.shape, device="cuda:0")
    if cfg.use_inference_mask:
        inference_mask = create_inference_mask().to("cuda:0")

    model.eval()
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            with autocast(cfg.use_amp):
                if cfg.use_tta:
                    y_preds = TTA(images, model)
                else:
                    y_preds = model(images)
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # Project patches onto the mask 
        #y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        y_preds = torch.sigmoid(y_preds)

        y_preds = y_preds.to("cuda:0").squeeze(1) 
        if cfg.use_inference_mask:
            y_preds = y_preds * inference_mask[None] # shape=(batch,H,W)

        start_idx = step*cfg.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i]
            if cfg.use_inference_mask:
                mask_count[y1:y2, x1:x2] += inference_mask
            else:
                mask_count[y1:y2, x1:x2] += 1

    print(f'mask_count_min: {mask_count.min()}')

    # Divide raw pred with mask_count to account for patches overlap (patches seperate by stride)
    mask_pred /= (mask_count + 1e-7)
    mask_pred = mask_pred.cpu().numpy()

    # L1/Hessian Denosing
    if cfg.use_denoising:
        mask_pred = xp.array(mask_pred)
        mask_pred = denoise_image(mask_pred, iter_num=250)
        mask_pred = mask_pred.get()

    return losses.avg, mask_pred

def TTA(x: torch.Tensor, model: nn.Module):
    #x.shape=(batch,c,h,w)

    shape = x.shape

    if cfg.use_channel_tta:

        # Channel TTA
        tta_chan_stride = 5
        #num_split_chans = (cfg.in_chans - cfg.z_dims) // tta_chan_stride + 1 # (25 - 20) // 5 + 1 = 2
        num_split_chans = (cfg.in_chans - cfg.z_dims) // tta_chan_stride # (25 - 20) // 5  = 2
        if num_split_chans != 1:
            x = [x[:, tta_chan_stride*i:cfg.z_dims + tta_chan_stride*i] for i in range(num_split_chans)]
            x = torch.cat(x,dim=0)

        # 90 degree Rotation TTA
        x = [torch.rot90(x,k=i,dims=(-2,-1)) for i in range(4)]
        x = torch.cat(x,dim=0)

        x = model(x)
        #x = torch.sigmoid(x)

        x = x.reshape(4,shape[0]*num_split_chans, *x.shape[1:])
        x = [torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
        x = torch.stack(x,dim=0)
        x = x.mean(0) # [32,224,224] 

        x = x.reshape(num_split_chans, shape[0], *x.shape[1:])
        x = x.mean(0)
        #x=x.unsqueeze(1) # Make output [32,1,224,224,]

    else:

        # Rotation
        x=[x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
        x=torch.cat(x,dim=0)

        x=model(x)
        #x=torch.sigmoid(x)

        x=x.reshape(4,shape[0],*shape[2:])
        x=[torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]

        x=torch.stack(x,dim=0)
        x=x.mean(0) # [32,224,224] 
        x=x.unsqueeze(1) # Make output [32,1,224,224,]

    return x

# }}}

# {{{ Main

if __name__ == "__main__":

    cfg = pyrallis.parse(config_class=Config)
    cfg_init(cfg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Logger = init_logger(log_file=cfg.log_path)

    Logger.info('\n\n-------- exp_info -----------------')

    if cfg.use_wandb:
        hparams = {}
        for key, value in vars(cfg).items():
            hparams[key] = value

        wandb.init(project="Vesuvius Unet Training", config=hparams)

    Logger.info(f'\nStart Training. Validate on fold-{cfg.validation_fold}\n')

    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
    valid_xyxys = np.stack(valid_xyxys)

    train_dataset = CustomDataset(train_images, cfg, labels=train_masks, transform=get_transforms(data='train', cfg=cfg), mode='train')
    valid_dataset = CustomDataset(valid_images, cfg, labels=valid_masks, transform=get_transforms(data='valid', cfg=cfg), mode='valid')

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train_batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.valid_batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    model = build_model(cfg)
    model.to(device)

    #if cfg.use_wandb:
    #    # Wandb Magic 
    #    wandb.watch(model)

    optimizer = AdamW(model.parameters(), lr=cfg.initial_lr)
    scheduler = get_scheduler(cfg, optimizer)

    DiceLoss = smp.losses.DiceLoss(mode='binary')
    LovaszLoss = smp.losses.LovaszLoss(mode='binary')
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()

    alpha = 0.5
    beta = 1 - alpha
    TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False, alpha=alpha, beta=beta)

    valid_mask_gt = cv2.imread(cfg.dataset_path + f"train/{cfg.validation_fold}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255

    pad0 = (cfg.tile_size - valid_mask_gt.shape[0] % cfg.tile_size)
    pad1 = (cfg.tile_size - valid_mask_gt.shape[1] % cfg.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    if cfg.metric_direction == 'minimize':
        best_score = np.inf
    elif cfg.metric_direction == 'maximize':
        best_score = -1

    best_loss = np.inf

    # Training Loop
    for epoch in range(cfg.epochs):

        start_time = time.time()

        # train
        train_loss = train_fn(train_loader, model, criterion, optimizer, device)

        # eval
        val_loss, mask_pred = valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt)

        scheduler_step(scheduler, val_loss, epoch)

        best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

        # score = val_loss
        score = best_dice

        elapsed = time.time() - start_time

        Logger.info(f'Epoch {epoch+1} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  time: {elapsed:.0f}s')
        Logger.info(f'Epoch {epoch+1} - Best F0.5 Score: {score:.4f}')

        if cfg.metric_direction == 'minimize':
            update_best = score < best_score
        elif cfg.metric_direction == 'maximize':
            update_best = score > best_score

        # Save the last model
        torch.save({'model': model.state_dict(),
                    'preds': mask_pred},
                    cfg.model_dir + f'{cfg.model_name}_{cfg.backbone}_fold{cfg.validation_fold}_last.pth')

        if update_best:
            best_loss = val_loss
            best_score = score

            Logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            Logger.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            
            torch.save({'model': model.state_dict(),
                        'preds': mask_pred},
                        cfg.model_dir + f'{cfg.model_name}_{cfg.backbone}_fold{cfg.validation_fold}_best.pth')

        if cfg.use_wandb:
            wandb.log({f'train loss fold-{cfg.validation_fold}': train_loss, f'valid loss fold-{cfg.validation_fold}': val_loss, f'F0.5 fold-{cfg.validation_fold}': score})

    if cfg.select_best_model:
        best_model_path = cfg.model_dir + f'{cfg.model_name}_{cfg.backbone}_fold{cfg.validation_fold}_best.pth'
        new_model_path = cfg.final_models_dir + f'/{cfg.model_name}_{cfg.backbone}_fold{cfg.validation_fold}.pth'
        shutil.copy(best_model_path, new_model_path)
    else:
        last_model_path = cfg.model_dir + f'{cfg.model_name}_{cfg.backbone}_fold{cfg.validation_fold}_last.pth'
        new_model_path = cfg.final_models_dir + f'/{cfg.model_name}_{cfg.backbone}_fold{cfg.validation_fold}.pth'
        shutil.copy(last_model_path, new_model_path)


# }}}











