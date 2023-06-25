"""
A Single script that can handle ensemble of different types of models (e.g. Unet, ResNet3D, SE-Resnet3D) and different data pipelines and generate submission file
"""

# {{{ Module Imports for all models

# Specify System Paths for Kaggle Notebook
import sys
sys.path.append('/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4')
sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append('/kaggle/input/segmentation-models-pytorch/segmentation_models.pytorch-master')
sys.path.append('/kaggle/input/segmentation-models-pytorch-v2')
sys.path.append('/kaggle/input/resnet3d')
sys.path.append('/kaggle/input/seresnet3d')
sys.path.append('/kaggle/input/einops/einops-master')

# Generic Imports
import pickle
import warnings
import pandas as pd
import os
import gc
import sys
import math
import time
import random
import datetime
import importlib
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
from functools import partial
import hashlib

# Computer Vision
import cv2
import PIL.Image as Image
import imageio

# ML modules
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

# TIMM
import timm
from timm.models.resnet import resnet34d

# Plotting
import matplotlib.pyplot as plt

# Monitoring
from tqdm.auto import tqdm

# Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

# Logging
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

# Model Imports
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from felix_work.resnet3d import generate_model
from felix_work.seresnet3d import Resnet3d

# For importing pre-trained model. See https://github.com/pytorch/pytorch/issues/33288
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from paths import dataset_path, models_path

# }}}

# {{{ Core Config for all models

class Config:

    # === Core Paths === 
    #comp_name = 'vesuvius'
    #root_path = '/kaggle/input/'
    #comp_name = 'vesuvius-challenge-ink-detection'
    #dataset_path = f'{root_path}{comp_name}/'

    dataset_path = dataset_path
    models_path = models_path

    target_size = 1 # target classes

    num_workers = 2
    seed = 38
    use_tta: bool = False
    use_denoising: bool = False
    use_th_search = False

    #device_ids = [0,1]
    device_ids = [0]

# }}}

# {{{ Functions for all models

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# L1/Hessian denoising
# Reference https://www.kaggle.com/code/brettolsen/improving-performance-with-l1-hessian-denoising

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

# {{{ Dataset

def read_image_and_binary_mask(fragment_id):
    images = []

    start = cfg.start_slice
    end = cfg.end_slice
    idxs = range(start, end)

    for i in tqdm(idxs):
        
        image = cv2.imread(cfg.dataset_path + f"{mode}/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    binary_mask = cv2.imread(cfg.dataset_path + f"{mode}/{fragment_id}/mask.png", 0)

    binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)
    binary_mask = (binary_mask / 255).astype(int)
    
    return images, binary_mask
    
def get_transforms(cfg):

    aug = A.Compose([
        A.Resize(cfg.input_size, cfg.input_size),
        A.Normalize(
            mean= [0] * cfg.in_chans,
            std= [1] * cfg.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ])

    return aug

def normalization(x: torch.Tensor) -> torch.Tensor:
    """input.shape=(batch,f1,f2,...)"""

    #[batch,f1,f2]->dim[1,2]
    dim = list(range(1, x.ndim))
    mean = x.mean(dim = dim,keepdim = True)
    std = x.std(dim = dim, keepdim = True)
    return (x - mean) / (std + 1e-9)

class CustomDataset(Dataset):

    def __init__(self, images, cfg, labels=None, transform=None, xys=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xys = xys

    def __len__(self):
        # return len(self.xyxys)
        return len(self.images)

    def __getitem__(self, idx):
        # x1, y1, x2, y2 = self.xyxys[idx]
        image = self.images[idx]
        
        if cfg.pipeline_version == '0.1':
            image = image.astype(np.float32) / np.iinfo(image.dtype).max
            image = image * np.iinfo(np.uint16).max
            image = image/65535.0
            image = (image - 0.45)/0.225
            data = self.transform(image=image)
            image = data['image']
        elif cfg.pipeline_version == '0.2':
            image = image.astype(np.float32) / np.iinfo(image.dtype).max
            image = image * np.iinfo(np.uint16).max
            image = image/image.max()*255
            image = torch.from_numpy(image).permute(2,0,1).to(torch.float32) / 255
            image[image > 0.78] = 0.78
        else:
            data = self.transform(image=image)
            image = data['image']

        return image, self.xys[idx]

def make_test_dataset(fragment_id):
    test_images, binary_mask = read_image_and_binary_mask(fragment_id) # 'a' and 'b'
    
    x1_list = list(range(0, test_images.shape[1]-cfg.tile_size+1, cfg.stride))
    y1_list = list(range(0, test_images.shape[0]-cfg.tile_size+1, cfg.stride))
    
    test_images_list = []
    xyxys = []
    for y1 in y1_list:
        for x1 in x1_list:

            if binary_mask[y1:y1+cfg.tile_size, x1:x1+cfg.tile_size].max() > 0:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size

                if np.all(test_images[y1:y2, x1:x2]==0):
                    continue

                test_images_list.append(test_images[y1:y2, x1:x2])
                xyxys.append((x1, y1, x2, y2))

    xyxys = np.stack(xyxys)
            
    test_dataset = CustomDataset(test_images_list, cfg, transform=get_transforms(cfg), xys=xyxys)
    
    test_loader = DataLoader(test_dataset,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    
    return test_loader, xyxys

def TTA(x: torch.Tensor, model: nn.Module):
    #x.shape=(batch,c,h,w)

    shape=x.shape

    if cfg.use_chan_tta:

        # Channel TTA
        tta_chan_stride = 5
        num_split_chans = (cfg.in_chans - cfg.z_dims) // tta_chan_stride + 1 # (25 - 20) // 5 + 1 = 2
        if num_split_chans != 1:
            x = [x[:, tta_chan_stride*i:cfg.z_dims + tta_chan_stride*i] for i in range(num_split_chans)]
            x = torch.cat(x,dim=0)

        # 90 degree Rotation TTA
        x = [x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
        x = torch.cat(x,dim=0)

        x = model(x)
        x = torch.sigmoid(x)

        x = x.reshape(4,shape[0]*num_split_chans, *x.shape[1:])
        x = [torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
        x = torch.stack(x,dim=0)
        x = x.mean(0) # [32,224,224] 

        x = x.reshape(num_split_chans, shape[0], *x.shape[1:])
        x = x.mean(0)

    else:
        x = [x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
        x = torch.cat(x,dim=0)
        x = model(x)
        x = torch.sigmoid(x)
        x = x.reshape(4,shape[0],*shape[2:])
        x = [torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
        x = torch.stack(x,dim=0)
        x = x.mean(0) # [32,224,224] 
        x = x.unsqueeze(1) # Make output [32,1,224,224,]

    return x
# }}}

# {{{ Inference Mask
def create_inference_mask():
    """
    It is used to mask out the edge pixels for model prediction

    If is of dimension input_size x input_size with zeros on the border with edge_size
    """
    mask = np.zeros((cfg.input_size, cfg.input_size))
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
        if cfg.model_depth == 152:
            self.encoder = generate_model_v2(model_depth=model_depth, n_input_channels=1)
            self.decoder = Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)
        else:
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

class SE_Resnet3D(nn.Module):

    def __init__(self, model_depth=101, SE=True):
        super().__init__()
        model = Resnet3d(model_depth, SE=True)
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, x):

        x = normalization(x.reshape(-1,*x.shape[2:])).reshape(x.shape)

        if x.ndim==4:
            x=x[:,None]

        feat_maps = self.encoder.get_each_layer_features(x) 
        pred_mask = self.decoder.forward(feat_maps) 

        return pred_mask

"""
Unet Model with Attention Pooling along the z (depth) dimension
"""

class SmpUnetDecoder(UnetDecoder):
    """
    Customized Unet Decoder. But why need to reinvent the wheel?
    """

    def __init__(self, **kwargs):
        super(SmpUnetDecoder, self).__init__(**kwargs)

    def forward(self, encoder):
        feature = encoder[::-1]  # reverse channels to start from head of encoder
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

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

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

            self.z_offsets = [0,2,4,6,8]

            # Get Encoder from segmentation_models_pytorch
            self.encoder = get_encoder(
                cfg.backbone,
                in_channels=cfg.z_dims,
                depth=5,
                #weights="imagenet",
                weights=None,
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

        B, C, H, W = x.shape # (8, 12, 224, 224)
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

        elif cfg.backbone[:3] == "mit":
            self.encoder = smp.Unet(
                encoder_name=cfg.backbone, 
                encoder_weights=weight,
                classes=cfg.target_size,
                activation=None,
            )
            #if cfg.in_chans == 6:
            #    out_channels=self.encoder.encoder.patch_embed1.proj.out_channels
            #    self.encoder.encoder.patch_embed1.proj=nn.Conv2d(cfg.in_chans,out_channels,7,4,3)
        else:
            self.encoder = smp.Unet(
                encoder_name=cfg.backbone, 
                encoder_weights=weight,
                in_channels=cfg.in_chans, # 6 -> middle 6 slices out of a total of 65
                classes=cfg.target_size,
                activation=None,
            )

    def forward(self, image):

        if cfg.backbone[:3] == "mit" and cfg.in_chans == 6:
            input_1, input_2 = image.split(3, dim=1)
            output_1 = self.encoder(input_1)
            output_2 = self.encoder(input_2)
            output = (output_1 + output_2) / 2
        else:
            output = self.encoder(image)
            output = output.squeeze(-1)

        return output
    

def build_model(cfg, model_path):
    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    if cfg.pipeline_version == '0.1':
        model = SegModel(cfg.model_depth)
    elif cfg.pipeline_version == '0.2':
        model = SE_Resnet3D(model_depth=101, SE=True)
    else:
        model = CustomModel(cfg)

    print(f'Load model from path: {model_path}')
    if cfg.pipeline_version == '0.2':
        model.load_state_dict(torch.load(model_path,map_location="cpu"))
    else:
        model.load_state_dict(torch.load(model_path,map_location="cpu")['model'])
    
    return model

class EnsembleModel(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.models = nn.ModuleList()
        self.weights = torch.tensor(weights)[:,None,None,None,None]

    def __call__(self, x):
        """
        Weighted average of models
        """
        x = [model(x) for model in self.models]
        x = torch.stack(x, dim=0)
        x = x * self.weights.to(device=x.device)
        return torch.sum(x, dim=0)

    def add_model(self, model):
        self.models.append(model)

def build_ensemble_model(cfg, model_path, cv_folds, weights):

    model = EnsembleModel(weights)

    for fold in cv_folds:
        path = model_path.format(fold=fold, backbone=cfg.backbone)
        _model = build_model(cfg, path)
        model.add_model(_model)
    
    return model
# }}}

# Main 

# Global Config
cfg = Config

mode = 'test'
fragment_ids = sorted(os.listdir(cfg.dataset_path + mode)) # ['a','b']

#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model_template_list = [{
    'alias': 'seresnet3d-101',
    'weight': 0.45,
    'config': {
        'model_name': 'seresnet3d_101',
        'backbone': 'se_resnet3d_101',
        'in_chans': 25,
        'z_dims': 20,
        'start_slice': 15,
        'end_slice': 40,
        'input_size': 256,
        'tile_size': 256,
        'edge_size': 32, # Do not predict on edge pixels
        'stride': (256-32*2) // 4, # 256 - 32*2 = 192
        'batch_size': 8,
        'pipeline_version': '0.2',
        'inference_mode': 'eval',
        'use_attn_pooling': False,
        'use_chan_tta': True,
        'use_inference_mask': True,
        'model': {
            'ensemble': False,
            'model_path': cfg.models_path + '/SE-resnet3d-101_epoch_14_tl0.345_vl0.335_vp0.737.pth'
        }
    }
}, {
    'alias': 'resnet3d',
    'weight': 0.2,
    'config': {
        'model_name': 'resnet3d_224',
        'backbone': 'resnet3d',
        'in_chans': 18,
        'start_slice': 22,
        'end_slice': 40,
        'model_depth': 34,
        'input_size': 224,
        'tile_size': 224,
        'edge_size': 16,
        'stride': (224-16*2) // 2, 
        'batch_size': 8, 
        'pipeline_version': '1.0',
        'inference_mode': 'train',
        'use_attn_pooling': False,
        'use_chan_tta': False,
        'use_inference_mask': True,
        'model': {
            'ensemble': True,
            'folds': [1,3,4,5],
            'weights': [0.3,0.3,0.2,0.2],
            'model_path': cfg.models_path + '/resnet34-224-models/Unet_resnet3d_fold{fold}.pth'
        }
    }
}, {
    'alias': 'resnet3d',
    'weight': 0.1,
    'config': {
        'model_name': 'resnet3d_192',
        'backbone': 'resnet3d',
        'in_chans': 18,
        'start_slice': 22,
        'end_slice': 40,
        'model_depth': 34,
        'input_size': 192,
        'tile_size': 192,
        'edge_size': 16,
        'stride': (192-16*2) // 2,
        'batch_size': 8,
        'pipeline_version': '1.0',
        'inference_mode': 'train',
        'use_attn_pooling': False,
        'use_chan_tta': False,
        'use_inference_mask': True,
        'model': {
            'ensemble': True,
            'folds': [1,2,3],
            'weights': [0.4,0.3,0.3],
            'model_path': cfg.models_path + '/resnet34-192-models/Unet_resnet3d_fold{fold}.pth'
        }
    }
}, {
    'alias': 'mit-b3',
    'weight': 0.2,
    'config': {
        'model_name': 'mit_b3_attnPool',
        'backbone': 'mit_b3',
        'in_chans': 11,
        'z_dims': 3,
        'start_slice': 25,
        'end_slice': 36,
        'input_size': 224,
        'tile_size': 224,
        'edge_size': 32,
        'stride': (224-32*2) // 2,
        'batch_size': 8,
        'pipeline_version': '1.0',
        'inference_mode': 'eval',
        'use_attn_pooling': True,
        'use_chan_tta': False,
        'use_inference_mask': True,
        'model': {
            'ensemble': True,
            'folds': [1,3,4,5],
            'weights': [0.3,0.3,0.2,0.2],
            'model_path': cfg.models_path + '/mit-b3-attnPool-exp048-models/Unet_mit_b3_fold{fold}.pth'
        }
    }
}, {
    'alias': 'unet',
    'weight': 0.15,
    'config': {
        'model_name': 'mit_b3_6chans',
        'backbone': 'mit_b3',
        'in_chans': 6,
        'start_slice': 29,
        'end_slice': 35,
        'input_size': 224,
        'tile_size': 224,
        'edge_size': 24,
        'stride': (224-24*2) // 2,
        'batch_size': 8,
        'pipeline_version': '1.0',
        'inference_mode': 'eval',
        'use_attn_pooling': False,
        'use_chan_tta': False,
        'use_inference_mask': True,
        'model': {
            'ensemble': True,
            'folds': [1,2,3],
            'weights': [0.4,0.3,0.3],
            'model_path': cfg.models_path + '/mit-b3-models/Unet_mit_b3_fold{fold}.pth'
        }
    }
}]

pred_masks_dict = defaultdict(list)

fixed_TH = 0.55 # Arbitrary confidence threshold

ensemble_weights = []

a_file = cfg.dataset_path + f"test/a/mask.png"
with open(a_file,'rb') as f:
    hash_md5 = hashlib.md5(f.read()).hexdigest()
is_skip_test = hash_md5 == '0b0fffdc0e88be226673846a143bb3e0'

is_skip_test = False

debug = False

if is_skip_test:
    submit_df = pd.DataFrame({
        'Id': ['a', 'b'],
        'Predicted':['1 2', '1 2']
    })
    submit_df.to_csv('submission.csv', index=False)

else:


    pred_paths_dict = defaultdict(list) # Store prediction on disk
    for model_template in model_template_list:

        # Register model specific config
        for k, v in model_template['config'].items():
            setattr(cfg,k,v)

        # Initialize model
        model_path = model_template['config']['model']['model_path']
        if model_template['config']['model']['ensemble']:
            folds = model_template['config']['model']['folds']
            weights = model_template['config']['model']['weights']
            model = build_ensemble_model(cfg, model_path, folds, weights)
        else:
            model = build_model(cfg, model_path)
            
        model = nn.DataParallel(model, device_ids=Config.device_ids)
        model = model.cuda()
        if cfg.inference_mode == 'eval':
            model.eval()

        # Register ensemnble weights
        ensemble_weights.append(model_template['weight'])

        for fragment_id in fragment_ids:

            test_loader, xyxys = make_test_dataset(fragment_id)

            # Load mask for test fragment
            binary_mask = cv2.imread(cfg.dataset_path + f"{mode}/{fragment_id}/mask.png", 0)
            binary_mask = (binary_mask / 255).astype(int)

            # Save height and weight for the original mask prior to padding
            ori_h = binary_mask.shape[0]
            ori_w = binary_mask.shape[1]

            # Pad mask such that it is divisible by 224 which is the patch size
            pad0 = (cfg.tile_size - binary_mask.shape[0] % cfg.tile_size)
            pad1 = (cfg.tile_size - binary_mask.shape[1] % cfg.tile_size)

            binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)

            mask_pred = np.zeros(binary_mask.shape)
            mask_count = np.zeros(binary_mask.shape)
            if cfg.use_inference_mask:
                inference_mask = create_inference_mask()

            for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
                images = images.cuda()
                #images = images.to(device)
                batch_size = images.size(0)

                with torch.no_grad():
                    with autocast():
                        if cfg.use_tta:
                            y_preds = TTA(images, model)
                        else:
                            y_preds = model(images)
                            y_preds = torch.sigmoid(y_preds)

                for k, (x1, y1, x2, y2) in enumerate(xys):
                    if cfg.use_inference_mask:
                        mask_pred[y1:y2, x1:x2] += y_preds[k].squeeze(0).cpu().numpy() * inference_mask
                        mask_count[y1:y2, x1:x2] += inference_mask
                    else:
                        mask_pred[y1:y2, x1:x2] += y_preds[k].squeeze(0).cpu().numpy()
                        mask_count[y1:y2, x1:x2] += 1

            print(f'mask_count_min: {mask_count.min()}')
            mask_pred /= (mask_count + 1e-7)

            # Cut out the region with original height and width
            mask_pred = mask_pred[:ori_h, :ori_w]
            binary_mask = binary_mask[:ori_h, :ori_w]

            mask_pred *= binary_mask

            if cfg.use_th_search:

                # Store the prediction on disk for thresholding computation in the next stage
                mask_pred = (mask_pred*255).astype(np.uint8) 
                path = f"_{cfg.model_name}_{fragment_id}"+".png"
                cv2.imwrite(path, mask_pred, [cv2.IMWRITE_PNG_COMPRESSION,3])
                pred_paths_dict[cfg.model_name].append(path)

            else:

                pred_masks_dict[fragment_id].append(mask_pred)

                mask_pred = (mask_pred >= fixed_TH).astype(int)

            del mask_pred, mask_count, binary_mask, xyxys, test_loader, images, y_preds
            gc.collect()
            torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Generate Submission

    if cfg.use_th_search:

        total_weight = np.sum(ensemble_weights)

        # Read prediction stored on disk
        pred_outputs = []
        for model_idx, (model_name, pred_paths) in enumerate(pred_paths_dict.items()):
            print(f"Reading prediction file from {model_name}")
            for fragment_idx, pred_path in enumerate(pred_paths):
                pred_img = cv2.imread(pred_path, 0).astype(np.uint8)
                shape = pred_img.shape
                pred_img = pred_img.flatten()
                cache = np.zeros(pred_img.shape[0], dtype=np.uint8)
                cache[pred_img.argsort()] = (np.arange(pred_img.shape[0]) / pred_img.shape[0]*255).astype(np.uint8)
                pred_img = cache.reshape(shape)

                del cache
                gc.collect()

                if model_idx == 0:
                    pred_outputs.append(pred_img * ensemble_weights[0] / total_weight)
                else:
                    pred_outputs[fragment_idx] += pred_img * ensemble_weights[model_idx] / total_weight

        # Search for Threshold
        th_percentile = 0.035

        TH = [output.flatten() for output in pred_outputs] 
        TH = np.concatenate(TH)
        TH.sort()
        TH:float = TH[-int(len(TH)*th_percentile)]

        results = []
        for fragment_id, mask_pred in zip(fragment_ids, pred_outputs):

        
            mask_pred = (mask_pred >= TH).astype(np.uint8)
            results.append((fragment_id, rle(mask_pred)))

        del pred_outputs
        gc.collect()
        torch.cuda.empty_cache()

    else:

        results = []

        for fragment_id, pred_list in pred_masks_dict.items():

            avg_pred = np.average(pred_list, axis=0, weights=ensemble_weights)

            avg_pred = (avg_pred >= fixed_TH).astype(int)

            inklabels_rle = rle(avg_pred)

            results.append((fragment_id, inklabels_rle))

            del avg_pred
            gc.collect()
            torch.cuda.empty_cache()
            
        del pred_masks_dict
        gc.collect()
        torch.cuda.empty_cache()

    sub = pd.DataFrame(results, columns=['Id', 'Predicted'])

    sample_sub = pd.read_csv(cfg.dataset_path + 'sample_submission.csv')
    sample_sub = pd.merge(sample_sub[['Id']], sub, on='Id', how='left')

    sample_sub.to_csv("submission.csv", index=False)
