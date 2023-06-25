from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import sys
import time
import torch as tc
import random
from torch.utils.data import DataLoader, Dataset
import torch
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from tqdm.auto import tqdm

import torch,copy
import torch.nn as nn
from torch.optim import AdamW

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import json
import os,warnings
from typing import Tuple

import numpy as np
import torch as tc
import os,datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from typing import List, Optional,Tuple
from timm.scheduler import CosineLRScheduler

warnings.filterwarnings("ignore")

PATH={"RAW_DATA_DIR":"./vc_raw_data/",
"CLEAN_DATA_DIR":"./clean_data/",
"TRAIN_DATA_CLEAN_PATH":"./clean_data/",
"CHECKPOINT_DIR":"./checkpoints/",
"MODEL_DIR":"./final_model/"}

for v in PATH.values():
    if not os.path.exists(v):
        os.mkdir(v)


class CFG:
    # ============== comp xxexp name =============
    comp_dataset_path = '/root/autodl-tmp/'
    model_dir = "/root/model_data/"
    image_type=".png"
    
    #if not os.path.exists(model_dir):
    #    os.mkdir(model_dir)
    
    # ============== model cfg =============
    backbone = 'SE-resnet3d-101'

    chan_start=17 #16
    in_chans = 16 #12
    load_chans=26 #26
    split_in_chan=1
    # ============== training cfg =============
    train_fragment_id=["1","2a","2b","3"]
    cp_rate=[1,0.75,0.5]
    cp_sample_rate=[1,1,1]
    
    val_cp_rate=1
    mean_output=False
    label_size=96#
    ex_size = 0
    model_input_size=label_size+ex_size
    train_load_size=model_input_size
    stride = label_size // 2
    filtering_size=0


    train_batch_size = 128 # 32
    valid_batch_size = train_batch_size * 2
    use_amp = True
    
    scheduler = True
    epochs = 50+20 # 30
    total_per_epoch=10000
    
    lr = 2e-4
    max_grad_norm=10000
    # ============== fold =============
    valid_id = "2a"

    TTA=False

    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-4
    num_workers = 4

    # ============== augmentation =============
    exponent_arg=True
    mixup_rate=0.
    noise_rate=0.1
    z_resize_rate=0.2
    assert load_chans*(1-z_resize_rate)>=in_chans
    
    rotate = A.Compose([A.Rotate(5,p=1)])
    
    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.ChannelDropout((1,6),p=0.3,fill_value=128),
        A.CoarseDropout(max_holes=1, max_width=int(model_input_size * 0.3), max_height=int(model_input_size * 0.3), 
                        mask_fill_value=0, p=0.5),
        
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),

        A.GaussNoise(var_limit=[1,10],p=0.4),

        A.GaussianBlur(p=0.3),#
        A.MotionBlur(p=0.3),#
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25),#
        
        ToTensorV2(transpose_mask=True)
        ])

    valid_aug = A.Compose([
        ToTensorV2(transpose_mask=True),
        ])

def add_noise(x:tc.Tensor,randn_rate=0.1,x_already_normed=False):
    """input.shape=(batch,f1,f2,...) output's var will be normalizate  """
    if x_already_normed:
        x_std=1
    else: 
        dim=list(range(1,x.ndim))
        x_std=x.std(dim=dim,keepdim=True)
    cache=(x_std**2+(x_std*randn_rate)**2)**0.5
    #https://blog.csdn.net/chaosir1991/article/details/106960408
    
    return (x+tc.randn(size=x.shape,device=x.device)*randn_rate*x_std)/(cache+1e-9)

def min_max_normalization(x:tc.Tensor)->tc.Tensor:
    """input.shape=(batch,f1,...)"""
    shape=x.shape
    if x.ndim>2:
        x=x.reshape(x.shape[0],-1)
    
    min_=x.min(dim=-1,keepdim=True)[0]
    max_=x.max(dim=-1,keepdim=True)[0]
    if min_.mean()==0 and max_.mean()==1:
        return x.reshape(shape)
    
    x=(x-min_)/(max_-min_+1e-9)
    return x.reshape(shape)

def normalization(x:tc.Tensor)->tc.Tensor:
    """input.shape=(batch,f1,f2,...)"""
    #[batch,f1,f2]->dim[1,2]
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    return (x-mean)/(std+1e-9)


def print_selective(*args,pass_=False):
    if not pass_:
        print(*args)
        
def input_selective(*args,pass_=False):
    if not pass_:
        input(*args)
        
def get_now_time(remove_year=False):
    cache=str(datetime.datetime.now()).split(":")
    cache="_".join(cache).split(" ")
    cache="_".join(cache).split("-")
    if remove_year:
        cache=cache[1:]
    cache="_".join(cache).split(".")
    return cache[0]
################################################################################################

class Module(nn.Module):
    def __init__(self,loss_fc=nn.BCELoss(reduction="mean")):
        super().__init__()
        self.epoch=0
        self.loss_fc=loss_fc
        self.scaler=GradScaler()
        self.scheduler=None
        
    def forward(self)->None:
        pass
    
    def AMP_forward(self,*args,**kwargs)->tc.Tensor:
        with autocast():
            x=self.forward(*args,**kwargs)
        if isinstance(x,Tuple):
            x=[i.to(tc.float32) for i in x]
        else :
            x=x.to(tc.float32)
        return x
        
    def init_optimizer(self,parameters:List=None,lr=1e-4,weight_decay=1e-2,
                       init_scheduler=False,lr_min=1e-6,warmup=None,nsteps=None):
        """warmup = epochs_warmup * nbatch  # number of warmup steps
            nsteps = epochs * nbatch        # number of total steps"""
        if bool(parameters):
            self.optimizer=tc.optim.AdamW(parameters,lr=lr,betas=(0.5, 0.999),
                                          weight_decay=weight_decay)
        else :
            self.optimizer=tc.optim.AdamW(self.parameters(),lr=lr,betas=(0.5, 0.999),
                                          weight_decay=weight_decay)
        self.optimizer_list=[self.optimizer]
        
        if init_scheduler:
            self.scheduler = CosineLRScheduler(self.optimizer,
                        warmup_t=warmup, warmup_lr_init=0.0, warmup_prefix=True, # 1 epoch of warmup
                        t_initial=(nsteps - warmup),lr_min=lr_min)                # 3 epochs of cosine
            #is you want to use 
            #can this ->self.scheduler.step(iepoch * nbatch + ibatch + 1)
    
    def backward(self,loss:tc.Tensor,retain_graph=False,updata=True,inputs:Optional[list]=None):
        if not bool(inputs):
            inputs=list(self.parameters())
        loss.backward(inputs=inputs,retain_graph=retain_graph)
        if updata:
            self.optimize_parameters()
    
    def AMP_backward(self,loss:tc.Tensor,retain_graph=False,max_grad_norm:float=None):
        with autocast():
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward(inputs=list(self.parameters()),retain_graph=retain_graph)
            if bool(max_grad_norm):
                tc.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"],
                                            max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
    def optimize_parameters(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    
    def load_checkpoint(self,load_path:str,data_save=False,whole_save=False,all_pass=False,
                        device=None):
        init_ot=False
        if not os.path.exists(load_path):
            print_selective("load path error",pass_=all_pass)
            return None
        print_selective("loading checkpoint...",pass_=all_pass)
        
        ###########################################load#################################################
        model_dict = self.state_dict()
        if bool(device):
            modelCheckpoint=tc.load(load_path,map_location=device)
        else :
            modelCheckpoint = tc.load(load_path)
        if "state_dict" in modelCheckpoint.keys():
            pretrained_dict = modelCheckpoint['state_dict']
        else :
            pretrained_dict=modelCheckpoint
                    
        ######################################## load optimizer
        try:
            for x in range(len(self.optimizer_list)):
                self.optimizer_list[x].load_state_dict(modelCheckpoint['optimizer'][x])
        except:
            print_selective("not loaded optimizer",pass_=all_pass)
        ######################################## load epoch
        if "id" in modelCheckpoint.keys():
            self.id=modelCheckpoint["id"]
        if "epoch" in modelCheckpoint.keys():
            self.epoch=modelCheckpoint["epoch"]
        else :
            self.epoch=0
        ######################################## load data
        # 过滤操作
        new_dict={}
        flexible_load_num=0
        for k,v in pretrained_dict.items():
            if k in model_dict.keys() and model_dict[k].shape==v.shape:
                v=v.to(model_dict[k].device)
                new_dict[k]=v
            else :
                try:
                    print_selective(f"flexible loading: {k} data_v: {v.shape} model_v: {model_dict[k].shape}",pass_=all_pass)
                except:
                    print_selective(f"flexible loading: {k} data_v: {v.shape} model_v: None",pass_=all_pass)
                v:tc.Tensor
                #cache=v.clone()
                if k not in model_dict.keys():
                    print_selective("key error",pass_=all_pass)
                    continue
                #model_dict[k]*=0
                model_dict[k]=tc.randn(model_dict[k].shape,device=model_dict[k].device)*1e-3#*1e-4
                a=v.shape
                b=model_dict[k].shape
                try:
                    v=v.to(model_dict[k].device)
                    if v.ndim==5:
                        model_dict[k][:a[0],:a[1],:a[2],:a[3],:a[4]]=v[:b[0],:b[1],:b[2],:b[3],:b[4]]
                    elif v.ndim==4:
                        model_dict[k][:a[0],:a[1],:a[2],:a[3]]=v[:b[0],:b[1],:b[2],:b[3]]
                    elif v.ndim==3:
                        model_dict[k][:a[0],:a[1],:a[2]]=v[:b[0],:b[1],:b[2]]
                    elif v.ndim==2:
                        model_dict[k][:a[0],:a[1]]=v[:b[0],:b[1]]
                    elif v.ndim==1:
                        model_dict[k][:a[0]]=v[:b[0]]
                    else:
                        print_selective("can not match in",model_dict[k].shape,v.shape,pass_=all_pass)
                        continue
                    print_selective("load finish",pass_=all_pass)
                    flexible_load_num+=1
                except:
                    print_selective("load error",pass_=all_pass)
                init_ot=True

        #new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来,更新了多少的参数
        print_selective(f"Total : {len(pretrained_dict)}, flexible num:{flexible_load_num} ,well num:{len(new_dict)} "+\
                        f"new tensor:{len(pretrained_dict)-flexible_load_num-len(new_dict)}",pass_=all_pass)
        self.load_state_dict(model_dict)
        print_selective("loaded finished!",pass_=all_pass)
        if init_ot:
            self.init_optimizer()
        # 如果不需要更新优化器那么设置为false
        
        ################################################################################################
        print_selective("load checkpoint Succeeded",pass_=all_pass)
        if data_save and not whole_save:
            self.save()
        if whole_save:
            tc.save(self,"whole_model.pth")
            print_selective("save whole model Succeeded",pass_=all_pass)
            
    def save(self,suffix="l_None_p_None",epoch_updata=1,name=None,file_path="G:/model_data/"):
        self.epoch+=epoch_updata
        if len(self.optimizer_list):
            optimizer_data=[x.state_dict() for x in self.optimizer_list]
        else :
            optimizer_data=[]
        
        if not os.path.exists(file_path):
            os.mkdir(f"{file_path}")
        if not bool(name):
            name=get_now_time(remove_year=True)+f"epoch{self.epoch}_{suffix}"
        
        tc.save({"epoch":self.epoch,"state_dict":self.state_dict(),
                    "optimizer":optimizer_data,"id":self.id,"epoch":self.epoch
                         },file_path+name+".pth")



def fbeta_score(preds, targets, threshold, beta=0.5, smooth=1e-5):
    preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
    y_true_count = targets.sum()
    
    ctp = preds_t[targets==1].sum()
    cfp = preds_t[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def find_TH_fbeta_score(preds:tc.Tensor,valid_mask_gt:tc.Tensor,show=False):
    preds=preds.flatten()
    valid_mask_gt=valid_mask_gt.flatten()
    bast_fbeta=0
    bast_TH=0
    for threshold in np.arange(0.1, 0.95, 0.05):
        fbeta=fbeta_score(preds,valid_mask_gt,threshold)
        if bast_fbeta<fbeta:
            bast_fbeta=fbeta
            bast_TH=threshold
        if show:
            print(f"Threshold : {threshold:.2f}\tFBeta : {fbeta:.6f}")
    if show:
        print(dice_coef_torch(preds,valid_mask_gt.flatten()))
        print(bast_fbeta,bast_TH)
    return bast_fbeta,bast_TH

########################################################################################################
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

    return 1-dice

def TTA(x:tc.Tensor,model):
    #x.shape=(batch,c,h,w)
    if CFG.TTA:
        batch=x.shape[0]
        x=[tc.rot90(x,k=i,dims=(-2,-1)) for i in range(4)]
        x=tc.cat(x,dim=0)

        x=model.AMP_forward(x)
        x=model.predict(model_output=x)
        if 4*batch==x.numel():
            x=x.reshape(4,batch)
            return x.mean(0)[:,None]
        else:
            x=x.reshape(4,batch,*x.shape[1:])
            x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
            x=tc.stack(x,dim=0)
            return x.mean(0)
    else :
        x=model.AMP_forward(x)
        x=model.predict(model_output=x)
        return x