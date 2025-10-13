import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributed import all_reduce
from torch import nn
import numpy as np
import math
from inspect import isfunction
from misc import get_obj_from_str,all_gather
from torch.optim.lr_scheduler import MultiStepLR

import copy
from easydict import EasyDict as edict
import yaml
import os 
from einops import rearrange
from models.loss.tools import CategoricalPooling3D
from models.loss.sscMetrics import SSCMetrics
from models.loss.ssc_loss import sem_scal_loss, CE_ssc_loss,  geo_scal_loss
#from layers.Voxel_Level.gen_denoise import Denoise
#from utils.loss import *
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1,mask = None):
    if mask is not None:
        x = x*mask.float()
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes,eps = 1e-30):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = x_onehot.float().clamp(min=eps) #torch.log(x_onehot.float().clamp(min=eps))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)

    return alphas

def gaussian_kernel(size, sigma):
    kernel = torch.Tensor(size, size)
    mean = size // 2
    sum_val = 0.0

    for i in range(size):
        for j in range(size):
            kernel[i, j] = torch.exp(-((i - mean)**2 + (j - mean)**2) / (2 * sigma**2))
            sum_val += kernel[i, j]

    # 归一化核
    kernel /= sum_val
    kernel = kernel.view(1, 1, size, size)
    return kernel

def gaussian_kernel_3d(size, sigma):
    """生成 3D 高斯核"""
    sigma = torch.tensor(sigma, dtype=torch.float32)
    kernel = torch.Tensor(size, size, size)
    mean = size // 2
    sum_val = 0.0

    for i in range(size):
        for j in range(size):
            for k in range(size):
                kernel[i, j, k] = torch.exp(-((i - mean)**2 + (j - mean)**2 + (k - mean)**2) / (2 * sigma**2))
                sum_val += kernel[i, j, k]

    # 归一化核
    kernel /= sum_val
    kernel = kernel.view(1, 1, size, size, size)
    return kernel

class wrapper(torch.nn.Module):
    def __init__(self, config):
        super(wrapper, self).__init__()
        self.mode  = config.mode
        self.config = config
        self.lr =config.lr
        self.weight_decay = config.weight_decay
        self.dataset = config.dataset
        self.milestones = config.milestones
        self.num_classes = config.n_classes
        self.class_names = config.class_names
        self.class_weights = config.class_weights
        self.sep_loss = config.sep_loss
        self.repeat = 1
        print("sep_loss?:",self.sep_loss)
        self.vfnet = get_obj_from_str(config.vfnet)(config)
        self.mae = get_obj_from_str(config.mae)(config) 

        self.eps = config.eps if "eps" in config else 1e-30
        self.metrics = {"val": SSCMetrics(self.num_classes), "val_c":SSCMetrics(self.num_classes),"val_n": SSCMetrics(self.num_classes)
                        , "val_m": SSCMetrics(self.num_classes), "val_f":SSCMetrics(self.num_classes)}
        self.b = 0 #0.45

        self.constant_std = 0.5
        self.downsample = self.vfnet.fold_size
        self.D = self.vfnet.n_bins
        self.cam_depth_range = [0, 51.2, 0.2*256/self.D] #[2.0, 58.0, 0.5] 
        
        
        print(self.dataset,"num_cls:",self.num_classes,self.eps)

    def device(self):
        return self.vfnet.device
    def final_loss(self,x,target,aux = "",loss = 0,loss_dict = {},b = 0):
        class_weight = self.class_weights.type_as(x)
        loss_ssc = CE_ssc_loss(x , target, class_weight,b = b)
        #print(b)
        loss += loss_ssc
        loss_dict["loss_ssc" + aux] = loss_ssc.detach()
    
        loss_sem_scal = sem_scal_loss(x, target,b = b)
        loss += loss_sem_scal
        loss_dict[ "loss_sem_scal"+ aux] = loss_sem_scal.detach()
        
        loss_geo_scal = geo_scal_loss(x, target,b = b)
        loss += loss_geo_scal
        loss_dict[ "loss_geo_scal"+ aux] = loss_geo_scal.detach()
        return loss,loss_dict

    def get_gt_depth(self,batch,valid_mask):
        img,proj_points,fov_masks = batch["img"],batch["proj_uvd"][0],batch["fov_mask"][0] 
        occ = batch["voxel_label"].long().flatten(1)
        valid_mask = fov_masks & (occ !=0) & (occ!= 255)
        #(valid_mask.flatten(1)) & (occ !=0) & (occ!= 255)
        #fov_masks & (occ !=0) & (occ!= 255)
        gt_depth = torch.zeros_like(img)[:,:1]

        for i,(pts,m) in enumerate(zip(proj_points,valid_mask)):
            valid = torch.nonzero(m)[:,0]
            pts = pts[valid]
            depth_order = torch.argsort(pts[:, 2], descending=True)
            pts = pts[depth_order]
            # fill in
            x_idx = pts[:, 0].round().long().clamp(0,gt_depth.shape[-1] -1)
            y_idx = pts[:, 1].round().long().clamp(0,gt_depth.shape[-2] -1)
            gt_depth[i,0,y_idx,x_idx] = pts[:, 2] 
        return gt_depth
    def forward(self,  batch,step_type = "train", *args, **kwargs):
        loss,loss_dict = 0,dict()
        vis = {}
        y_true  = batch["voxel_label"].long()
        b,device = y_true.shape[0],y_true.device
        cond = self.vfnet(batch)
        valid_mask = cond["valid"]

        if step_type == "test":
            outputs = self.mae(cond)
            ssc_pred = outputs["ssc_pred"]
            y_pred = ssc_pred.detach().softmax(1).argmax(1)
            return y_pred
        
        #print((valid_mask!=0).sum())
        if "depth_volume" in cond :
            depth_volume = cond["depth_volume"]
            gt_depth = self.get_gt_depth(batch,valid_mask )
            depth_loss = self.get_klv_depth_loss(gt_depth,depth_volume)
            depth_loss = depth_loss *0.001
            loss += depth_loss
            loss_dict["depth_loss"] = depth_loss.detach()
        if "ssc_pred" in cond :
            t1 = copy.deepcopy(y_true)
            t1[~valid_mask] = 255
            ssc_coarse = cond["ssc_pred"]
            loss,loss_dict = self.final_loss(ssc_coarse,t1,aux = "_coarse",loss= loss, loss_dict = loss_dict,b = self.b) #,b = self.b
            if step_type == "val":
                coarse = ssc_coarse.detach().softmax(1).argmax(1)
                self.metrics["val_c"].add_batch(self.all_gather(coarse),self.all_gather(t1))
            
        if self.mae is not None:
            for i in range(self.repeat):
                outputs = self.mae(cond)
                cond["ssc_pred"] = outputs["ssc_pred"]
            ssc_pred = outputs["ssc_pred"]
            y_pred = ssc_pred.detach().softmax(1)
            y_pred[:,1:] = y_pred[:,1:]#*3
            y_pred = y_pred.argmax(1)
            t2 = copy.deepcopy(y_true)
            if self.sep_loss:
                t2[valid_mask] = 255
                y_pred[valid_mask] = ssc_coarse.detach().softmax(1).argmax(1)[valid_mask]
            loss,loss_dict = self.final_loss(ssc_pred,t2,loss= loss, loss_dict = loss_dict,b = self.b)
            if step_type == "val":
                self.update_metric(y_pred,y_true)
            vis["y_pred"] = y_pred
        vis["y_true"] = y_true
        cond["vis"] = vis
        return cond,loss,loss_dict
    
    def all_gather(self,x_):
        x = all_gather(x_)#.flatten()
        x = [p.to(x_.device) for p in x]
        x = torch.cat(x).cpu().numpy()
        return x

    def update_metric(self,y_pred,y_true):
        preds = self.all_gather(y_pred)
        gts = self.all_gather(y_true)
        self.metrics["val"].add_batch( preds, gts)
        self.metrics["val_n"].add_batch( preds[:,:64,96:160], gts[:,:64,96:160])
        self.metrics["val_m"].add_batch(preds[:,:128,64:192], gts[:,:128,64:192])
        self.metrics["val_f"].add_batch( preds[:,170:], gts[:,170:])
    @classmethod
    def init_from_path(cls_ ,path):
        with open(os.path.join(path,"conf.yaml"), "r") as f:
            config = edict(yaml.safe_load(f))
        generator  = cls_(config)
        weight  = torch.load(os.path.join(path,"checkpoints","last.ckpt"), map_location='cpu')["model"]
        mis,uxp = generator.load_state_dict(weight,strict = False)
        print("load form:",path ,mis,uxp)
        return generator
        
    def configure_optimizers(self):
        param_dicts = [ 
            {"params": [p for n, p in self.named_parameters() if "net_rgb" not in n  and p.requires_grad]},
            {
            "params": [p for n, p in self.named_parameters() if "net_rgb" in n and p.requires_grad],
            "lr": self.lr/10,
            },
            ]#backbone.layer1
        #print(param_dicts[1])
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
        return optimizer, scheduler


    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)
        #print(depth_gaussian_labels.shape,depth_values.shape)
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))        
        
        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]
        
        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
        
        return depth_loss


def generate_guassian_depth_target(depth, stride, cam_depth_range, constant_std=None):
    depth = depth.flatten(0, 1)
    B, tH, tW = depth.shape
    kernel_size = stride
    center_idx = kernel_size * kernel_size // 2
    H = tH // stride
    W = tW // stride
    
    unfold_depth = F.unfold(depth.unsqueeze(1), kernel_size, dilation=1, padding=0, stride=stride) #B, Cxkxk, HxW
    unfold_depth = unfold_depth.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous() # B, H, W, kxk
    valid_mask = (unfold_depth != 0) # BN, H, W, kxk
    
    if constant_std is None:
        valid_mask_f = valid_mask.float() # BN, H, W, kxk
        valid_num = torch.sum(valid_mask_f, dim=-1) # BN, H, W
        valid_num[valid_num == 0] = 1e10
        
        mean = torch.sum(unfold_depth, dim=-1) / valid_num
        var_sum = torch.sum(((unfold_depth - mean.unsqueeze(-1))**2) * valid_mask_f, dim=-1) # BN, H, W
        std_var = torch.sqrt(var_sum / valid_num)
        std_var[valid_num == 1] = 1 # set std_var to 1 when only one point in patch
    else:
        std_var = torch.ones((B, H, W)).type_as(depth).float() * constant_std

    unfold_depth[~valid_mask] = 1e10
    min_depth = torch.min(unfold_depth, dim=-1)[0] #BN, H, W
    min_depth[min_depth == 1e10] = 0
    
    # x in raw depth 
    x = torch.arange(cam_depth_range[0] - cam_depth_range[2] / 2, cam_depth_range[1], cam_depth_range[2])
    # normalized by intervals
    dist = Normal(min_depth / cam_depth_range[2], std_var / cam_depth_range[2]) # BN, H, W, D
    cdfs = []
    for i in x:
        cdf = dist.cdf(i)
        cdfs.append(cdf)
    
    cdfs = torch.stack(cdfs, dim=-1)
    depth_dist = cdfs[..., 1:] - cdfs[...,:-1]
    
    return depth_dist, min_depth