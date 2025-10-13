
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.ops as ops
from functools import reduce
from einops import rearrange
from einops.layers.torch import Rearrange
import time

import copy
import datetime
import os
from .mmdet_wrapper import MMDetWrapper
from .unet3d import UNet3D
from .transformer import DeformableTransformerEncoder



def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def generate_meshgrid(features):
    b, c, h, w = features.shape
    v_coords, u_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    u_coords = u_coords.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    v_coords = v_coords.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    meshgrid = torch.cat((u_coords, v_coords), dim=1)
    
    return meshgrid

def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)

def pix2cam(p_pix, depth, K):
    p_pix = torch.cat([p_pix * depth, depth], dim=1)  # bs, 3, h, w
    return K.inverse() @ p_pix.flatten(2)

def cam2vox(p_cam, E, vox_origin, vox_size, offset=0.5):
    p_wld = E.inverse() @ F.pad(p_cam, (0, 0, 0, 1), value=1)
    p_vox = (p_wld[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(0).unsqueeze(0)) / vox_size - offset
    return p_vox

def pix2vox(p_pix, depth, K, E, vox_origin, vox_size, offset=0.5, downsample_z=1):
    p_cam = pix2cam(p_pix, depth, K)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size, offset)
    if downsample_z != 1:
        p_vox[..., -1] /= downsample_z
    return p_vox
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum = 0.1):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
        )
        self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.main(x)
        x = self.relu(self.norm(x))
        return x

class vfnet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.dim_2d = config.dim_unet #config.dim_2d
        dim = config.dim_unet
        self.fold_size = 8
        self.pool_size = 1 
        self.stride =  config.strides[0]
        s = 2
        bn_momentum =  0.0003
        self.n_bins = int(256//s)
        self.register_buffer('boundaries', torch.linspace(0, 51.2, self.n_bins + 1))

        self.freeze_backbone  = config.freeze_backbone
        self.build_rgb_net(config.net_rgb)
        self.mini_unet = SimpleUnet(int(self.dim_2d + self.fold_size**2),dim)
        self.hard = config.hard_assign if "hard_assign" in config else False
        #nn.Sequential(nn.Conv2d(int(self.dim_2d + self.fold_size**2),dim, kernel_size=3,padding = 1),nn.BatchNorm2d(dim),nn.ReLU(),
        #                            nn.Conv2d(dim,dim,kernel_size=3,padding = 1),nn.BatchNorm2d(dim),nn.ReLU(),
        #                            nn.Conv2d(dim,dim,kernel_size=3,padding = 1))  
        ##SimpleUnet(int(self.dim_2d + self.fold_size**2),dim//2)
        self.depth_net = nn.Sequential(nn.Conv2d(dim,dim,1),nn.ReLU(),nn.Conv2d(dim,self.n_bins,1))
        self.smooth = nn.Identity()
        #nn.Conv3d(dim//2,dim,1)
        d = dim //2
        s = self.stride//4
        self.transformer = DeformableTransformerEncoder(num_layers = 3, embed_dim = dim, num_heads = 4, num_levels =1)
        #nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3,padding = 1),nn.BatchNorm3d(dim),nn.ReLU()
        #                ,nn.Conv3d(dim, dim, kernel_size=3,padding = 1),nn.BatchNorm3d(dim),nn.ReLU(),
        #                nn.Conv3d(dim, dim, kernel_size=3,padding = 1),nn.BatchNorm3d(dim),nn.ReLU())
        self.unet3d = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3,padding = 1),nn.BatchNorm3d(dim,momentum =bn_momentum),nn.ReLU()
                        ,nn.Conv3d(dim, dim, kernel_size=3,padding = 1),nn.BatchNorm3d(dim,momentum =bn_momentum),nn.ReLU(),
                        nn.Conv3d(dim, dim, kernel_size=3,padding = 1),nn.BatchNorm3d(dim,momentum =bn_momentum),nn.ReLU())
        
        # UNet3D(dim)
        self.head =  nn.Sequential(Upsample(dim, d),Upsample(d, d),nn.Conv3d(d, d, kernel_size=1),nn.ReLU(),nn.Conv3d(d, int(config.n_classes*(s**3)), kernel_size=1),
                            Rearrange("b (p1 p2 p3 c) h w d -> b c (h p1) (w p2) (d p3)",p1=s,p2=s,p3=s))
        #

        self.fc_2d = nn.Sequential(
            nn.Linear(int(self.dim_2d*(self.pool_size**2)), dim,bias = True),nn.ReLU(),nn.Linear(dim,dim,bias = True),nn.ReLU(),nn.Linear(dim,dim,bias = True))
        input_dim = int(self.stride**3)
        self.fc_3d = nn.Sequential(
            nn.Conv3d(input_dim, dim,1,stride = 1 ,padding = 0),
            nn.BatchNorm3d(dim,momentum =bn_momentum),
            nn.ReLU(),
            nn.Conv3d(dim, dim,3,stride = 1 ,padding = 1),
            nn.BatchNorm3d(dim,momentum =bn_momentum),
            nn.ReLU(),
            nn.Conv3d(dim, dim,3,stride = 1 ,padding = 1))

        self.feat_shape = [int(i//self.stride) for i in config.full_scene_size]
        self.scene_shape = config.full_scene_size
        print(self.scene_shape,self.stride)

        if self.hard:
            image_shape = [384,1280]
            image_grid = generate_grid(image_shape)
            image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
            self.register_buffer('image_grid', image_grid)
            self.register_buffer("voxel_origin" ,torch.tensor([0, -25.6, -2]))
            self.voxel_size = 0.2
        
    def build_rgb_net(self,name):
        self.project_res = [4, 8, 16,32]
        self.net_rgb = MMDetWrapper(
            checkpoint_path = os.path.join(os.getcwd(),"backbone_pth","maskdino_r50_50e_300q_panoptic_pq53.0.pth"),freeze= self.freeze_backbone,scales = self.project_res)
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.net_rgb.hidden_dims, self.dim_2d, 1),
                nn.BatchNorm2d(self.dim_2d),
                nn.ReLU(inplace=True),
            ) for _ in self.project_res
        ])

    def forward_2d(self,img_2):
        if self.freeze_backbone:
            with torch.no_grad():
                x_rgb = self.net_rgb(img_2)
        else:
            x_rgb = self.net_rgb(img_2)
        fs = list()
        fs_raw = list()
        for ii, s in enumerate(self.project_res):
            f = self.out_projs[ii](x_rgb["1_" + str(s)].contiguous()).contiguous()
            fs_raw.append(f)
            if s != self.fold_size:
                f = F.interpolate(f,scale_factor=s/self.fold_size, mode='bilinear')
            fs.append(f)
        fs = sum(fs)
        return fs,fs_raw
    def pool(self,fs,boxes):
        b,n,_ = boxes.shape
        roi_boxes = [b[...,:4] for b in boxes]
        roi_output = ops.roi_pool(
                fs,  
                roi_boxes,  
                output_size= self.pool_size,  
                spatial_scale=1/self.fold_size
            )
        w,h,d = self.feat_shape
        x = self.fc_2d(roi_output.flatten(1)).view(b,n,-1)
        x = rearrange(x,"b (w h d) c  -> b c w h d",w=w,h=h,d=d).contiguous()
        return x
    def hard_assign(self, depth,K,E):
        w,h,d = self.scene_shape
        b = depth.shape[0]
        x = torch.zeros(b,w,h,d).to(depth)
        vol_pts = pix2vox(
            self.image_grid,
            depth,
            K,
            E,
            self.voxel_origin,
            self.voxel_size,).long()

        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))
        assert vol_pts.shape[0] == 1
        keep = torch.nonzero(keep)
        geom = vol_pts[keep[:,0],keep[:,1]]
        x[keep[:,0],geom[:, 0], geom[:, 1], geom[:, 2]] = 1
        return x
    def get_vf(self,depth,K,E,sample_point =None):
        w,h,d = self.scene_shape
        heat_maps = list()
        sample_point = copy.deepcopy(sample_point)
        sample_point,sample_d = sample_point[...,:2],sample_point[...,2:]
        sample_point[...,0] = sample_point[...,0]/depth.shape[-1]
        sample_point[...,1] = sample_point[...,1]/depth.shape[-2]
        sample_point = sample_point.clamp(0,1)*2-1
        
        dep = F.grid_sample(depth, sample_point.unsqueeze(1), mode='bilinear', align_corners=True,padding_mode = 'zeros')
        dep = rearrange(dep,"b c k n -> b n (c k)").contiguous()
        dis = (dep - sample_d).abs() 
        ratio = (dis /dep)

        invalid =  ((dis > 0.2) & (ratio > 0.1)) | (dis > 1)
        x = (2 - 2*((10*dis).sigmoid()))
        x[invalid] = 0   
        #print(x.shape,invalid.sum())
        x = rearrange(x,"b (w h d) c  -> b c w h d",w=w,h=h,d=d).contiguous()
        if self.hard:
            x = self.hard_assign(depth,K,E)
        else:
            x = x[:,0]
        heat_maps.append(x)
        x = rearrange(x,"b (w p1) (h p2) (d p3) ->b (p1 p2 p3) w h d ",p1 =  self.stride,p2 =  self.stride,p3 =  self.stride).contiguous()
        return x,heat_maps
    def sample(self,img,sample_point):
        sample_point = copy.deepcopy(sample_point).float()
        sample_point = sample_point[...,:2]
        sample_point[...,0] = sample_point[...,0]/img.shape[-1]
        sample_point[...,1] = sample_point[...,1]/img.shape[-2]
        sample_point = sample_point.clamp(0,1)*2-1
        w,h,d = self.scene_shape

        sample = F.grid_sample(img, sample_point.unsqueeze(1), mode='bilinear', align_corners=True,padding_mode = 'zeros')
        sample = rearrange(sample,"b c k n -> b n (c k)").contiguous()
        sample = rearrange(sample,"b (w h d) c  -> b c w h d",w=w,h=h,d=d).contiguous()
        return sample

    def get_anchor(self,batch):
        w,h,d = self.scene_shape
        depth_map = batch["depth"]
        K,E = batch["K"].float(),batch["E"].float()
        fov_mask = batch["fov_mask"][0]
        fov_mask = rearrange(fov_mask[...,None],"b (w h d) c -> b c w h d",w=w,h=h,d=d).contiguous()
        
        sample_point,sample_d = batch[ "proj_uvd"][0][...,:2],batch[ "proj_uvd"][0][...,2:]
        depth = self.sample(depth_map,sample_point)
        sample_d = rearrange(sample_d,"b (w h d) c  -> b c w h d",w=w,h=h,d=d).contiguous()
        dis = sample_d - depth
        valid = dis<3.5 #0.5  #

        return  (valid.bool()&fov_mask)[:,0] #fov_mask[:,0] #

    def norm(self,coord,x):
        _,_,d,h,w = x.shape
        coord[...,0] = coord[...,0]/w
        coord[...,1] = coord[...,1]/h
        coord[...,2] = coord[...,2]/d
        coord = coord.clamp(0,1)
        return coord*2-1

    def get_depth(self,feat_2d,depth):
        d = rearrange(depth,"b c (h p1) (w p2) -> b (p1 p2 c) h w",p1 = self.fold_size,p2= self.fold_size)
        x = torch.cat((feat_2d,d),1)
        x = self.mini_unet(x)
        depth_volume = self.depth_net(x).softmax(1)
        return x,depth_volume

    def forward(self,batch):
        imgs,depth,fov_mask = batch["img"],batch["depth"],batch["fov_mask"][1]
        K,E = batch["K"].float(),batch["E"].float()
        img_2 = imgs[:,:3]
        b = imgs.shape[0]
        h,w,d = self.feat_shape
        feat_2d,feat_2d_raw = self.forward_2d(img_2)
        x,depth_volume = self.get_depth(feat_2d,depth)
        geo,heat_maps = self.get_vf(depth,K,E,sample_point = batch[ "proj_uvd"][0])

        if self.hard:
            geo = geo.mean(1,keepdim = True) >0.05
            src = self.pool(feat_2d,batch["box_xyxy"][0].float())*geo
        else:
            src = self.pool(feat_2d,batch["box_xyxy"][0].float()) + self.fc_3d(geo) 
        #print("good luck")
        
        reference_points = batch[ "proj_uvd"][1]
        reference_points[...,0] = reference_points[...,0]/1280 
        reference_points[...,1] = reference_points[...,1]/384
        reference_points[...,2] = reference_points[...,2]/51.2
        reference_points = reference_points.clamp(0,1)
        src = self.transformer(src,value=x,value_dpt_dist=depth_volume,reference_points=reference_points,fov_mask = fov_mask.flatten())
        ssc_pred = self.head(self.unet3d(src))

        cond = dict()
        cond["heat_maps"] = heat_maps
        cond["ssc_pred"] = ssc_pred
        cond["depth_volume"] = depth_volume
        cond["src"] = src
        cond["valid"] = self.get_anchor(batch)
        return cond


def convbn_2d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=pad, bias=False),
        nn.BatchNorm2d( out_channels)
    )


class SimpleUnet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SimpleUnet, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            convbn_2d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.redir1 = convbn_2d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return self.out(conv6) 