

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
import copy
from functools import reduce

#from .unet3d import UNet3D
from .unet3d import UNet3D
########3d dit########
from models.loss.tools import CategoricalPooling3D
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
def inverse_sigmoid(x, eps=1e-4):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class HMAE(nn.Module):
    def __init__(
        self,config
    ):
        super().__init__()
        dim= config.dim_unet
        self.s = config.strides[0]
        self.n_classes = config.n_classes
        self.unet3d = UNet3D(dim)
        self.pool = CategoricalPooling3D(self.n_classes)
        d = dim //2
        s = self.s//4
        self.head =  nn.Sequential(Upsample(dim, d),Upsample(d, d),nn.Conv3d(d, d, kernel_size=1),nn.ReLU(),nn.Conv3d(d, int(config.n_classes*(s**3)), kernel_size=1),
                            Rearrange("b (p1 p2 p3 c) h w d -> b c (h p1) (w p2) (d p3)",p1=s,p2=s,p3=s))

        self.fc = nn.Sequential(nn.Embedding(self.n_classes + 2,int(dim//(self.s**2))),
                        Rearrange("b (h p1) (w p2) (d p3) c -> b (p1 p2 p3 c) h w d ",p1=self.s,p2=self.s,p3=self.s),
                        nn.Conv3d(int(dim*self.s), dim, kernel_size=1),nn.ReLU(),nn.Conv3d(dim, dim, kernel_size=1))
        self.t_embedder = nn.Sequential(
                        Rearrange("b (h p1) (w p2) (d p3) -> b (p1 p2 p3) h w d ",p1=self.s,p2=self.s,p3=self.s),
                        nn.Conv3d(self.s**3, dim//2, kernel_size=3,padding = 1),nn.ReLU(),
                        nn.Conv3d(dim//2, dim, kernel_size=3,padding = 1),nn.ReLU(),nn.Conv3d(dim, 2*dim, kernel_size=1, bias=False))
        self.BN = nn.BatchNorm3d(dim,affine  = False)

        self.c = config.c
        self.max_v,self.min_v = config.max_v,config.min_v
        print(self.c,self.max_v,self.min_v)

    def get_src(self,x):
        x = copy.deepcopy(x)
        x[x == 255] = self.n_classes
        x = self.fc(x)
        return x

    def smooth(self,x,ks = (3,1),pads= (1,0),p1 = 0.6,p2 = 0.1):
        x = copy.deepcopy(x)
        k1,k2 = ks
        pad1, pad2 = pads
        b, h, w, d = x.shape
        k = rearrange(x,"b h w (d c) -> (b d) c h w",c =1).contiguous()
        #padded_map = F.pad(k, (pad1, pad1, pad2 ,pad2), mode='constant', value=0)
        unfolded_map = F.unfold(
            k.float(),  
            kernel_size=(k1,k2),
            padding=(pad1,pad2)
        )  
        #print(unfolded_map.shape,k.shape,(k1,k2),(pad1,pad2))
        unfolded_map = rearrange(unfolded_map,"(b d) c (h w) -> b c h w d",h = h,w=w,d=d).contiguous().long()
        '''
        unfolded_map = rearrange(unfolded_map,"(b d) c (h w) -> (b h w) c d",h = h,w=w,d=d).contiguous()
        unfolded_map = F.pad(unfolded_map, (1, 1), mode='constant', value=0)
        unfolded_map = F.unfold(
            unfolded_map.float(),  
            kernel_size=3,
            padding=0
        )  
        unfolded_map = rearrange(unfolded_map,"(b h w) c d -> b c h w d",h = h,w=w,d=d).contiguous().long()
        '''
        random_matrix = torch.rand_like(x.float())
        mask = torch.where(
            x == 0,  
            random_matrix < p1,  # 值为0的点以概率p1为True
            random_matrix < p2   # 值不为0的点以概率p2为True
        )
        invalid = torch.nonzero((x == 255)|(~mask))
        unfolded_map[invalid[:,0],:,invalid[:,1],invalid[:,2],invalid[:,3]] = x[invalid[:,0],invalid[:,1],invalid[:,2],invalid[:,3]][:,None]

        #print(unfolded_map.shape)
        random_indices = torch.randint(0, int(k1*k2), (b, h, w, d)).to(x.device)
        selected_values = torch.gather(
            unfolded_map,  
            1,       
            random_indices.unsqueeze(1)  
        ).squeeze(1)  
        return selected_values.to(x)
    
    def depth_pool(self,x):
        max_var,min_var = self.max_v,self.min_v
        b, h, w, d = x.shape
        # 使用 meshgrid 生成 h, w, d 的索引
        b_idx = torch.arange(b, dtype=torch.long,device = x.device)
        h_idx = torch.arange(h, dtype=torch.long,device = x.device)  # [h]
        w_idx = torch.arange(w, dtype=torch.long,device = x.device)  # [w]
        d_idx = torch.arange(d, dtype=torch.long,device = x.device)  # [d]
        # 生成网格索引 [h, w, d]
        grid_b,grid_h, grid_w, grid_d = torch.meshgrid(b_idx,h_idx, w_idx, d_idx, indexing='ij')
        # 将网格索引展平为 [h * w * d, 3]
        batch_indices = torch.stack([grid_b, grid_h, grid_w, grid_d], dim=-1)  

        depth_idx = batch_indices[...,1:2].float()
        bias = (torch.rand_like(depth_idx)*2 -1)
        #scale = (depth_idx*(max_var-min_var)/h) + min_var
        bias = bias*max_var  #*scale
        #print(max_var)
        #print(torch.rand_like(depth_idx).mean(),torch.rand_like(depth_idx).max(),torch.rand_like(depth_idx).min(),depth_idx.min(),depth_idx.max(),bias.min(),bias.max(),bias.median())

        depth_idx  = (depth_idx + bias).clamp(0,h-1).long()
        batch_indices[...,1:2] = depth_idx
        new = x[batch_indices[...,0],batch_indices[...,1],batch_indices[...,2],batch_indices[...,3]]
        #print(new.shape)
        return new,bias[...,0]


    def sm(self,x):
        for i in range(self.c):
            x = self.smooth(x,ks = (3,3),pads= (1,1),p1 = 1,p2 = 1)
        x,bias = self.depth_pool(x)
        return x,bias

    def forward(self, cond):
        outputs = {}
        x,valid,ssc_coarse = cond["src"],cond["valid"],cond["ssc_pred"]
        valid_mask = valid
        anchor = ssc_coarse.detach().softmax(1).argmax(1)
        anchor[~valid_mask] = 255
        anchor,bias = self.sm(anchor)
        bias[~valid_mask] = 0
        #print(bias.shape)
        t = self.t_embedder(bias)
        src = self.get_src(anchor)
        scale,shift = t.chunk(2, dim=1)
        src = modulate(self.BN(src), shift, scale)
        #print(src.shape)
        x = x + src
        x = self.unet3d(x)
        ssc_pred = self.head(x)
        outputs.update({"ssc_pred":ssc_pred,"ssc_coarse":ssc_coarse,"valid_mask":valid_mask,"anchor":anchor})
        return outputs

