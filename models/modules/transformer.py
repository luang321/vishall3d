
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from einops import rearrange
from einops.layers.torch import Rearrange

'''
from dfa3D.ops.multi_scale_3D_deform_attn import (
    WeightedMultiScaleDeformableAttnFunction,
    MultiScaleDepthScoreSampleFunction, MultiScale3DDeformableAttnFunction
)'''
from .dfa import  MultiScale3DDeformableAttnFunction_fp32 as MultiScale3DDeformableAttnFunction

def constant_init(model, val):
    """将模型中所有的权重初始化为常数值"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.constant_(param, val)
class DeformableAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=4, num_levels=4, num_points=8, im2col_step=64, batch_first=True):
        super(DeformableAttention, self).__init__()

        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.sampling_offsets_depth = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 1)

            
        self.init_smpl_off_weights()

    def init_smpl_off_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets_depth, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([(thetas.cos() + thetas.sin()) / 2], -1)
        grid_init = grid_init.view(self.num_heads, 1, 1, 1).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets_depth.bias.data = grid_init.view(-1)

    def forward(self, query,
                value=None,
                value_dpt_dist=None,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            value_dpt_dist(Tensor): The depth distribution of each image feature (value), with shape
                `(bs, num_key,  D)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        _, _, dim_depth = value_dpt_dist.shape
        value_dpt_dist = value_dpt_dist.view(bs, num_value, 1, dim_depth).repeat(1,1,self.num_heads, 1)
        spatial_shapes_3D = self.get_spatial_shape_3D(spatial_shapes, dim_depth)

        sampling_offsets_uv = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        sampling_offsets_depth = self.sampling_offsets_depth(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 1)
        sampling_offsets = torch.cat([sampling_offsets_uv, sampling_offsets_depth], dim = -1)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 3:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes_3D[..., 1], spatial_shapes_3D[..., 0], spatial_shapes_3D[..., 2]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            sampling_locations_ref = reference_points.repeat(1,1,num_heads,num_levels,num_points,1,1)
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
            sampling_locations_ref = sampling_locations_ref.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        
        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output, depth_score = MultiScale3DDeformableAttnFunction.apply(
                value, value_dpt_dist, spatial_shapes_3D, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output
    def get_spatial_shape_3D(self, spatial_shape, depth_dim):
        spatial_shape_depth = spatial_shape.new_ones(*spatial_shape.shape[:-1], 1) * depth_dim
        spatial_shape_3D = torch.cat([spatial_shape, spatial_shape_depth], dim=-1)
        return spatial_shape_3D.contiguous()

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_levels, im2col_step= 8):
        super(DeformableTransformerEncoderLayer, self).__init__()
        self.attn = DeformableAttention( embed_dims=embed_dim, num_heads=num_heads, num_levels=num_levels,  im2col_step=im2col_step)
        ff_dim = int(2*embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self,x,
                value=None,
                value_dpt_dist=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None):
        attn_output = self.attn(query = x, value = value,value_dpt_dist = value_dpt_dist,reference_points = reference_points
                        ,spatial_shapes = spatial_shapes,level_start_index = level_start_index)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
def flatten_fov_from_voxels(x3d, fov_mask):
    assert x3d.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    return x3d.flatten(2)[..., fov_mask].transpose(1, 2)

def index_fov_back_to_voxels(x3d, fov, fov_mask):
    assert x3d.shape[0] == fov.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    fov_concat = torch.zeros_like(x3d).flatten(2)
    fov_concat[..., fov_mask] = fov.transpose(1, 2)
    return torch.where(fov_mask, fov_concat, x3d.flatten(2)).reshape(*x3d.shape)

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, num_levels, im2col_step= 64):
        super(DeformableTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(embed_dim, num_heads, num_levels, im2col_step= im2col_step)
            for _ in range(num_layers)
        ])

    def forward(self, scene_embed,
                value=None,
                value_dpt_dist=None,
                reference_points=None,
                fov_mask = None):
        assert len(value) == 1
        bs, c, h, w = value.shape
        value = rearrange(value,"b c h w -> b (h w) c").contiguous()
        value_dpt_dist = rearrange(value_dpt_dist,"b c h w -> b (h w) c").contiguous()

        spatial_shapes = torch.as_tensor([(h, w),], dtype=torch.long, device=scene_embed.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        reference_points = reference_points[:, fov_mask]
        x = flatten_fov_from_voxels(scene_embed, fov_mask)
        for layer in self.layers:
            x = layer(x,value = value,value_dpt_dist = value_dpt_dist,reference_points = reference_points[:,:,None]
                        ,spatial_shapes = spatial_shapes,level_start_index = level_start_index)

        scene_embed = index_fov_back_to_voxels(scene_embed, x, fov_mask) + scene_embed
        return scene_embed

