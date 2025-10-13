from importlib import import_module

import torch
import torch.nn as nn

from mmengine.config import Config
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS

class AdaptiveBatchNorm(nn.Module):
    def __init__(self, num_features, fc_dim, original_bn):
        super(AdaptiveBatchNorm, self).__init__()
        self.num_features = num_features
        # Standard BatchNorm layer
        self.batchnorm = nn.BatchNorm2d(num_features,affine = False)
        
        # Initialize with original BatchNorm statistics
        self.register_buffer('old_mean', original_bn.weight)
        self.register_buffer('old_bias', original_bn.bias)

        # Define FC layers to generate mean and bias adjustment factors
        self.fc_mean = nn.Linear(fc_dim, num_features)
        self.fc_bias = nn.Linear(fc_dim, num_features)

        # Initialize FC layers with the BatchNorm mean and bias to stabilize forward propagation
        self.fc_mean.weight.data.fill_(0)
        self.fc_mean.bias.data = original_bn.weight.clone()
        self.fc_bias.weight.data.fill_(0)
        self.fc_bias.bias.data = original_bn.bias.clone()
        self.eps = 1e-5

    def forward(self, x, fc_input =None):
        # Apply BatchNorm first
        x = self.batchnorm(x)
        # Generate adaptive mean and bias from fc_input
        if fc_input is not None:
            mean_adj = self.fc_mean(fc_input).view(1, -1, 1, 1)
            bias_adj = self.fc_bias(fc_input).view(1, -1, 1, 1)
        else:
            mean_adj = self.old_mean[None, :, None, None]
            bias_adj = self.old_bias[None, :, None, None]
        # Scale and shift
        x = x * mean_adj + bias_adj
        return x

def replace_batchnorm_with_adaptive(module, fc_dim):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            setattr(module, name, AdaptiveBatchNorm(num_features, fc_dim, child))
        else:
            replace_batchnorm_with_adaptive(child, fc_dim)


from mmdet.models.backbones.resnet import BasicBlock
class CustomBasicBlock(BasicBlock):
    def __init__(self, **kwargs):
        super(CustomBasicBlock, self).__init__(**kwargs)
    def forward(self, x,addbn_input =None):
        # 修改 BasicBlock 的前向传播逻辑
        # 你可以在此插入额外的操作，如自适应批归一化等
        def _inner_forward(x,addbn_input =None):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out,addbn_input)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out,addbn_input)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x,addbn_input)
        out = self.relu(out)
        return out
from mmdet.models.backbones.resnet import Bottleneck
class CustomBottleneck(Bottleneck):
    def __init__(self, **kwargs):
        super(CustomBottleneck, self).__init__(**kwargs)
    def forward(self, x,addbn_input =None):
        # 修改 BasicBlock 的前向传播逻辑
        # 你可以在此插入额外的操作，如自适应批归一化等
        def _inner_forward(x,addbn_input =None):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out,addbn_input)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out,addbn_input)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x,addbn_input)
        out = self.relu(out)
        return out
from mmdet.models.backbones import ResNet
class CustomResNet(ResNet):
    def __init__(self, **kwargs):
        super(CustomResNet, self).__init__(**kwargs)

    def forward(self, x,addbn_input = None):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x,addbn_input)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x,addbn_input)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
class MMDetWrapper(nn.Module):

    def __init__(self,
                 checkpoint_path,
                 freeze=True,
                 scales = [4, 8, 16]):
        super().__init__()
        import_module("maskdino")
        config = Config.fromfile("maskdino/configs/maskdino_r50_8xb2-panoptic-export.py")
        self.hidden_dims = config.model.panoptic_head.decoder.hidden_dim
        self.model = MODELS.build(config.model)
        #for n,p in self.model.named_parameters():
        #    print(n)
        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu'))
            )  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM
        self.model.panoptic_head.predictor = None
        self.scales = scales

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, x):
        # TODO: The following is only devised for the MaskDINO implementation.
        feats = self.model.extract_feat(x)
        mask_feat, _, multi_scale_feats = self.model.panoptic_head.pixel_decoder.forward_features(
            feats, masks=None)
        feats = (feats[0], *multi_scale_feats[:int(len(self.scales)-1)])
        feats ={f"1_{s}": f for s,f in zip(self.scales,feats)}
        return feats
    def filter_topk_queries(self, queries):
        scores = self.class_embed(queries)
        indices = scores.max(-1)[0].topk(self.filter_topk, sorted=False)[1]
        return self._batch_indexing(queries, indices), indices

    def pred_box(self, bbox_embed, hs, reference):
        delta_unsig = bbox_embed(hs)
        outputs_unsig = delta_unsig + inverse_sigmoid(reference)
        return outputs_unsig.sigmoid()

    def _batch_indexing(self, x, indices):
        """
        Args:
            x: shape (B, N, ...)
            indices: shape (B, N')
        Returns:
            shape (B, N', ...)
        """
        return torch.stack([q[i] for q, i in zip(x, indices)])
