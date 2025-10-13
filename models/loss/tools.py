import torch
from torch import nn
import torch.nn.functional as F
import random
import copy

import functools
import operator

class CategoricalPooling3D:
    def __init__(self, num_classes):
        print(num_classes)
        self.num_classes = num_classes

    def __call__(self, label, kernel_size, stride):
        if kernel_size == 1:
            return label.long()
        ds = kernel_size
        volume_size = (ds ** 3) if isinstance(ds,int) else functools.reduce(operator.mul, ds)
        # Calculate pooled counts of specific values
        count_0 = F.avg_pool3d((label == 0).float(), kernel_size=kernel_size, stride = stride)[:,0] * volume_size 
        count_255 = F.avg_pool3d((label == 255).float(), kernel_size=kernel_size, stride = stride)[:,0] * volume_size 
        
        # Calculate pooled labels excluding 0 and 255

        l = copy.deepcopy(label)
        l[l==255] = 0

        one_hot_label = F.one_hot(l[:,0].long(), num_classes= self.num_classes).float()
        one_hot_label = one_hot_label.permute(0, 4, 1, 2, 3)
        pooled_label = F.avg_pool3d(one_hot_label, kernel_size=kernel_size, stride = stride)
        empty_t = 0.96 * volume_size 

        label_downscale = torch.argmax(pooled_label[:,1:self.num_classes], dim=1) +1
        empty_mask = (count_0 + count_255) > empty_t
  
        label_downscale[empty_mask & (count_0>count_255)] = 0
        label_downscale[empty_mask & (count_0<=count_255)] = 255
        return label_downscale.long()[:,None]


