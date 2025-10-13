# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import pickle
import torch
from torch import autograd
import misc as utils
import torch
from tqdm import tqdm
import numpy as np 
#from vis import draw_once
import torch.nn.functional as F
import matplotlib.pyplot as plt

def init_engine(config):
    train_engine = globals()[config.train_engine]
    val_engine = globals()[config.val_engine]
    return train_engine,val_engine
# 创建一个钩子函数来捕获梯度
def normalize_to_255(image):
    image = (image - image.min()) / (image.max() - image.min())  # 归一化到 0-1
    image = (image * 255).astype(np.uint8)                      # 转换为 0-255 的整数
    return image

def draw(image,output_dir):
    plt.figure(figsize=(5, 5))
    plt.imshow(normalize_to_255(image), cmap='gray')  # 使用灰度颜色映射
    #plt.title("Mean Gradient (Grayscale)")
    plt.colorbar()
    plt.savefig(output_dir,os.path.join(output_dir, "mean_gradient.png"))
    plt.close()

def vis_grad(model,data_loader, epoch):
    model.train()
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 5

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        gradients,loss,loss_dict= model.training_step(batch,epoch = epoch)
        loss_dict.update({"loss":loss})
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update( **loss_dict_reduced)
        metric_logger.update(lr=model.optimizer.param_groups[0]["lr"])
        del loss,loss_dict_reduced

    gradients = torch.stack(gradients)
    gradients = utils.all_gather(gradients)
    gradients = torch.cat(gradients,0)
    mean_grad = gradients.mean(dim=0).numpy()
    std_grad = gradients.std(dim=0).numpy()
    
    if utils.is_main_process():
        draw(mean_grad,os.path.join(output_dir, "mean_gradient.png"))
        draw(std_grad,os.path.join(output_dir, "std_gradient.png"))

    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch(model,data_loader, epoch):
    model.train()
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 5

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        #with torch.autograd.set_detect_anomaly(True):
        loss,loss_dict= model.training_step(batch,epoch = epoch)
        loss_dict.update({"loss":loss})
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            print(loss_dict_reduced)
            sys.exit(1)
        metric_logger.update( **loss_dict_reduced)
        metric_logger.update(lr=model.optimizer.param_groups[0]["lr"])
        del loss,loss_dict_reduced
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_noloss(model,data_loader,path = None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'val:'
    print_freq = 10
    for ii,batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if ii<=0 and path is not None:
            _ = model.validation_step(batch,path =path )
        else:
            _ = model.validation_step(batch )
        if ii ==1:
            break
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stats.update(model.validation_epoch_end())
    return test_stats


@torch.no_grad()
def test_standard(model,data_loader, path = None,epoch = None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'test:'
    print_freq = 10
    print(len(data_loader))
    for ii,batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        model.testing_step(batch,epoch = epoch)
    test_stats = {}
    return test_stats

@torch.no_grad()
def evaluate_standard(model,data_loader, path = None,epoch = None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'val:'
    print_freq = 10
    for ii,batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if ii<=0 and path is not None:
            loss,loss_dict = model.validation_step(batch,path =path )
        else:
            loss,loss_dict = model.validation_step(batch )
        loss_dict.update({"loss" : loss})
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)#统一分布式训练的结果
        metric_logger.update( **loss_dict_reduced)
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stats.update(model.validation_epoch_end())
    return test_stats

@torch.no_grad()
def evaluate_tqdm(model,data_loader, path = None,epoch = None):
    model.eval()
    for ii,batch in tqdm(enumerate(data_loader)):
        if ii<=0 and path is not None:
            loss,loss_dict = model.validation_step(batch,path =path )
        else:
            loss,loss_dict = model.validation_step(batch )
        loss_dict.update({"loss" : loss})
    test_stats = {}
    test_stats.update(model.validation_epoch_end())
    return test_stats