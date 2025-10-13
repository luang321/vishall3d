import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,4"
from datasets.dm import DataModule

from tqdm import tqdm
import numpy as np
import torch
import misc as utils

from engine import init_engine
import json
import time
import datetime
import yaml
from easydict import EasyDict as edict
from datasets.dm import DataModule
from models.trainer import  trainer
def main(config_path):
    print(config_path)
    with open(config_path, "r") as f:
        config = edict(yaml.safe_load(f)) 
        config.config_path = config_path
    config.resume = False 
    config.eval = False#
    config.start_epoch = 1
    config.save_path = config.save_path + config.exp_prefix
    if "val_rate" not in config:
        config.val_rate = 2 
    exp_name = config.exp_prefix
    print(exp_name)
    logdir = os.path.join(os.getcwd(), 'output',config.dataset)
    utils.init_distributed_mode(config)
    utils.mkdir_if_missing(logdir)
    
    utils.init_training_paths(config, logdir, exp_name)
    utils.init_torch(config)
    train_one_epoch,evaluate = init_engine(config)
    utils.init_dataset_config(config)

        
    data_module = DataModule(config)
    print(int(config.batch_size / config.n_gpus))
    trian_loader = data_module.train_dataloader()#.trainval_dataloader()#.train_dataloader()
    val_loader = data_module.val_dataloader() #.val_dataloader()#val_dataloader() #.test_dataloader()#
 
    model = trainer(config )#"pretrain"
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)

    if config.resume:
        if  "resume_ckpt" in config:
            checkpoint = torch.load(config.resume_ckpt, map_location='cpu')
        else:
            checkpoint = torch.load(config.paths.model_path, map_location='cpu')
        
        config.start_epoch = checkpoint['epoch'] + 1
        print("resume from:" ,checkpoint['epoch'])

    if config.eval:
        test_stats = evaluate(model, val_loader)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': config.start_epoch,
                     'n_parameters': n_parameters,
                     }
        if config.paths.output_dir and utils.is_main_process():
            with  open(os.path.join(config.paths.output_dir , "log_val.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return
    start_val =config.start_val if  "start_val" in config else 0
    print(f"Start training, start val:{start_val}")
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs+1):
        if config.distributed:
            data_module.sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, trian_loader , epoch= epoch)

        model.update_scheduler()
        checkpoint_paths = [config.paths.model_path,]
        if (epoch + 1) % config.val_rate ==  0 and epoch>start_val:
            checkpoint_paths.append(os.path.join(config.paths.output_dir, f'checkpoints/checkpoint{epoch:04}.pth'))
            new_vis_dir = os.path.join(config.paths.output_dir,f"vis_{epoch}")
            utils.mkdir_if_missing(new_vis_dir)
            test_stats = evaluate(model,  val_loader,path = new_vis_dir,epoch = epoch)
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,}
            if  utils.is_main_process():
                with  open(os.path.join(config.paths.output_dir , "log_val.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n" + "\n")
        #break

        for checkpoint_path in checkpoint_paths:
            save_dict = model.get_save_dict()
            save_dict["epoch"] = epoch
            utils.save_on_master(save_dict, checkpoint_path)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters,}
        if utils.is_main_process():
            with  open(os.path.join(config.paths.output_dir , "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    main("models/confs/conf.yaml")


