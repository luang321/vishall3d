import torch
import torch.nn as nn
import numpy as np 
import copy
import os
from misc import reduce_dict,all_gather,is_main_process
import torch.nn.functional as F
from misc import reduce_dict,all_gather,get_obj_from_str
import pickle
import yaml


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)



def to_device(input,device,k = "NO_KEY"):
    if isinstance(input,torch.Tensor):
        #print(k,input.shape)
        return input.to(device)
    elif isinstance(input,dict):
        out_dict = {k:to_device(v,device,k=k) for k,v in input.items()}
        return out_dict
    elif isinstance(input,list):
        out_list = [to_device(x,device,k=k) for x in input]
        return out_list
    elif isinstance(input,str) or isinstance(input,int) or isinstance(input,float):
        return input
    else:
        print(type(input),input)
        raise ValueError
gradients = []
def backward_hook(grad_input, grad_output):
    grad_input_resized = F.interpolate(grad_input, scale_factor=0.25, mode='bilinear', align_corners=False)
    gradients.append(grad_input_resized.mean(1,keepdim = True).cpu())


def get_inv_map():
  '''
  remap_lut to remap classes of semantic kitti for training...
  :return:
  '''
  config_path = "datasets/semantic-kitti.yaml"
  dataset_config = yaml.safe_load(open(config_path, 'r'))
  # make lookup table for mapping
  inv_map = np.zeros(20, dtype=np.int32)
  inv_map[list(dataset_config['learning_map_inv'].keys())] = list(dataset_config['learning_map_inv'].values())

  return inv_map

class trainer(nn.Module):
    def __init__(
        self,
        config,
    ):

        super().__init__()
        self.device = torch.device(config.device)
        self.distributed = config.distributed
        self.device_ids = [config.gpu]
        self.init_generator(config)
        self.clip = config.clip if "clip" in config else 10
        self.inv_map = get_inv_map()
        print("clip:",self.clip)
        
    def init_generator(self,config):
        generator = get_obj_from_str(config.generator)(config).to(self.device)
        self.generator = generator
        self.generator_without_ddp = generator
        if self.distributed :
            self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            self.generator = torch.nn.parallel.DistributedDataParallel(self.generator, device_ids=[self.device_ids],find_unused_parameters=True)
            self.generator_without_ddp = self.generator.module
        n_parameters_g = sum(p.numel() for p in self.generator_without_ddp.parameters() if p.requires_grad)
        print("num_p generator:",n_parameters_g)
        self.optimizer_g, self.lr_scheduler_g = self.generator_without_ddp.configure_optimizers()
        self.optimizer = self.optimizer_g
        self.save_path = config.save_path if "save_path" in config else None
        self.metrics = self.generator_without_ddp.metrics
        print(self.save_path)
    def update_scheduler(self):
        self.lr_scheduler_g.step()
   
    def get_save_dict(self):
        save_dict = {
                    'model_g': self.generator_without_ddp.state_dict(),
                    'optimizer_g':self.optimizer_g.state_dict(),
                    'lr_scheduler_g': self.lr_scheduler_g.state_dict(),
                }
        return save_dict
    def load_save_dicts(self,checkpoint):
        mis,uxp = self.generator_without_ddp.load_state_dict(checkpoint['model_g'],strict=False)
        print("g:",mis,uxp)
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
           

    def training_step(self,batch,**kwargs):
        batch = to_device(batch,self.device)
        cond,loss,loss_dict  = self.generator(batch ,step_type = "train",**kwargs)
        self.optimizer_g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip)
        self.optimizer_g.step()
        return loss,loss_dict
    def validation_step(self, batch,path= None):
        batch = to_device(batch,self.device)
        cond,loss,loss_dict  = self.generator(batch,step_type = "val")
        return loss,loss_dict
    def testing_step(self,batch,epoch):
        pred =  self.generator(batch,step_type = "test").detach().cpu().numpy()
        save_path = os.path.join(self.save_path,str(epoch))
        output_voxels = self.inv_map[pred].astype(np.uint16)
        sequence_id = batch['sequence'][0]
        frame_id = batch["frame_id"][0]
        save_folder = "{}/sequences/{}/predictions".format(save_path, sequence_id)
        save_file = os.path.join(save_folder, "{}.label".format(frame_id))
        os.makedirs(save_folder, exist_ok=True)
        with open(save_file, 'wb') as f:
            output_voxels.tofile(f)
            print('\n save to {}'.format(save_file))

    def save_vis(self,f_ids,batch,path):
        #y_preds,occs,path,occs_gt= None
        for ii,f_id in enumerate(f_ids):
            out_dict = {}
            for k,v in batch.items():
                out_dict[k] = v.long().cpu().numpy().astype(np.uint16)
            filepath = os.path.join(path,f"{f_id}_{ii}" + ".pkl")
            with open(filepath, "wb") as handle:
                pickle.dump(out_dict, handle)
                print("wrote to", filepath)

    def validation_epoch_end(self):
        val_result = dict()
        for prefix, metric in self.metrics.items():
            stats = metric.get_stats()
            for i, class_name in enumerate(self.generator_without_ddp.class_names):
                val_result["{}_SemIoU/{}".format(prefix, class_name)] = stats["iou_ssc"][i]

            val_result["{}/mIoU".format(prefix)] = stats["iou_ssc_mean"]
            val_result["{}/IoU".format(prefix)] = stats["iou"]
            val_result["{}/Precision".format(prefix)] = stats["precision"]
            val_result["{}/Recall".format(prefix)] = stats["recall"]
            metric.reset()
        return val_result


