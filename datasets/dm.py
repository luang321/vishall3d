from torch.utils.data.dataloader import DataLoader
import torch
from torch.utils.data import DistributedSampler
from torch import nn

#import pytorch_lightning as pl
from datasets.kitti_dataset import SemKITTI
from datasets.kitti_360 import KITTI360
from datasets.psuedo_dataset import PsuedoKITTI
import random
class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, real_dataset, synthetic_dataset, real_ratio=0.25):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.real_ratio = real_ratio
        self.length = max(len(real_dataset), len(synthetic_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random() < self.real_ratio:
            return self.real_dataset[idx % len(self.real_dataset)]
        else:
            return self.synthetic_dataset[idx % len(self.synthetic_dataset)]
        
def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


def convert_to_tensor_or_list(item):
    if isinstance(item, dict):
        return {key: convert_to_tensor_or_list(val) for key, val in item.items()}
    elif isinstance(item, (list, tuple)):
        return [convert_to_tensor_or_list(i) for i in item]
    elif isinstance(item, torch.Tensor):
        return item
    try:
        return torch.tensor(item)
    except Exception:
        return item
def stack_or_list(data):
    if isinstance(data[0], torch.Tensor):
        return torch.stack(data)
    elif isinstance(data[0], (dict, list, tuple)):
        if isinstance(data[0], dict):
            return {key: stack_or_list([d[key] for d in data]) for key in data[0]}
        elif isinstance(data[0], (list, tuple)):
            return [stack_or_list([d[i] for d in data]) for i in range(len(data[0]))]
    else:
        return data  
def collate_fn(batch):
    batch = [convert_to_tensor_or_list(item) for item in batch]
    return stack_or_list(batch)

class DataModule(nn.Module):
    def __init__(
        self,
        config,

    ):
        super().__init__()
        self.dataset  =  SemKITTI if config.dataset == "kitti" else KITTI360 
        self.config = config
        self.batch_size = int(config.batch_size / config.n_gpus)
        self.batch_size_val = int(config.batch_size_val / config.n_gpus)
        self.num_workers = int(config.num_workers_per_gpu)
        self.distributed = config.distributed

    def train_dataloader(self):
        train_ds = self.dataset(
            self.config, imageset = 'train'
        )#,color_jitter=(0.4, 0.4, 0.4)
        if self.distributed:
            self.sampler_train = DistributedSampler(train_ds)
        else:
            self.sampler_train = torch.utils.data.RandomSampler(train_ds)
        batch_sampler_train = torch.utils.data.BatchSampler(self.sampler_train, self.batch_size, drop_last=True)
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler_train,
            num_workers=self.num_workers,
            pin_memory=True,
            #worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
    
    def trainval_dataloader(self):
        train_ds = self.dataset(
            self.config, imageset = 'trainval'
        )#,color_jitter=(0.4, 0.4, 0.4)
        if self.distributed:
            self.sampler_train = DistributedSampler(train_ds)
        else:
            self.sampler_train = torch.utils.data.RandomSampler(train_ds)
        batch_sampler_train = torch.utils.data.BatchSampler(self.sampler_train, self.batch_size, drop_last=True)
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler_train,
            num_workers=self.num_workers,
            pin_memory=True,
            #worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
    def val_dataloader(self):
        val_ds = self.dataset(
            self.config, imageset = 'val'
        )#,color_jitter=(0.4, 0.4, 0.4)
        if self.distributed:
            sampler_val = DistributedSampler(val_ds, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_ds)
        return DataLoader(
            val_ds,
            sampler=sampler_val,
            batch_size=self.batch_size_val,
            drop_last=False,
            num_workers= self.num_workers,
            pin_memory=True,
            #worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
    def psuedo_dataloader(self):
        train_ds = PsuedoKITTI(
            self.config, imageset = 'avoid_over_fit'
        )
        
        if self.distributed:
            self.sampler_train = DistributedSampler(train_ds)
        else:
            self.sampler_train = torch.utils.data.RandomSampler(train_ds)
        batch_sampler_train = torch.utils.data.BatchSampler(self.sampler_train, self.batch_size, drop_last=True)
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler_train,
            num_workers=self.num_workers,
            pin_memory=True,
            #worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def mixed_dataloader(self):
        synthetic_dataset = PsuedoKITTI(
            self.config, imageset = 'avoid_over_fit'
        )
        real_dataset = self.dataset(
            self.config, imageset = 'train'
        )
        train_ds = MixedDataset(real_dataset, synthetic_dataset, real_ratio=0.66)
        if self.distributed:
            self.sampler_train = DistributedSampler(train_ds)
        else:
            self.sampler_train = torch.utils.data.RandomSampler(train_ds)
        batch_sampler_train = torch.utils.data.BatchSampler(self.sampler_train, self.batch_size, drop_last=True)
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler_train,
            num_workers=self.num_workers,
            pin_memory=True,
            #worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
        
    def test_dataloader(self):
        test_ds = self.dataset(
            self.config, imageset = 'test'
        )

        if self.distributed:
            sampler_val = DistributedSampler(test_ds, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(test_ds)
        return DataLoader(
            test_ds,
            sampler=sampler_val,
            batch_size=self.batch_size_val,
            drop_last=False,
            num_workers= 0,
            pin_memory=True,
            #worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
