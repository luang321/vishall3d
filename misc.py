import os

import time
from collections import defaultdict, deque
import datetime
import pickle
import torch
import torch.distributed as dist
import importlib
import numpy as np
import random
import shutil
from easydict import EasyDict as edict
import yaml
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    print("build:",cls,"from",module)
    return getattr(importlib.import_module(module, package=None), cls)


def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """
    if is_main_process():
        if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

        # check if not exist, then make
        if not os.path.exists(directory):
            os.makedirs(directory)
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)







class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            #print(type(values),values.dtype,type(world_size))
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(conf):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        conf.rank = int(os.environ["RANK"])
        conf.world_size = int(os.environ['WORLD_SIZE'])
        conf.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        conf.rank = int(os.environ['SLURM_PROCID'])
        conf.gpu = conf.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        conf.distributed = False
        conf.gpu = 0
        return
    conf.dist_url = 'env://'
    conf.distributed = True
    conf.n_gpus = conf.world_size
    torch.cuda.set_device(conf.gpu)
    conf.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        conf.rank, conf.dist_url), flush=True)
    torch.distributed.init_process_group(backend=conf.dist_backend, init_method=conf.dist_url,
                                         world_size=conf.world_size, rank=conf.rank)
    torch.distributed.barrier()
    print("rank",conf.rank)
    setup_for_distributed(conf.rank == 0)


def init_torch(conf):
    
    
    if conf.seed is None:
        seed = (
                os.getpid()
                # + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )  + get_rank()
        #logger.info("Using a generated random seed {}".format(seed))
    else:
        seed = conf.seed + get_rank()
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.allow_tf32 = True
    # ma    ke the code deterministic
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def save_yaml_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)  # 加载 YAML 文件内容

    with open(output_file, 'w') as f:
        yaml.dump(data, f)  # 将内容写入到另一个 YAML 文件

def init_training_paths(config, logdir, exp_name):
    """
    Simple function to store and create the relevant paths for the project,
    based on the base = current_working_dir (cwd). For this reason, we expect
    that the experiments are run from the root folder.

    data    =  ./data
    output  =  ./output/<name>/time/
    weights =  ./output/<name>/time/weights
    results =  ./output/<name>/time/results
    logs    =  ./output/<name>/time/log

    Args:
        conf_name (str): configuration experiment name (used for storage into ./output/<conf_name>)
    """

    # make paths
    

    paths = edict()
    paths.output_dir = os.path.join(logdir, exp_name)
    save_config_path = os.path.join(paths.output_dir,"conf.yaml")
    mkdir_if_missing(paths.output_dir )
    if not os.path.exists(save_config_path) and is_main_process():
        with open(save_config_path, 'w') as f:
            yaml.dump(dict(config), f)  # 将内容写入到另一个 YAML 文件

    
    mkdir_if_missing(os.path.join(paths.output_dir, "checkpoints"))
    paths.model_path  = os.path.join(paths.output_dir, "checkpoints/last.ckpt")#last.ckpt

    config.paths = paths

    return paths

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

def init_dataset_config(config):
    if config.dataset == "kitti":
        config.class_names = [
                "empty",'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
                'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
                'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
            ]
        config.full_scene_size = (256, 256, 32)
        config.n_classes = 20

        complt_num_per_class= np.asarray([7632350044, 15783539,  125136, 118809, 646799, 821951, 262978, 283696, 204750, 61688703, 4502961, 44883650, 2269923, 56840218, 15719652, 158442623, 2061623, 36970522, 1151988, 334146])
        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        #config.class_weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0))
        config.class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))#
        print(torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)),config.class_weights)
        config.data_path = "/data/datasets/kitti/semantic/sequences"
        
        # 0: unlabeld 1: "car" 2: "bicycle" 3: "motorcycle" 4: "truck" 5: "other-vehicle" 6: "person"
        # 7: "bicyclist" 8: "motorcyclist" 9: "road" 10: "parking" 11: "sidewalk" 12: "other-ground" 13: "building"
        # 14: "fence" 15: "vegetation" 16: "trunk" 17: "terrain" 18: "pole" 19: "traffic-sign"

    elif config.dataset ==  'carla':
        config.class_names = ['building', 'barrier', 'other', 'pedestrian', 'pole', 'road', 'ground', 'sidewalk', 'vegetation', 'vehicle']
        config.full_scene_size = [256, 256, 16]
        config.n_classes = 11

        complt_num_per_class= np.asarray([4.16659328e+09, 4.23097440e+07,  3.33326810e+07, 8.17951900e+06, 9.05663000e+05, 3.08392300e+06, 2.35769663e+08, 8.76012450e+07, 1.12863867e+08, 2.98168940e+07, 1.38396550e+07])
        epsilon_w = 0.001  # eps to avoid zero division
        config.class_weights = torch.from_numpy(1 / np.log(complt_num_per_class + epsilon_w))

        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        config.class_weights =  torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0))
        print(torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)),config.class_weights)

        config.data_path = "/data/datasets/carlasc/Cartesian"
        #0 : Free, 1 : Building, 2 : Barrier, 3 : Other ,4 : Pedestrian ,5 : Pole, 6 : Road, 7 : Ground ,8 : Sidewalk ,9 : Vegetation 10 : Vehicle
    elif config.dataset ==  'kitti360':
        config.class_names = [
                'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
                'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
                'pole', 'traffic-sign', 'other-structure', 'other-object'
            ]
        config.full_scene_size = (256, 256, 32)
        config.n_classes = 19
        complt_num_per_class= np.asarray([2264087502, 20098728, 104972, 96297, 1149426, 4051087, 125103, 105540713, 16292249, 45297267,
                                        14454132, 110397082, 6766219, 295883213, 50037503, 1561069, 406330, 30516166, 1950115])
        config.class_weights =  torch.from_numpy(1 / np.log(complt_num_per_class + 0.00001))
        print(config.class_weights)
        config.data_path = "/data/datasets/kitti-360"
    else:
        raise ValueError("不存在的数据集")
    return config