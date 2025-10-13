import glob
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


from .helpers import vox2box,vox2pix

SPLITS = {
    'train':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync'),
    'trainval':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync','2013_05_28_drive_0006_sync'),
    'val': ('2013_05_28_drive_0006_sync', ),
    'test': ('2013_05_28_drive_0009_sync', ),
}



class KITTI360(Dataset):
    def __init__(
        self,
        config,color_jitter = None, imageset='train'
    ):
        super().__init__()
        root =  config.data_path
        self.data_root = osp.join(root,"unzips","sscbench-kitti") #data_root 
        self.depth_root = osp.join(root,"depth","sequences")
        self.label_root = osp.join(self.data_root,"preprocess","labels")

        self.sequences = SPLITS[imageset]
        self.split = imageset
        self.strides = [config.init_stride, ] + config.strides

        self.vox_origin = np.array((0, -25.6, -2))
        self.voxel_size = 0.2
        self.scene_size = (51.2, 51.2, 6.4)
        self.img_shape = (1408, 376)
        self.img_W = 1408 #1216 #1220#1216#
        self.img_H = 384 #376#368 #370#320
        self.scans = []
        calib = self.read_calib()
        P = calib['P2']
        T_velo_2_cam = calib['Tr']
        proj_matrix = P @ T_velo_2_cam
        self.projected_assets = self.proj_box(P,T_velo_2_cam)
        print(self.split,self.sequences)
        for sequence in self.sequences:
            glob_path = osp.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            for voxel_path in glob.glob(glob_path):
                self.scans.append({
                    'sequence': sequence,
                    'P': P,
                    'T_velo_2_cam': T_velo_2_cam,
                    'proj_matrix': proj_matrix,
                    'voxel_path': voxel_path,
                })

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def proj_box(self,P2,T_velo_2_cam):
        box_xyxys, fov_masks,proj_uvds = list(),list(),list()
        for ii,s in enumerate(self.strides):
            projected_pix, fov_mask , pix_z = vox2pix(
                        T_velo_2_cam,
                        P2[0:3, 0:3],
                        self.vox_origin,
                        self.voxel_size *s,
                        self.img_W,
                        self.img_H,
                        self.scene_size,
                    )
            print(projected_pix.shape)
            uvd = np.concatenate((projected_pix, pix_z[:,None]), axis=-1).astype(np.float32)
            if ii >=1 :
                box_xyxy_2, _ = vox2box(
                        T_velo_2_cam,
                        P2[0:3, 0:3],
                        self.vox_origin,
                        self.voxel_size *s,
                        self.img_W,
                        self.img_H,
                        self.scene_size,
                    )
                print(box_xyxy_2.shape)
                box_xyxys.append(box_xyxy_2)
            fov_masks.append(fov_mask)
            proj_uvds.append(uvd)
        return (box_xyxys, fov_masks,proj_uvds)

    def __len__(self):
        return len(self.scans)

    def getrgb(self,path):
        img = Image.open(path).convert("RGB")
        # Image augmentation
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        h,w,_ = img.shape
        img = np.pad(img, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w),
                                 (0, 0)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)  

        #img = img[:self.img_H, :self.img_W, :]  # crop image
        return  self.transforms(img)
    def getdepth(self,path):
        depth = np.load(path)
        #depth = depth[:self.img_H, :self.img_W][None]
        h,w = depth.shape
        depth = np.pad(depth, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)[None]

        return depth

    def __getitem__(self, idx):
        scan = self.scans[idx]
        sequence = scan['sequence']
        P = scan['P']
        T_velo_2_cam = scan['T_velo_2_cam']
        proj_matrix = scan['proj_matrix']

        filename = osp.basename(scan['voxel_path'])
        frame_id = osp.splitext(filename)[0]
        box_xyxys, fov_masks,proj_uvds = self.projected_assets
        data = {
            'frame_id': frame_id,
            'sequence': sequence,
            'cam_pose': T_velo_2_cam,
            'proj_matrix': proj_matrix,
            "box_xyxy":box_xyxys,
            "fov_mask":fov_masks,
            "proj_uvd":proj_uvds,
            "E":T_velo_2_cam,
            "K":P[:3, :3],
        }
        target_1_path = osp.join(self.label_root, sequence, frame_id + '_1_1.npy')
        target = np.load(target_1_path)
        data ["voxel_label"] = target
        if self.depth_root is not None:
            depth_path = osp.join(self.depth_root, sequence, frame_id + '.npy')
            #depth = np.load(depth_path)[:self.img_shape[1], :self.img_shape[0]]
            data['depth'] = self.getdepth(depth_path) #depth

        # Compute the masks, each indicate the voxels of a local frustum


        img_path = osp.join(self.data_root, 'data_2d_raw', sequence, 'image_00/data_rect',
                            frame_id + '.png')
        #img = Image.open(img_path).convert('RGB')
        #img = np.asarray(img, dtype=np.float32) / 255.0
        #img = img[:self.img_shape[1], :self.img_shape[0]]  # crop image
        data['img'] = self.getrgb(img_path)#self.transforms(img)  # (3, H, W)
        ndarray_to_tensor(data)
        return data

    @staticmethod
    def read_calib():
        P = np.array([
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]).reshape(3, 4)

        cam2velo = np.array([
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        out = {}
        out['P2'] = P
        out['Tr'] = np.identity(4)
        out['Tr'][:3, :4] = V2C
        return out
def ndarray_to_tensor(data: dict):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                v = v.astype('float32')
            data[k] = torch.from_numpy(v)