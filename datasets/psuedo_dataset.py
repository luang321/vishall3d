import os
import numpy as np
from torch.utils import data
import yaml
import pathlib
import torch
import glob
from PIL import Image
import pickle
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from .helpers import vox2box,vox2pix
def read_positive_frame_ids(file_path):
    positive_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == '1':
                positive_ids.append(parts[0])
    return positive_ids
class PsuedoKITTI(data.Dataset):
    def __init__(self, config,color_jitter = None, imageset='avoid_over_fit'):
        with open("datasets/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.config = config
        remapdict = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml["learning_map_inv"]

        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())

        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.learning_map = remap_lut

        self.scene_size = (51.2, 51.2, 6.4)#whd
        self.vox_origin = np.array([0, -25.6, -2])#xyz
        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1280 #1216 #1220#1216#
        self.img_H = 384#368 #370#320
        self.strides = [config.init_stride, ] + config.strides
        #onfig.strides

        self.imageset = imageset
        self.data_path = config.psuedo_path
        
        frame_ids = read_positive_frame_ids(os.path.join(self.data_path,imageset,"labels.txt"))
        
        calib = self.read_calib("/data/datasets/kitti/semantic/sequences/00/calib.txt")
        P2 = calib["P2"]
        P3 = calib["P3"]
        T_velo_2_cam = calib["Tr"]
        self.projected_assets =self.proj_box(P2,P3,T_velo_2_cam)
        self.scans=[]
        for ii,frame_id in enumerate(frame_ids):

            voxel_name = os.path.join(self.data_path,"sample",frame_id+".pkl")
            depth_name_2 = os.path.join(self.data_path, "depth2_render",frame_id+".npy")
            rgb_name_2 = os.path.join(self.data_path,imageset,'imgs',frame_id+".png")
            scan = {
                    "frame_id":frame_id,
                    "T_velo_2_cam": T_velo_2_cam,
                    "cam_E":P2[:3,:3],
                    "voxel_path":str(voxel_name),   #str(filename),
                    "rgb_path_2":str(rgb_name_2),
                    "depth_path_2":str(depth_name_2),
                }
            self.scans.append(scan)
        print(color_jitter)
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    def proj_box(self,P2,P3,T_velo_2_cam):
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
                box_xyxys.append(box_xyxy_2)
            fov_masks.append(fov_mask)
            proj_uvds.append(uvd)
        return box_xyxys, fov_masks,proj_uvds
    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1
        return uncompressed
    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])
        #print(calib_all)
        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.scans)

    def getvox(self,path,invalid):
        with open(path, "rb") as handle:
            b = pickle.load(handle)            
            voxel_label= b["y_pred"].astype(np.int32)
        voxel_label = voxel_label.reshape((256, 256, 32))
        invalid = invalid.reshape((256,256,32))
        voxel_label[invalid == 1]=255
        return voxel_label
    def getrgb(self,path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_W, self.img_H), resample=Image.BILINEAR)
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        h,w,_ = img.shape
        img = np.pad(img, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w),
                                 (0, 0)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)  

        #img = img[:self.img_H, :self.img_W, :]  # crop image
        return self.normalize_rgb(img)
    def getdepth(self,path):
        depth = np.load(path)
        #depth = depth[:self.img_H, :self.img_W][None]
        h,w = depth.shape
        depth = np.pad(depth, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)[None]

        return depth
    def __getitem__(self, index):
        box_xyxys, fov_masks,proj_uvds = self.projected_assets
        scan = self.scans[index]
        rgb2 = self.getrgb(scan["rgb_path_2"])
        depth2 = self.getdepth(scan["depth_path_2"])
        voxel_label = self.getvox(scan["voxel_path"],invalid = ~fov_masks[0])
        frame_id = scan["frame_id"]
        data = {
            "frame_id": frame_id,
            "voxel_label": voxel_label,
            "depth":depth2,
            "img":rgb2,
            "box_xyxy":box_xyxys,
            "fov_mask":fov_masks,
            "proj_uvd":proj_uvds,
            "E":scan["T_velo_2_cam"],
            "K":scan["cam_E"]
        }
        return data 
def count_png_files(directory):
        # 匹配所有 .png 文件（不区分大小写）
    png_files = glob.glob(os.path.join(directory, "*.png"))
    return len(png_files)

def save_rgb_array_as_image(np_array, path):
    """
    将 numpy RGB 图像保存为图片，支持 (H, W, 3) 或 (3, H, W) 形状。
    输入必须是 RGB，值在 0~255。
    """
    if np_array.ndim != 3:
        raise ValueError(f"图像必须是3维的数组，当前形状为 {np_array.shape}")

    # 如果是 (3, H, W)，转换为 (H, W, 3)
    if np_array.shape[0] == 3 and np_array.shape[1] != 3:
        np_array = np.transpose(np_array, (1, 2, 0))

    if np_array.dtype != np.uint8:
        np_array = np.clip(np_array, 0, 255).astype(np.uint8)

    image = Image.fromarray(np_array, mode='RGB')
    image.save(path)