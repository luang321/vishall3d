import numpy as np
from . import fusion as fusion

def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale

def compute_CP_mega_matrix(target, is_binary=False):
    """
    Parameters
    ---------
    target: (H, W, D)
        contains voxels semantic labels

    is_binary: bool
        if True, return binary voxels relations else return 4-way relations
    """
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i//2 for i in target.shape]
    if is_binary:
        matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
    else:
        matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = np.array([
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N)  * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0 # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0 # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0 # diff
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0 # same
    return matrix


def vox2pix(cam_E, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    """
    compute the 2D projection of voxels centroids
    
    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )#whd lidar coord
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return projected_pix, fov_mask, pix_z



def vox2box(cam_E, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    """
    compute the 2D projection of voxels centroids
    
    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )#whd lidar coord
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    w = voxel_size / 2  
    corners_offset = np.array([
        [ 0,  0,  0],
        [ w,  w,  w],
        [ w,  w, -w],
        [ w, -w,  w],
        [ w, -w, -w],
        [-w,  w,  w],
        [-w,  w, -w],
        [-w, -w,  w],
        [-w, -w, -w]
    ])
    corners = vox_coords[:,None] + corners_offset[None]
    n,p,c = corners.shape
    corners = corners.reshape(-1,3)

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, corners, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    projected_pix = projected_pix.reshape(n,p,2)
    pix_x, pix_y = projected_pix[..., 0], projected_pix[..., 1]
    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2].reshape(n,p)


    '''
    fov_mask = (pix_z[:,0] > 0)
    inside = (pix_x >= 0) + (pix_x < img_W) + (pix_y >= 0) + (pix_y < img_H)
    inside = np.sum(inside, axis=-1) >3
    fov_mask = np.logical_and(fov_mask,inside)
    '''
    fov_mask = np.logical_and(pix_x[:,0] >= 0,
                np.logical_and(pix_x[:,0] < img_W,
                np.logical_and(pix_y[:,0] >= 0,
                np.logical_and(pix_y[:,0] < img_H,
                pix_z[:,0] > 0))))

    min_x = np.min(pix_x, axis=1)
    max_x = np.max(pix_x, axis=1)
    min_y = np.min(pix_y, axis=1)
    max_y = np.max(pix_y, axis=1)
    box_2d = np.stack((min_x, min_y, max_x, max_y), axis=1)

    return box_2d, fov_mask

def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                np.logical_and(pix_x < max_x,
                np.logical_and(pix_y >= min_y,
                np.logical_and(pix_y < max_y,
                pix_z > 0))))
    return valid_pix

def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies
    
    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ="NYU" or "kitti" (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size
    
    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels  
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists
