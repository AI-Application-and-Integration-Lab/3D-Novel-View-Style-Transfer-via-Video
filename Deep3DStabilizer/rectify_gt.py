import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trn
import argparse, sys, os, csv, time, datetime
import scipy
from tqdm import tqdm
from scipy.optimize import linprog, minimize
from scipy.spatial.transform import Rotation as R
from path import Path
from PIL import Image
import options
from imageio import imread, imwrite

from warper import Warper, inverse_pose
from sequence_io import *
from smooth import smooth_trajectory, get_smooth_depth_kernel, generate_right_poses, generate_LR_poses
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter as scipy_gaussian
import warnings


def get_cropping_area(warp_maps, h, w):
    border_t = warp_maps[:, 0, :, 1][warp_maps[:, 0, :, 1] >= 0]
    border_b = warp_maps[:, -1, :, 1][warp_maps[:, -1, :, 1] >= 0]
    border_l = warp_maps[:, :, 0, 0][warp_maps[:, :, 0, 0] >= 0]
    border_r = warp_maps[:, :, -1, 0][warp_maps[:, :, -1, 0] >= 0]
    
    t = int(torch.ceil(torch.clamp(torch.max(border_t), 0, h))) if border_t.shape[0] != 0 else 0
    b = int(torch.floor(torch.clamp(torch.min(border_b), 0, h)))if border_b.shape[0] != 0 else 0
    l = int(torch.ceil(torch.clamp(torch.max(border_l), 0, w))) if border_l.shape[0] != 0 else 0
    r = int(torch.floor(torch.clamp(torch.min(border_r), 0, w)))if border_r.shape[0] != 0 else 0
    return t, b, l, r

@torch.no_grad()
def compute_warp_maps(root, warper, compensate_poses, post_process=False):
    # compute all warp maps
    batch_size = 40
    batch_begin = 0
    warp_maps = []
    computed_depths = []
    ds = []
    W, H = warper.width, warper.height
    w, h = warper.width, warper.height
    crop_t, crop_b, crop_l, crop_r = 0, H, 0, W

    # post processing
    if post_process:
        smooth_depth = get_smooth_depth_kernel().to(device)

    imgs_list = sorted(list(glob.glob(root/'images'/'*.png')))
    n = len(imgs_list)
    while batch_begin < n:
    
        batch_end = min(n, batch_begin + batch_size)
        # imgs = seq_io.load_snippet(batch_begin, batch_end)['imgs'].to(device)
        segment = list(range(batch_begin, batch_end))
        depths = []
        for idx in segment:
            depth = np.load(root/'depths/{:05}.npy'.format(idx))
            depths.append(depth)
        depths = np.stack(depths, axis=0)
        depths = torch.from_numpy(depths).float().to(device)

        if post_process:
            # load error maps
            # error_maps = seq_io.load_errors(segment)
            # thresh = 0.5
            # error_maps[error_maps > thresh] = 1
            # error_maps[error_maps < thresh] = 0
           
            # # remove the noise in error map
            # for i in range(error_maps.shape[0]):
            #     eroded_map = np.expand_dims(binary_erosion(error_maps[i].squeeze(0), iterations=1), 0)
            #     error_maps[i] = binary_dilation(eroded_map, iterations=8)
            
            # spatial-variant smoother according to the error map
            # softmasks = scipy_gaussian(error_maps, sigma=[0, 0, 7, 7])
            # softmasks = torch.from_numpy(softmasks).float().to(device)

            # scaling should change
            # scale_depth = 4.306856
            # depths /= scale_depth

            smoothed_depths = smooth_depth(depths) #smooth_depths(depths)
            # depths = depths * (1 - softmasks) + smoothed_depths * softmasks
            depths = smoothed_depths

        (root/'depths_smooth').makedirs_p()
        for i, idx in enumerate(range(batch_begin, batch_end)):
            np.save(root/'depths_smooth'/'{:05}.npy'.format(idx), depths[i].cpu().detach().numpy())
        # compute warping fields
        batch_warps, _, batch_com_depths, _, _ = warper.project_pixel(depths, compensate_poses[batch_begin:batch_end])

        batch_warps = (batch_warps + 1) / 2

        batch_warps[..., 0] *= (W - 1) 
        batch_warps[..., 1] *= (H - 1)
        t, b, l, r = get_cropping_area(batch_warps, H, W)
        crop_t = max(crop_t, t); crop_b = min(crop_b, b); crop_l = max(crop_l, l); crop_r = min(crop_r, r)
        
        batch_warps[..., 0] *= (w - 1) / (W - 1)
        batch_warps[..., 1] *= (h - 1) / (H - 1)

        inverse_warps = warper.inverse_flow(batch_warps)
        inverse_warps[..., 0] = inverse_warps[..., 0] * 2 / (w - 1) - 1
        inverse_warps[..., 1] = inverse_warps[..., 1] * 2 / (h - 1) - 1
        
        warp_maps.append(inverse_warps.detach().cpu())
        computed_depths.append(batch_com_depths.detach().cpu())
        batch_begin = batch_end
    
    warp_maps = torch.cat(warp_maps, 0)
    computed_depths = torch.cat(computed_depths, 0)
    return warp_maps, computed_depths, (crop_t, crop_b, crop_l, crop_r)

@torch.no_grad()
def run(opt):
    root = Path(opt.output_dir)/opt.name
    print('=> load camera trajectory')
    poses = np.load(root/'poses.npy')
    # scale_depth(pose) : chess-01 = 4.306856, fire-01 = 3.858711, redkitchen-04 = 2.974208, stairs-03 = 7.158402
    poses_L, poses_R, comp_L, comp_R  = generate_LR_poses(poses, opt, scale_depth = 7.158402)
    compensate_poses_L = torch.from_numpy(comp_L).float().to(device)
    compensate_poses_R = torch.from_numpy(comp_R).float().to(device)

    # change intrinsic & image size here
    intrinsic = torch.from_numpy(np.load(root/'intrinsic.npy')).float()
    setattr(opt, 'height', 480)
    setattr(opt, 'width', 640)

    warper = Warper(opt, intrinsic).to(device)
    # compute all warping maps
    warp_maps_L, computed_depths_L, (crop_t_L, crop_b_L, crop_l_L, crop_r_L) = compute_warp_maps(root, warper, compensate_poses_L, True)
    warp_maps_R, computed_depths_R, (crop_t_R, crop_b_R, crop_l_R, crop_r_R) = compute_warp_maps(root, warper, compensate_poses_R, True)

    crop_w_L, crop_h_L = crop_r_L - crop_l_L, crop_b_L - crop_t_L
    crop_w_R, crop_h_R = crop_r_R - crop_l_R, crop_b_R - crop_t_R
    H, W = warper.height, warper.width


    imgs_list = sorted(list(glob.glob(root/'images'/'*.png')))
    n = len(imgs_list)

    print('=> warping frames')
    poses = torch.from_numpy(poses).float().to(device)

    # create video writer with crop size
    # print('=> The output video will be saved as {}'.format(root/'output_L.avi'))
    # video_writer = cv.VideoWriter(root/'output_L.avi', cv.VideoWriter_fourcc(*'MJPG'), int(30), (W,H))
    # seq_io.create_video_writer((crop_w, crop_h))
    print('processing left eye warping results')
    # forward warping
    batch_begin = 0
    batch_size = 40
    poses_L = torch.from_numpy(poses_L).float().to(device)

    while batch_begin < n:
        # warp frames parallelly
        batch_end = min(n, batch_begin + batch_size)
        # print(batch_begin, batch_end)
        imgs = []
        for i in range(batch_begin, batch_end):
            img = imread(imgs_list[i]).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            tensor_img = torch.from_numpy(img).float() / 255 
            imgs.append(tensor_img)
        imgs = torch.stack(imgs, 0).to(device)
        
        warp = F.interpolate(warp_maps_L[batch_begin:batch_end].to(device).permute(0, 3, 1, 2),
                        (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        reproj_imgs = F.grid_sample(imgs, warp)
        reproj_depths = F.grid_sample(computed_depths_L[batch_begin:batch_end].to(device), warp)

        (root/'images_L').makedirs_p()
        reproj_imgs = (reproj_imgs * 255.).detach().cpu().numpy()
        reproj_imgs = reproj_imgs.transpose(0, 2, 3, 1).astype(np.uint8)
        for i, idx in enumerate(range(batch_begin, batch_end)):
            imwrite(root/'images_L'/'{:05}.png'.format(idx), reproj_imgs[i])

        (root/'depths_L').makedirs_p()
        for i, idx in enumerate(range(batch_begin, batch_end)):
            np.save(root/'depths_L'/'{:05}.npy'.format(idx), reproj_depths[i].cpu().detach().numpy())

        batch_begin = batch_end

    poses_L = poses_L.detach().cpu().numpy()
    
    np.save(root/'poses_L.npy', poses_L)

    # seq_io.create_video_writer((W, H), 'output_R.avi')
    # seq_io.create_video_writer((crop_w, crop_h))

    # forward warping
    print('processing right eye warping results')
    batch_begin = 0
    poses_R = torch.from_numpy(poses_R).float().to(device)
    # smooth_poses = torch.from_numpy(smooth_poses).float().to(device)

    while batch_begin < n:
        # warp frames parallelly
        batch_end = min(n, batch_begin + batch_size)
        # print(batch_begin, batch_end)
        imgs = []
        for i in range(batch_begin, batch_end):
            img = imread(imgs_list[i]).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            tensor_img = torch.from_numpy(img).float() / 255 
            imgs.append(tensor_img)
        imgs = torch.stack(imgs, 0).to(device)
        # imgs = trn.Resize((H, W))(imgs)
        
        warp = F.interpolate(warp_maps_R[batch_begin:batch_end].to(device).permute(0, 3, 1, 2),
                        (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        reproj_imgs = F.grid_sample(imgs, warp)
        reproj_depths = F.grid_sample(computed_depths_R[batch_begin:batch_end].to(device), warp)
        
        (root/'images_R').makedirs_p()
        reproj_imgs = (reproj_imgs * 255.).detach().cpu().numpy()
        reproj_imgs = reproj_imgs.transpose(0, 2, 3, 1).astype(np.uint8)
        for i, idx in enumerate(range(batch_begin, batch_end)):
            imwrite(root/'images_R'/'{:05}.png'.format(idx), reproj_imgs[i])

        (root/'depths_R').makedirs_p()
        for i, idx in enumerate(range(batch_begin, batch_end)):
            np.save(root/'depths_R'/'{:05}.npy'.format(idx), reproj_depths[i].cpu().detach().numpy())

        batch_begin = batch_end

    poses_R = poses_R.detach().cpu().numpy()
    np.save(root/'poses_R.npy', poses_R)
    # smooth_poses = smooth_poses.detach().cpu().numpy()
    # seq_io.save_poses_stab(smooth_poses)
    print('=> Done!')

if __name__ == '__main__':
    opt = options.Options().parse()
    global device
    device = torch.device(opt.cuda)
    # global H_pad, W_pad
    # H_pad, W_pad = 576, 960
    # global opt_for_refine
    # opt_for_refine = opts_helper(opt)
    
    run(opt)

