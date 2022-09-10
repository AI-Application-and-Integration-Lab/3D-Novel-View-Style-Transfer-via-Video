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
def compute_warp_maps(seq_io, warper, compensate_poses, post_process=False):
    # compute all warp maps
    batch_begin = 0
    warp_maps = []
    # inv_warp_maps = []
    computed_depths = []
    ds = []
    W, H = seq_io.origin_width, seq_io.origin_height

    w, h = seq_io.width, seq_io.height # for deep3d
    # w, h = seq_io.origin_width, seq_io.origin_height # for colmap
    crop_t, crop_b, crop_l, crop_r = 0, H, 0, W
    
    # xv, yv = np.meshgrid(np.linspace(-1, 1, W_pad), np.linspace(-1, 1, H_pad))
    # xv = np.expand_dims(xv, axis=2)
    # yv = np.expand_dims(yv, axis=2)
    # grid_pad = np.expand_dims(np.concatenate((xv, yv), axis=2), axis=0)
    # grid_pad = np.repeat(grid_pad, 1, axis=0)
    # grid_pad = torch.from_numpy(grid_pad).float().to(device)

    # post processing
    if post_process:
        smooth_depth = get_smooth_depth_kernel().to(device)

    while batch_begin < len(seq_io):
    
        batch_end = min(len(seq_io), batch_begin + seq_io.batch_size)
        # imgs = seq_io.load_snippet(batch_begin, batch_end)['imgs'].to(device)
        segment = list(range(batch_begin, batch_end))
        depths = seq_io.load_depths(segment).to(device)

        if post_process:
            # load error maps
            error_maps = seq_io.load_errors(segment)
            thresh = 0.5
            error_maps[error_maps > thresh] = 1
            error_maps[error_maps < thresh] = 0
           
            # remove the noise in error map
            for i in range(error_maps.shape[0]):
                eroded_map = np.expand_dims(binary_erosion(error_maps[i].squeeze(0), iterations=1), 0)
                error_maps[i] = binary_dilation(eroded_map, iterations=8)
            
            # spatial-variant smoother according to the error map
            softmasks = scipy_gaussian(error_maps, sigma=[0, 0, 7, 7])
            softmasks = torch.from_numpy(softmasks).float().to(device)

            smoothed_depths = smooth_depth(depths) #smooth_depths(depths)
            depths = depths * (1 - softmasks) + smoothed_depths * softmasks

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
    # inv_warp_maps = torch.cat(inv_warp_maps, 0)
    computed_depths = torch.cat(computed_depths, 0)
    return warp_maps, computed_depths, (crop_t, crop_b, crop_l, crop_r)

# @torch.no_grad()
# def warp_neighbors(seq_io, warper, compensate_poses, segment, post_process=False):
#     # compute all warp maps
#     imgs = torch.stack([seq_io.load_image(i) for i in segment], 0).to(device)
    
#     W, H = seq_io.origin_width, seq_io.origin_height
#     w, h = seq_io.width, seq_io.height
    
#     # post processing
#     if post_process:
#         smooth_depth = get_smooth_depth_kernel().to(device)

#     depths = seq_io.load_depths(segment).to(device)

#     if post_process:
#         # load error maps
#         error_maps = seq_io.load_errors(segment)
#         thresh = 0.5
#         error_maps[error_maps > thresh] = 1
#         error_maps[error_maps < thresh] = 0
       
#         # remove the noise in error map
#         for i in range(error_maps.shape[0]):
#             eroded_map = np.expand_dims(binary_erosion(error_maps[i].squeeze(0), iterations=1), 0)
#             error_maps[i] = binary_dilation(eroded_map, iterations=8)
        
#         # spatial-variant smoother according to the error map
#         softmasks = scipy_gaussian(error_maps, sigma=[0, 0, 7, 7])
#         softmasks = torch.from_numpy(softmasks).float().to(device)

#         smoothed_depths = smooth_depth(depths) #smooth_depths(depths)
#         depths = depths * (1 - softmasks) + smoothed_depths * softmasks

#     # compute warping fields
#     batch_warps, _, batch_com_depths, _, _ = warper.project_pixel(depths, compensate_poses)
#     batch_warps = (batch_warps + 1) / 2

#     batch_warps[..., 0] *= (W - 1) 
#     batch_warps[..., 1] *= (H - 1)
#     # t, b, l, r = get_cropping_area(batch_warps, H, W)
#     # crop_t = max(crop_t, t); crop_b = min(crop_b, b); crop_l = max(crop_l, l); crop_r = min(crop_r, r)
    
#     batch_warps[..., 0] *= (w - 1) / (W - 1)
#     batch_warps[..., 1] *= (h - 1) / (H - 1)

#     inverse_warps = warper.inverse_flow(batch_warps)
#     inverse_warps[..., 0] = inverse_warps[..., 0] * 2 / (w - 1) - 1
#     inverse_warps[..., 1] = inverse_warps[..., 1] * 2 / (h - 1) - 1
    
#     # warp_maps.append(inverse_warps.detach().cpu())
#     # computed_depths.append(batch_com_depths.detach().cpu())
#     # batch_begin = batch_end
    
#     # warp_maps = torch.cat(warp_maps, 0)
#     # computed_depths = torch.cat(computed_depths, 0)
#     return imgs, inverse_warps, batch_com_depths

@torch.no_grad()
def run(opt):
    # refinement module from pixelsynth
    # encoder = get_encoder(opt_for_refine).to(device)
    # decoder = get_decoder(opt_for_refine).to(device)
    
    # pretrained_dict_enc = {
    #     k[21:]: v
    #     for k, v in torch.load(opt.old_model)["state_dict"].items()
    #     if "encoder" in k
    # }

    # pretrained_dict_dec = {
    #     k[23:]: v
    #     for k, v in torch.load(opt.old_model)["state_dict"].items()
    #     if "projector" in k
    # }

    # encoder.load_state_dict(pretrained_dict_enc)
    # decoder.load_state_dict(pretrained_dict_dec)


    seq_io = SequenceIO(opt, preprocess=False)
    warper = Warper(opt, seq_io.get_intrinsic(True)).to(device)
    # ori_h, ori_w = seq_io.origin_height, seq_io.origin_width
    # warper = Warper(opt, torch.from_numpy(np.load(Path(opt.output_dir)/opt.name/'K.npy')).float(), im_size=(ori_h, ori_w)).to(device)
    
    print('=> load camera trajectory')
    poses = seq_io.load_poses()
    # smooth_poses, comp = smooth_trajectory(poses, opt)
    # poses_R, comp_R = generate_right_poses(poses, opt)
    poses_L, poses_R, comp_L, comp_R  = generate_LR_poses(poses, opt)
    # compensate_poses = torch.from_numpy(comp).float().to(device)
    compensate_poses_L = torch.from_numpy(comp_L).float().to(device)
    compensate_poses_R = torch.from_numpy(comp_R).float().to(device)
    # compute all warping maps
    warp_maps_L, computed_depths_L, (crop_t_L, crop_b_L, crop_l_L, crop_r_L) = compute_warp_maps(seq_io, warper, compensate_poses_L, False)
    warp_maps_R, computed_depths_R, (crop_t_R, crop_b_R, crop_l_R, crop_r_R) = compute_warp_maps(seq_io, warper, compensate_poses_R, False)

    crop_w_L, crop_h_L = crop_r_L - crop_l_L, crop_b_L - crop_t_L
    crop_w_R, crop_h_R = crop_r_R - crop_l_R, crop_b_R - crop_t_R
    H, W = seq_io.origin_height, seq_io.origin_width

    # xv, yv = np.meshgrid(np.linspace(-1, 1, W_pad), np.linspace(-1, 1, H_pad))
    # xv = np.expand_dims(xv, axis=2)
    # yv = np.expand_dims(yv, axis=2)
    # grid_pad = np.expand_dims(np.concatenate((xv, yv), axis=2), axis=0)
    # grid_pad = np.repeat(grid_pad, 1, axis=0)
    # grid_pad = torch.from_numpy(grid_pad).float().to(device)

    # i_range = torch.arange(0, H_pad).view(1, H_pad, 1).expand(1, H_pad, W_pad).float()
    # j_range = torch.arange(0, W_pad).view(1, 1, W_pad).expand(1, H_pad, W_pad).float()
    # grid_pad = torch.cat((j_range, i_range), 0).unsqueeze(0).permute(0, 2, 3, 1).to(device)


    print('=> warping frames')
    poses = torch.from_numpy(poses).float().to(device)

    # create video writer with crop size
    # seq_io.create_video_writer((W, H), 'output_L.avi')
    # # seq_io.create_video_writer((crop_w, crop_h))

    # # forward warping
    batch_begin = 0
    seq_io.need_resize = False
    poses_L = torch.from_numpy(poses_L).float().to(device)

    while batch_begin < len(seq_io):
        # warp frames parallelly
        batch_end = min(len(seq_io), batch_begin + seq_io.batch_size)
        # print(batch_begin, batch_end)

        imgs = seq_io.load_snippet(batch_begin, batch_end)['imgs'].to(device)
        # imgs = trn.Resize((H, W))(imgs)
        
        warp = F.interpolate(warp_maps_L[batch_begin:batch_end].to(device).permute(0, 3, 1, 2),
                        (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        reproj_imgs = F.grid_sample(imgs, warp)
        reproj_depths = F.grid_sample(computed_depths_L[batch_begin:batch_end].to(device), warp)
        # reproj_imgs = F.grid_sample(imgs_pad, inv_warp)
        # reproj_depths = F.grid_sample(depths_pad, inv_warp)

        seq_io.write_images_stab(reproj_imgs, list(range(batch_begin, batch_end)), 'images_L')
        # seq_io.write_images(reproj_imgs[..., crop_t:crop_b, crop_l:crop_r])
        # seq_io.write_images(reproj_imgs)
        # seq_io.write_background(warp, list(range(batch_begin, batch_end)))
        seq_io.save_depths_stab(reproj_depths, list(range(batch_begin, batch_end)), 'depths_L')

        batch_begin = batch_end

    poses_L = poses_L.detach().cpu().numpy()
    seq_io.save_poses_LR(poses_L, 'poses_L.npy')

    # seq_io.create_video_writer((W, H), 'output_R.avi')
    # seq_io.create_video_writer((crop_w, crop_h))

    # forward warping
    batch_begin = 0
    seq_io.need_resize = False
    poses_R = torch.from_numpy(poses_R).float().to(device)
    # smooth_poses = torch.from_numpy(smooth_poses).float().to(device)

    while batch_begin < len(seq_io):
        # warp frames parallelly
        batch_end = min(len(seq_io), batch_begin + seq_io.batch_size)
        # print(batch_begin, batch_end)

        imgs = seq_io.load_snippet(batch_begin, batch_end)['imgs'].to(device)
        # imgs = trn.Resize((H, W))(imgs)
        
        warp = F.interpolate(warp_maps_R[batch_begin:batch_end].to(device).permute(0, 3, 1, 2),
                        (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        reproj_imgs = F.grid_sample(imgs, warp)
        reproj_depths = F.grid_sample(computed_depths_R[batch_begin:batch_end].to(device), warp)
        # reproj_imgs = F.grid_sample(imgs_pad, inv_warp)
        # reproj_depths = F.grid_sample(depths_pad, inv_warp)

        seq_io.write_images_stab(reproj_imgs, list(range(batch_begin, batch_end)), 'images_R')
        # seq_io.write_images(reproj_imgs[..., crop_t:crop_b, crop_l:crop_r])
        # seq_io.write_images(reproj_imgs)
        # seq_io.write_background(warp, list(range(batch_begin, batch_end)))
        seq_io.save_depths_stab(reproj_depths, list(range(batch_begin, batch_end)), 'depths_R')

        batch_begin = batch_end

    poses_R = poses_R.detach().cpu().numpy()
    seq_io.save_poses_LR(poses_R, 'poses_R.npy')
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

