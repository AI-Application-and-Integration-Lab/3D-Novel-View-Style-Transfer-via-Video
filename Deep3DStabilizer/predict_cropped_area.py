import argparse, sys, os, csv, time, datetime, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms as trn
import cv2 as cv
from tqdm import tqdm
from path import Path
import options
from PIL import Image
from imageio import imread, imwrite
from sequence_io import SequenceIO
from models import *
from loss import Loss
from skimage.transform import resize as imresize
from models.layers import disp_to_depth
from warper import Warper, pose_vec2mat, inverse_pose
import warnings
from smooth import smooth_trajectory, get_smooth_depth_kernel

# torch.set_printoptions(threshold=384*900)

mean = 0.45
std = 0.225

def load_mask(filename):
    mask = Image.open(filename)
    # mask = trn.Resize((seq_io.height, seq_io.width))(mask)
    mask = trn.ToTensor()(mask).unsqueeze(0)
    mask = (mask < 0.5).to(torch.bool)
    return mask

def load_image(filename, resize=False):
    img = imread(filename).astype(np.float32)
    if resize:
        img = resize_image(img)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (torch.from_numpy(img).float() / 255. - mean) / std
    return tensor_img
def resize_image(img):
    H, W, _ = img.shape
    if H > W:
        a, b = H, W
    else:
        a, b = W, H

    a_depth = 384
    if a >= 1024:
        a_flow = 1024
    else:
        a_flow = int(np.round(a / 64) * 64)

    b_flow = int(np.round(b * a_flow / a / 64) * 64)
    b_depth = int(np.round(b * a_depth / a / 32) * 32)
    if H > W:
        img = imresize(img, (a_depth, b_depth))
    else:
        img = imresize(img, (b_depth, a_depth))
    return img

def predict_keyframe_depth(outpainted_paths, keyframe_indices, img_paths, bg_mask_paths, model_paths, opt):
    # smooth_depth = get_smooth_depth_kernel().to(device)
    model_id = 0
    dispnet = torch.load(model_paths[model_id])
    imgs = []
    depths = []
    for i, filename in enumerate(outpainted_paths):
        idx = keyframe_indices[i]
        print(idx)
        outpnt_img = load_image(filename, False).unsqueeze(0).to(device)
        origin_img = load_image(img_paths[idx], False).unsqueeze(0).to(device)
        bg_mask = load_mask(bg_mask_paths[idx]).to(device)
        img = torch.where(bg_mask.expand(-1,3,-1,-1), outpnt_img, origin_img)
        imgs.append(img)

        img = ((img * std + mean) * 255.).detach().squeeze(0).cpu().numpy()
        img = img.transpose(1, 2, 0)#.astype(np.uint8)
        img = resize_image(img)
        img = np.transpose(img, (2, 0, 1))
        img = ((torch.from_numpy(img).float() / 255. - mean) / std).unsqueeze(0).to(device)

        modelname_split = model_paths[model_id].split('_')
        begin, end = int(modelname_split[2]), int(modelname_split[4][:5])
        while idx >= end:
            model_id += 1
            dispnet = torch.load(model_paths[model_id])
            modelname_split = model_paths[model_id].split('_')
            begin, end = int(modelname_split[2]), int(modelname_split[4][:5])
        # dispnet['encoder'].eval()
        # dispnet['decoder'].eval()
        with torch.no_grad():
            d_feature = dispnet['encoder'](img)
            d_output = dispnet['decoder'](d_feature)
        depth = [d_output['disp', s] for s in opt.scales]
        depth = [d * opt.max_depth + opt.min_depth for d in depth]
        # depth = smooth_depth(depth)
        depths.append(depth[0])
    return imgs, depths

def project_keyframe_to_frame(imgs, depths, bg_masks, kf_imgs, kf_depths, rel_poses):
    w, h = seq_io.width, seq_io.height
    W, H = seq_io.origin_width, seq_io.origin_height

    batch_warps, _, batch_com_depths, _, _ = warper.project_pixel(kf_depths, rel_poses)
    batch_warps = (batch_warps + 1) / 2
    
    batch_warps[..., 0] *= (w - 1)
    batch_warps[..., 1] *= (h - 1)

    inverse_warps = warper.inverse_flow(batch_warps)
    inverse_warps[..., 0] = inverse_warps[..., 0] * 2 / (w - 1) - 1
    inverse_warps[..., 1] = inverse_warps[..., 1] * 2 / (h - 1) - 1


    warp = F.interpolate(inverse_warps.to(device).permute(0, 3, 1, 2),
                        (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

    proj_imgs = F.grid_sample(kf_imgs, warp)

    proj_depths = F.grid_sample(batch_com_depths.to(device), warp)

    warp_x = warp[:,:,:,0].unsqueeze(1)
    warp_y = warp[:,:,:,1].unsqueeze(1)
    
    is_bg = bg_masks & (warp_x >= -1.0) & (warp_x <= 1.0) & (warp_y >= -1.0) & (warp_y <= 1.0)
    new_bg_mask = bg_masks & ((warp_x < -1.0) | (warp_x > 1.0) | (warp_y < -1.0) | (warp_y > 1.0))

    proj_imgs = torch.where(is_bg.expand(-1,3,-1,-1), proj_imgs, imgs)

    proj_depths = torch.where(is_bg, proj_depths, depths)

    return proj_imgs, proj_depths, new_bg_mask

def warp_other_frame(img_paths, depth_paths, bg_mask_paths, kf_min_imgs, kf_min_depths, kf_max_imgs, kf_max_depths, kf_min_indices, kf_max_indices, smooth_poses, model_paths):
    model_id = -1
    prev_model_id = -1  
    dispnet = None
    for i, kf_max_idx in enumerate(kf_max_indices):
        if i == len(kf_max_indices) - 1:
            end_kf_min = len(img_paths)
        else:
            end_kf_min = kf_min_indices[i+1]
        begin_kf_min = kf_min_indices[i]
        print("Processing from {} to {}, local max: {}".format(begin_kf_min, end_kf_min, kf_max_idx))

        prev_img = kf_max_imgs[i]
        prev_d = kf_max_depths[i].squeeze(1)
        prev_pose = torch.from_numpy(smooth_poses[kf_max_idx]).float().unsqueeze(0).to(device)
        
        
        for j in range(kf_max_idx-1, begin_kf_min, -1):
            # print(j)
            # projection and outpainting
            img = load_image(img_paths[j], False).unsqueeze(0).to(device)
            d = torch.from_numpy(np.load(depth_paths[j])).to(device)
            # d = trn.ToPILImage()(d)
            # d = trn.Resize((seq_io.height, seq_io.width))(d)
            # d = trn.ToTensor()(d).to(device)
            bg_mask = load_mask(bg_mask_paths[j]).to(device)
            cur_pose = torch.from_numpy(smooth_poses[j]).float().unsqueeze(0).to(device)
            rel_pose = inverse_pose(cur_pose) @ prev_pose
            proj_imgs, proj_depths, new_bg_mask = project_keyframe_to_frame(
                img, 
                d, 
                bg_mask, 
                prev_img, 
                prev_d, 
                rel_pose
            )
            # Need refinement inout_painting and depth reprediction
            while (new_bg_mask == True).sum().item() > 0:
                nbor_count = torch.zeros_like(proj_imgs).to(device)
                pix_color_avg = torch.zeros_like(proj_imgs).to(device)

                mask_up = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_up[:,:,1:,:] = new_bg_mask[:,:,:-1,:]
                img_up = torch.zeros_like(proj_imgs).to(device)
                img_up[:,:,1:,:] = proj_imgs[:,:,:-1,:]
                nbor_count = torch.where(mask_up.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_up.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_up)

                mask_down = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_down[:,:,:-1,:] = new_bg_mask[:,:,1:,:]
                img_down = torch.zeros_like(proj_imgs).to(device)
                img_down[:,:,:-1,:] = proj_imgs[:,:,1:,:]
                nbor_count = torch.where(mask_down.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_down.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_down)

                mask_left = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_left[:,:,:,1:] = new_bg_mask[:,:,:,:-1]
                img_left = torch.zeros_like(proj_imgs).to(device)
                img_left[:,:,:,1:] = proj_imgs[:,:,:,:-1]
                nbor_count = torch.where(mask_left.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_left.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_left)
                
                mask_right = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_right[:,:,:,:-1] = new_bg_mask[:,:,:,1:]
                img_right = torch.zeros_like(proj_imgs).to(device)
                img_right[:,:,:,:-1] = proj_imgs[:,:,:,1:]
                nbor_count = torch.where(mask_right.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_right.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_right)
                
                pix_color_avg = pix_color_avg / nbor_count

                is_edge = new_bg_mask & ~(mask_up & mask_down & mask_left & mask_right)
                proj_imgs = torch.where(is_edge.expand(-1,3,-1,-1), pix_color_avg, proj_imgs)
                new_bg_mask = torch.where(is_edge, False, new_bg_mask)
            
            prev_img = proj_imgs

            proj_imgs = ((proj_imgs * std + mean) * 255.).detach().squeeze(0).cpu().numpy()
            proj_imgs = proj_imgs.transpose(1, 2, 0)#.astype(np.uint8)
            proj_imgs_rsz = resize_image(proj_imgs)
            proj_imgs_rsz = np.transpose(proj_imgs_rsz, (2, 0, 1))
            proj_imgs_rsz = ((torch.from_numpy(proj_imgs_rsz).float() / 255. - mean) / std).unsqueeze(0).to(device)
            
            for k in range(len(model_paths)):
                modelname_split = model_paths[k].split('_')
                begin, end = int(modelname_split[2]), int(modelname_split[4][:5])
                if j < end:
                    # print(begin, end)
                    model_id = k
                    break
            if model_id != prev_model_id:
                dispnet = torch.load(model_paths[model_id])
                prev_model_id = model_id
            # dispnet['encoder'].eval()
            # dispnet['decoder'].eval()
            with torch.no_grad():
                d_feature = dispnet['encoder'](proj_imgs_rsz)
                d_output = dispnet['decoder'](d_feature)
            depth = [d_output['disp', s] for s in opt.scales]
            depth = [d * opt.max_depth + opt.min_depth for d in depth][0]

            imwrite(os.path.join(root, 'outpainted_images/{:05}.png'.format(j)), proj_imgs.astype(np.uint8))
            np.save(os.path.join(root, 'outpainted_depths/{:05}.npy'.format(j)), depth.cpu().detach().numpy())
            # imwrite(os.path.join(root, 'outpainted_images_f/{:05}_f.png'.format(i+j)), proj_imgs_f[j])
            # np.save(os.path.join(root, 'outpainted_depths_f/{:05}_f.npy'.format(i+j)), proj_depths_f[j].cpu().detach().numpy())
            # imwrite(os.path.join(root, 'outpainted_images_b/{:05}_b.png'.format(i+j)), proj_imgs_b[j])
            # np.save(os.path.join(root, 'outpainted_depths_b/{:05}_b.npy'.format(i+j)), proj_depths_b[j].cpu().detach().numpy())
            
            prev_d = depth.squeeze(1)
            prev_pose = torch.from_numpy(smooth_poses[j]).float().unsqueeze(0).to(device)

        prev_img = kf_max_imgs[i]
        prev_d = kf_max_depths[i].squeeze(1)
        prev_pose = torch.from_numpy(smooth_poses[kf_max_idx]).float().unsqueeze(0).to(device)
        for j in range(kf_max_idx+1, end_kf_min, 1):
            # print(j)
            img = load_image(img_paths[j], False).unsqueeze(0).to(device)
            d = torch.from_numpy(np.load(depth_paths[j])).to(device)
            # d = trn.ToPILImage()(d)
            # d = trn.Resize((seq_io.height, seq_io.width))(d)
            # d = trn.ToTensor()(d).to(device)
            bg_mask = load_mask(bg_mask_paths[j]).to(device)
            cur_pose = torch.from_numpy(smooth_poses[j]).float().unsqueeze(0).to(device)
            rel_pose = inverse_pose(cur_pose) @ prev_pose
            proj_imgs, proj_depths, new_bg_mask = project_keyframe_to_frame(
                img, 
                d, 
                bg_mask, 
                prev_img, 
                prev_d, 
                rel_pose
            )
            # Need refinement inout_painting and depth reprediction
            while (new_bg_mask == True).sum().item() > 0:
                nbor_count = torch.zeros_like(proj_imgs).to(device)
                pix_color_avg = torch.zeros_like(proj_imgs).to(device)

                mask_up = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_up[:,:,1:,:] = new_bg_mask[:,:,:-1,:]
                img_up = torch.zeros_like(proj_imgs).to(device)
                img_up[:,:,1:,:] = proj_imgs[:,:,:-1,:]
                nbor_count = torch.where(mask_up.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_up.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_up)

                mask_down = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_down[:,:,:-1,:] = new_bg_mask[:,:,1:,:]
                img_down = torch.zeros_like(proj_imgs).to(device)
                img_down[:,:,:-1,:] = proj_imgs[:,:,1:,:]
                nbor_count = torch.where(mask_down.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_down.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_down)

                mask_left = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_left[:,:,:,1:] = new_bg_mask[:,:,:,:-1]
                img_left = torch.zeros_like(proj_imgs).to(device)
                img_left[:,:,:,1:] = proj_imgs[:,:,:,:-1]
                nbor_count = torch.where(mask_left.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_left.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_left)
                
                mask_right = torch.ones_like(new_bg_mask).to(torch.bool).to(device)
                mask_right[:,:,:,:-1] = new_bg_mask[:,:,:,1:]
                img_right = torch.zeros_like(proj_imgs).to(device)
                img_right[:,:,:,:-1] = proj_imgs[:,:,:,1:]
                nbor_count = torch.where(mask_right.expand(-1,3,-1,-1), nbor_count, nbor_count+1)
                pix_color_avg = torch.where(mask_right.expand(-1,3,-1,-1), pix_color_avg, pix_color_avg + img_right)
                
                pix_color_avg = pix_color_avg / nbor_count

                is_edge = new_bg_mask & ~(mask_up & mask_down & mask_left & mask_right)
                proj_imgs = torch.where(is_edge.expand(-1,3,-1,-1), pix_color_avg, proj_imgs)
                new_bg_mask = torch.where(is_edge, False, new_bg_mask)
            prev_img = proj_imgs

            proj_imgs = ((proj_imgs * std + mean) * 255.).detach().squeeze(0).cpu().numpy()
            proj_imgs = proj_imgs.transpose(1, 2, 0)#.astype(np.uint8)
            proj_imgs_rsz = resize_image(proj_imgs)
            proj_imgs_rsz = np.transpose(proj_imgs_rsz, (2, 0, 1))
            proj_imgs_rsz = ((torch.from_numpy(proj_imgs_rsz).float() / 255. - mean) / std).unsqueeze(0).to(device)
     
            for k in range(len(model_paths)):
                modelname_split = model_paths[k].split('_')
                begin, end = int(modelname_split[2]), int(modelname_split[4][:5])
                if j < end:
                    # print(begin, end)
                    model_id = k
                    break
            if model_id != prev_model_id:
                dispnet = torch.load(model_paths[model_id])
                prev_model_id = model_id
            # dispnet['encoder'].eval()   
            # dispnet['decoder'].eval()
            with torch.no_grad():
                d_feature = dispnet['encoder'](proj_imgs_rsz)
                d_output = dispnet['decoder'](d_feature)
            depth = [d_output['disp', s] for s in opt.scales]
            depth = [d * opt.max_depth + opt.min_depth for d in depth][0]

            imwrite(os.path.join(root, 'outpainted_images/{:05}.png'.format(j)), proj_imgs.astype(np.uint8))
            np.save(os.path.join(root, 'outpainted_depths/{:05}.npy'.format(j)), depth.cpu().detach().numpy())
            # imwrite(os.path.join(root, 'outpainted_images_f/{:05}_f.png'.format(i+j)), proj_imgs_f[j])
            # np.save(os.path.join(root, 'outpainted_depths_f/{:05}_f.npy'.format(i+j)), proj_depths_f[j].cpu().detach().numpy())
            # imwrite(os.path.join(root, 'outpainted_images_b/{:05}_b.png'.format(i+j)), proj_imgs_b[j])
            # np.save(os.path.join(root, 'outpainted_depths_b/{:05}_b.npy'.format(i+j)), proj_depths_b[j].cpu().detach().numpy())
            
            prev_d = depth.squeeze(1)
            prev_pose = torch.from_numpy(smooth_poses[j]).float().unsqueeze(0).to(device)

    # bs = 8
    # kf_idx = -1
    # for i in range(0, len(img_paths), bs):
    #     imgs = []
    #     depths = []
    #     bg_masks = []
    #     kf_imgs_front = []
    #     kf_imgs_back = []
    #     kf_depths_front = []
    #     kf_depths_back = []
    #     rel_poses_f = []
    #     rel_poses_b = []

    #     for j in range(bs):
    #         if i + j == 0:
    #             kf_idx = 0
    #         if kf_idx < len(kf_indices) and i + j == kf_indices[kf_idx]:
    #             kf_idx += 1
    #         img = load_image(img_paths[i+j], False).unsqueeze(0).to(device)
    #         d = torch.from_numpy(np.load(depth_paths[i+j])).to(device)
    #         # d = trn.ToPILImage()(d)
    #         # d = trn.Resize((seq_io.height, seq_io.width))(d)
    #         # # d = trn.Resize((288, 384))(d)
    #         # d = trn.ToTensor()(d).to(device)
    #         bg_mask = load_mask(bg_mask_paths[i+j]).to(device)
    #         imgs.append(img)
    #         depths.append(d)
    #         bg_masks.append(bg_mask)

    #         cur_pose = torch.from_numpy(smooth_poses[i+j]).float().unsqueeze(0).to(device)
            
    #         if kf_idx == 0:
    #             kf_img_f = load_image(img_paths[0], True).unsqueeze(0).to(device)
    #             kf_d_f = torch.from_numpy(np.load(depth_paths[0]))
    #             kf_d_f = trn.ToPILImage()(kf_d_f)
    #             kf_d_f = trn.Resize((seq_io.height, seq_io.width))(kf_d_f)
    #             # d = trn.Resize((288, 384))(d)
    #             kf_d_f = trn.ToTensor()(kf_d_f).to(device)
    #             rel_pose_f = inverse_pose(cur_pose) @ torch.from_numpy(smooth_poses[0]).float().unsqueeze(0).to(device)
    #         else :
    #             kf_img_f = kf_imgs[kf_idx-1]
    #             kf_d_f = kf_depths[kf_idx-1].squeeze(1)
    #             kf_f_pose = torch.from_numpy(smooth_poses[kf_indices[kf_idx-1]]).float().unsqueeze(0).to(device)
    #             rel_pose_f = inverse_pose(cur_pose) @ kf_f_pose

    #         if kf_idx == len(kf_indices):
    #             kf_img_b = load_image(img_paths[-1], True).unsqueeze(0).to(device)
    #             kf_d_b = torch.from_numpy(np.load(depth_paths[-1]))
    #             kf_d_b = trn.ToPILImage()(kf_d_b)
    #             kf_d_b = trn.Resize((seq_io.height, seq_io.width))(kf_d_b)
    #             # d = trn.Resize((288, 384))(d)
    #             kf_d_b = trn.ToTensor()(kf_d_b).to(device)
    #             rel_pose_b = inverse_pose(cur_pose) @ torch.from_numpy(smooth_poses[-1]).float().unsqueeze(0).to(device)
    #         else :
    #             kf_img_b = kf_imgs[kf_idx]
    #             kf_d_b = kf_depths[kf_idx].squeeze(1)
    #             kf_b_pose = torch.from_numpy(smooth_poses[kf_indices[kf_idx]]).float().unsqueeze(0).to(device)
    #             rel_pose_b = inverse_pose(cur_pose) @ kf_b_pose

    #         kf_imgs_front.append(kf_img_f)
    #         kf_depths_front.append(kf_d_f)
    #         kf_imgs_back.append(kf_img_b)
    #         kf_depths_back.append(kf_d_b)
    #         rel_poses_f.append(rel_pose_f)
    #         rel_poses_b.append(rel_pose_b)

    #     imgs = torch.cat(imgs, 0)
    #     depths = torch.cat(depths, 0)
    #     bg_masks = torch.cat(bg_masks, 0)
    #     kf_imgs_front = torch.cat(kf_imgs_front, 0)
    #     kf_depths_front = torch.cat(kf_depths_front, 0)
    #     kf_imgs_back = torch.cat(kf_imgs_back, 0)
    #     kf_depths_back = torch.cat(kf_depths_back, 0)
    #     rel_poses_f = torch.cat(rel_poses_f, 0)
    #     rel_poses_b = torch.cat(rel_poses_b, 0)

    #     proj_imgs, proj_depths, proj_imgs_f, proj_depths_f, proj_imgs_b, proj_depths_b = project_keyframe_to_frame(\
    #             imgs, depths, bg_masks, kf_imgs_front, kf_depths_front, \
    #             kf_imgs_back, kf_depths_back, rel_poses_f, rel_poses_b)
    #     # output_dir_imgs = os.path.join(root, 'outpainted_images')
    #     # output_dir_depths = os.path.join(root, 'outpainted_depths')
    #     # output_dir_imgs_f = os.path.join(root, 'outpainted_images_f')
    #     # output_dir_depths_f = os.path.join(root, 'outpainted_depths_f')
    #     # output_dir_imgs_b = os.path.join(root, 'outpainted_images_b')
    #     # output_dir_depths_b = os.path.join(root, 'outpainted_depths_b')

    #     proj_imgs = ((proj_imgs * std + mean) * 255.).detach().cpu().numpy()
    #     proj_imgs = proj_imgs.transpose(0, 2, 3, 1).astype(np.uint8)
    #     proj_imgs_f = ((proj_imgs_f * std + mean) * 255.).detach().cpu().numpy()
    #     proj_imgs_f = proj_imgs_f.transpose(0, 2, 3, 1).astype(np.uint8)
    #     proj_imgs_b = ((proj_imgs_b * std + mean) * 255.).detach().cpu().numpy()
    #     proj_imgs_b = proj_imgs_b.transpose(0, 2, 3, 1).astype(np.uint8)
    #     for j in range(proj_imgs.shape[0]):
    #         imwrite(os.path.join(root, 'outpainted_images/{:05}.png'.format(i+j)), proj_imgs[j])
    #         np.save(os.path.join(root, 'outpainted_depths/{:05}.npy'.format(i+j)), proj_depths[j].cpu().detach().numpy())
    #         imwrite(os.path.join(root, 'outpainted_images_f/{:05}_f.png'.format(i+j)), proj_imgs_f[j])
    #         np.save(os.path.join(root, 'outpainted_depths_f/{:05}_f.npy'.format(i+j)), proj_depths_f[j].cpu().detach().numpy())
    #         imwrite(os.path.join(root, 'outpainted_images_b/{:05}_b.png'.format(i+j)), proj_imgs_b[j])
    #         np.save(os.path.join(root, 'outpainted_depths_b/{:05}_b.npy'.format(i+j)), proj_depths_b[j].cpu().detach().numpy())


if __name__ == '__main__':
    opt = options.Options().parse()
    global device
    device = torch.device(opt.cuda)
    global seq_io
    seq_io = SequenceIO(opt, preprocess=False)
    global warper
    warper = Warper(opt, seq_io.get_intrinsic(True)).to(device)
    root = os.path.join(opt.output_dir, opt.name)
    ## Load stabalized poses
    smooth_poses = np.load(os.path.join(root, 'poses_stab.npy'))

    ## Load stabalized images
    img_dir = os.path.join(root, 'images_stab')
    print(f"Input Images: {img_dir}")
    img_glob  = os.path.join(img_dir,"*.png")
    img_paths = sorted(glob.glob(img_glob))

    ## Load depths
    depth_dir = os.path.join(root, 'depths_stab')
    print(f"Input Depths: {depth_dir}")
    depth_glob  = os.path.join(depth_dir,"*.npy")
    depth_paths = sorted(glob.glob(depth_glob))

    ## Load background masks
    bg_mask_dir = os.path.join(root, "background_mask")
    print(f"Background Masks: {bg_mask_dir}")
    bg_mask_glob  = os.path.join(bg_mask_dir,"*.png")
    bg_mask_paths = sorted(glob.glob(bg_mask_glob))

    ## Load dispnet
    model_dir = os.path.join(root, "models")
    print(f"Dispnet: {model_dir}")
    model_glob  = os.path.join(model_dir,"*.pth")
    model_paths = sorted(glob.glob(model_glob))

    ## Load outpainted keyframe
    # Minimal cropped area keyframe
    outpnt_min_dir = '../pixelsynth/outputs_local_min'
    print(f"Outpainted Keyframe: {outpnt_min_dir}")
    outpnt_min_glob  = os.path.join(outpnt_min_dir,"*.png")
    outpnt_min_paths = sorted(glob.glob(outpnt_min_glob))

    outpnt_max_dir = '../pixelsynth/outputs_local_max'
    print(f"Outpainted Keyframe: {outpnt_max_dir}")
    outpnt_max_glob  = os.path.join(outpnt_max_dir,"*.png")
    outpnt_max_paths = sorted(glob.glob(outpnt_max_glob))

    keyframe_min_indices = []
    for i in range(len(outpnt_min_paths)):
        idx = int(outpnt_min_paths[i][-9:-4])
        keyframe_min_indices.append(idx)

    keyframe_max_indices = []
    for i in range(len(outpnt_max_paths)):
        idx = int(outpnt_max_paths[i][-9:-4])
        keyframe_max_indices.append(idx)

    kf_min_imgs,  kf_min_depths = predict_keyframe_depth(outpnt_min_paths, keyframe_min_indices, img_paths, bg_mask_paths, model_paths, opt)
    kf_max_imgs,  kf_max_depths = predict_keyframe_depth(outpnt_max_paths, keyframe_max_indices, img_paths, bg_mask_paths, model_paths, opt)
    # print(kf_depths, len(kf_depths), len(kf_depths) == len(keyframe_indices))
    # print(kf_imgs)
    for i, _ in enumerate(kf_min_depths):
        idx = keyframe_min_indices[i]
        d = kf_min_depths[i]
        # print(d.shape)
        img = kf_min_imgs[i]
        # print(img)
        kf_img_dir = os.path.join(root, "outpainted_images")
        kf_depth_dir = os.path.join(root, "outpainted_depths")
        img = ((img * std + mean) * 255.).squeeze(0).detach().cpu().numpy()
        img = img.transpose(1, 2, 0).astype(np.uint8)
        # img = imresize(img, (seq_io.origin_height, seq_io.origin_width))
        
        imwrite(os.path.join(kf_img_dir,'{:05}.png'.format(idx)), img)
        np.save(os.path.join(kf_depth_dir,'{:05}.npy'.format(idx)), d.cpu().detach().numpy())
    for i, _ in enumerate(kf_max_depths):
        idx = keyframe_max_indices[i]
        d = kf_max_depths[i]
        # print(d.shape)
        img = kf_max_imgs[i]
        # print(img)
        kf_img_dir = os.path.join(root, "outpainted_images")
        kf_depth_dir = os.path.join(root, "outpainted_depths")
        img = ((img * std + mean) * 255.).squeeze(0).detach().cpu().numpy()
        img = img.transpose(1, 2, 0).astype(np.uint8)
        # img = imresize(img, (seq_io.origin_height, seq_io.origin_width))
        
        imwrite(os.path.join(kf_img_dir,'{:05}.png'.format(idx)), img)
        np.save(os.path.join(kf_depth_dir,'{:05}.npy'.format(idx)), d.cpu().detach().numpy())
        
    warp_other_frame(
        img_paths, 
        depth_paths, 
        bg_mask_paths, 
        kf_min_imgs, 
        kf_min_depths, 
        kf_max_imgs, 
        kf_max_depths, 
        keyframe_min_indices, 
        keyframe_max_indices, 
        smooth_poses,
        model_paths,
    )