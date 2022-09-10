from __future__ import absolute_import, division, print_function
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warper import *

def gradient_x(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx

def gradient_y(img):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy

def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum() # L1-loss (Deep3D)
    return mean_value


class Loss(nn.Module):
    def __init__(self, opt, warper, seq_io=None):
        super(Loss, self).__init__()
        
        self.opt = opt
        global device
        device = torch.device(opt.cuda)
        self.scales = opt.scales
        self.intervals = opt.intervals
        # self.kf_indices = seq_io.kf_indices
        self.ssim_weight = opt.ssim_weight

        self.ssim = SSIM(window_size=3).to(device)
        self.warper = warper

        self.width = opt.width
        self.height = opt.height
        
        self.weights = {
            'photo':opt.photometric_loss,
            'flow': opt.flow_loss,
            'geo':  opt.geometry_loss,
            'ds': opt.disparity_smoothness_loss,
            # 'photo_kf':opt.photometric_loss_keyframe,
            # 'flow_kf': opt.flow_loss_keyframe,
            # 'geo_kf':  opt.geometry_loss_keyframe,
            # 'ds_kf': opt.disparity_smoothness_loss_keyframe
        }

        self.pixel_map = self.warper.pixel_map.permute(0, 3, 1, 2)
        self.pixel_map_norm = torch.zeros_like(self.pixel_map).to(device)
        self.pixel_map_norm[:,0,:,:] = self.pixel_map[:,0,:,:] * 2 / (self.width - 1) - 1
        self.pixel_map_norm[:,1,:,:] = self.pixel_map[:,1,:,:] * 2 / (self.height - 1) - 1

    def gradient_x(self, img):
        gx = torch.zeros_like(img).to(device)
        gx[:,:,:,:-1] += img[:,:,:,1:] - img[:,:,:,:-1]
        gx[:,:,:,1:] += img[:,:,:,1:] - img[:,:,:,:-1]
        gx[:,:,:,1:-1] /= 2.
        gx = gx.abs()
        return gx

    def gradient_y(self, img):
        gy = torch.zeros_like(img).to(device)
        gy[:,:,:-1,:] += img[:,:,1:,:] - img[:,:,:-1,:]
        gy[:,:,1:,:] += img[:,:,1:,:] - img[:,:,:-1,:]
        gy[:,:,1:-1,:] /= 2.
        gy = gy.abs()
        return gy

    def preprocess_minibatch_weights(self, items):
        opt = self.opt
        
        self.bs = items['imgs'].size(0)
        self.interval_weights = {}
        self.adaptive_weights = {}
        
        # compute the weights for different source view
        alpha_sum = 0
        self.alpha = {}
        self.beta = {}
        for i in self.intervals:
            if i >= self.bs: continue
            self.alpha[i] = np.power(opt.adaptive_alpha, i)
            self.beta[i] = np.power(opt.adaptive_beta, i)

        alpha_sum = sum([self.alpha[k] for k in self.alpha.keys()])
        beta_sum = sum([self.beta[k] for k in self.beta.keys()])

        for k in self.alpha.keys():
            self.alpha[k] /= alpha_sum
            self.beta[k] /= beta_sum

        # h, w = self.height, self.width

        # self.alpha_kf = {}
        # self.beta_kf = {}
        # for i, idx in enumerate(self.kf_indices):
        #     if idx >= items["end"]: continue 
        #     flow12 = items[('flow_fwd_kf', idx)].clone()
        #     flow21 = items[('flow_bwd_kf', idx)].clone()
        #     valid12 = (flow12[:,:,:,0] >= -1.) & (flow12[:,:,:,0] <= 1.) & (flow12[:,:,:,1] >= -1.) & (flow12[:,:,:,1] <= 1.)
        #     valid21 = (flow21[:,:,:,0] >= -1.) & (flow21[:,:,:,0] <= 1.) & (flow21[:,:,:,1] >= -1.) & (flow21[:,:,:,1] <= 1.)
        #     valid_sum12 = torch.sum(valid12.float(), dim=(2,1))
        #     valid_sum21 = torch.sum(valid21.float(), dim=(2,1))
        #     valid_sum = torch.min((valid_sum12 + valid_sum21) / 2)
        #     valid_sum = valid_sum.item()
        #     if valid_sum < h * w * self.opt.keyframe_thr:
        #         self.alpha_kf[idx] = 0.
        #         self.beta_kf[idx] = 0.
        #     else:
        #         pow_val = math.ceil(np.power(float(h) * w / valid_sum, 2))
        #         self.alpha_kf[idx] = np.power(opt.adaptive_alpha, pow_val)
        #         self.beta_kf[idx] = np.power(opt.adaptive_beta, pow_val)
        #     # self.alpha_kf[idx] = 0.3
        #     # self.beta_kf[idx] = 0.3

            
        # alpha_sum = sum([self.alpha_kf[k] for k in self.alpha_kf.keys()])
        # beta_sum = sum([self.beta_kf[k] for k in self.beta_kf.keys()])

        # for k in self.alpha_kf.keys():
        #     self.alpha_kf[k] /= alpha_sum
        #     self.beta_kf[k] /= beta_sum
        # print("geo and photo weight: local {}, global {}".format(self.alpha, self.alpha_kf))
        # print("photo weight: local {}, global {}".format(self.beta, self.beta_kf))
    
    def compute_loss_terms(self, items):
        # compute the loss of a given snippet with multiple frame intervals
        bs = items['imgs'].size(0)
        h, w = self.height, self.width

        loss_items = {}
        for key in self.weights.keys():
            loss_items[key] = 0

        poses = items['poses']
        poses_inv = items['poses_inv']
        
        for i in self.intervals:
            if i >= bs: continue
            pair_item = {'img1':    items['imgs'][:-i],
                         'img2':    items['imgs'][i:],
                         'mask1':   items['mask'][:-i],
                         'mask2':   items['mask'][i:],
                         'depth1':  [depth[:-i] for depth in items['depths']],
                         'depth2':  [depth[i:] for depth in items['depths']],
                         'pose21':  poses_inv[:-i] @ poses[i:],
                         'pose12':  poses_inv[i:] @ poses[:-i],
                         'flow12':  items[('flow_fwd', i)],
                         'flow21':  items[('flow_bwd', i)]}
            pair_item['alpha'] = self.alpha[i]
            pair_item['beta'] = self.beta[i]
            pair_loss, err_mask = self.compute_pairwise_loss(pair_item)

            for name in loss_items.keys():
                if name not in pair_loss.keys(): continue
                loss_items[name] += pair_loss[name] 
            try:
                m = err_mask.size(0)
                n = items['err_mask'].size(0)
                items['err_mask'][:m-n] += err_mask
            except Exception as e:
                items['err_mask'] = err_mask

        # Here we calculate losses between each frame and keyframe
        # for i, idx in enumerate(self.kf_indices):
        #     if idx >= items['end']: continue
        #     kf_poses = items['kf_poses'][i]
        #     kf_poses_inv = inverse_pose(kf_poses)
        #     pair_item = {'img1':    items['imgs'],
        #                  'img2':    items['kf_imgs'][i].unsqueeze(0).expand(bs,-1,-1,-1),
        #                  'mask1':   items['mask'],
        #                  'mask2':   items['kf_mask'][i].unsqueeze(0).expand(bs,-1,-1,-1),
        #                  'depth1':  [depth for depth in items['depths']],
        #                  'depth2':  [depth[i] for depth in items['kf_depths']],
        #                  'pose21':  poses_inv @ kf_poses,
        #                  'pose12':  kf_poses_inv @ poses,
        #                  'flow12':  items[('flow_fwd_kf', idx)],
        #                  'flow21':  items[('flow_bwd_kf', idx)],
        #                  'begin':  items['begin'],
        #                  'end':  items['end'],
        #                  'kf_idx':  idx}
            
        #     pair_item['alpha'] = self.alpha_kf[idx]
        #     pair_item['beta'] = self.beta_kf[idx]
        #     pair_loss, err_mask = self.compute_frame_to_keyframe_loss(pair_item)

        #     for name in loss_items.keys():
        #         if name not in pair_loss.keys(): continue
        #         loss_items[name] += pair_loss[name] 
            # try:
            #     m = err_mask.size(0)
            #     n = items['err_mask'].size(0)
            #     items['err_mask'][:m-n] += err_mask
            # except Exception as e:
            #     items['err_mask'] = err_mask

        return loss_items


    def compute_pairwise_loss(self, item):
        # compute the loss a given snippet with a frame interval
        img1, img2 = item['img1'], item['img2']
        pose12, pose21 = item['pose12'], item['pose21']
        input_flow12 = item['flow12'].permute(0, 3, 1, 2)
        input_flow21 = item['flow21'].permute(0, 3, 1, 2)

        bs = img1.size(0)
        loss_items = {}
        for key in self.weights.keys():
            loss_items[key] = 0

        for scale in self.scales:
            depth1_scaled = item['depth1'][scale]
            depth2_scaled = item['depth2'][scale]

            ret1 = self.warper.inverse_warp(img2, depth1_scaled, depth2_scaled, pose12)
            ret2 = self.warper.inverse_warp(img1, depth2_scaled, depth1_scaled, pose21)

            rec1, mask1, projected_depth1, computed_depth1, warp_sample1, pt1, pt12 = ret1
            rec2, mask2, projected_depth2, computed_depth2, warp_sample2, pt2, pt21 = ret2

            # mask1 *= item['mask1']
            # mask2 *= item['mask2']

            # geometry loss
            diff_depth1 = ((computed_depth1 - projected_depth1).abs() /
                           (computed_depth1 + projected_depth1).abs()).clamp(0, 1)
            diff_depth2 = ((computed_depth2 - projected_depth2).abs() /
                           (computed_depth2 + projected_depth2).abs()).clamp(0, 1)
            
            diff_depth1 *= item['alpha']
            diff_depth2 *= item['alpha']

            loss_items['geo'] += mean_on_mask(diff_depth1, mask1)
            loss_items['geo'] += mean_on_mask(diff_depth2, mask2)
            
            # photometric loss
            diff_img1 = (img1 - rec1).abs()
            diff_img2 = (img2 - rec2).abs()
            if self.ssim_weight > 0:
                ssim_map1 = self.ssim(img1, rec1)
                ssim_map2 = self.ssim(img2, rec2)
                diff_img1 = (1-self.ssim_weight)*diff_img1 + self.ssim_weight*ssim_map1
                diff_img2 = (1-self.ssim_weight)*diff_img2 + self.ssim_weight*ssim_map2

            loss_items['photo'] += mean_on_mask(diff_img1 * item['alpha'], mask1)
            loss_items['photo'] += mean_on_mask(diff_img2 * item['alpha'], mask2)
            
            warp_flow1 = warp_sample1.permute(0, 3, 1, 2)
            warp_flow2 = warp_sample2.permute(0, 3, 1, 2)

            # flow
            diff_flow1 = (warp_flow1 - input_flow12).abs().sum(1, keepdim=True)
            diff_flow2 = (warp_flow2 - input_flow21).abs().sum(1, keepdim=True)

            diff_flow1 *= item['beta']
            diff_flow2 *= item['beta']
            loss_items['flow'] += mean_on_mask(diff_flow1, mask1)
            loss_items['flow'] += mean_on_mask(diff_flow2, mask2)
            
            # disparity smoothness on consistent part
            # disp1 = warp_flow1 - self.pixel_map_norm
            # disp[:,0,:,:] = (disp[:,0,:,:] + 1)* (self.width - 1) / 2
            # disp[:,1,:,:] = (disp[:,1,:,:] + 1)* (self.height - 1) / 2
            # disp_gradients_x = self.gradient_x(disp1)
            # disp_gradients_y = self.gradient_y(disp1)

            # image_gradients_x = self.gradient_x(img1)
            # image_gradients_y = self.gradient_y(img1)

            # weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
            # weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

            # smoothness_x = disp_gradients_x * weights_x
            # smoothness_y = disp_gradients_y * weights_y

            # disparity sharpness on edge part
            # weights_x = -torch.log(1 + torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
            # weights_y = -torch.log(1 + torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

            # sharpness_x = disp_gradients_x * weights_x
            # sharpness_y = disp_gradients_y * weights_y

            # sum of smoothness and sharpness
            # loss_items['ds'] += mean_on_mask(smoothness_x + smoothness_y, mask1)

            # disp2 = warp_flow2 - self.pixel_map_norm
            # disp[:,0,:,:] = (disp[:,0,:,:] + 1)* (self.width - 1) / 2
            # disp[:,1,:,:] = (disp[:,1,:,:] + 1)* (self.height - 1) / 2
            # disp_gradients_x = self.gradient_x(disp2)
            # disp_gradients_y = self.gradient_y(disp2)

            # image_gradients_x = self.gradient_x(img2)
            # image_gradients_y = self.gradient_y(img2)

            # weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
            # weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

            # smoothness_x = disp_gradients_x * weights_x
            # smoothness_y = disp_gradients_y * weights_y

            # disparity sharpness on edge part
            # weights_x = -torch.log(1 + torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
            # weights_y = -torch.log(1 + torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

            # sharpness_x = disp_gradients_x * weights_x
            # sharpness_y = disp_gradients_y * weights_y

            # sum of smoothness and sharpness
            # loss_items['ds'] += mean_on_mask(smoothness_x + smoothness_y, mask2)
            
            # return error mask for post-processing
            err_mask = torch.abs(diff_img1.mean(1, keepdim=True)) * mask1

        return loss_items, err_mask

    # def compute_frame_to_keyframe_loss(self, item):
    #     # compute the loss a given snippet with a frame interval
    #     img1, img2 = item['img1'], item['img2']
    #     pose12, pose21 = item['pose12'], item['pose21']
    #     input_flow12 = item['flow12'].permute(0, 3, 1, 2)
    #     input_flow21 = item['flow21'].permute(0, 3, 1, 2)

    #     bs = img1.size(0)
    #     loss_items = {}
    #     for key in self.weights.keys():
    #         loss_items[key] = 0

    #     for scale in self.scales:
    #         depth1_scaled = item['depth1'][scale]
    #         depth2_scaled = item['depth2'][scale]

    #         ret1 = self.warper.inverse_warp(img2, depth1_scaled, depth2_scaled, pose12)
    #         ret2 = self.warper.inverse_warp(img1, depth2_scaled, depth1_scaled, pose21)

    #         rec1, mask1, projected_depth1, computed_depth1, warp_sample1, pt1, pt12 = ret1
    #         rec2, mask2, projected_depth2, computed_depth2, warp_sample2, pt2, pt21 = ret2

    #         # mask1 *= item['mask1']
    #         # mask2 *= item['mask2']

    #         if item['kf_idx'] < item['end'] and item['kf_idx'] >= item['begin']:
    #             mask1[item['kf_idx']-item['begin']] *= 0.
    #             mask2[item['kf_idx']-item['begin']] *= 0.

    #         # geometry loss
    #         diff_depth1 = ((computed_depth1 - projected_depth1).abs() /
    #                        (computed_depth1 + projected_depth1).abs()).clamp(0, 1)
    #         diff_depth2 = ((computed_depth2 - projected_depth2).abs() /
    #                        (computed_depth2 + projected_depth2).abs()).clamp(0, 1)
            
    #         diff_depth1 *= item['alpha']
    #         # diff_depth2 *= item['alpha']

    #         loss_items['geo_kf'] += mean_on_mask(diff_depth1, mask1)
    #         # loss_items['geo_kf'] += mean_on_mask(diff_depth2, mask2)
            
    #         # photometric loss
    #         diff_img1 = (img1 - rec1).abs()
    #         diff_img2 = (img2 - rec2).abs()
    #         if self.ssim_weight > 0:
    #             ssim_map1 = self.ssim(img1, rec1)
    #             ssim_map2 = self.ssim(img2, rec2)
    #             diff_img1 = (1-self.ssim_weight)*diff_img1 + self.ssim_weight*ssim_map1
    #             diff_img2 = (1-self.ssim_weight)*diff_img2 + self.ssim_weight*ssim_map2

    #         loss_items['photo_kf'] += mean_on_mask(diff_img1 * item['alpha'], mask1)
    #         # loss_items['photo_kf'] += mean_on_mask(diff_img2 * item['alpha'], mask2)
            
    #         warp_flow1 = warp_sample1.permute(0, 3, 1, 2)
    #         warp_flow2 = warp_sample2.permute(0, 3, 1, 2)

    #         # flow
    #         diff_flow1 = (warp_flow1 - input_flow12).abs().sum(1, keepdim=True)
    #         diff_flow2 = (warp_flow2 - input_flow21).abs().sum(1, keepdim=True)

    #         diff_flow1 *= item['beta']
    #         diff_flow2 *= item['beta']
    #         loss_items['flow_kf'] += mean_on_mask(diff_flow1, mask1)
    #         # loss_items['flow_kf'] += mean_on_mask(diff_flow2, mask2)
            
    #         # disparity smoothness on consistent part
    #         disp1 = warp_flow1 - self.pixel_map_norm
    #         # disp[:,0,:,:] = (disp[:,0,:,:] + 1)* (self.width - 1) / 2
    #         # disp[:,1,:,:] = (disp[:,1,:,:] + 1)* (self.height - 1) / 2
    #         disp_gradients_x = self.gradient_x(disp1)
    #         disp_gradients_y = self.gradient_y(disp1)

    #         image_gradients_x = self.gradient_x(img1)
    #         image_gradients_y = self.gradient_y(img1)

    #         weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    #         weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    #         smoothness_x = disp_gradients_x * weights_x
    #         smoothness_y = disp_gradients_y * weights_y

    #         # disparity sharpness on edge part
    #         # weights_x = -torch.log(1 + torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    #         # weights_y = -torch.log(1 + torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    #         # sharpness_x = disp_gradients_x * weights_x
    #         # sharpness_y = disp_gradients_y * weights_y

    #         # sum of smoothness and sharpness
    #         loss_items['ds_kf'] += mean_on_mask(smoothness_x + smoothness_y, mask1)
            
    #         # return error mask for post-processing
    #         err_mask = torch.abs(diff_img1.mean(1, keepdim=True)) * mask1

    #     # return loss_items, err_mask


    def forward(self, items):
        bs = items['imgs'].size(0)
        loss_items = self.compute_loss_terms(items)

        loss_items['full'] = 0
        for key in self.weights.keys():
            loss_items['full'] += self.weights[key] * loss_items[key]
        
        return loss_items

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, window_size=3, alpha=1, beta=1, gamma=1):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(window_size, 1)
        self.mu_y_pool   = nn.AvgPool2d(window_size, 1)
        self.sig_x_pool  = nn.AvgPool2d(window_size, 1)
        self.sig_y_pool  = nn.AvgPool2d(window_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(window_size, 1)

        self.refl = nn.ReflectionPad2d(window_size//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.C3 = self.C2 / 2
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if alpha == 1 and beta == 1 and gamma == 1:
            self.run_compute = self.compute_simplified
        else:
            self.run_compute = self.compute
        

    def compute(self, x, y):
        
        x = self.refl(x)
        y = self.refl(y)
        
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        l = (2 * mu_x * mu_y + self.C1) / \
            (mu_x * mu_x + mu_y * mu_y + self.C1)
        c = (2 * sigma_x * sigma_y + self.C2) / \
            (sigma_x + sigma_y + self.C2)
        s = (sigma_xy + self.C3) / \
            (torch.sqrt(sigma_x * sigma_y) + self.C3)

        ssim_xy = torch.pow(l, self.alpha) * \
                  torch.pow(c, self.beta) * \
                  torch.pow(s, self.gamma)
        return torch.clamp((1 - ssim_xy) / 2, 0, 1)

    def compute_simplified(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    def forward(self, x, y):
        return self.run_compute(x, y)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
