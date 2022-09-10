import argparse, sys, os, csv, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from path import Path

from sequence_io import SequenceIO
from models import *
from loss import Loss
from models.layers import disp_to_depth
from warper import Warper, pose_vec2mat, inverse_pose
import warnings

warnings.filterwarnings('ignore')

class GeometryOptimizer:
    
    def __init__(self, opt):

        self.opt = opt
        global device
        device = torch.device(opt.cuda)

        self.output_dir = Path(opt.output_dir)/opt.name
        (self.output_dir/'depths').makedirs_p()
        (self.output_dir/'models').makedirs_p()
        self.seq_io = SequenceIO(opt)
        print('=> sequence length = {}'.format(len(self.seq_io)))
       
        warper = Warper(opt, self.seq_io.get_intrinsic(True)).to(device)

        # model
        self.load_model()

        # Loss
        self.loss_function = Loss(opt, warper, self.seq_io)
        # self.kf_indices = self.seq_io.kf_indices

    def load_model(self):
        opt = self.opt
       
        input_channel = 3
        output_channel = 1
        dispnet_encoder = ResnetEncoder(18, True, input_channel).to(device)
        dispnet_decoder = DepthDecoder(dispnet_encoder.num_ch_enc, opt.scales, 
                                       num_output_channels=output_channel,
                                       h=opt.height, w=opt.width).to(device)
        dispnet_encoder.eval() #train()
        dispnet_decoder.train()

        self.dispnet = {'encoder': dispnet_encoder, 'decoder': dispnet_decoder}

        posenet_encoder = ResnetEncoder(18, True, input_channel).to(device)
        posenet_decoder = PoseDecoder(posenet_encoder.num_ch_enc, 1, output_channel).to(device)
        posenet_encoder.eval() #train()
        posenet_decoder.train()

        self.posenet = {'encoder': posenet_encoder, 'decoder': posenet_decoder}
        
        self.optim_params = [
                #{'params': dispnet_encoder.parameters(), 'initial_lr': opt.learning_rate},
                {'params': dispnet_decoder.parameters(), 'initial_lr': opt.learning_rate},
                #{'params': posenet_encoder.parameters(), 'initial_lr': opt.learning_rate},
                {'params': posenet_decoder.parameters(), 'initial_lr': opt.learning_rate}
        ]
        
        self.optimizer = optim.Adam(self.optim_params, betas=(0.9, 0.99))

    def run(self):
        opt = self.opt
        h, w = opt.height, opt.width
        print('=> optimize depths for each frame')
        self.n_iter = 0
        snippet_len = 1 + int(np.ceil((len(self.seq_io) - opt.batch_size) / (opt.batch_size - max(opt.intervals))))
        self.pbar = tqdm(total=opt.init_num_epochs + opt.num_epochs*snippet_len)
        
        # batch for initialization
        self.batch_idx = -1
        begin, end = self.get_batch_indices(self.batch_idx)
        init_batch_items = self.load_batch(begin, end)
        
        self.loss_function.preprocess_minibatch_weights(init_batch_items)
        # self.kf_depths = [torch.zeros(len(self.kf_indices),1,h,w).to(device).float() for s in self.opt.scales]
        # self.kf_poses = torch.zeros(len(self.kf_indices),4,4).to(device).float()

        for ep in range(opt.init_num_epochs):
            depths = self.optimize_snippet(init_batch_items, ep)
        # self.save_model('dispnet_begin_{:05}_end_{:05}.pth'.format(begin, end))

        if opt.num_epochs == 0: return
        
        for self.batch_idx in range(snippet_len):
            begin, end = self.get_batch_indices(self.batch_idx)
            batch_items = self.load_batch(begin, end)
            self.loss_function.preprocess_minibatch_weights(batch_items)
            for ep in range(opt.num_epochs):
                items = self.optimize_snippet(batch_items, ep)

            # save depth & pose results
            if  end == 0: end = None
            self.save_results(items, begin, end)
            # self.save_model('dispnet_begin_{:05}_end_{:05}.pth'.format(begin, end))
        self.pbar.close()

    def get_batch_indices(self, batch_idx):
        self.batch_idx = batch_idx
        if batch_idx <= 0:
            begin = 0
            self.prefix = 0
        else:
            begin = self.end - max(self.opt.intervals)
            self.prefix = max(self.opt.intervals)
        end = min(begin + self.opt.batch_size, len(self.seq_io))
        self.begin = begin
        self.end = end
        return begin, end

    def load_batch(self, begin, end):
        batch = self.seq_io.load_snippet(begin, end, load_flow=True)
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        batch["begin"] = begin
        batch["end"] = end
        return batch

    def optimize_snippet(self, items, ep):
        h, w = self.opt.height, self.opt.width
        begin, end = items["begin"], items["end"]
        bs = end - begin
        #d_features = self.dispnet['encoder'](items['imgs'][self.prefix:])
        if ep == 0:
            with torch.no_grad():
                self.d_features = self.dispnet['encoder'](items['imgs'][self.prefix:])
        d_outputs = self.dispnet['decoder'](self.d_features)

        depths = [d_outputs['disp', s] for s in self.opt.scales]
        depths = [depth * self.opt.max_depth + self.opt.min_depth for depth in depths]
        if self.prefix > 0: 
            depths = [torch.cat([self.fix_depths[s][-self.prefix:], depths[s]], 0) for s in self.opt.scales]
        items['depths'] = depths
        # for i, idx in enumerate(self.kf_indices):
        #     if idx < end:
        #         if idx >= begin:
        #             for s in self.opt.scales:
        #                 self.kf_depths[s][i,:,:,:] = depths[s][idx-begin,:,:,:].detach()
        # items['kf_depths'] = [self.kf_depths[s].unsqueeze(1).expand(-1,bs,-1,-1,-1) for s in self.opt.scales]
        if ep == 0:
            with torch.no_grad():
                self.p_features = self.posenet['encoder'](items['imgs'][max(0, self.prefix-1):])
        #p_features = self.posenet['encoder'](items['imgs'][max(0, self.prefix-1):])
        poses = self.posenet['decoder'](self.p_features)
        poses = pose_vec2mat(poses, self.opt.rotation_mode)
        poses = inverse_pose(poses[0].view(-1, 4, 4)).expand_as(poses) @ poses
        try:    poses = self.poses[-1].expand_as(poses).to(device) @ poses # repaired, original is self.poses[-1]...
        except: pass

        if self.prefix > 0:
            poses = torch.cat([self.poses[-self.prefix:].to(device), poses[1:]], 0)
        
        items['poses'] = poses
        items['poses_inv'] = inverse_pose(items['poses'])
        # for i, idx in enumerate(self.kf_indices):
        #     if idx < end:
        #         if idx >= begin:
        #             self.kf_poses[i,:,:] = poses[idx-begin,:,:].detach()

        # items['kf_poses'] = self.kf_poses.unsqueeze(1).expand(-1,bs,-1,-1)

        loss_items = self.loss_function(items)
        # if ep % 50 == 0:
        #     print("depths:")
        #     print(items['depths'])
        #     print("poses:")
        #     print(items['poses'])
        #     print("loss:", loss_items['full'])
            # print("photo loss:", loss_items['photo'])
            # print("flow loss:", loss_items['flow'])
            # print("geo loss:", loss_items['geo'])
            # print("ds loss:", loss_items['ds'])
            # print("photo loss kf:", loss_items['photo_kf'])
            # print("flow loss kf:", loss_items['flow_kf'])
            # print("geo loss kf:", loss_items['geo_kf'])
            # print("ds loss kf:", loss_items['ds_kf'])
        self.optimizer.zero_grad()
        loss_items['full'].backward()
        self.optimizer.step()

        self.n_iter += 1
        self.pbar.update(1)
        return items

    def save_results(self, items, start_idx, end):
        # depth
        save_indices = list(range(start_idx, end))
        self.fix_depths = [items['depths'][s][-self.prefix:].detach() for s in self.opt.scales]
        self.seq_io.save_depths(items['depths'], save_indices)

        # pose
        poses = items['poses'].cpu().detach()
        try:    self.poses = torch.cat([self.poses[:-self.prefix], poses], dim=0)
        except: self.poses = poses
        self.seq_io.save_poses(self.poses)

        # mask
        err_masks = items['err_mask'].cpu().detach()
        self.seq_io.save_errors(err_masks, save_indices[:items['err_mask'].size(0)])

    def save_model(self, file_name):
        torch.save(self.dispnet, self.output_dir/'models'/file_name)

if __name__ == '__main__':
    import options
    opt = options.Options().parse()
    geo_optim = GeometryOptimizer(opt)
    geo_optim.run()

