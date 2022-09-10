import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from imageio import imread, imwrite
from path import Path
import random, os, sys, glob, subprocess
from skimage.transform import resize as imresize
from skimage import color
import cv2 as cv
from flow import *

class SequenceIO(data.Dataset):
    def __init__(self, opt, preprocess=True): 
        self.opt = opt
        global device
        device = torch.device(opt.cuda)

        self.input_video = opt.video_path
        self.root = Path(opt.output_dir)/opt.name
        self.dynamic_mask_dir = self.root/'masks'
        self.root.makedirs_p()
        self.batch_size = opt.batch_size
        self.mean = opt.img_mean
        self.std = opt.img_std
        self.preprocess = preprocess
        # self.kf_indices = None
        if preprocess:
            self.extract_frames()
            self.load_video()
            self.generate_flows()
            self.generate_dynamic_mask()
        else:
            self.load_video()
            # self.generate_flows()

        self.load_intrinsic()

    def extract_frames(self):
        (self.root/'images').makedirs_p()
        os.system('ffmpeg -y -hide_banner -loglevel panic -i "{}" {}/%05d.png'.format(self.input_video, self.root/'images'))

    def generate_dynamic_mask(self):
        import dynamic_mask_generation

        self.dynamic_mask_dir.makedirs_p()
        args, _ = dynamic_mask_generation.get_parser().parse_known_args()
        args.input = [self.root/'images/*.png']
        args.output = self.dynamic_mask_dir

        dynamic_mask_generation.dynamic_mask_generation(args)

    def load_video(self): 
        self.image_names = sorted(list(glob.glob(self.root/'images/*.png')))
        # self.image_names_L = sorted(list(glob.glob(self.root/'images_L/*.png')))
        # self.image_names_R = sorted(list(glob.glob(self.root/'images_R/*.png')))

        # get frame size
        sample_image = imread(self.image_names[0])
        self.origin_size = sample_image.shape[:2]
        self.origin_height, self.origin_width = self.origin_size
        #self.height, self.width = self.opt.height, self.opt.width
        
        if self.origin_height > self.origin_width:
            a, b = self.origin_height, self.origin_width
        else:
            a, b = self.origin_width, self.origin_height

        a_depth = self.opt.depth_size
        if a >= 1024:
            a_flow = 1024
        else:
            a_flow = int(np.round(a / 64) * 64)

        b_flow = int(np.round(b * a_flow / a / 64) * 64)
        b_depth = int(np.round(b * a_depth / a / 32) * 32)

        if self.origin_height > self.origin_width:
            self.height, self.width = a_depth, b_depth
            self.flow_height, self.flow_width = a_flow, b_flow
        else:
            self.height, self.width = b_depth, a_depth
            self.flow_height, self.flow_width = b_flow, a_flow
        self.image_flow_names = [f.replace('images', 'images_flow') for f in self.image_names]

        if self.preprocess:
            (self.root/'images_flow').makedirs_p()
            for f, f_flow in zip(self.image_names, self.image_flow_names):
                image = imread(f)
                image_flow = imresize(image, (self.flow_height, self.flow_width))
                imwrite(f_flow, (image_flow*255.).astype(np.uint8))

        setattr(self.opt, 'height', self.height)
        setattr(self.opt, 'width', self.width)

        self.need_resize = True

        # get fps
        p = subprocess.check_output(['ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {}'.format(self.input_video)], shell=True)
        exec('self.fps = int({})'.format(p.decode('utf-8').rstrip('\n')))

    def get_intrinsic(self, resize=False):
        return self.intrinsic_res if resize else self.intrinsic

    def load_intrinsic(self):
        self.intrinsic = torch.FloatTensor([[500., 0, self.origin_width*0.5], [0, 500., self.origin_height*0.5], [0, 0, 1]])
        if self.need_resize:
            self.intrinsic_res = self.intrinsic.clone()
            self.intrinsic_res[0] *= (self.width / self.origin_width)
            self.intrinsic_res[1] *= (self.height / self.origin_height)

    def generate_flows(self):
        print('=> preparing optical flow. it would take a while.')
        flow = FlowProcessor(self.opt).to(device)
        
        flow.compute_sequence(self.image_flow_names, self.root/'flows', pre_homo=True, consistency_thresh=1.0, 
                intervals=self.opt.intervals)

        del flow
        # self.kf_indices = kf_indices

    def load_flow_snippet(self, begin, end, interval):
        w, h, W, H = self.width, self.height, self.origin_width, self.origin_height
        
        flows = []
        for i in range(begin, end - interval):
            flow_item = np.load(self.root/'flows'/str(interval)/'{}.npy'.format(os.path.split(self.image_names[i])[-1]))
            flow_item = torch.from_numpy(flow_item).float().to(device)
            flow_f = flow_item[:, :2]
            flow_b = flow_item[:, 2:4]
            flow_fb = torch.cat((flow_f, flow_b), 0)
            flow_fb = normalize_for_grid_sample(flow_fb).permute(0, 3, 1, 2)

            flows.append(flow_fb)

        flows = torch.cat(flows, 0)
        flows = F.interpolate(flows, (h, w), mode='area')
        flows = flows.reshape(-1, 2, 2, h, w).permute(1, 0, 3, 4, 2)

        return flows

    def load_flow_keyframe(self, begin, end, kf_index):
        w, h, W, H = self.width, self.height, self.origin_width, self.origin_height
        
        flows = []
        for i in range(begin, end):
            flow_item = np.load(self.root/'flows'/f'kf_{kf_index}'/'{}.npy'.format(os.path.split(self.image_names[i])[-1]))
            flow_item = torch.from_numpy(flow_item).float().to(device)
            flow_f = flow_item[:, :2]
            flow_b = flow_item[:, 2:4]
            flow_fb = torch.cat((flow_f, flow_b), 0)
            flow_fb = normalize_for_grid_sample(flow_fb).permute(0, 3, 1, 2)

            flows.append(flow_fb)

        flows = torch.cat(flows, 0)
        flows = F.interpolate(flows, (h, w), mode='area')
        flows = flows.reshape(-1, 2, 2, h, w).permute(1, 0, 3, 4, 2)

        return flows

    def load_depth_files(self, index, size):
        depth_path = self.root/'depths/{:05}.npy'.format(index)
        depth = np.load(depth_path)
        return torch.from_numpy(depth).float()

    def __len__(self):
        return len(self.image_names)

    def load_mask(self, index):
        return (torch.from_numpy(imread(self.dynamic_mask_dir/os.path.split(self.image_names[index])[-1])).float() / 255.).to(device).unsqueeze(0)

    def load_snippet(self, begin, end, load_flow=False):
        items = {}
        items['imgs'] = torch.stack([self.load_image(i) for i in range(begin, end)], 0)
        items['mask'] = torch.stack([self.load_mask(i) for i in range(begin, end)], 0)
        items['mask'] = F.interpolate(items['mask'], (self.height, self.width), mode='area')
        # items['imgs_L'] = torch.stack([self.load_image_L(i) for i in range(begin, end)], 0)
        # items['imgs_R'] = torch.stack([self.load_image_R(i) for i in range(begin, end)], 0)
        if load_flow:
            for i in self.opt.intervals:
                flows = self.load_flow_snippet(begin, end, i)
                items[('flow_fwd', i)] = flows[0]
                items[('flow_bwd', i)] = flows[1]
        
        return items

    def create_video_writer(self, crop_size, filename):
        print('=> The output video will be saved as {}'.format(self.root/filename))
        self.video_writer = cv.VideoWriter(self.root/filename, cv.VideoWriter_fourcc(*'MJPG'), int(self.fps), crop_size)

    def write_images(self, imgs):
        # write torch.Tensor images into cv.VideoWriter
        imgs = ((imgs * self.std + self.mean) * 255.).detach().cpu().numpy()
        imgs = imgs.transpose(0, 2, 3, 1).astype(np.uint8)[..., ::-1]

        for i in range(imgs.shape[0]):
            self.video_writer.write(imgs[i])

    def write_images_stab(self, imgs, indices, dirname):
        (self.root/dirname).makedirs_p()
        # (self.root/'images_warpy').makedirs_p()
        imgs = ((imgs * self.std + self.mean) * 255.).detach().cpu().numpy()
        imgs = imgs.transpose(0, 2, 3, 1).astype(np.uint8)
        
        # warp[warp < 0.0] = 0.0
        # warp[warp > 1.0] = 1.0
        for i, idx in enumerate(indices):
            # img_rsz = imresize(imgs[i], (256,256))
            imwrite(self.root/dirname/'{:05}.png'.format(idx), imgs[i])
            # imwrite(self.root/'images_warpx/{:05}.png'.format(idx), warp[i,:,:,0])
            # imwrite(self.root/'images_warpy/{:05}.png'.format(idx), warp[i,:,:,1])

    def write_background(self, warp_map, indices):
        (self.root/'background_mask').makedirs_p()
        # (self.root/'flow_stab').makedirs_p()
        # warp = warp_map.detach().cpu().numpy()
        background = (warp_map.abs().max(dim=-1)[0] <= 1).detach().cpu().numpy()
        # print(background.shape)
        for i, idx in enumerate(indices):
            np.save(self.root/'background_mask/{:05}.npy'.format(idx),background[i])
            # np.save(self.root/'flow_stab/{:05}.npy'.format(idx),warp[i])

    def load_image(self, index, normalize = True):
        img = imread(self.image_names[index]).astype(np.float32)
        if self.need_resize:
            img = imresize(img, (self.height, self.width))
        img = np.transpose(img, (2, 0, 1))
        if normalize:
            tensor_img = (torch.from_numpy(img).float() / 255 - self.mean) / self.std
        else:
            tensor_img = torch.from_numpy(img).float() / 255
        return tensor_img

    def load_image_L(self, index):
        img = imread(self.image_names_L[index]).astype(np.float32)
        if self.need_resize:
            img = imresize(img, (self.height, self.width))
        img = np.transpose(img, (2, 0, 1))
        tensor_img = (torch.from_numpy(img).float() / 255 - self.mean) / self.std
        return tensor_img

    def load_image_R(self, index):
        img = imread(self.image_names_R[index]).astype(np.float32)
        if self.need_resize:
            img = imresize(img, (self.height, self.width))
        img = np.transpose(img, (2, 0, 1))
        tensor_img = (torch.from_numpy(img).float() / 255 - self.mean) / self.std
        return tensor_img

    def save_depths(self, depths, indices):
        (self.root/'depths').makedirs_p()
        for i, idx in enumerate(indices):
            np.save(self.root/'depths/{:05}.npy'.format(idx), depths[0][i].cpu().detach().numpy())

    def save_depths_stab(self, depths, indices, dir_name):
        (self.root/dir_name).makedirs_p()
        for i, idx in enumerate(indices):
            np.save(self.root/dir_name/'{:05}.npy'.format(idx), depths[i].cpu().detach().numpy())

    def load_depths(self, indices):
        depths = []
        for idx in indices:
            depth = np.load(self.root/'depths/{:05}.npy'.format(idx))
            depths.append(depth)
        depths = np.stack(depths, axis=0)
        depths = torch.from_numpy(depths).float()
        return depths

    def save_errors(self, errors, indices):
        (self.root/'errors').makedirs_p()
        for i, idx in enumerate(indices):
            np.save(self.root/'errors/{:05}.npy'.format(idx), errors[i].cpu().detach().numpy())

    def load_errors(self, indices):
        errors = []
        for idx in indices:
            try:
                error = np.load(self.root/'errors/{:05}.npy'.format(idx))
            except:
                # the last frame has no corresponding error map
                error = np.zeros(errors[-1].shape)
            errors.append(error)
        errors = np.stack(errors, axis=0)
        return errors

    def load_poses(self):
        return np.load(self.root/'poses.npy')

    def load_poses_stab(self):
        return np.load(self.root/'poses_stab.npy')

    def save_poses(self, poses):
        np.save(self.root/'poses.npy', poses.numpy())

    def save_poses_stab(self, poses):
        np.save(self.root/'poses_stab.npy', poses)

    def save_poses_LR(self, poses, filename):
        np.save(self.root/filename, poses)

    def save_H(self, H_inv, imgs, reproj_imgs, indices):
        (self.root/'images_warpx').makedirs_p()
        (self.root/'images_warpy').makedirs_p()
        imgs = ((imgs * self.std + self.mean) * 255.).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        reproj_imgs = ((reproj_imgs * self.std + self.mean) * 255.).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for i, idx in enumerate(indices):
            np.save(self.root/'images_warpx/{:05}_H_inv.npy'.format(idx), H_inv[i])
            processed = cv.warpPerspective(reproj_imgs[i],H_inv[i],(960, 576))
            imwrite(self.root/'images_warpy/{:05}.png'.format(idx), processed)


    # def save_model(self, model, file_name):
    #     (self.root/'models').makedirs_p()
    #     torch.save(model, self.root/'models'/file_name)
