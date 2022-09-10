# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Depth Regressor options")

        # RUN COMMAND
        self.parser.add_argument('video_path',
                                 type=str,
                                 help='path to input video')
        self.parser.add_argument("--name",
                                 type=str,
                                 help="the name of the video folder to process",
                                 default="test")

        self.parser.add_argument("--output_dir",
                                 type=str,
                                 help='output depths directory',
                                 default='outputs')
        

        # TRAINING options
        self.parser.add_argument("--depth_size",
                                 type=int,
                                 help="input image width",
                                 default=384)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--intervals",
                                 nargs="+",
                                 type=int,
                                 help='the interval of  nearby view for supvervision',
                                 default=[1, 4, 9]) # [1, 2, 3] for kitti, [1, 4, 9] else
        self.parser.add_argument('--rotation_mode',
                                 type=str,
                                 choices=['euler', 'quat'],
                                 default='quat',
                                 help='the rotation mode of pose vector')
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=1e-3)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=10.0)
        self.parser.add_argument('--img_mean',
                                 type=float,
                                 help='normalized mean',
                                 default=0.45)
        self.parser.add_argument('--img_std',
                                 type=float,
                                 help='normalized standard deviation',
                                 default=0.225)
        self.parser.add_argument('--keyframe_thr',
                                 type=float,
                                 help='the threshold of warping rate to decide keyframe',
                                 default=0.75)
        # LOSS WEIGHTS
        self.parser.add_argument('--photometric_loss',
                                 type=float,
                                 help='the weight of photometric loss',
                                 default=1.0)
        self.parser.add_argument('--geometry_loss',
                                 type=float,
                                 help='the weight of geometry consistency loss',
                                 default=1.0)
        self.parser.add_argument('--ssim_weight', 
                                 type=float,
                                 help='ssim weight',
                                 default=0.5)
        self.parser.add_argument('--flow_loss',
                                 type=float,
                                 help='the weight of flow consistency loss',
                                 default=10.0) # original: 10.0
        self.parser.add_argument('--disparity_smoothness_loss',
                                 type=float,
                                 help='the weight of disparity smoothness consistency loss',
                                 default=0.0)
        self.parser.add_argument('--photometric_loss_keyframe',
                                 type=float,
                                 help='the weight of photometric loss',
                                 default=0.0)
        self.parser.add_argument('--geometry_loss_keyframe',
                                 type=float,
                                 help='the weight of geometry consistency loss',
                                 default=0.0)
        self.parser.add_argument('--flow_loss_keyframe',
                                 type=float,
                                 help='the weight of flow consistency loss',
                                 default=0.0)
        self.parser.add_argument('--disparity_smoothness_loss_keyframe',
                                 type=float,
                                 help='the weight of disparity smoothness consistency loss',
                                 default=0.0)
        self.parser.add_argument('--adaptive_alpha',
                                 type=float,
                                 default=1.2)
        self.parser.add_argument('--adaptive_beta',
                                 type=float,
                                 default=0.9) # 1.2 for kitti, 0.85 else
        
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=60)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=2e-4)
        self.parser.add_argument("--init_num_epochs",
                                 type=int,
                                 help="number of epochs for initialization",
                                 default=300)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=200)
        
        # SYSTEM
        self.parser.add_argument('--cuda',
                                 default='cuda',
				 help='indicate cuda device')
        self.parser.add_argument('--img_extension',
                                 choices=['jpg', 'png'],
                                 default='png',
                                 help='the data type of input frames')
        self.parser.add_argument('--save_together',
                                 action='store_true',
                                 dest='save_together',
                                 help='save all result video in a single directory')

        # recitification
        self.parser.add_argument('--stability',
                                 type=int,
                                 help='std of gaussian filter for smoothing trajectory',
                                 default=12)
        self.parser.add_argument('--smooth_window',
                                 type=int,
                                 help='window size of moving average filter',
                                 default=59)
        self.parser.add_argument('--post_process',
                                 action='store_true',
                                 default=True,
                                 help='handle dynamic objects in post processing')

        # refinement from "pixelsynth"
        self.parser.add_argument("--old_model", type=str, default="")
        self.parser.add_argument("--short_name", type=str, default="")
        self.parser.add_argument("--result_folder", type=str, default="")
        self.parser.add_argument("--test_folder", type=str, default="")
        self.parser.add_argument("--model_setting", type=str, choices=("train", "gen_paired_img", "gen_img", \
                                    "gen_scene", 'get_gen_order', 'gen_two_imgs'), default="train")
        self.parser.add_argument("--dataset_folder", type=str, default="")
        self.parser.add_argument("--demo_img_name", type=str, default="")
        self.parser.add_argument("--gt_folder", type=str, default="")
        # self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--num_views", type=int, default=2)
        self.parser.add_argument("--num_workers", type=int, default=1)
        self.parser.add_argument(
            "--sampling_mixture_temp",
            type=float,
            default=1.0,
            help="mixture sampling temperature",
        )
        self.parser.add_argument(
            "--num_samples",
            type=int,
            default=1,
            help="num samples from which to optimize",
        )
        self.parser.add_argument(
            "--sampling_logistic_temp",
            type=float,
            default=1.0,
            help="logistic sampling temperature",
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="temperature for vqvae",
        )
        self.parser.add_argument(
            "--temp_eps",
            type=float,
            default=0.05,
            help="max / min temp (distance from 0 & 1) when drawing during sampling",
        )
        self.parser.add_argument(
            "--rotation",
            type=float,
            default=0.3,
            help="rotation (in radians) of camera for image generation",
        )
        self.parser.add_argument(
            "--decoder_truncation_threshold",
            type=float,
            default=2,
            help="resample if value above this drawn for decoder sampling",
        )
        self.parser.add_argument(
            "--homography", action="store_true", default=False
        ) 
        self.parser.add_argument(
            "--load_autoregressive", action="store_true", default=False
        ) 
        self.parser.add_argument("--no_outpainting", action="store_true", default=False)
        self.parser.add_argument(
            "--render_ids", type=int, nargs="+", default=[1]
        )
        self.parser.add_argument(
            "--directions", type=str, nargs="+", default=[], help="directions for scene generation"
        )
        self.parser.add_argument(
            "--direction", type=str, default="", help="direction for image generation"
        )
        self.parser.add_argument(
            "--background_smoothing_kernel_size", type=int, default=13
        )
        self.parser.add_argument(
            "--normalize_before_residual", action="store_true", default=False
        )
        self.parser.add_argument(
            "--sequential_outpainting", action="store_true", default=False
        )
        self.parser.add_argument(
            "--pretrain", action="store_true", default=False
        )
        self.parser.add_argument(
            "--val_rotation",
            type=int,
            default=10,
            help="size of rotation in single l/r direction in degrees for validation",
        )
        self.parser.add_argument(
            "--num_visualize_imgs", type=int, default=10
        )
        self.parser.add_argument(
            "--eval_iters", type=int, default=3600
        )
        self.parser.add_argument(
            "--eval_real_estate", action="store_true", default=False
        )
        self.parser.add_argument(
            "--intermediate", action="store_true", default=False
        )
        self.parser.add_argument(
            "--gen_fs", action="store_true", default=False
        )
        self.parser.add_argument(
            "--gen_order", action="store_true", default=False
        )
        self.parser.add_argument(
            "--gt_histogram", 
            type=str,
            help="rgb",
        )
        self.parser.add_argument(
            "--pred_histogram", 
            type=str,
            help="rgb",
        )
        self.parser.add_argument(
            "--image_type",
            type=str,
            default="both",
            choices=(
                "both"
            ),
        )
        self.parser.add_argument("--gpu_ids", type=str, default="0")
        self.parser.add_argument("--images_before_reset", type=int, default=100)
        self.parser.add_argument(
            "--test_input_image", action="store_true", default=False
        )
        self.parser.add_argument(
            "--use_custom_testset", action="store_true", default=False
        )
        self.parser.add_argument(
            "--use_fixed_testset", action="store_true", default=False
        )
        self.parser.add_argument(
            "--use_videos", action="store_true", default=False
        )
        self.parser.add_argument("--autoregressive", type=str, default="")
        self.parser.add_argument(
            "--num_split",
            type=int,
            default=1,
            help='number to split autoregressive steps into'
        )
        self.parser.add_argument(
            "--vqvae",action="store_true", default=False,
        )
        self.parser.add_argument(
            "--load_vqvae",action="store_true", default=False,
        )
        self.parser.add_argument("--vqvae_path", type=str, default="")
        self.parser.add_argument("--use_gt", action="store_true", default=False)
        self.parser.add_argument("--save_data", action="store_true", default=False)
        self.parser.add_argument("--dataset", type=str, default="")
        self.parser.add_argument(
            "--use_higher_res", action="store_true", default=False
        )
        self.parser.add_argument(
            "--use_3_discrim", action="store_true", default=False
        )
        self.parser.add_argument(
            "--max_rotation",
            type=int,
            default=10,
            help="size of rotation in single l/r direction in degrees (double for max difference)",
        )
        self.parser.add_argument(
            "--num_beam_samples",
            type=int,
            default=1,
            help="number of samples per beam",
        )
        self.parser.add_argument(
            "--num_beams",
            type=int,
            default=1,
            help="number of beams to sample",
        )

        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
