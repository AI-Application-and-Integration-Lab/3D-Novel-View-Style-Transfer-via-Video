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

def load_image(img_path):
    img = imread(img_path).astype(np.float32)
    img = imresize(img, (288, 384))	
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (torch.from_numpy(img).float() / 255 - 0.45) / 0.225
    return tensor_img

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("argument must be __input_path__ __output_path__ __model_path__    !!")
		sys.exit()

	opt = options.Options().parse()

	img_glob = os.path.join(sys.argv[1], "*.png")
	img_paths = sorted(glob.glob(img_glob))
    
    input_img = []

    for img in img_paths:
        img_tensor = load_image(img)
        input_img.append(img_tensor)

    input_img = torch.cat(input_img)

    model_dir = os.path.join(sys.argv[3], "models")
    print(f"Dispnet: {model_dir}")
    model_glob  = os.path.join(model_dir,"*.pth")
    model_paths = sorted(glob.glob(model_glob))

    model_begin = []
    model_end = []

  	for k in range(len(model_paths)):
		modelname_split = model_paths[k].split('_')
		begin, end = int(modelname_split[2]), int(modelname_split[4][:5])
		model_begin.append(begin)
		model_end.append(end)

    model_id  = 0
    dispnet = torch.load(model_paths[model_id])
    for i in range(input_img.shape[0]):

    	if i == model_end[model_id]:
    		model_id += 1
    		dispnet = torch.load(model_paths[model_id])

		with torch.no_grad():
			d_feature = dispnet['encoder'](input_img[i:i+1,:,:,:].cuda())
			d_output = dispnet['decoder'](d_feature)
		depth = [d_output['disp', s] for s in opt.scales]
		depth = [d * opt.max_depth + opt.min_depth for d in depth][0]

		np.save(os.path.join(sys.argv[2], '{:05}.npy'.format(i)), depth.cpu().detach().numpy())



