import numpy as np
import cv2
import glob
import os
import sys
from path import Path
import torch
import torchvision.transforms as trn
import torch.nn.functional as F
import lpips
import random

random.seed(1024)

from flow import *
from loss import mean_on_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
flow_processor = FlowProcessor(None).to(device)

dataset_name = sys.argv[1]


input_dir = Path(f'/home/ai2lab/Documents/al777/StyleTransfer/stylescene/colmap_tat/{dataset_name}/dense/ibr3d_pw_0.25')

input_imgs = sorted(glob.glob(input_dir/'im_*.png'))[:250]
if "-col" in dataset_name:
	input_imgs = sorted(glob.glob(input_dir/'im_*.png'))[100:300]

style_ids = random.sample(list(range(21)), 10)
print(style_ids)
model_name = sys.argv[2]
stylized_dir = [Path(f'/media/ai2lab/Data/results/for_quantitative_results/{model_name}/t_1/{style_id}/tat_subseq_{dataset_name}_0.25_n4') for style_id in style_ids]

if model_name == '0':
	input_dir = Path(f'/home/ai2lab/Documents/al777/StyleTransfer/stylescene/colmap_tat/{dataset_name}-col/dense/ibr3d_pw_0.25')
	input_imgs = sorted(glob.glob(input_dir/'im_*.png'))[100:300]
	model_name = '0'
	stylized_dir = [Path(f'/media/ai2lab/Data/results/for_quantitative_results/{model_name}/t_1/{style_id}/tat_subseq_{dataset_name}-col_0.25_n4') for style_id in style_ids]


if 'Gao' in model_name:
	input_dir = Path(f'./outputs/{dataset_name}/images_L')
	input_imgs = sorted(glob.glob(input_dir/'*.png'))[:250]
	stylized_dir = [Path(f'/media/ai2lab/Data/results/for_quantitative_results/Gao/{dataset_name}/images_L/{style_id}') for style_id in style_ids]

total_score = 0.0
temporal = 1
thr = 0.005


print(input_dir)
if not os.path.exists(os.path.join('error_maps', dataset_name)):
    os.mkdir(os.path.join('error_maps', dataset_name))
if not os.path.exists(os.path.join('error_maps', dataset_name,model_name)):
    os.mkdir(os.path.join('error_maps', dataset_name,model_name))
if not os.path.exists(os.path.join('error_maps', dataset_name,model_name,f'thr_{thr}')):
    os.mkdir(os.path.join('error_maps', dataset_name,model_name,f'thr_{thr}'))
if not os.path.exists(os.path.join('warp_log', dataset_name)):
    os.mkdir(os.path.join('warp_log', dataset_name))
if not os.path.exists(os.path.join('warp_log', dataset_name,model_name)):
    os.mkdir(os.path.join('warp_log', dataset_name,model_name))

for idx, style_id in enumerate(style_ids):
	score = 0.0
	# print(style_id)
	stylized_imgs = sorted(glob.glob(stylized_dir[idx]/'*es.jpg'))
	stylized_imgs += sorted(glob.glob(stylized_dir[idx]/'*.png'))
	# print(stylized_imgs)
	n = min(len(input_imgs),len(stylized_imgs))
	# print(style_id)
	for i in range(n-temporal):
		img1 = Image.open(input_imgs[i]).convert('RGB')
		img1 = trn.ToTensor()(img1).unsqueeze(0)[:,:,:464,:640].to(device) #480 for other data
		img2 = Image.open(input_imgs[i+temporal]).convert('RGB')
		img2 = trn.ToTensor()(img2).unsqueeze(0)[:,:,:464,:640].to(device)
		# print(img1.shape)
		# print(img2.shape)
		h, w = img1.shape[-2:]
		sty1 = lpips.im2tensor(lpips.load_image(stylized_imgs[i]))[:,:,:h,:w].to(device)
		sty2 = lpips.im2tensor(lpips.load_image(stylized_imgs[i+temporal]))[:,:,:h,:w].to(device)
		flow12, _, mask1, _, _ = flow_processor.get_flow_forward_backward(
                        img1, img2, pre_homo=True, consistency_thresh=1.0)
		# flow12, _, mask1, _, _ = flow_processor.get_flow_forward_backward(
  #                       sty1, sty2, pre_homo=False, consistency_thresh=1.0)
		flow12 = normalize_for_grid_sample(flow12)
		# print(flow12.shape)
		# flow12 = F.interpolate(flow12, (480, 640), mode='area')
		# mask1 = F.interpolate(mask1, (480, 640), mode='area')
		# sty21 = F.grid_sample(img2, flow12)
		sty21 = F.grid_sample(sty2, flow12)
		# sty1 = img1

		sty21 *= mask1
		sty1 *= mask1

		# sty21 = sty21[:,:,24:-24,32:-32]
		# sty1 = sty1[:,:,24:-24,32:-32]
		# sty2 = sty2[:,:,24:-24,32:-32]
		# mask1 = mask1[:,24:-24,32:-32].unsqueeze(1).to(device)

		with torch.no_grad():
			loss = loss_fn_vgg(sty1,sty21)
			# print(loss.item())
		diff1 = (sty1 - sty21).pow(2)
		error_map = diff1 > thr

		# loss = mean_on_mask(diff1, mask1)
		# print(loss.item())
		score += loss

		# if i % 20 == 0 and idx == 0:
		# 	sty21 = sty21[0].detach().cpu().numpy().transpose(1,2,0)
		# 	sty1 = sty1[0].detach().cpu().numpy().transpose(1,2,0)
		# 	sty2 = sty2[0].detach().cpu().numpy().transpose(1,2,0)
		# 	sty1 = Image.fromarray(((sty1+1)/2*255).astype(np.uint8))
		# 	# sty1 = Image.fromarray(sty1.astype(np.uint8))
		# 	sty1.save(f'warp_log/{dataset_name}/{model_name}/{i:05}.png')
		# 	sty21 = Image.fromarray(((sty21+1)/2*255).astype(np.uint8))
		# 	# sty21 = Image.fromarray(sty21.astype(np.uint8))
		# 	sty21.save(f'warp_log/{dataset_name}/{model_name}/{i:05}_w.png')
		# 	sty2 = Image.fromarray(((sty2+1)/2*255).astype(np.uint8))
		# 	# sty21 = Image.fromarray(sty21.astype(np.uint8))
		# 	sty2.save(f'warp_log/{dataset_name}/{model_name}/{i+temporal:05}.png')
		# 	# mask1 = mask1[0,0].detach().cpu().numpy()
		# 	# mask1 = Image.fromarray((mask1*255).astype(np.uint8))
		# 	# mask1.save(f'error_maps/{dataset_name}/mask_{i}.png')
		# 	error_map = error_map[0,0].float().detach().cpu().numpy()
		# 	error_map = Image.fromarray((error_map*255).astype(np.uint8))
		# 	error_map.save(f'error_maps/{dataset_name}/{model_name}/thr_{thr}/{i:05}.png')

	score /= (n-temporal)
	# print(f"style_id {style_id}: {score.item()}")
	total_score += score

total_score /= len(style_ids)
print(f"total_score of model {model_name} with temporal {temporal}: {total_score.item()}")





