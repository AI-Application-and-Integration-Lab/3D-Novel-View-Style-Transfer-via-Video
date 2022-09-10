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

from flow import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

flow_processor = FlowProcessor(None).to(device)

dataset_name = sys.argv[1]

input_dir = Path(f'./outputs/{dataset_name}/images')

input_imgs = sorted(glob.glob(input_dir/'*.png'))[:250]

style_ids = list(range(10))
# stylized_dir = [Path(f'/media/ai2lab/Data/results/for_quantitative_results/50000/t_1/{style_id}/tat_subseq_{dataset_name}_0.25_n4') for style_id in style_ids]
stylized_dir = [Path(f'stylize_log/{dataset_name}/images_L/{style_id}') for style_id in style_ids]
total_score = 0.0
temporal = 1

print(input_dir)

for style_id in style_ids:
	score = 0.0
	# stylized_imgs = sorted(glob.glob(stylized_dir[style_id]/'*es.jpg'))[:250]
	# stylized_imgs = sorted(glob.glob(stylized_dir[style_id]/'*.png'))[:250]
	# stylized_imgs = sorted(glob.glob(stylized_dir[style_id]/'*.png'))[:250]
	n = 249
	# print(style_id)
	for i in range(n-temporal):
		# img1 = Image.open(input_imgs[i]).convert('RGB')
		# img1 = trn.ToTensor()(img1).unsqueeze(0)[:,:,:480,:640].to(device)
		# img2 = Image.open(input_imgs[i+temporal]).convert('RGB')
		# img2 = trn.ToTensor()(img2).unsqueeze(0)[:,:,:480,:640].to(device)
		# print(img1.shape)
		# print(img2.shape)
		# h, w = img1.shape[-2:]
		sty1 = lpips.im2tensor(lpips.load_image(stylized_dir[style_id]/f"{i+1:06}.png")).to(device)
		sty2 = lpips.im2tensor(lpips.load_image(stylized_dir[style_id]/f"{i:06}_L_w.png")).to(device)
		# flow12, _, mask1, _, _ = flow_processor.get_flow_forward_backward(
  #                       img1, img2, pre_homo=True, consistency_thresh=1.0)
		# flow12 = normalize_for_grid_sample(flow12)
		# print(flow12.shape)
		# flow12 = F.interpolate(flow12, (480, 640), mode='area')
		# mask1 = F.interpolate(mask1, (480, 640), mode='area')
		# sty21 = F.grid_sample(img2, flow12)
		# sty21 = F.grid_sample(sty2, flow12)
		# sty1 = img1

		# sty21 *= mask1
		# sty1 *= mask1

		sty2 = sty2[:,:,24:-24,32:-32]
		sty1 = sty1[:,:,24:-24,32:-32]

		with torch.no_grad():
			loss = loss_fn_vgg(sty1,sty2)
			# print(loss.item())
			score += loss

		# sty21 = sty21[0].detach().cpu().numpy().transpose(1,2,0)
		# sty1 = sty1[0].detach().cpu().numpy().transpose(1,2,0)
		# sty1 = Image.fromarray(((sty1+1)/2*255).astype(np.uint8))
		# # sty1 = Image.fromarray(sty1.astype(np.uint8))
		# sty1.save('1.png')
		# sty21 = Image.fromarray(((sty21+1)/2*255).astype(np.uint8))
		# # sty21 = Image.fromarray(sty21.astype(np.uint8))
		# sty21.save('21.png')
		# mask1 = mask1[0].detach().cpu().numpy()
		# mask1 = Image.fromarray((mask1*255).astype(np.uint8))
		# mask1.save('mask.png')
		# s

	score /= (n-temporal)
	# print(f"style_id {style_id}: {score.item()}")
	total_score += score

total_score /= len(style_ids)
print(f"total_score: {total_score.item()}")

