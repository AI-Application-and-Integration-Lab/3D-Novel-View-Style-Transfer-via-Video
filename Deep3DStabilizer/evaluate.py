from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
import os
import glob
import cv2 as cv

eval_metric = 'SSIM' # 'PSNR' or 'SSIM'

score = 0.0

gt_dir = [
	# 'outputs/7scene-chess-01-gt',
	# 'outputs/7scene-fire-01-gt',
	# 'outputs/7scene-redkitchen-04-gt',
	# 'outputs/7scene-stairs-03-gt',
	# '../uni-freiburg-dataset/car052',
	'../ETH3D/processed_gt/delivery_area_rig_stereo_pairs_gt',
	'../ETH3D/processed_gt/electro_rig_stereo_pairs_gt',
	# '../ETH3D/processed_gt/forest_rig_stereo_pairs_gt',
	'../ETH3D/processed_gt/terrains_rig_stereo_pairs_gt',
]
out_dir = [
	# 'outputs/7scene-chess-01',
	# 'outputs/7scene-fire-01',
	# 'outputs/7scene-redkitchen-04',
	# 'outputs/7scene-stairs-03',
	# 'outputs/car052',
	'outputs/eth_delivery_2',
	'outputs/eth_electro_2',
	# 'outputs/eth_forest',
	'outputs/eth_terrians_2',
]

for i, _ in enumerate(gt_dir):
	score_scene = 0.0
	# imgs_gt = sorted(list(glob.glob(os.path.join(gt_dir[i], 'images_L', '*.png'))))
	# imgs_out = sorted(list(glob.glob(os.path.join(out_dir[i], 'images_L', '*.png'))))
	# for j in range(len(imgs_gt)):
	# 	img_gt = cv.imread(imgs_gt[j])[48:-48, 64:-64, :]
	# 	img_out = cv.imread(imgs_out[j])[48:-48, 64:-64, :]
	# 	if eval_metric == 'SSIM':
	# 		score_img = structural_similarity(img_gt, img_out, multichannel=True)
	# 	if eval_metric == 'PSNR':
	# 		score_img = peak_signal_noise_ratio(img_gt, img_out)
	# 	# print(j, ":", score_img)
	# 	score_scene += score_img
	
	if i == -1:
		imgs_gt = sorted(list(glob.glob(os.path.join(gt_dir[i], 'images_R_sub', '*.png'))))
	else:
		imgs_gt = sorted(list(glob.glob(os.path.join(gt_dir[i], 'images_R', '*.png'))))
	imgs_out = sorted(list(glob.glob(os.path.join(out_dir[i], 'images_R', '*.png'))))
	for j in range(len(imgs_gt)):
		img_gt = cv.imread(imgs_gt[j])
		img_out = cv.imread(imgs_out[j])
		H, W, _ = img_out.shape
		if img_gt.shape[0] != H or img_gt.shape[1] != W:
			# img_gt = img_gt[:H,:W,:]
			img_out = cv.resize(img_out, (img_gt.shape[1], img_gt.shape[0]), cv.INTER_NEAREST)
		if eval_metric == 'SSIM':
			score_img = structural_similarity(img_gt[48:-48, 64:-64, :], img_out[48:-48, 64:-64, :], multichannel=True)
		if eval_metric == 'PSNR':
			score_img = peak_signal_noise_ratio(img_gt[48:-48, 64:-64, :], img_out[48:-48, 64:-64, :])
		# print(j, ":", score_img)
		score_scene += score_img

	# score_scene /= (2 * len(imgs_gt))
	score_scene /= (len(imgs_gt))
	score += score_scene
	print("Eval of {}: {}".format(out_dir[i].split('/')[1], score_scene))

score /= len(gt_dir)
print(score)