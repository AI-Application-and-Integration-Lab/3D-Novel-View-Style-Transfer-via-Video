import numpy as np
import cv2 as cv
from path import Path
import os
import glob
import sys

if 'gt' not in sys.argv[2]:
	print("please add gt in your output name parameter")
	raise ValueError 
data_dir = Path(sys.argv[1])
out_dir = Path('outputs')/sys.argv[2]
out_dir.makedirs_p()
out_img_dir = out_dir/'images'
out_img_dir.makedirs_p()
out_depth_dir = out_dir/'depths'
out_depth_dir.makedirs_p()

# data_imgs = sorted(list(glob.glob(data_dir/'*.color.png')))
# data_depths = sorted(list(glob.glob(data_dir/'*.depth.png')))
# data_poses = sorted(list(glob.glob(data_dir/'*.txt')))
data_imgs = sorted(list(glob.glob(data_dir/'im_*.png')))
data_depths = sorted(list(glob.glob(data_dir/'dm_*.npy')))
data_K = np.load(data_dir/'Ks.npy')
data_R = np.load(data_dir/'Rs.npy')
data_t = np.load(data_dir/'ts.npy')


poses = []
img = cv.imread(data_imgs[0])
origin_h, origin_w, _ = img.shape

for i in range(len(data_imgs)):
	img = cv.imread(data_imgs[i])
	img = cv.resize(img, (640,480))
	cv.imwrite(out_img_dir/f'{i:05}.png', img)
	# d = cv.imread(data_depths[i], cv.IMREAD_ANYDEPTH)
	# d = np.expand_dims(d.astype(np.float32) / 1000., axis=0)
	d = np.load(data_depths[i])
	d[d==0] = 1e-3
	d = cv.resize(d, (640,480))
	d = np.expand_dims(d.astype(np.float32), axis=0)
	np.save(out_depth_dir/f'{i:05}.npy', d)
	# f = open(data_poses[i])
	# lines = f.readlines()
	# lines = [line.split() for line in lines]
	pose = np.zeros((4,4))
	# for x in range(4):
	# 	for y in range(4):
	# 		pose[x,y] = float(lines[x][y])
	pose[:3,:3] = data_R[i]
	pose[:3,3] = data_t[i]
	pose[3,3] = 1.
	poses.append(np.linalg.inv(pose))
	# poses.append(pose)

poses = np.array(poses)
np.save(out_dir/'poses.npy', poses)
data_K[0,0,:] *= (640. / origin_w)
data_K[0,1,:] *= (480. / origin_h)
np.save(out_dir/'intrinsic.npy', data_K[0])