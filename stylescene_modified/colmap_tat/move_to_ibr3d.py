import numpy as np
import glob, os, shutil
from PIL import Image
import sys

des = os.path.join(sys.argv[1],'ibr3d_pw_0.25')
if not os.path.exists(des):
	os.mkdir(des)
src = os.path.join(sys.argv[1],'images')
W, H = None, None
skip = 1
for i, f in enumerate(sorted(glob.glob(os.path.join(src,"*.png")))):
	if i % skip != 0:
		continue
	if i >= 500:
		continue
	s = f
	img = Image.open(s)
	W, H = img.size
	d = os.path.join(des,f'im_{i//skip:08}.png')
	# img = img.resize((W//2, H//2), Image.BICUBIC)
	img = img.resize((W, H), Image.BICUBIC)
	img.save(d)
	# shutil.copy2(s,d)

src = os.path.join(sys.argv[1],'depths')

for i, f in enumerate(sorted(glob.glob(os.path.join(src,"*.npy")))):
	if i % skip != 0:
		continue
	if i >= 500:
		continue
	d = np.load(f)
	d = d.reshape(d.shape[1], d.shape[2])
	#print(d.shape)
	# d = np.array(Image.fromarray(d).resize((W//2, H//2), Image.BICUBIC))
	d = np.array(Image.fromarray(d).resize((W, H), Image.BICUBIC))
	#print(d.shape)
	np.save(os.path.join(des, f'dm_{i//skip:08}.npy'),d)

p = np.load(os.path.join(sys.argv[1],'poses.npy'))
Rs = []
ts = []

for i in range(p.shape[0]):
	if i % skip != 0:
		continue
	if i >= 500:
		continue
	p_i = np.linalg.inv(p[i])
	R = p_i[:3,:3]
	t = p_i[:3,3]
	Rs.append(R)
	ts.append(t)
Rs = np.array(Rs)
ts = np.array(ts)
print(Rs.shape, ts.shape)

np.save(os.path.join(des, 'Rs.npy'),Rs)
np.save(os.path.join(des, 'ts.npy'),ts)

Ks = []

for i in range(p.shape[0]):
	if i % skip != 0:
		continue
	if i >= 500:
		continue
	K = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
	# K[0] /= 2
	# K[1] /= 2
	Ks.append(K)

Ks = np.array(Ks)
np.save(os.path.join(des, 'Ks.npy'),Ks)
