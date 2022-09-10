import numpy as np
from path import Path
import sys
from scipy.spatial.transform import Rotation as R

poses = np.load(Path(sys.argv[1])/'poses.npy')
f = open(Path(sys.argv[1])/'poses_TUM.txt', "w")

for i in range(poses.shape[0]):
	line = str(i)
	line += " "
	r = R.from_matrix(poses[i,:3,:3])
	t = poses[i,:3,3]
	r_quat = r.as_quat()
	line += "{} {} {} {} {} {} {}\n".format(t[0], t[1], t[2], r_quat[0], r_quat[1], r_quat[2], r_quat[3])
	f.write(line)
