import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def mat2vec(poses_mat, rotation_mode):
    
    r = R.from_matrix(poses_mat[:, :3, :3])
    if rotation_mode == 'euler':
        r_vec = r.as_euler('xyz')
    else:
        r_vec = r.as_quat()
    t_vec = poses_mat[:, :3, 3]
    vec = np.concatenate([t_vec, r_vec], axis=1)
    return vec

a = np.load('outputs/7scene-chess-01/poses.npy')
b = np.load('outputs/7scene-chess-01/poses_stab.npy')
a_vec = mat2vec(a, 'quat')
b_vec = mat2vec(b, 'quat')

print(b_vec.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(a_vec[:,0], a_vec[:,1], a_vec[:,2], color='red', label='poses')
ax.plot(b_vec[:,0], b_vec[:,1], b_vec[:,2], color='blue', label='poses_stab')

plt.show()