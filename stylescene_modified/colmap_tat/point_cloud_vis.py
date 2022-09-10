import copy
import numpy as np
import open3d as o3d
import cv2 as cv
import random

pcd = o3d.geometry.PointCloud()
# for i in range(0,17,16):
# pcd = o3d.io.read_point_cloud("office-01/dense/fused.ply")
# pcd = o3d.io.read_triangle_mesh("fire-01-col/dense/delaunay_photometric.ply")
xyz_a = np.load('Truck/dense/ibr3d_pw_0.25/points.npy')[100:160:12].reshape((-1,3))
# xyz_b = np.load('redkitchen-04/dense/ibr3d_pw_0.25/points.npy')[92-12:92+12].reshape((-1,3))
# xyz = np.r_[xyz_a, xyz_b]
xyz = xyz_a
# color_a = np.array([cv.cvtColor(cv.imread(f'Truck/dense/ibr3d_pw_0.25/im_{(int(i)):08}.png'), cv.COLOR_BGR2RGB) for i in range(100,160,12)])
color_a = np.array([cv.cvtColor(cv.imread(f'/media/ai2lab/Data/results/for_quantitative_results/79999/t_1/18/tat_subseq_Truck_0.25_n4/00_s{i:04}_es.jpg'), cv.COLOR_BGR2RGB) for i in range(100,160,12)])
# print(color_a)
color_a = color_a.astype(np.float32).reshape((-1,3))
color_a /= 255.
# color_a = np.zeros_like(xyz_a)
# color_a[:,0] = .8
# color_b = np.zeros_like(xyz_b)
# color_b[:,1] = .8
# color = np.r_[color_a, color_b]
color = color_a
# print(xyz)
randlist = random.sample(list(range(xyz.shape[0])),10000)
pcd.points = o3d.utility.Vector3dVector(xyz[randlist])
pcd.colors = o3d.utility.Vector3dVector(color[randlist])
# vis = o3d.visualization.Visualizer()
# vis.create_window(visible = False)
# vis.add_geometry(pcd)
# img = vis.capture_screen_float_buffer(True)
# cv.imshow('1',np.asarray(img))
# cv.waitKey(0)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# xyz  = np.load('office-01/dense/ibr3d_pw_0.25/points.npy')[32:33]
# xyz = xyz.reshape((-1,3))
# pcd.points = o3d.utility.Vector3dVector(xyz)
# vis = o3d.visualization.Visualizer()
# vis.create_window(visible = False)
# vis.add_geometry(pcd)
# img = vis.capture_screen_float_buffer(True)
# cv.imshow('2',np.asarray(img))
# cv.waitKey(0)