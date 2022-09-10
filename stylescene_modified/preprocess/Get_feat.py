import os
from Models import encoder3
import torch
import numpy as np
import glob
import PIL.Image
import torch.nn.functional as F
import ext
from projection.z_buffer_manipulator import get_splatter, PtsManipulator
from argparse import Namespace

enc_net = encoder3().cuda()
enc_net.load_state_dict(torch.load("vgg_r31.pth"))
root = "../colmap_tat"

proj_args = Namespace(learn_default_feature=False, radius=2, rad_pow=2, tau=1.0, accumulation='alphacomposite')

dirs_train = [
# "training/Barn/dense/ibr3d_pw_0.25",
# "training/Caterpillar/dense/ibr3d_pw_0.25",
# "training/Church/dense/ibr3d_pw_0.25",
# "training/Ignatius/dense/ibr3d_pw_0.25",
# "training/Meetingroom/dense/ibr3d_pw_0.25",
# "intermediate/Family/dense/ibr3d_pw_0.25",
# "intermediate/Francis/dense/ibr3d_pw_0.25",
# "intermediate/Horse/dense/ibr3d_pw_0.25",
# "intermediate/Lighthouse/dense/ibr3d_pw_0.25",
# "intermediate/Panther/dense/ibr3d_pw_0.25",
# "advanced/Auditorium/dense/ibr3d_pw_0.25",
# "advanced/Ballroom/dense/ibr3d_pw_0.25",
# "advanced/Museum/dense/ibr3d_pw_0.25",
# "advanced/Temple/dense/ibr3d_pw_0.25",
# "advanced/Courtroom/dense/ibr3d_pw_0.25",
# "advanced/Palace/dense/ibr3d_pw_0.25"
]

dirs_test = [
# "training/Truck/dense/ibr3d_pw_0.25",
# "intermediate/M60/dense/ibr3d_pw_0.25",
# "intermediate/Playground/dense/ibr3d_pw_0.25",
# "intermediate/Train/dense/ibr3d_pw_0.25",
# "Ignatius/dense/ibr3d_pw_0.25",
# "Playground/dense/ibr3d_pw_0.25",
# "Courtroom/dense/ibr3d_pw_0.25",
"Truck/dense/ibr3d_pw_0.25",
# "Train/dense/ibr3d_pw_0.25",
# "Horse/dense/ibr3d_pw_0.25",
# "Lighthouse/dense/ibr3d_pw_0.25",
# "Auditorium/dense/ibr3d_pw_0.25",
# "Family/dense/ibr3d_pw_0.25",
# "Francis/dense/ibr3d_pw_0.25",
# "Meetingroom/dense/ibr3d_pw_0.25",
# "Palace/dense/ibr3d_pw_0.25",
# "Panther/dense/ibr3d_pw_0.25",
# "Museum/dense/ibr3d_pw_0.25",
# "fire-01/dense/ibr3d_pw_0.25",
# "fire-03/dense/ibr3d_pw_0.25",
# "office-01/dense/ibr3d_pw_0.25",
# "office-02/dense/ibr3d_pw_0.25",
# "redkitchen-04/dense/ibr3d_pw_0.25",
# "redkitchen-01/dense/ibr3d_pw_0.25",
# "head-01/dense/ibr3d_pw_0.25",
# "head-02/dense/ibr3d_pw_0.25",
# "chess-01/dense/ibr3d_pw_0.25",
# "chess-02/dense/ibr3d_pw_0.25",
# "stairs-03/dense/ibr3d_pw_0.25",
# "stairs-02/dense/ibr3d_pw_0.25",
# "pumpkin-02/dense/ibr3d_pw_0.25",
# "pumpkin-06/dense/ibr3d_pw_0.25",
# "Ignatius-col/dense/ibr3d_pw_0.25",
# "Playground-col/dense/ibr3d_pw_0.25",
# "Courtroom-col/dense/ibr3d_pw_0.25",
# "Truck-col/dense/ibr3d_pw_0.25",
# "Train-col/dense/ibr3d_pw_0.25",
# "Horse-col/dense/ibr3d_pw_0.25",
# "Auditorium-col/dense/ibr3d_pw_0.25",
# "Family-col/dense/ibr3d_pw_0.25",
# "Francis-col/dense/ibr3d_pw_0.25",
# "Meetingroom-col/dense/ibr3d_pw_0.25",
# "Palace-col/dense/ibr3d_pw_0.25",
# "Panther-col/dense/ibr3d_pw_0.25",
]

tgt_ind = {}
# tgt_ind["training/Truck/dense/ibr3d_pw_0.25"] = [172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
# tgt_ind["intermediate/M60/dense/ibr3d_pw_0.25"] = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
# tgt_ind["intermediate/Playground/dense/ibr3d_pw_0.25"] = [221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]
# tgt_ind["intermediate/Train/dense/ibr3d_pw_0.25"] = [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]

def load(p, height=None, width=None):
  im = PIL.Image.open(p)
  im = np.array(im)
  if (
    height is not None
    and width is not None
    and (im.shape[0] != height or im.shape[1] != width)
  ):
    raise Exception("invalid size of image")
  im = (im.astype(np.float32) / 255) * 2 - 1
  im = im.transpose(2, 0, 1)
  return im

def pad(im, pad_width):
  h, w = im.shape[-2:]
  mh = h % pad_width
  ph = 0 if mh == 0 else pad_width - mh
  mw = w % pad_width
  pw = 0 if mw == 0 else pad_width - mw
  shape = [s for s in im.shape]
  h += ph
  w += pw
  return h,w

def process_scene(scene, istrain):
  scene_path = os.path.join(root, scene)
  print(scene_path)    
  h,w = pad(np.load(glob.glob(os.path.join(scene_path, 'dm*.npy'))[0]), 16)
  srcs = sorted(glob.glob(os.path.join(scene_path, 'im*.png')))
  srcs = np.array([load(im) for im in srcs])
  nview = len(srcs)

  if scene in tgt_ind:
    select = [ind for ind in range(nview) if ind not in tgt_ind[scene]]
  else:
    select = list(range(nview))
  srcs = srcs[select]
  srcs = torch.from_numpy(srcs)

  feats = []
  for src in srcs:
    feat = F.interpolate((src.unsqueeze(0)+1)/2, (h,w), mode = "bilinear").cuda()
    feat = enc_net(feat)
    feats.append(feat.detach().cpu().numpy())
  feats = np.concatenate(feats, 0)
  np.save(os.path.join(scene_path, "r31.npy"), feats[0:1])

  src_dms = sorted(glob.glob(os.path.join(scene_path, 'dm*.npy')))
  src_dms = [np.load(im) for im in src_dms]
  src_Ks = np.load(os.path.join(scene_path,"Ks.npy"))
  src_Rs = np.load(os.path.join(scene_path,"Rs.npy"))
  src_ts = np.load(os.path.join(scene_path,"ts.npy"))

  points = []
  for idx in range(nview):
    src_dm = np.array([src_dms[idx]])
    src_K = np.array([src_Ks[idx]])
    src_R = np.array([src_Rs[idx]])
    src_t = np.array([src_ts[idx]])
    tgt_dm = src_dm[0]
    tgt_K = src_K[0]
    tgt_R = src_R[0]
    tgt_t = src_t[0]

    patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
    if istrain:
      sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map_train(
        tgt_dm,
        tgt_K,
        tgt_R,
        tgt_t,
        src_dm,
        src_K,
        src_R,
        src_t,
        patch,
        0.01,
        True,
      )
    else:
      sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map_test(
        tgt_dm,
        tgt_K,
        tgt_R,
        tgt_t,
        src_dm,
        src_K,
        src_R,
        src_t,
        patch,
        0.01,
        True,
      )
    points.append(sampling_maps)

  points = np.array(points)[:,0,...][select]
  np.save(os.path.join(scene_path, "points.npy"), points)

  # For Debugging
  # _, c, h_feat, w_feat = srcs.shape
  # mysplatter = get_splatter("xyblending", None, proj_args, size=(h_feat,w_feat), C=c, points_per_pixel=24)
  # for idx in range(nview):
  #   src_dm = np.array([src_dms[idx]])
  #   src_K = np.array([src_Ks[idx]])
  #   src_R = np.array([src_Rs[idx]])
  #   src_t = np.array([src_ts[idx]])
    
  #   tgt_dm = torch.from_numpy(src_dm[0]).float().unsqueeze(0).unsqueeze(0).cuda()
  #   ##tgt_feat = src_feat.float().unsqueeze(0).cuda()
  #   # F.interpolate(tgt_dm, (h_feat, w_feat), mode = "bilinear")

  #   tgt_K = torch.from_numpy(src_K[0]).float().unsqueeze(0).cuda()
  #   tgt_R = torch.from_numpy(src_R[0]).float().unsqueeze(0).cuda()
  #   tgt_t = torch.from_numpy(src_t[0]).float().unsqueeze(0).cuda()
  #   # tgt_K4 = torch.zeros((1,4,4)).float().cuda()
  #   # tgt_K4[:,:3,:3] = tgt_K
  #   # tgt_K4[:,3,3] = 1.0
  #   # tgt_RT = torch.zeros((1,4,4)).float().cuda()
  #   # tgt_RT[:,:3,:3] = tgt_R
  #   # tgt_RT[:,:3,3] = tgt_t
  #   # tgt_RT[:,3,3] = 1.0
    
  #   points_t = torch.from_numpy(points[idx:idx+1,:,:,:]).float().cuda().view(-1,3).permute(1,0).unsqueeze(0)
  #   srcs_t = srcs[idx:idx+1,:,:,:].float().cuda().permute(1,0,2,3).reshape(c,-1).unsqueeze(0)
  #   # stylescene, pixelsynth ver.
  #   xy_proj = tgt_K.bmm(tgt_R.bmm(points_t) + tgt_t.unsqueeze(-1))
  #   mask = (xy_proj[:, 2:3, :].abs() < 1e-3).detach()

  #   zs = xy_proj[:, 2:3, :]
  #   zs[mask] = 1e-3

  #   sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)
  #   sampler[mask.repeat(1, 3, 1)] = -10
  #   # Flip the ys
    
  #   sampler[:,0,:] = sampler[:,0,:] / float(w_feat) * 2 - 1
  #   sampler[:,1,:] = sampler[:,1,:] / float(h_feat) * 2 - 1
  #   sampler = sampler * torch.Tensor([1, 1, 1]).unsqueeze(0).unsqueeze(
  #       2
  #   ).to(sampler.device)
  #   pointcloud = sampler.permute(0, 2, 1).contiguous()
  #   result, background_mask, proj_d = mysplatter(pointcloud, srcs_t)
  #   result = result.detach().squeeze(0).cpu().numpy().transpose(1,2,0)
  #   result = PIL.Image.fromarray(((result+1)/2*255).astype(np.uint8))
  #   result.save('outputs/{:05}.png'.format(idx))
  print(scene_path)


for scene in dirs_train:
  process_scene(scene, istrain=True)
for scene in dirs_test:
  process_scene(scene, istrain=False)
