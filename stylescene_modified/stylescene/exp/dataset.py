import numpy as np
import PIL
import logging
import sys
import glob
sys.path.append("../")
import co
import ext

import cv2
import imageio
from pathlib import Path
import config
import os.path
import torch
import random


def load(p, height=None, width=None):
  if p.suffix == ".npy":
    return np.load(p)
  elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
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
  else:
    raise Exception("invalid suffix")


def load_resize_test(p, height=None, width=None):
  if p.suffix == ".npy":
    return np.load(p)
  elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
    im = PIL.Image.open(p).convert('RGB')
    im = im.resize((224,224)) 
    im = np.array(im)
    if (
      height is not None
      and width is not None
      and (im.shape[0] != height or im.shape[1] != width)
    ):
      raise Exception("invalid size of image")
    im = (im.astype(np.float32) / 255)
    if len(im.shape) == 2: 
      im = np.stack((im,im,im), 2)
    im = im.transpose(2, 0, 1)
    return im
  else:
    raise Exception("invalid suffix")



def load_resize_train(p, height=None, width=None):
  if p.suffix == ".npy":
    return np.load(p)
  elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
    try:
      im = PIL.Image.open(p).convert('RGB')
      im = im.resize((496,288)) 
      im = np.array(im)
    except:
      im = np.zeros((496,288,3))

    if (
      height is not None
      and width is not None
      and (im.shape[0] != height or im.shape[1] != width)
    ):
      raise Exception("invalid size of image")
    im = (im.astype(np.float32) / 255)
    if len(im.shape) == 2: 
      im = np.stack((im,im,im), 2)
    im = im.transpose(2, 0, 1)
    return im
  else:
    raise Exception("invalid suffix")


class Dataset(co.mytorch.BaseDataset):
  def __init__(
    self,
    *,
    name,
    tgt_im_paths,
    tgt_dm_paths,
    tgt_Ks,
    tgt_Rs,
    tgt_ts,
    tgt_counts,
    src_im_paths,
    src_dm_paths,
    src_Ks,
    src_Rs,
    src_ts,
    im_size=None,
    pad_width=None,
    patch=None,
    n_nbs=5,
    nbs_mode="sample",
    bwd_depth_thresh=0.1,
    invalid_depth_to_inf=True,
    **kwargs,
  ):
    super().__init__(name=name, **kwargs)

    self.tgt_im_paths = tgt_im_paths
    self.tgt_dm_paths = tgt_dm_paths
    self.tgt_Ks = tgt_Ks
    self.tgt_Rs = tgt_Rs
    self.tgt_ts = tgt_ts
    self.tgt_counts = tgt_counts

    self.src_im_paths = src_im_paths
    self.src_dm_paths = src_dm_paths
    self.src_Ks = src_Ks
    self.src_Rs = src_Rs
    self.src_ts = src_ts

    self.im_size = im_size
    self.pad_width = pad_width
    self.patch = patch
    self.n_nbs = n_nbs
    self.nbs_mode = nbs_mode
    self.bwd_depth_thresh = bwd_depth_thresh
    self.invalid_depth_to_inf = invalid_depth_to_inf

    # self.intervals = [0,1,2] # must have zero
    # print(tgt_dm_paths)
    tmp = np.load(tgt_dm_paths[0])
    self.height, self.width = tmp.shape
    del tmp

    n_tgt_im_paths = len(tgt_im_paths) if tgt_im_paths else 0
    shape_tgt_im = (
      self.load_pad(tgt_im_paths[0]).shape if tgt_im_paths else None
    )
    logging.info(
      f"  #tgt_im_paths={n_tgt_im_paths}, #tgt_counts={tgt_counts.shape}, tgt_im={shape_tgt_im}, tgt_dm={self.load_pad(tgt_dm_paths[0]).shape}"
    )

    # print(str(tgt_dm_paths[0])[:-15])
    path = str(tgt_dm_paths[0])[:-15].replace("long","pw_0.25")
    points = np.load(os.path.join(path,"points.npy"))
    emb  = np.load(os.path.join(path, "r31.npy"))
    H1, W1 = points.shape[1:3]
    H2, W2 = emb.shape[2:4]
    y = [int(round(y)) for y in np.array(list(range(H2))) / (H2 - 1) * (H1 - 1)]
    x = [int(round(x)) for x in np.array(list(range(W2))) / (W2 - 1) * (W1 - 1)]
    points = np.take(points, y, axis=1)
    points = np.take(points, x, axis=2)
    
    emb = emb.transpose(1,0,2,3)
    self.points = points.reshape(-1, 3)
    self.feats = emb.reshape(256, -1)
    
    # self.points = points
    
    
    # print("original point size: ", self.points.shape[0]*self.points.shape[1]*self.points.shape[2])
    if config.Train:
      if not config.use_local_pointcloud:
        self.points = self.points.reshape(-1, 3)
        skip = int(self.points.shape[0] // 100000)
        self.points = self.points[::skip, :].astype(np.float32)
        self.feats = self.feats.reshape(256, -1)
        self.feats = self.feats[:, ::skip].astype(np.float32)
      else:
        self.points = points
      self.style = sorted([os.path.join(config.styleroot,"style120/%d.jpg") % i for i in config.Test_style])
    else:
      # self.points = self.points.astype(np.float32)
      # self.feats = self.feats.astype(np.float32)
      # self.feats = emb 
      if not config.use_local_pointcloud:
        self.points = self.points.reshape(-1, 3)
        self.feats = self.feats.reshape(256, -1)
        ind = [int(ii / 1800000 * self.points.shape[0]) for ii in list(range(1800000))]
        self.points = self.points[ind, :]
        self.feats = self.feats[:, ind]
      else:
        self.points = points
        self.feats = None
      # else:
      #   self.global_feats = self.feats.reshape(256, -1)
      #   skip = int(self.global_feats.shape[-1] // 100000)
      #   self.global_feats = self.global_feats[:, ::skip].astype(np.float32)
      self.style = [os.path.join(config.styleroot,"style120/%d.jpg") % i for i in config.Test_style]
    print("point size: ", self.points.shape)
    del points
    del emb
    # print("feat size: ", self.feats.shape)

  def pad(self, im):
    if self.im_size is not None:
      shape = [s for s in im.shape]
      shape[-2] = self.im_size[0]
      shape[-1] = self.im_size[1]
      im_p = np.zeros(shape, dtype=im.dtype)
      sh = min(im_p.shape[-2], im.shape[-2])
      sw = min(im_p.shape[-1], im.shape[-1])
      im_p[..., :sh, :sw] = im[..., :sh, :sw]
      im = im_p
    if self.pad_width is not None:
      h, w = im.shape[-2:]
      mh = h % self.pad_width
      ph = 0 if mh == 0 else self.pad_width - mh
      mw = w % self.pad_width
      pw = 0 if mw == 0 else self.pad_width - mw
      shape = [s for s in im.shape]
      shape[-2] += ph
      shape[-1] += pw
      im_p = np.zeros(shape, dtype=im.dtype)
      im_p[..., :h, :w] = im
      im = im_p
    return im

  def load_pad(self, p):
    im = load(p)
    return self.pad(im)

  def base_len(self):
    if config.Train:
      return len(self.tgt_dm_paths) - 4
    return len(self.tgt_dm_paths)

  def base_getitem(self, idx, rng):
    ret = {}
    if config.Train:
      sidx = np.random.randint(0,len(self.style))
      ret["style"] = load_resize_train(Path(self.style[sidx]))
    else:
      ret["style"] = np.array([load_resize_test(Path(self.style[sidx])) for sidx in range(len(self.style))])
    
    # m = max(np.abs(np.max(self.points)), np.abs(np.min(self.points)))
    if config.Train:
      if config.use_local_pointcloud:
        m = max(np.abs(np.max(self.points)), np.abs(np.min(self.points)))
        ret["points"] = []
        ret["points2"] = []
        # ret["feats"] = []
        ret["imgs"] = []
        snippet_size = config.snippet_size
        snippet_half = snippet_size // 2
        # list_nbor = list(range(max(0, idx-2),min(len(self.tgt_dm_paths),idx+2)))
        # list_nbor.remove(idx)
        # idx_nbor = random.choice(list_nbor)
        for i in [idx, idx+1, idx+3]:
          if config.use_naive:
            snippet_idx = min(max(0, i - snippet_half),self.points.shape[0]-snippet_size) if snippet_half != 0 else i
            snippet = list(range(snippet_idx, snippet_idx + snippet_size))

            local_points = self.points[snippet]
            local_points = local_points.reshape(-1, 3)
            skip = int(local_points.shape[0] // 100000)
            local_points = local_points[::skip, :].astype(np.float32)
            local_imgs = np.array([self.load_pad(self.tgt_im_paths[s]) for s in snippet])

          else:
            # snippet_idx = min(max(0, (i - snippet_half) // snippet_half),(self.points.shape[0]+snippet_size-1)//snippet_size) if snippet_half != 0 else i
            # snippet = list(range(snippet_idx*snippet_half,min(self.feats.shape[0],snippet_idx*snippet_half+snippet_size)))
            
            # local_points = self.points[snippet_idx*snippet_size:(snippet_idx+1)*snippet_size]
            # local_points = local_points.reshape(-1, 3)
            # local_feats = self.feats[snippet].transpose(1,0,2,3).reshape(256,-1)
            snippet_idx = min(max(0, i - snippet_half),self.points.shape[0]-snippet_size) if snippet_half != 0 else i
            snippet = list(range(snippet_idx, snippet_idx + snippet_size))

            local_points = self.points[snippet]
            local_points = local_points.reshape(-1, 3)
            skip = int(local_points.shape[0] // 100000)
            local_points = local_points[::skip, :].astype(np.float32)
            frames_per_pc = config.frames_per_pc
            local_imgs = np.array([self.load_pad(self.tgt_im_paths[s//frames_per_pc*frames_per_pc]) for s in snippet])

          nbs = [idx+1]
          # ret["points"].append(local_points)
          ret["points2"].append(local_points / m)
          # ret["feats"].append(local_feats)
          ret["imgs"].append(local_imgs)

          tgt_dm = load(self.tgt_dm_paths[idx+1])
          tgt_dm = self.pad(tgt_dm)
          tgt_K = self.tgt_Ks[idx+1]
          tgt_R = self.tgt_Rs[idx+1]
          tgt_t = self.tgt_ts[idx+1]
          src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
          src_dms = self.pad(src_dms)
          src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
          src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
          src_ts = np.array([self.src_ts[ii] for ii in nbs])
          patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)

          if local_points.shape[0] < 1000000:
            sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            patch,
            self.bwd_depth_thresh,
            self.invalid_depth_to_inf,
            local_points
            )
          else:
            sampling_maps = []
            for pt in range(0,local_points.shape[0], 1000000):
              sampling_map, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
              tgt_dm,
              tgt_K,
              tgt_R,
              tgt_t,
              src_dms,
              src_Ks,
              src_Rs,
              src_ts,
              patch,
              self.bwd_depth_thresh,
              self.invalid_depth_to_inf,
              local_points[pt:pt+1000000,:]
              )
              sampling_maps.append(sampling_map)
            sampling_maps = np.concatenate(sampling_maps, 0)
          
          ret["points"].append(sampling_maps) 

        tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
        tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
        ret["src"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)
        ret["tgt"] = load(self.tgt_im_paths[idx])

      else:
        m = max(np.abs(np.max(self.points)), np.abs(np.min(self.points)))
        nbs = [idx]
        ret["points"] = self.points
        ret["points2"] = self.points / m
        ret["feats"] = self.feats
      
        tgt_dm = load(self.tgt_dm_paths[idx])
        tgt_dm = self.pad(tgt_dm)
        tgt_K = self.tgt_Ks[idx]
        tgt_R = self.tgt_Rs[idx]
        tgt_t = self.tgt_ts[idx]
        src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
        src_dms = self.pad(src_dms)
        src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        src_ts = np.array([self.src_ts[ii] for ii in nbs])
        patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
        
        if ret["points"].shape[0] < 1000000:
          sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
          tgt_dm,
          tgt_K,
          tgt_R,
          tgt_t,
          src_dms,
          src_Ks,
          src_Rs,
          src_ts,
          patch,
          self.bwd_depth_thresh,
          self.invalid_depth_to_inf,
          ret["points"]
          )
        else:
          sampling_maps = []
          for pt in range(0,ret["points"].shape[0], 1000000):
            sampling_map, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            patch,
            self.bwd_depth_thresh,
            self.invalid_depth_to_inf,
            ret["points"][pt:pt+1000000,:]
            )
            sampling_maps.append(sampling_map)
          sampling_maps = np.concatenate(sampling_maps, 0)
        
        ret["points"] = sampling_maps
        tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
        tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
        ret["src"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)
        ret["tgt"] = load(self.tgt_im_paths[idx])

    else:
      if config.use_local_pointcloud:
        m = max(np.abs(np.max(self.points)), np.abs(np.min(self.points)))
        if config.use_naive:
          snippet_size = config.snippet_size
          snippet_half = snippet_size//2
          # snippet_idx = min(max(0,(idx - snippet_size)//snippet_size*snippet_size),(self.points.shape[0]-1)//snippet_size*snippet_size)
          # snippet = list(range(snippet_idx, min(snippet_idx + 2*snippet_size, self.points.shape[0])))
          snippet_idx = min(max(0, idx - snippet_half),self.points.shape[0]-snippet_size) if snippet_half != 0 else idx
          snippet = list(range(snippet_idx, snippet_idx + snippet_size))
          local_points = self.points[snippet]
          local_points = local_points.reshape(-1, 3)
          # local_feats = self.feats[snippet]
          # local_feats = local_feats.transpose(1,0,2,3).reshape(256,-1)
          local_imgs = np.array([self.load_pad(self.tgt_im_paths[s]) for s in snippet])
          # m = max(np.abs(np.max(local_points)), np.abs(np.min(local_points)))
          # nbs = [idx]
          # ret["points"] = local_points
          # ret["points2"] = local_points / m
          # ret["feats"] = local_feats
        
        else:
          frames_per_pc = config.frames_per_pc
          snippet_size = config.snippet_size
          snippet_half = snippet_size // 2
          snippet_stride = config.snippet_stride
          # snippet_idx = min(max(0, (idx-snippet_half)//frames_per_pc),len(self.src_im_paths)//frames_per_pc - snippet_size // frames_per_pc)

          # snippet = list(range(snippet_idx,snippet_idx+snippet_size//frames_per_pc))
          snippet_idx = min(max(0, idx - snippet_half),self.points.shape[0]-snippet_size) if snippet_half != 0 else idx
          snippet = list(range(snippet_idx, snippet_idx + snippet_size))
          # local_points = self.points[snippet_idx*snippet_size:(snippet_idx+1)*snippet_size]
          local_points = self.points[snippet]
          local_points = local_points.reshape(-1, 3)
          local_imgs = np.array([self.load_pad(self.tgt_im_paths[s]) for s in snippet])
          # local_feats = self.feats[snippet]
          # local_feats = local_feats.transpose(1,0,2,3).reshape(256,-1)

        # m = max(np.abs(np.max(local_points)), np.abs(np.min(local_points)))
        nbs = [idx]
        # ret["points"] = local_points
        ret["points"] = local_points
        # ret["points_wrp"] = None
        ret["points2"] = local_points / m
        ret["imgs"] = local_imgs
        # ret["feats"] = local_feats
        # ret["global_feats"] = self.global_feats

      else:
        m = max(np.abs(np.max(self.points)), np.abs(np.min(self.points)))
        nbs = [idx]
        ret["points"] = self.points
        # ret["points_wrp"] = None
        ret["points2"] = self.points / m
        ret["feats"] = self.feats
    
      # original pose
      tgt_dm = load(self.tgt_dm_paths[idx])
      tgt_dm = self.pad(tgt_dm)
      tgt_K = self.tgt_Ks[idx]
      tgt_R = self.tgt_Rs[idx]
      tgt_t = self.tgt_ts[idx]
      src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
      src_dms = self.pad(src_dms)
      src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
      src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
      src_ts = np.array([self.src_ts[ii] for ii in nbs])
      patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)

      # stereo poses (Left eye)
      # eye_dist = 0.04
      # tgt_dm = load(self.tgt_dm_paths[idx])
      # tgt_dm = self.pad(tgt_dm)
      # tgt_K = self.tgt_Ks[idx]
      # tgt_R = self.tgt_Rs[idx]
      # tgt_t = self.tgt_ts[idx]
      # src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
      # src_dms = self.pad(src_dms)
      # src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
      # src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
      # src_ts = np.array([self.src_ts[ii] + self.src_Rs[ii] @ np.array([eye_dist,0.,0.]) for ii in nbs])
      # patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
      
      if ret["points"].shape[0] < 1000000:
        sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
        tgt_dm,
        tgt_K,
        tgt_R,
        tgt_t,
        src_dms,
        src_Ks,
        src_Rs,
        src_ts,
        patch,
        self.bwd_depth_thresh,
        self.invalid_depth_to_inf,
        ret["points"]
        )
      else:
        sampling_maps = []
        for pt in range(0,ret["points"].shape[0], 1000000):
          sampling_map, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
          tgt_dm,
          tgt_K,
          tgt_R,
          tgt_t,
          src_dms,
          src_Ks,
          src_Rs,
          src_ts,
          patch,
          self.bwd_depth_thresh,
          self.invalid_depth_to_inf,
          ret["points"][pt:pt+1000000,:]
          )
          sampling_maps.append(sampling_map)
        sampling_maps = np.concatenate(sampling_maps, 0)
      
      points_glob = ret["points"]
      ret["points"] = sampling_maps
      ret["points"] = ret["points"].reshape(ret["points2"].shape)

      # For temporal consistency checking
      timestamp = 10000
      if idx - timestamp >= 0:
        # original pose
        # nbs = [idx-timestamp]
        # tgt_dm = load(self.tgt_dm_paths[idx-timestamp])
        # tgt_dm = self.pad(tgt_dm)
        # tgt_K = self.tgt_Ks[idx-timestamp]
        # tgt_R = self.tgt_Rs[idx-timestamp]
        # tgt_t = self.tgt_ts[idx-timestamp]
        # src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
        # src_dms = self.pad(src_dms)
        # src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        # src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        # src_ts = np.array([self.src_ts[ii] for ii in nbs])
        # patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)

        # Right eye
        nbs = [idx-timestamp]
        tgt_dm = load(self.tgt_dm_paths[idx-timestamp])
        tgt_dm = self.pad(tgt_dm)
        tgt_K = self.tgt_Ks[idx-timestamp]
        tgt_R = self.tgt_Rs[idx-timestamp]
        tgt_t = self.tgt_ts[idx-timestamp]
        src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
        src_dms = self.pad(src_dms)
        src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        src_ts = np.array([self.src_ts[ii] - self.src_Rs[ii] @ np.array([eye_dist,0.,0.]) for ii in nbs])
        patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
        ret["points_wrp"] = points_glob
        
        if ret["points_wrp"].shape[0] < 1000000:
          sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
          tgt_dm,
          tgt_K,
          tgt_R,
          tgt_t,
          src_dms,
          src_Ks,
          src_Rs,
          src_ts,
          patch,
          self.bwd_depth_thresh,
          self.invalid_depth_to_inf,
          ret["points_wrp"]
          )
        else:
          sampling_maps = []
          for pt in range(0,ret["points_wrp"].shape[0], 1000000):
            sampling_map, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            patch,
            self.bwd_depth_thresh,
            self.invalid_depth_to_inf,
            ret["points_wrp"][pt:pt+1000000,:]
            )
            sampling_maps.append(sampling_map)
          sampling_maps = np.concatenate(sampling_maps, 0)
        
        ret["points_wrp"] = sampling_maps
        ret["points_wrp"] = ret["points_wrp"].reshape(ret["points2"].shape)

      tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
      tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
      ret["src"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)
      ret["tgt"] = load(self.tgt_im_paths[idx])
      ret["timestamp"] = timestamp
    
    return ret

