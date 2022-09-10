import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import torchvision
import itertools
import sys
sys.path.append("../")
sys.path.append("./")
import config
import co
import ext
from stylization.Criterion import LossCriterion
from stylization.vgg_models import encoder5 as loss_network
from stylization.vgg_models import encoder3
from projection.z_buffer_manipulator import get_splatter
from stylization.Matrix import MulLayer
from argparse import Namespace

if config.Train:
  proj_args = Namespace(learn_default_feature=False, radius=1, rad_pow=2, tau=1.0, accumulation='alphacomposite')
else:
  proj_args = Namespace(learn_default_feature=False, radius=2, rad_pow=2, tau=1.0, accumulation='alphacomposite')

class VGGPerceptualLoss(nn.Module):
  def __init__(self, inp_scale="-11"):
    super().__init__()
    self.inp_scale = inp_scale
    self.criterion = LossCriterion(["r11","r21","r31","r41"],["r41"],0.02,1.0)
    self.vgg5 = loss_network()
    self.vgg5.load_state_dict(torch.load("stylization/vgg_r51.pth"))
    for param in self.vgg5.parameters():
      param.requires_grad = False
    self.vgg5.cuda()


  def forward(self, es, ta, style):
    if self.inp_scale == "-11":
      es = (es + 1) / 2
      ta = (ta + 1) / 2
    es_feat = self.vgg5(es)
    style_feat = self.vgg5(style)
    ta_feat = self.vgg5(ta)
    loss,styleLoss,contentLoss,temporalLoss = self.criterion(es, style, ta, es_feat, style_feat, ta_feat)
    return [styleLoss, contentLoss, temporalLoss]


class UNet(nn.Module):
  def __init__(
    self,
    in_channels,
    enc_channels=[64, 128, 256],
    dec_channels=[128, 64],
    out_channels=3,
    n_enc_convs=2,
    n_dec_convs=2,
  ):
    super().__init__()

    self.encs = nn.ModuleList()
    self.enc_translates = nn.ModuleList()
    pool = False
    for enc_channel in enc_channels:
      stage = self.create_stage(
        in_channels, enc_channel, n_enc_convs, pool
      )
      self.encs.append(stage)
      translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
      self.enc_translates.append(translate)
      in_channels, pool = enc_channel, True

    self.decs = nn.ModuleList()
    for idx, dec_channel in enumerate(dec_channels):
      in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
      stage = self.create_stage(
        in_channels, dec_channel, n_dec_convs, False
      )
      self.decs.append(stage)

    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    if out_channels <= 0:
      self.out_conv = None
    else:
      self.out_conv = nn.Conv2d(
        dec_channels[-1], out_channels, kernel_size=1, padding=0
      )

  def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
    if padding is None:
      padding = kernel_size // 2
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
      nn.ReLU(inplace=True),
    )

  def create_stage(self, in_channels, out_channels, n_convs, pool):
    mods = []
    if pool:
      mods.append(nn.AvgPool2d(kernel_size=2))
    for _ in range(n_convs):
      mods.append(self.convrelu(in_channels, out_channels))
      in_channels = out_channels
    return nn.Sequential(*mods)

  def forward(self, x):
    outs = []
    for enc, enc_translates in zip(self.encs, self.enc_translates):
      x = enc(x)
      outs.append(enc_translates(x))

    for dec in self.decs:
      x0, x1 = outs.pop(), outs.pop()
      x = torch.cat((self.upsample(x0), x1), dim=1)
      x = dec(x)
      outs.append(x)

    x = outs.pop()
    if self.out_conv:
      x = self.out_conv(x)
    return x





class FixedNet(nn.Module):
  def __init__(self, enc_net, dec_net):
    super().__init__()
    self.enc_net = encoder3()
    self.enc_net.load_state_dict(torch.load("stylization/vgg_r31.pth"))
    self.dec_net = dec_net    
    state_path = "projection/net_dec.params"
    # state_path = "projection/net_dec_retrain_epoch_6_val_loss_17.6485225315094.params"
    state = torch.load(str(state_path))
    for k,v in list(state.items()):
      if k.find("dec_net") == -1:
        # state[k] = v # work only in our decoder model
        state.pop(k)
      else:
        state[k.replace("dec_net.","")] = v
        state.pop(k)
    # print(state.keys())
    self.dec_net.load_state_dict(state)
    self.matrix = MulLayer('r31')
  
  def forward(self, **kwargs):
    if self.training:
      return self.forward_train(**kwargs)
    else:
      return self.forward_eval(**kwargs)
  
  def forward_train(self, **kwargs):
    if config.use_local_pointcloud:
      src = kwargs["src"]
      style = kwargs["style"] 
      H, W = src.shape[-2:]
      style_feat = self.enc_net(style).detach()
      mysplatter = get_splatter("xyblending", None, proj_args, size=(H,W), C=256, points_per_pixel=128)
      points_batch = []
      points2_batch = []
      style_feat_batch = []
      feats_batch = []
      for i in range(len(kwargs["points"])):
        # print("interval {}".format(i))
        points = kwargs["points"][i].float()
        points2 = kwargs["points2"][i].float()
        # feats = kwargs["feats"][i].float()
        img = kwargs["imgs"][i].float().squeeze(0)
        style_feat = self.enc_net(style).detach()
        feats = self.enc_net((img+1)/2).detach().permute(1,0,2,3).reshape(256,-1).unsqueeze(0)
        skip = int(feats.shape[-1] // 100000)
        feats = feats[:,:,::skip]
        points_batch.append(points)
        points2_batch.append(points2)
        style_feat_batch.append(style_feat)
        feats_batch.append(feats)

      points = torch.cat(points_batch, dim=0)
      points2 = torch.cat(points2_batch, dim=0)
      style_feat = torch.cat(style_feat_batch, dim=0)
      feats = torch.cat(feats_batch, dim=0)

      feats = self.matrix(feats, style_feat, points2)
      proj_enc, _ = mysplatter(points, feats)
      proj_enc = self.dec_net(proj_enc)  
      # print(proj_enc.shape)
        # print(proj_enc.shape)
        # proj_encs.append(proj_enc)
      # proj_enc = torch.cat(proj_enc,0)  
      return {"out": proj_enc, "style": style}

    else:
      src = kwargs["src"]
      style = kwargs["style"] 
      points = kwargs["points"].float()
      points2 = kwargs["points2"].float()
      # feats = kwargs["feats"].float()
      img = kwargs["imgs"].float().squeeze(0)
      feats = self.enc_net((img+1)/2).detach().permute(1,0,2,3).reshape(256,-1).unsqueeze(0)
      H, W = src.shape[-2:]
      style_feat = self.enc_net(style).detach()
      feats = self.matrix(feats, style_feat, points2)    
      mysplatter = get_splatter("xyblending", None, proj_args, size=(H,W), C=256, points_per_pixel=128)
      proj_enc, _ = mysplatter(points, feats)
      proj_enc = self.dec_net(proj_enc)  
      return {"out": proj_enc, "style": style} 
  
  def forward_eval(self, **kwargs):
    nstyle = kwargs["style"].shape[1]
    src = kwargs["src"]
    H, W = src.shape[-2:]
    # print(H,W)
    if config.use_local_pointcloud:
      points2 = kwargs["points2"].float()
      # points2 = F.interpolate(points2.squeeze(0).permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
      # points2 = points2.reshape(1,-1,3)
    else:
      points2 = kwargs["points2"].float()
    # points2 = kwargs["points2"].float()
    
    proj_encs = []
    proj_encs_wrp = []
    mysplatter = get_splatter("xyblending", None, proj_args, size=(H,W), C=256, points_per_pixel=24)
    for i in range(nstyle):
      if config.use_local_pointcloud:
        points = kwargs["points"].float().clone()
        if "points_wrp" not in kwargs.keys():
          points_wrp = None
        else:
          points_wrp = kwargs["points_wrp"].float().clone()
        # print(points.shape)
        # points = F.interpolate(points.squeeze(0).permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
        # points = points.reshape(1,-1,3)
        img = kwargs["imgs"].float().squeeze(0)
        feats = self.enc_net((img+1)/2).detach().permute(1,0,2,3).reshape(256,-1).unsqueeze(0)
        # feats = kwargs["feats"].float()
        # print(feats.shape)
        # feats = F.interpolate(feats.squeeze(0), (H,W), mode='area').permute(1,0,2,3)
        # feats = feats.reshape(1,256,-1)
      else:
        points = kwargs["points"].float().clone()
        if "points_wrp" not in kwargs.keys():
          points_wrp = None
        else:
          points_wrp = kwargs["points_wrp"].float().clone()
        feats = kwargs["feats"].float()
        # img = kwargs["imgs"].float().squeeze(0)
        # feats = self.enc_net((img+1)/2).detach().permute(1,0,2,3).reshape(256,-1).unsqueeze(0)
        # glob_feats = None
      

      style = kwargs["style"][:,i,...]
      style_feat = self.enc_net(style).detach()   
      # print(points2.shape, feats.shape)
      feats = self.matrix(feats, style_feat, points2)  
      del style_feat
      del style
      proj_enc, _ = mysplatter(points, feats)
      proj_enc = self.dec_net(proj_enc)
      proj_encs.append(proj_enc)
      del points
      if points_wrp is not None:
        proj_enc_wrp, _ = mysplatter(points_wrp, feats)
        proj_enc_wrp = self.dec_net(proj_enc_wrp)
        proj_encs_wrp.append(proj_enc_wrp)
        del points_wrp

      del feats
    proj_encs = torch.stack(proj_encs,1)   
    if len(proj_encs_wrp) > 0:
      proj_encs_wrp = torch.stack(proj_encs_wrp,1) 
    else:
      proj_encs_wrp = None
    return {"out": proj_encs, "out_wrp": proj_encs_wrp, "timestamp": kwargs["timestamp"]}


def get_fixed_net(enc_net, dec_net, n_views):
  if dec_net == "unet4.64.3":
    dec_net = UNet(
      256,
      enc_channels=[256, 256, 256, 512],
      dec_channels=[256, 256, 256],
      out_channels=3,
      n_enc_convs=3,
      n_dec_convs=3,
    )
  else:
    raise Exception("invalid dec_net")

  return FixedNet(enc_net, dec_net)



