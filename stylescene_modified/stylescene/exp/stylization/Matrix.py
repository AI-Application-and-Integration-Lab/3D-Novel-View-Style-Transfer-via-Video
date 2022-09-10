import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../")
import config


from stylization.pointnet2_ssg_cls import PointNet2ClassificationSSG
from stylization.Pointlstm import PointLSTM


class CNN(nn.Module):
  def __init__(self,layer,matrixSize=32):
    super(CNN,self).__init__()
    self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(128,64,3,1,1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(64,matrixSize,3,1,1))
    self.pointnet = PointNet2ClassificationSSG()
    self.pointlstm = PointLSTM(pts_num=1024, in_channels=matrixSize+3, hidden_dim=matrixSize, offset_dim=3, num_layers=1)
    self.fc1 = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
    self.fc2 = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
    self.prev_pos = None


  def forward(self,feat,img,xyz,prev_hidden):
    # feat = feat.reshape(1,256,-1)
    outx = self.convs(img)
    b,c,h,w = outx.size()
    outx = outx.view(b,c,-1)
    outx = torch.bmm(outx,outx.transpose(1,2)).div(h*w)
    outx = outx.view(outx.size(0),-1)
    feat = feat.reshape(b,256,-1)
    
    skip = max(int(feat.shape[-1] // 50000),1)
    # skip = 1
    xyz = xyz[:,::skip,:].contiguous()
    feat = feat[:,:,::skip].contiguous()
    # print(xyz.shape, feat.shape)
    xyz, feat = self.pointnet(xyz, feat)

    xyz = xyz.permute(0,2,1).unsqueeze(0)
    feat = feat.unsqueeze(0)
    feat = torch.cat([xyz,feat], dim=2) # (1, T, c+3, n_points)
    if config.Train:
      out = self.pointlstm(feat, prev_hidden)
      outy = out[0][0].squeeze(-1).squeeze(0)[:,3:,:] # (T, c, n_points)
      hidden_y = out[1][0] # hidden state & cell
    else:
      out = self.pointlstm.forward_eval(feat, self.prev_pos, prev_hidden)
      outy = out[0][0].squeeze(-1).squeeze(0)[:,3:,:] # (T, c, n_points)
      hidden_y = out[1][0]
      self.prev_pos = xyz[:,0].unsqueeze(-1)
    
    # outy = feat
    
    b,c,p = outy.size()
    outy = outy.view(b,c,-1)
    outy = torch.bmm(outy,outy.transpose(1,2)).div(p)
    outy = outy.view(outy.size(0),-1)

    return self.fc1(outx), self.fc2(outy), hidden_y
    # return self.fc1(outx), self.fc2(outy), None


class MulLayer(nn.Module):
  def __init__(self,layer,matrixSize=64):
    super(MulLayer,self).__init__()
    self.cnet = CNN(layer,matrixSize)
    self.matrixSize = matrixSize
    self.compress = nn.Conv2d(256,matrixSize,1,1,0)
    self.unzip = nn.Conv2d(matrixSize,256,1,1,0)
    self.transmatrix = None
    self.prev_hidden = None

  def forward(self,cF,sF,points,trans=True):
    b = cF.shape[0]
    cF = cF.reshape(b,256,-1,1) # (1,256,-1,1) for original
    cMean = torch.mean(cF,dim=2,keepdim=True)
    cMeanC = cMean.expand_as(cF)
    cF = cF - cMeanC
     
    sMean = torch.mean(sF.view(b,256,-1,1),dim=2,keepdim=True)
    sMeanC = sMean.expand_as(cF)
    sMeanS = sMean.expand_as(sF)
    sF = sF - sMeanS

    compress_pointcloud = self.compress(cF).view(b,self.matrixSize,-1)
    cMatrix, sMatrix, prev_hidden = self.cnet(cF, sF, points, self.prev_hidden) 
    if not config.Train:
      self.prev_hidden = prev_hidden
    sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
    cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
    transmatrix = torch.bmm(sMatrix,cMatrix)

    compress_pointcloud = torch.bmm(transmatrix,compress_pointcloud).view(b,self.matrixSize,-1)
    feat = (self.unzip(compress_pointcloud.view(b,self.matrixSize,-1,1)) + sMeanC).squeeze(-1)
    return feat
