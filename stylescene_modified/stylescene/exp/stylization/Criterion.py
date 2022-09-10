import torch
import torch.nn as nn

class styleLoss(nn.Module):
  def forward(self,input,target):
    ib,ic,ih,iw = input.size()
    iF = input.view(ib,ic,-1)
    iMean = torch.mean(iF,dim=2)
    iCov = GramMatrix()(input)

    tb,tc,th,tw = target.size()
    tF = target.view(tb,tc,-1)
    tMean = torch.mean(tF,dim=2)
    tCov = GramMatrix()(target)

    loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iCov,tCov)
    return loss/tb

class GramMatrix(nn.Module):
  def forward(self,input):
    b, c, h, w = input.size()
    f = input.view(b,c,h*w) # bxcx(hxw)
    # torch.bmm(batch1, batch2, out=None)   #
    # batch1: bxmxp, batch2: bxpxn -> bxmxn #
    G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
    return G.div_(c*h*w)

class LossCriterion(nn.Module):
  def __init__(self,style_layers,content_layers,style_weight,content_weight):
    super(LossCriterion,self).__init__()

    self.style_layers = style_layers
    self.content_layers = content_layers
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.temporal_weight = 12.0

    self.styleLosses = [styleLoss()] * len(style_layers)
    self.contentLosses = [nn.MSELoss()] * len(content_layers)
    self.temporalLoss = nn.L1Loss()

    

  def forward(self,t,s,c,tF,sF,cF):
    # content loss
    totalContentLoss = 0
    for i,layer in enumerate(self.content_layers):
      tf_i = tF[layer]
      cf_i = cF[layer]
      cf_i = cf_i.detach().expand(tf_i.shape[0],-1,-1,-1)
      loss_i = self.contentLosses[i]
      totalContentLoss += loss_i(tf_i,cf_i)
    totalContentLoss = totalContentLoss * self.content_weight

    # style loss
    totalStyleLoss = 0
    for i,layer in enumerate(self.style_layers):
      tf_i = tF[layer]
      sf_i = sF[layer]
      sf_i = sf_i.detach().expand(tf_i.shape[0],-1,-1,-1)
      loss_i = self.styleLosses[i]
      totalStyleLoss += loss_i(tf_i,sf_i)
    totalStyleLoss = totalStyleLoss * self.style_weight

    # consistency loss
    totalTemporalLoss = 0
    for i in range(t.shape[0]-1):
      totalTemporalLoss += self.temporalLoss(t[i:i+1], t[i+1:i+2])
    for i in range(t.shape[0]-3):
      totalTemporalLoss += self.temporalLoss(t[:1], t[i+3:i+4])
    totalTemporalLoss = totalTemporalLoss * self.temporal_weight

    loss = totalStyleLoss + totalContentLoss + totalTemporalLoss

    return loss,totalStyleLoss,totalContentLoss,totalTemporalLoss

# class SSIM(nn.Module):
#     """Layer to compute the SSIM loss between a pair of images
#     """
#     def __init__(self, window_size=3, alpha=1, beta=1, gamma=1):
#         super(SSIM, self).__init__()
#         self.mu_x_pool   = nn.AvgPool2d(window_size, 1)
#         self.mu_y_pool   = nn.AvgPool2d(window_size, 1)
#         self.sig_x_pool  = nn.AvgPool2d(window_size, 1)
#         self.sig_y_pool  = nn.AvgPool2d(window_size, 1)
#         self.sig_xy_pool = nn.AvgPool2d(window_size, 1)

#         self.refl = nn.ReflectionPad2d(window_size//2)

#         self.C1 = 0.01 ** 2
#         self.C2 = 0.03 ** 2
#         self.C3 = self.C2 / 2
        
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         if alpha == 1 and beta == 1 and gamma == 1:
#             self.run_compute = self.compute_simplified
#         else:
#             self.run_compute = self.compute
        

#     def compute(self, x, y):
        
#         x = self.refl(x)
#         y = self.refl(y)
        
#         mu_x = self.mu_x_pool(x)
#         mu_y = self.mu_y_pool(y)

#         sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
#         sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
#         sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

#         l = (2 * mu_x * mu_y + self.C1) / \
#             (mu_x * mu_x + mu_y * mu_y + self.C1)
#         c = (2 * sigma_x * sigma_y + self.C2) / \
#             (sigma_x + sigma_y + self.C2)
#         s = (sigma_xy + self.C3) / \
#             (torch.sqrt(sigma_x * sigma_y) + self.C3)

#         ssim_xy = torch.pow(l, self.alpha) * \
#                   torch.pow(c, self.beta) * \
#                   torch.pow(s, self.gamma)
#         return torch.clamp((1 - ssim_xy) / 2, 0, 1)

#     def compute_simplified(self, x, y):
#         x = self.refl(x)
#         y = self.refl(y)

#         mu_x = self.mu_x_pool(x)
#         mu_y = self.mu_y_pool(y)

#         sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
#         sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
#         sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

#         SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
#         SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

#         return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

#     def forward(self, x, y):
#         return self.run_compute(x, y)