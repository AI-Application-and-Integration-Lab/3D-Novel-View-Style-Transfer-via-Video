import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from models.stylize.utils import repackage_hidden
from loss import TVLoss
from smooth import smooth_trajectory, get_smooth_kernel, generate_right_poses, generate_LR_poses
# from scipy.ndimage import binary_dilation, binary_erosion
# from scipy.ndimage import gaussian_filter as scipy_gaussian


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram



def vgg_norm(var):
    dtype = torch.cuda.FloatTensor
    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    normed = var.sub(mean).div(std)
    return normed

def vgg_denorm(var):
    dtype = torch.cuda.FloatTensor
    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    normed = var.mul(std).add(mean)
    return normed






class VideoNet(nn.Module):
    def __init__(self, styler, vgg, flow):
        super(VideoNet, self).__init__()
        self.styler = styler
        self.vgg = vgg
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.flow_processor = flow
        # self.flownet.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        for parm in self.flow_processor.model.parameters():
            param.requires_grad = False
        self.tvloss = TVLoss()
        '''
        for name, param in self.styler.named_parameters():
            if  'style' in name:
                param.requires_grad = False
        '''


    def set_train(self):
        for name, param in self.styler.named_parameters():
            param.requires_grad = True

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size()[:2] == target.size()[:2])
        assert (target.requires_grad is False)
        input_gram = gram_matrix(input)
        target_gram = gram_matrix(target)
        return self.mse_loss(input_gram, target_gram)

    def concat(self, content1, content2):
        content = torch.cat([content1, content2], dim=1)
        return content

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        # if flo.size(2) < x.size(2):
        #     scale_factor = x.size(2) // flo.size(2)
        #     flo = torch.nn.functional.upsample(flo, size=x.size()[-2:], mode='bilinear')  * scale_factor
        # elif flo.size(2) > x.size(2):
        #     scale_factor = flo.size(2) // x.size(2)
        #     flo = torch.nn.functional.avg_pool2d(flo, scale_factor)  / scale_factor

        # B, C, H, W = x.size()
        # # mesh grid
        # xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        # yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        # grid = torch.cat((xx,yy),1).float()

        # if x.is_cuda:
        #     grid = grid.cuda()
        # vgrid = Variable(grid) + flo

        # # scale grid to [-1,1]
        # vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        # vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        # vgrid = vgrid.permute(0,2,3,1)
        # output = nn.functional.grid_sample(x, vgrid)

        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid)
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1

        output = F.grid_sample(x, flo)

        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = F.grid_sample(mask, flo)
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask

    def mask_occlusion(self, img1, img2_, alpha=20.0):
        return torch.exp( -alpha * torch.sum(img1 - img2_, dim=1).pow(2)  ).unsqueeze(1)

    def calc_temporal_loss(self, img1, img2, flow, mask):
        return self.l1_loss(mask * self.warp(img2, flow),  Variable(mask.data * img1.data, requires_grad=False))

    def temporal_loss(self, content1, content2, img1, img2):
        with torch.no_grad():
            H, W = content2.shape[-2:]
            # contents = self.concat(content1, content2)
            flowout = self.flow_processor.get_flow(content1, content2, pre_homo=False, grid_normalize=True)
            flowout = F.interpolate(flowout.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            # flowout = self.flownet(contents)
            mask = self.mask_occlusion(content1, self.warp(content2, flowout))
        return self.calc_temporal_loss(img1, img2, flowout, mask)



    def forward(self, content1, content2, style, prev_state1, prev_state2, bank):
        H, W = content2.shape[-2:]
        with torch.no_grad():
            # contents = self.concat(content1, content2)
            flowout = self.flow_processor.get_flow(content1, content2, pre_homo=False, grid_normalize=True)
            flowout = F.interpolate(flowout.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            mask = self.mask_occlusion(content1, self.warp(content2, flowout))
            g_t1, return_state1, return_state2 = self.styler(vgg_norm(content1), prev_state1, prev_state2, bank)

        g_t2, prev_state1, prev_state2 = self.styler(vgg_norm(content2), repackage_hidden(return_state1), repackage_hidden(return_state2), bank)

        content_feat = self.vgg(vgg_norm(Variable(content2.data, requires_grad=False)))[2]
        style_feats = self.vgg(vgg_norm(style))
        output_feats = self.vgg(vgg_norm(g_t2))

        loss_c = self.calc_content_loss(output_feats[2], Variable(content_feat.data, requires_grad=False))
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(output_feats[i], style_feats[i].data)
        loss_t = self.calc_temporal_loss(g_t1, g_t2, flowout, mask)

        return loss_c, loss_s, loss_t, g_t1, g_t2, return_state1, return_state2 # original g_t1 => now g_t2

    def forward_LR_video(self, content1L, content2L, content1R, style, prev_state1, prev_state2, bank):
        H, W = content2L.shape[-2:]
        with torch.no_grad():
            # contents = self.concat(content1, content2)
            flow12L = self.flow_processor.get_flow(content1L, content2L, pre_homo=False, grid_normalize=True)
            flow12L = F.interpolate(flow12L.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            mask12L = self.mask_occlusion(content1L, self.warp(content2L, flow12L))
            flowLR = self.flow_processor.get_flow(content1L, content1R, pre_homo=False, grid_normalize=True)
            flowLR = F.interpolate(flowLR.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskLR = self.mask_occlusion(content1L, self.warp(content1R, flowLR))
            g_t1, return_state1, return_state2 = self.styler(vgg_norm(content1L), prev_state1, prev_state2, bank)

        g_t2, prev_state1, prev_state2 = self.styler(vgg_norm(content2L), repackage_hidden(return_state1), repackage_hidden(return_state2), bank)
        g_tR, _, _ = self.styler(vgg_norm(content1R), repackage_hidden(return_state1), repackage_hidden(return_state2), bank)
        
        style_feats = self.vgg(vgg_norm(style))

        content2L_feat = self.vgg(vgg_norm(Variable(content2L.data, requires_grad=False)))[2]
        output2L_feats = self.vgg(vgg_norm(g_t2))

        content1R_feat = self.vgg(vgg_norm(Variable(content1R.data, requires_grad=False)))[2]
        output1R_feats = self.vgg(vgg_norm(g_tR))

        loss_c = self.calc_content_loss(output2L_feats[2], Variable(content2L_feat.data, requires_grad=False))
        loss_c += self.calc_content_loss(output1R_feats[2], Variable(content1R_feat.data, requires_grad=False))
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(output2L_feats[i], style_feats[i].data)
            loss_s += self.calc_style_loss(output1R_feats[i], style_feats[i].data)
        loss_t = self.calc_temporal_loss(g_t1, g_t2, flow12L, mask12L)
        loss_t += self.calc_temporal_loss(g_t1, g_tR, flowLR, maskLR)

        return loss_c, loss_s, loss_t, g_t2, g_tR, return_state1, return_state2 # original g_t1 => now g_t2

    def forward_eval(self, contentL2, contentR2, contentL1, contentR1, style, prev_state1, prev_state2, bank, stylizeL1, stylizeR1, post_process=False):
        H, W = contentL2.shape[-2:]
        with torch.no_grad():
            g_t1, return_state1, return_state2 = self.styler(vgg_norm(contentL2), prev_state1, prev_state2, bank)
            g_t2, prev_state1, prev_state2 = self.styler(vgg_norm(contentR2), repackage_hidden(return_state1), repackage_hidden(return_state2), bank)

        maskL, maskR = None, None
        stylizeL21, stylizeR21 = None, None
        # if post_process:
        with torch.no_grad():
            flowLR = self.flow_processor.get_flow(contentL2, contentR2, pre_homo=False, grid_normalize=True)
            flowLR = F.interpolate(flowLR.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskLR = self.mask_occlusion(contentL2, self.warp(contentR2, flowLR))
            flowRL = self.flow_processor.get_flow(contentR2, contentL2, pre_homo=False, grid_normalize=True)
            flowRL = F.interpolate(flowRL.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskRL = self.mask_occlusion(contentR2, self.warp(contentL2, flowRL))
            if stylizeL1 is not None and stylizeR1 is not None:
                flow21L = self.flow_processor.get_flow(contentL2, contentL1, pre_homo=False, grid_normalize=True)
                flow21L = F.interpolate(flow21L.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
                mask21L = self.mask_occlusion(contentL2, self.warp(contentL1, flow21L))
                flow21R = self.flow_processor.get_flow(contentR1, contentR2, pre_homo=False, grid_normalize=True)
                flow21R = F.interpolate(flow21R.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
                mask21R = self.mask_occlusion(contentR2, self.warp(contentR1, flow21R))
            else:
                flow21L, flow21R, mask21L, mask21R = None, None, None, None

            if flow21L is not None and flow21R is not None:
                stylizeL21 = self.warp(stylizeL1, flow21L)
                stylizeR21 = self.warp(stylizeR1, flow21R)

        return g_t1, g_t2, return_state1, return_state2, maskL, maskR, stylizeL21, stylizeR21