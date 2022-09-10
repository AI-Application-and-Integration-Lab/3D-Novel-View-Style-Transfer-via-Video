import torch.nn as nn
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
from loss import TVLoss
import torch.nn.functional as F


def affine_loss(output, M):
    loss_affine = 0.0
    N = output.size(0)
    for i in range(3):
        # V = output[]
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse_tensor_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine


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

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t) / (ch * h * w)
    return gram



class Net(nn.Module):
    def __init__(self, styler, vgg, flow):
        super(Net, self).__init__()
        self.styler = styler
        self.vgg = vgg
        if self.vgg is not None:
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss(TVLoss_weight=1)
        self.flow_processor = flow

    def set_train(self):
        for name, param in self.styler.named_parameters():
            param.requires_grad = True

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (target.requires_grad is False)
        input_gram = gram_matrix(input)
        target_gram = gram_matrix(target)
        return self.mse_loss(input_gram, target_gram)

    def calc_temporal_loss_LR_video(self, g_ts, flowout, mask):
        loss = 0.
        loss += self.l1_loss(mask[0] * self.warp(g_ts[1], flowout[0]),  Variable(mask[0].data * g_ts[0].data, requires_grad=False))
        loss += self.l1_loss(mask[1] * self.warp(g_ts[0], flowout[1]),  Variable(mask[1].data * g_ts[1].data, requires_grad=False))
        loss += self.l1_loss(mask[2] * self.warp(g_ts[0], flowout[2]),  Variable(mask[2].data * g_ts[2].data, requires_grad=False))
        loss += self.l1_loss(mask[3] * self.warp(g_ts[1], flowout[3]),  Variable(mask[3].data * g_ts[3].data, requires_grad=False))
        return loss

    def calc_temporal_loss_LR_image(self, g_ts, flowout, mask):
        loss = 0.
        loss += self.l1_loss(mask[0] * self.warp(g_ts[1], flowout[0]),  Variable(mask[0].data * g_ts[0].data, requires_grad=False))
        loss += self.l1_loss(mask[1] * self.warp(g_ts[0], flowout[1]),  Variable(mask[1].data * g_ts[1].data, requires_grad=False))
        return loss

    def concat(self, content1, content2):
        content = torch.cat([content1, content2], dim=1)
        return content

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, H, W, 2] flow

        """
        output = F.grid_sample(x, flo)

        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = F.grid_sample(mask, flo)
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask

    def mask_occlusion(self, img1, img2_, alpha=50.0):
        return torch.exp( -alpha * torch.sum(img1 - img2_, dim=1).pow(2)  ).unsqueeze(1)
    # def forward(self, content, style, bank, alpha=1.0):
    #     content = vgg_norm(content)
    #     style = vgg_norm(style)
    #     output = self.styler(content, bank)
    #     content_feat = self.vgg(Variable(content.data, requires_grad=False))[2]
    #     style_feats = self.vgg(style)
    #     output_feats = self.vgg(vgg_norm(output))
    #     loss_c = self.calc_content_loss(output_feats[2], Variable(content_feat.data, requires_grad=False))
    #     loss_s = 0
    #     for i in range(4):
    #         loss_s += self.calc_style_loss(output_feats[i], style_feats[i])
    #     loss_t = self.tv_loss(output)
    #     save_image(output.data.clone(), 'out.jpg')
    #     return loss_c, loss_s, loss_t * 10
    def forward(self, contentL2, contentR2, contentL1, contentR1, style, bank, debug=False, alpha=1.0):
        assert 0 <= alpha <= 1
        H, W = contentL2.shape[-2:]
        with torch.no_grad():
            # contents = self.concat(contentL2, contentR2)
            flowLR = self.flow_processor.get_flow(contentL2, contentR2, pre_homo=False, grid_normalize=True)
            flowLR = F.interpolate(flowLR.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskLR = self.mask_occlusion(contentL2, self.warp(contentR2, flowLR))
            flowRL = self.flow_processor.get_flow(contentR2, contentL2, pre_homo=False, grid_normalize=True)
            flowRL = F.interpolate(flowRL.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskRL = self.mask_occlusion(contentR2, self.warp(contentL2, flowRL))
            flow12L = self.flow_processor.get_flow(contentL1, contentL2, pre_homo=False, grid_normalize=True)
            flow12L = F.interpolate(flow12L.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            mask12L = self.mask_occlusion(contentL1, self.warp(contentL2, flow12L))
            flow12R = self.flow_processor.get_flow(contentR1, contentR2, pre_homo=False, grid_normalize=True)
            flow12R = F.interpolate(flow12R.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            mask12R = self.mask_occlusion(contentR1, self.warp(contentR2, flow12R))
            style_feats = self.vgg(vgg_norm(style))
            contentL2_feat = self.vgg(vgg_norm(Variable(contentL2.data, requires_grad=False)))[2]
            contentR2_feat = self.vgg(vgg_norm(Variable(contentR2.data, requires_grad=False)))[2]
            g_tL1 = self.styler(vgg_norm(contentL1), bank)
            g_tR1 = self.styler(vgg_norm(contentR1), bank)

        g_tL2 = self.styler(vgg_norm(contentL2), bank)
        g_tR2 = self.styler(vgg_norm(contentR2), bank)

        output_feats_L = self.vgg(vgg_norm(g_tL2))
        output_feats_R = self.vgg(vgg_norm(g_tR2))

        loss_c = self.calc_content_loss(output_feats_L[2], Variable(contentL2_feat.data, requires_grad=False))
        loss_c += self.calc_content_loss(output_feats_R[2], Variable(contentR2_feat.data, requires_grad=False))
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(output_feats_L[i], style_feats[i].data)
            loss_s += self.calc_style_loss(output_feats_R[i], style_feats[i].data)

        g_ts = [g_tL2, g_tR2, g_tL1, g_tR1]
        flowout = [flowLR, flowRL, flow12L, flow12R]
        mask = [maskLR, maskRL, mask12L, mask12R]
        # loss_tc = self.calc_temporal_loss(g_t1, g_t2, flowout, mask)
        loss_tc = self.calc_temporal_loss_LR_video(g_ts, flowout, mask)
        loss_tv = self.tv_loss(g_tL2)
        loss_tv += self.tv_loss(g_tR2)
        return loss_c, loss_s, loss_tc, loss_tv, g_ts

    def forward_paired_images(self, contentL, contentR, style, bank, debug=False, alpha=1.0):
        assert 0 <= alpha <= 1
        H, W = contentL.shape[-2:]
        with torch.no_grad():
            # contents = self.concat(contentL2, contentR2)
            flowLR = self.flow_processor.get_flow(contentL, contentR, pre_homo=False, grid_normalize=True)
            flowLR = F.interpolate(flowLR.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskLR = self.mask_occlusion(contentL, self.warp(contentR, flowLR))
            flowRL = self.flow_processor.get_flow(contentR, contentL, pre_homo=False, grid_normalize=True)
            flowRL = F.interpolate(flowRL.permute(0,3,1,2), (H,W), mode='area').permute(0,2,3,1)
            maskRL = self.mask_occlusion(contentR, self.warp(contentL, flowRL))
            
            style_feats = self.vgg(vgg_norm(style))
            contentL_feat = self.vgg(vgg_norm(Variable(contentL.data, requires_grad=False)))[2]
            contentR_feat = self.vgg(vgg_norm(Variable(contentR.data, requires_grad=False)))[2]

        g_tL = self.styler(vgg_norm(contentL), bank)
        g_tR = self.styler(vgg_norm(contentR), bank)

        output_feats_L = self.vgg(vgg_norm(g_tL))
        output_feats_R = self.vgg(vgg_norm(g_tR))

        loss_c = self.calc_content_loss(output_feats_L[2], Variable(contentL_feat.data, requires_grad=False))
        loss_c += self.calc_content_loss(output_feats_R[2], Variable(contentR_feat.data, requires_grad=False))
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(output_feats_L[i], style_feats[i].data)
            loss_s += self.calc_style_loss(output_feats_R[i], style_feats[i].data)

        g_ts = [g_tL, g_tR]
        flowout = [flowLR, flowRL]
        mask = [maskLR, maskRL]
        # loss_tc = self.calc_temporal_loss(g_t1, g_t2, flowout, mask)
        loss_tc = self.calc_temporal_loss_LR_image(g_ts, flowout, mask)
        loss_tv = self.tv_loss(g_tL)
        loss_tv += self.tv_loss(g_tR)
        return loss_c, loss_s, loss_tc, loss_tv, g_ts, flowout

    def evaluate(self, content, bank):
        with torch.no_grad():
            output = self.styler(vgg_norm(content), bank)
        return output
