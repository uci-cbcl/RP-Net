import sys
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm
from torch import nn
from net.unet import dice_loss, binary_dice_loss
from net.unet import Unet_2D, up_conv, conv_block, Attention_block
import torch
import numpy as np
from torch.nn.init import xavier_uniform_
import math
from net.modules import *
thismodule = sys.modules[__name__]


bn_momentum = 0.1
affine = True

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.InstanceNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1, stride=1),
            nn.InstanceNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.InstanceNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32))

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64))

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))


        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.dsv = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='trilinear'),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.preBlock(x)
        out_pool, _ = self.maxpool1(out)
        out1 = self.forw1(out_pool)
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)
        # out3_pool, _ = self.maxpool4(out3)
        # out4 = self.forw4(out3_pool)

        # rev3 = self.path1(out4)
        # comb3 = self.back3(torch.cat((rev3, out3), 1))

        dsv = self.dsv(out3)

        return {'d1': out, 'd2': out1, 'd3': out2, 'd4': out3, 'dsv': dsv}


class U_Net_Encoder_FU3D(Unet_2D):
    def __init__(self, cfg, img_ch=5, output_ch=6, resnet_type=None):
        super().__init__(cfg, img_ch, output_ch)

        self.p_num = [24, 32, 64, 64]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=self.img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64 + self.p_num[0], ch_out=128)
        self.Conv3 = conv_block(ch_in=128 + self.p_num[1], ch_out=256)
        self.Conv4 = conv_block(ch_in=256 + self.p_num[2], ch_out=512)
        self.Conv5 = conv_block(ch_in=512 + self.p_num[3], ch_out=1024)


    def forward(self, x, features):
        # x = data['slice']
        # features = data['features']
        p1 = features['d1']
        p2 = features['d2']
        p3 = features['d3']
        p4 = features['d4']
        glob_feat = features['glob_feat']

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = torch.cat((x2, p1), dim=1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = torch.cat((x3, p2), dim=1)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = torch.cat((x4, p3), dim=1)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = torch.cat((x5, p4), dim=1)
        x5 = self.Conv5(x5)

        res = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

        return res


# class U_Net_Encoder(Unet_2D):
#     def __init__(self, cfg, img_ch=5, output_ch=6, resnet_type=None):
#         super().__init__(cfg, img_ch, output_ch)

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=self.img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)


#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         res = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

#         return res


# class U_Net_Decoder(Unet_2D):
#     def __init__(self, cfg, img_ch=5, output_ch=6, resnet_type=None):
#         super().__init__(cfg, img_ch, output_ch)

#         self.p_num = np.array([24, 32, 64, 64])

#         self.Up5 = up_conv(ch_in=1024 + self.p_num[3], ch_out=512)
#         self.Up_conv5 = conv_block(ch_in=1024 + self.p_num[2], ch_out=512)

#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv4 = conv_block(ch_in=512 + self.p_num[1], ch_out=256)

#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv3 = conv_block(ch_in=256 + self.p_num[0], ch_out=128)

#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv2 = conv_block(ch_in=192, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64, self.output_ch, kernel_size=1, stride=1, padding=0)

#         # self.Conv_1x1.bias.data.fill_(-4.59)


#     def forward(self, x, features_2D, features_3D):
#         # x = data['slice']
#         # features = data['features']
#         p1 = features_3D['d1']
#         p2 = features_3D['d2']
#         p3 = features_3D['d3']
#         p4 = features_3D['d4']
#         glob_feat = features_3D['glob_feat']

#         x1 = features_2D['x1']
#         x2 = features_2D['x2']
#         x3 = features_2D['x3']
#         x4 = features_2D['x4']
#         x5 = features_2D['x5']

#         # decoding + concat path
#         x5 = torch.cat((x5, p4), dim=1)
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5, p3), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4, p2), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3, p1), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2, glob_feat), dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


# class U_Net(Unet_2D):
#     def __init__(self, cfg, img_ch=5, output_ch=6, resnet_type=None):
#         super().__init__(cfg, img_ch, output_ch)

#         self.encoder = U_Net_Encoder(cfg=cfg, img_ch=img_ch, output_ch=output_ch, resnet_type=resnet_type)
#         self.decoder = U_Net_Decoder(cfg=cfg, img_ch=img_ch, output_ch=output_ch, resnet_type=resnet_type)  


class AttentionLayer(nn.Module):
    def __init__(self, num_feat_2D, num_feat_3D, num_feat, num_embed):
        super(AttentionLayer, self).__init__()
        self.global_pooling_3D = torch.nn.Sequential(
            torch.nn.Conv3d(num_feat_3D, num_feat, 1, bias=False),
            # torch.nn.InstanceNorm3d(num_feat),
            # torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveMaxPool3d((None, num_embed, num_embed)),
        ) 

        self.global_pooling_2D = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat_2D, num_feat, 1, bias=False),
            # torch.nn.BatchNorm2d(num_feat),
            # torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveMaxPool2d((num_embed, num_embed)),
        )

        self.w_q = torch.nn.Sequential(
            torch.nn.Linear(num_feat * num_embed ** 2, 256, bias=False),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.ReLU(inplace=True),
        )
        self.w_k = torch.nn.Sequential(
            torch.nn.Linear(num_feat * num_embed ** 2, 256, bias=False),
            # torch.nn.BatchNorm1d(512),
            # torch.nn.ReLU(inplace=True),
        )


    def forward(self, feat_2D, feat_3D):
        B, _, _, _ = feat_2D.shape
        _, C, D, H, W = feat_3D.shape

        feat_2D_sig = self.global_pooling_2D(feat_2D)
        # feat_2D_sig = getattr(self, k_2d)(feat_2D_sig) 

        feat_3D_sig = self.global_pooling_3D(feat_3D)
        # feat_3D_sig = getattr(self, k_3d)(feat_3D_sig)

        feat_2D_sig = feat_2D_sig.view(B, -1)
        feat_3D_sig = feat_3D_sig.permute(0, 1, 3, 4, 2).contiguous()
        feat_3D_sig = feat_3D_sig.view(-1, D)

        # feat_2D_sig = self.w_q(feat_2D_sig)
        # feat_3D_sig = feat_3D_sig.permute(0, 2, 1, 3, 4).contiguous()
        # feat_3D_sig = feat_3D_sig.view(D, -1)
        # feat_3D_sig = self.w_k(feat_3D_sig)
        # feat_3D_sig = feat_3D_sig.transpose(0, 1).contiguous()

        _, C = feat_2D_sig.shape

        slice_att = torch.matmul(feat_2D_sig, feat_3D_sig)
        slice_att = slice_att / math.sqrt(C)
        slice_att = slice_att.softmax(dim=1)
        slice_att_v = slice_att

        slice_att = slice_att.view(B, 1, D, 1, 1)

        fuse_attention = feat_3D * slice_att
        fuse_attention = fuse_attention.sum(dim=2)

        return fuse_attention, slice_att_v


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_head, num_feat_2D, num_feat_3D, num_feat, num_embed):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_head = num_head

        for i in range(num_head):
            l = AttentionLayer(num_feat_2D, num_feat_3D, num_feat, num_embed)
            setattr(self, 'att_layer_{}'.format(i), l)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(num_head * num_feat_3D, num_feat_3D, 1, bias=False),
            torch.nn.BatchNorm2d(num_feat_3D),
            torch.nn.ReLU(inplace=True),
        )


    def forward(self, feat_2D, feat_3D):
        num_head = self.num_head
        fuse_attention, slice_att_v = [], []

        for i in range(num_head):
            l = getattr(self, 'att_layer_{}'.format(i))
            f, att = l(feat_2D, feat_3D)
            fuse_attention.append(f)
            slice_att_v.append(att.unsqueeze(0))

        fuse_attention = torch.cat(fuse_attention, dim=1)
        fuse_attention = self.conv(fuse_attention)
        slice_att_v = torch.cat(slice_att_v, dim=0)
        slice_att_v = slice_att_v.permute(1, 2, 0).contiguous()

        return fuse_attention, slice_att_v


class U_Net(Unet_2D):
    def __init__(self, cfg, img_ch=5, output_ch=6, resnet_type=None, feature_scale=1.):
        super().__init__(cfg, img_ch, output_ch)

        self.p_num = [24, 32, 64, 64]
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        # filters = [64, 96, 128, 192, 256]
        basic_block = cfg.get('base_block', 'conv_block')
        basic_block = getattr(thismodule, basic_block)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = basic_block(ch_in=self.img_ch, ch_out=filters[0], normalization_type=cfg['unet_normalize_type'])
        self.Conv2 = basic_block(ch_in=filters[0] + self.p_num[0], ch_out=filters[1], normalization_type=cfg['unet_normalize_type'])
        self.Conv3 = basic_block(ch_in=filters[1] + self.p_num[1], ch_out=filters[2], normalization_type=cfg['unet_normalize_type'])
        self.Conv4 = basic_block(ch_in=filters[2] + self.p_num[2], ch_out=filters[3], normalization_type=cfg['unet_normalize_type'])
        self.Conv5 = basic_block(ch_in=filters[3] + self.p_num[3], ch_out=filters[4], normalization_type=cfg['unet_normalize_type'])

        self.self_attention1 = MultiHeadAttentionLayer(2, filters[0], self.p_num[0], 2, 16)
        self.self_attention2 = MultiHeadAttentionLayer(2, filters[1], self.p_num[1], 2, 8)
        self.self_attention3 = MultiHeadAttentionLayer(4, filters[2], self.p_num[2], 4, 4)
        self.self_attention4 = MultiHeadAttentionLayer(4, filters[3], self.p_num[3], 4, 4)

        # self.self_attention1 = AttentionLayer(64, self.p_num[0], 4, 16)
        # self.self_attention2 = AttentionLayer(128, self.p_num[1], 8, 8)
        # self.self_attention3 = AttentionLayer(256, self.p_num[2], 16, 4)
        # self.self_attention4 = AttentionLayer(512, self.p_num[3], 16, 4)  

        # self.self_attention_back1 = AttentionLayer(64, self.p_num[0], 24, 16)
        # self.self_attention_back2 = AttentionLayer(128, self.p_num[0], 32, 16)
        # self.self_attention_back3 = AttentionLayer(256, self.p_num[1], 64, 8)
        # self.self_attention_back4 = AttentionLayer(512, self.p_num[2], 64, 4)

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3], normalization_type=cfg['unet_normalize_type'])
        self.Up_conv5 = basic_block(ch_in=filters[4], ch_out=filters[3], normalization_type=cfg['unet_normalize_type'])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2], normalization_type=cfg['unet_normalize_type'])
        self.Up_conv4 = basic_block(ch_in=filters[3], ch_out=filters[2], normalization_type=cfg['unet_normalize_type'])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1], normalization_type=cfg['unet_normalize_type'])
        self.Up_conv3 = basic_block(ch_in=filters[2], ch_out=filters[1], normalization_type=cfg['unet_normalize_type'])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0], normalization_type=cfg['unet_normalize_type'])
        self.Up_conv2 = basic_block(ch_in=filters[1] + 64, ch_out=filters[0], normalization_type=cfg['unet_normalize_type'])

        self.Conv_1x1 = nn.Conv2d(filters[0], self.output_ch, kernel_size=1, stride=1, padding=0)

        # self.Conv_1x1.bias.data.fill_(-4.59)

    def forward(self, x, features):
        # x = data['slice']
        # features = data['features']
        p1 = features['d1']
        p2 = features['d2']
        p3 = features['d3']
        p4 = features['d4']
        glob_feat = features['glob_feat']
        # activation = features['activation']

        # x = torch.cat((x, activation), dim=1)
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2_att, slice_att_1 = self.self_attention1(x2, p1)
        x2 = torch.cat((x2, x2_att), dim=1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3_att, slice_att_2 = self.self_attention2(x3, p2)
        x3 = torch.cat((x3, x3_att), dim=1)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4_att, slice_att_3 = self.self_attention3(x4, p3)
        x4 = torch.cat((x4, x4_att), dim=1)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5_att, slice_att_4 = self.self_attention4(x5, p4)
        x5 = torch.cat((x5, x5_att), dim=1)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        # d5_att, slice_att_back_4 = self.self_attention_back4(d5, p3)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # d4_att, slice_att_back_3 = self.self_attention_back3(d4, p2)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # d3_att, slice_att_back_2 = self.self_attention_back2(d3, p1)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2, glob_feat), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return {'seg_2d': d1, 'slice_att_1': slice_att_1, 'slice_att_2': slice_att_2,
                'slice_att_3': slice_att_3, 'slice_att_4': slice_att_4,
                # 'slice_att_back_4': slice_att_back_4, 'slice_att_back_3': slice_att_back_3,
                # 'slice_att_back_2': slice_att_back_2
               }


class AttU_Net(Unet_2D):
    def __init__(self, cfg, img_ch=5, output_ch=6, resnet_type=None):
        super().__init__(cfg, img_ch, output_ch)
        self.p_num = [24, 32, 64, 64]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=self.img_ch, ch_out=64, normalization_type=cfg['unet_normalize_type'])
        self.Conv2 = conv_block(ch_in=64 + self.p_num[0], ch_out=128, normalization_type=cfg['unet_normalize_type'])
        self.Conv3 = conv_block(ch_in=128 + self.p_num[1], ch_out=256, normalization_type=cfg['unet_normalize_type'])
        self.Conv4 = conv_block(ch_in=256 + self.p_num[2], ch_out=512, normalization_type=cfg['unet_normalize_type'])
        self.Conv5 = conv_block(ch_in=512 + self.p_num[3], ch_out=1024, normalization_type=cfg['unet_normalize_type'])

        self.self_attention1 = MultiHeadAttentionLayer(2, 64, self.p_num[0], 2, 16)
        self.self_attention2 = MultiHeadAttentionLayer(2, 128, self.p_num[1], 2, 8)
        self.self_attention3 = MultiHeadAttentionLayer(4, 256, self.p_num[2], 4, 4)
        self.self_attention4 = MultiHeadAttentionLayer(4, 512, self.p_num[3], 4, 4)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, normalization_type=cfg['unet_normalize_type'])
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256, normalization_type=cfg['unet_normalize_type'])
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, normalization_type=cfg['unet_normalize_type'])

        self.Up4 = up_conv(ch_in=512, ch_out=256, normalization_type=cfg['unet_normalize_type'])
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128, normalization_type=cfg['unet_normalize_type'])
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, normalization_type=cfg['unet_normalize_type'])

        self.Up3 = up_conv(ch_in=256, ch_out=128, normalization_type=cfg['unet_normalize_type'])
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64, normalization_type=cfg['unet_normalize_type'])
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, normalization_type=cfg['unet_normalize_type'])

        self.Up2 = up_conv(ch_in=128, ch_out=64, normalization_type=cfg['unet_normalize_type'])
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32, normalization_type=cfg['unet_normalize_type'])
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, normalization_type=cfg['unet_normalize_type'])

        self.Conv_1x1 = nn.Conv2d(128, self.output_ch, kernel_size=1, stride=1, padding=0)
        # self.Conv_1x1.bias.data.fill_(-4.59)

    def forward(self, x, features):
        # x = data['slice']
        # features = data['features']
        p1 = features['d1']
        p2 = features['d2']
        p3 = features['d3']
        p4 = features['d4']
        glob_feat = features['glob_feat']

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2_att, slice_att_1 = self.self_attention1(x2, p1)
        x2 = torch.cat((x2, x2_att), dim=1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3_att, slice_att_2 = self.self_attention2(x3, p2)
        x3 = torch.cat((x3, x3_att), dim=1)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4_att, slice_att_3 = self.self_attention3(x4, p3)
        x4 = torch.cat((x4, x4_att), dim=1)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5_att, slice_att_4 = self.self_attention4(x5, p4)
        x5 = torch.cat((x5, x5_att), dim=1)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d2 = torch.cat((d2, glob_feat), dim=1)
        d1 = self.Conv_1x1(d2)


        return {'seg_2d': d1, 'slice_att_1': slice_att_1, 'slice_att_2': slice_att_2,
                'slice_att_3': slice_att_3, 'slice_att_4': slice_att_4,
                # 'slice_att_back_4': slice_att_back_4, 'slice_att_back_3': slice_att_back_3,
                # 'slice_att_back_2': slice_att_back_2
               }


class LGCANet_V3(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(LGCANet_V3, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.context_net = FeatureNet(cfg, 1, len(cfg['roi_names']))

        unet_model = cfg['net_UNet']
        self.unet = getattr(thismodule, unet_model)(cfg, img_ch=1, output_ch=len(cfg['roi_names']), feature_scale=cfg.get('feature_scale', 1))
        # self.view_pooling = ViewPoolingLayer(cfg)
        # self.att_fuse = AttentionFuseLayer(cfg)


    def forward(self, data):
        volume = data['volume']
        slice = data['slice']
        slice_num = data['slice_num']
        slice_weight = data['slice_weight']
        
        # 3D context net
        features_3D = data_parallel(self.context_net, (volume))
        dsv = features_3D['dsv']
        # features_2D = data_parallel(self.unet.encoder, slice)
        p4 = features_3D['d4']
        B, C, H, W = slice.shape
        glob_feat = torch.nn.AdaptiveAvgPool3d((1, 1, 1))(p4)
        glob_feat = glob_feat.view(glob_feat.shape[0], -1)
        glob_feat = glob_feat.expand(B, -1)
        glob_feat = glob_feat.view(glob_feat.shape[0], glob_feat.shape[1], 1, 1)
        glob_feat = glob_feat.expand(-1, -1, H, W)
        
        # view pooling features
        # features = self.view_pooling(features, data)
        # features = data_parallel(self.view_pooling, (features, data))
        for k, feat in features_3D.items():
            features_3D[k] = feat.expand(torch.cuda.device_count(), -1, -1, -1, -1)
        features_3D['glob_feat'] = glob_feat
        # features_3D = data_parallel(self.view_pooling, (features_3D, slice, slice_num, slice_weight))

        # features_3D = data_parallel(self.att_fuse, (features_3D, features_2D, slice, slice_num, slice_weight))
        
        # 2D unet
        output = data_parallel(self.unet, (slice, features_3D))
        # output = data_parallel(self.unet.decoder, (slice, features_2D, features_3D))
        output['dsv'] = dsv

        return output


    def loss(self, pred, target):
        pred_2D = pred['seg_2d']
        target_2D = target['mask']

        _, num_class, _, _ = pred_2D.shape
        pred_2D = pred_2D.permute(0, 2, 3, 1).contiguous().view(-1, num_class)
        target_2D = target_2D.permute(0, 2, 3, 1).contiguous().view(-1, num_class)

        unet_dice = dice_loss(pred_2D, target_2D)
        
        pred_3d = pred['dsv']
        target_3d = target['downsampled_volume_mask']
        pred_3d = pred_3d.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_class)
        target_3d = target_3d.permute(1, 2, 3, 0).contiguous().view(-1, num_class)
        dsv_loss = dice_loss(pred_3d, target_3d)

        loss_dice = []
        for i in range(len(unet_dice)):
            loss_dice.append(dsv_loss[i] + unet_dice[i])

        return {'unet_dice': unet_dice, 'loss_dice': loss_dice}


    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()
