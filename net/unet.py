from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .modules import *

import copy
import torch.nn.functional as F
from torchvision import models
from torch.nn import init
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
# from .seg_hrnet import HRNET

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class GHMC(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
                                    target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


class GHMDice(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMDice, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        pred = pred.view(-1)
        target = target.view(-1)
        label_weight = label_weight.view(-1)

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        I = (pred * target).sum()
        S = pred.sum() + target.sum()
        g = torch.abs(2 * I / S * pred.detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = 1 - (2 * pred * target * weights).sum() / S
        return loss * self.loss_weight


def dice_loss(pred, target):
    N, C = pred.shape
    pred = pred.sigmoid()
    losses = []
    alpha = 0.5
    beta  = 0.5

    for i in range(C):
        p0 = (pred[:, i]).float()
        p1 = 1 - p0
        g0 = target[:, i]
        g1 = 1 - target[:, i]

        num = torch.sum(p0 * g0)
        den = num + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
        
        loss = 1 - num / (den + 1e-5)
        if g0.sum() == 0:
            loss = loss * 0

        losses.append(loss)

    return losses


def binary_dice_loss(pred, target, k=5):
    N, C = pred.shape
    pred = pred.sigmoid()
    losses = []
    alpha = 0.5
    beta  = 0.5

    for i in range(C):
        p0 = (pred[:, i]).float()
        g0 = target[:, i]

        # loss = GHMDice()(p0, g0, torch.ones_like(p0))

        # p0, g0 = topk_neg(p0, g0, k=k)

        # foreground
        num = torch.sum(p0 * g0)
        den = torch.sum(p0) + torch.sum(g0) + 1e-5

        loss_fore = 1 - num / (den + 1e-5)

        # background
        loss_back = - torch.sum((1 - p0) * (1 - g0)) / (torch.sum(1 - p0) + torch.sum(1 - g0) + 1e-5)

        loss = loss_fore + loss_back

        if g0.sum() == 0:
            loss = loss * 0
        # else:
        #     loss = loss / weight[i]

        losses.append(loss)

    return losses


def topk_dice_loss(pred, target, k=5):
    N, C = pred.shape
    pred = pred.sigmoid()
    losses = []
    alpha = 0.5
    beta  = 0.5

    for i in range(C):
        p0 = (pred[:, i]).float()
        g0 = target[:, i]

        # loss = GHMDice()(p0, g0, torch.ones_like(p0))

        # p0, g0 = topk_neg(p0, g0, k=k)

        # foreground
        num = torch.sum(p0 * g0)
        den = torch.sum(p0) + torch.sum(g0) + 1e-5

        loss_fore = 1 - num / (den + 1e-5)

        # background
        loss_back = - torch.sum((1 - p0) * (1 - g0)) / (torch.sum(1 - p0) + torch.sum(1 - g0) + 1e-5)

        loss = loss_fore + loss_back

        if g0.sum() == 0:
            loss = loss * 0
        # else:
        #     loss = loss / weight[i]

        losses.append(loss)

    return losses


def topk_neg(pred, target, k):
    base = 1000

    pred = pred.view(-1)
    target = target.view(-1)

    neg = pred[target == 0]
    pos = pred[target == 1]

    neg_gt = target[target == 0]
    pos_gt = target[target == 1]

    _, indicis = torch.sort(neg, descending=True)
    topk = int(base * k)
    neg = neg[indicis[:topk]]
    neg_gt = neg_gt[indicis[:topk]]

    _, indicis = torch.sort(pos, descending=False)
    pos = pos[indicis[:base]]
    pos_gt = pos_gt[indicis[:base]]

    return torch.cat((pos, neg)), torch.cat((pos_gt, neg_gt))


def dice_loss_bootstrap(pred, target):
    N, C = pred.shape
    pred = pred.sigmoid()
    losses = []
    alpha = 0.5
    beta  = 0.5

    for i in range(C):
        p0 = (pred[:, i]).float()
        p1 = 1 - p0
        g0 = target[:, i]
        g1 = 1 - target[:, i]

        num = torch.sum(p0 * g0)
        den = num + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
        
        dice_loss = 1 - num / (den + 1e-5)
        if g0.sum() == 0:
            dice_loss = dice_loss * 0

        neg_loss = nll_neg_bootstrap_loss(p0, g0, None)

        loss = dice_loss + neg_loss

        losses.append(loss)

    return losses


def nll_neg_bootstrap_loss(p, g, label_weight):
    # bootstrap negative loss
    p = p.sigmoid()
    balance_weight = 0.1
    # g = ((1 - balance_weight) * g + balance_weight * p).detach()
    neg_loss = - torch.mean((1 - g) * torch.log(torch.clamp(1 - p, min=1e-8)))

    return neg_loss


class Unet_2D(nn.Module):
    def __init__(self, cfg, img_ch=5, output_ch=6, t=2, pretrained=True, resnet_type='resnet18'):
        super(Unet_2D, self).__init__()
        self.cfg = cfg
        self.img_ch = img_ch
        self.t = t
        self.pretrained = pretrained
        self.resnet_type = resnet_type
        self.final_activation = cfg['final_activation']
        self.output_ch = output_ch

    def loss(self, pred, target):
        pred = pred['seg_2d']
        target = target['mask']
        
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.output_ch)
        target = target.permute(0, 2, 3, 1).contiguous().view(-1, self.output_ch)
        unet_dice = dice_loss(pred, target)
        # label_weight = torch.ones_like(pred)

        # ghm_loss = 0
        # ghm_losses = []
        # for i in range(self.output_ch):
        #     pred_one = pred[:, i]
        #     target_one = target[:, i]
        #     label_weight = torch.ones_like(pred_one)
        #     l = GHMC()(pred_one, target_one, label_weight)
        #     # l = nll_neg_bootstrap_loss(pred_one, target_one, None)
        #     ghm_loss += l
        #     ghm_losses.append(l)

        return {'unet_dice': unet_dice, 'loss_dice': unet_dice}

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()


class U_Net(Unet_2D):
    def __init__(self, cfg, img_ch=1, output_ch=6, resnet_type=None):
        super().__init__(cfg, img_ch, output_ch)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        num_feats = [64, 128, 256, 512, 1024]
        # num_feats = [num // 2 for num in num_feats]

        if cfg['mask_feature_map'] == 'x':
            self.Conv1 = conv_block(ch_in=self.img_ch + 1, ch_out=num_feats[0], normalization_type=cfg['unet_normalize_type'])
        else:
            self.Conv1 = conv_block(ch_in=self.img_ch, ch_out=num_feats[0], normalization_type=cfg['unet_normalize_type'])

        if cfg['mask_feature_map'] == 'x2':
            self.Conv2 = conv_block(ch_in=num_feats[0] + 1, ch_out=num_feats[1], normalization_type=cfg['unet_normalize_type'])
        else:
            self.Conv2 = conv_block(ch_in=num_feats[0], ch_out=num_feats[1], normalization_type=cfg['unet_normalize_type'])

        if cfg['mask_feature_map'] == 'x3':
            self.Conv3 = conv_block(ch_in=num_feats[1] + 1, ch_out=num_feats[2], normalization_type=cfg['unet_normalize_type'])
        else:
            self.Conv3 = conv_block(ch_in=num_feats[1], ch_out=num_feats[2], normalization_type=cfg['unet_normalize_type'])

        if cfg['mask_feature_map'] == 'x4':
            self.Conv4 = conv_block(ch_in=num_feats[2] + 1, ch_out=num_feats[3], normalization_type=cfg['unet_normalize_type'])
        else:
            self.Conv4 = conv_block(ch_in=num_feats[2], ch_out=num_feats[3], normalization_type=cfg['unet_normalize_type'])

        if cfg['mask_feature_map'] == 'x5':
            self.Conv5 = conv_block(ch_in=num_feats[3] + 1, ch_out=num_feats[4], normalization_type=cfg['unet_normalize_type'])
        else:
            self.Conv5 = conv_block(ch_in=num_feats[3], ch_out=num_feats[4], normalization_type=cfg['unet_normalize_type'])

        self.Up5 = up_conv(ch_in=num_feats[4], ch_out=num_feats[3], normalization_type=cfg['unet_normalize_type'])
        self.Up_conv5 = conv_block(ch_in=num_feats[3] * 2, ch_out=num_feats[3], normalization_type=cfg['unet_normalize_type'])

        self.Up4 = up_conv(ch_in=num_feats[3], ch_out=num_feats[2], normalization_type=cfg['unet_normalize_type'])
        self.Up_conv4 = conv_block(ch_in=num_feats[2] * 2, ch_out=num_feats[2], normalization_type=cfg['unet_normalize_type'])


        # self.Conv_1x1.bias.data.fill_(-4.59)

    def forward(self, x, mask, do_last_conv=True):
        # encoding path
        if self.cfg['mask_feature_map'] == 'x':
            x = torch.cat([x, mask], dim=1)
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        if self.cfg['mask_feature_map'] == 'x2':
            x2 = torch.cat([x2, F.avg_pool2d(mask, 2)], dim=1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        if self.cfg['mask_feature_map'] == 'x3':
            x3 = torch.cat([x3, F.avg_pool2d(mask, 4)], dim=1)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        return {'d4': d4}