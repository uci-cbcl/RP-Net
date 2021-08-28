"""
RP-Net for Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
from .modules import *
from .unet import U_Net
import numpy as np
import torchvision
from torchvision.models.resnet import BasicBlock


class ResNet18(nn.Module):
    def __init__(self, use_pretrained=False):
        super().__init__()
        resnet_net = torchvision.models.resnet18(pretrained=use_pretrained)
        modules = list(resnet_net.children())[:-5]
        modules.append(nn.Sequential(
            BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128))),
            BasicBlock(128, 128)
        ))
        modules.append(nn.Sequential(
            BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256))),
            BasicBlock(256, 256)
        ))
        modules.append(nn.Sequential(
            BasicBlock(256, 512, downsample=nn.Sequential(nn.Conv2d(256, 512, 1), nn.BatchNorm2d(512))),
            BasicBlock(512, 512)
        ))
        self.backbone = nn.Sequential(*modules)
        self.backbone.out_channels = 512

    def forward(self, x, mask):
        output = self.backbone(x)

        return {'d4': output}


class ContextCorrelationEncoder(nn.Module):
    def __init__(self, cfg, in_channels=3, radius=5):
        super().__init__()
        self.radius = cfg['mask_refinement_correlation_radius']
        num_feat = 64
        self.w_k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.w_q = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.w_context = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.q = nn.Sequential(
            nn.Conv2d(in_channels + (self.radius * 2 + 1) ** 2, num_feat, 1),
            nn.BatchNorm2d(num_feat),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(2 * in_channels, num_feat, 1),
            nn.BatchNorm2d(num_feat),
            nn.ReLU(inplace=True)
        )


    def forward(self, fm1, fm2):
        fm1 = self.w_k(fm1)
        fm2 = self.w_q(fm2)
        corr = Correlation(fm1, fm2, r=self.radius)
        corr = self.q(torch.cat([corr, fm1], dim=1))
        # corr = self.q(torch.cat([fm1, fm2], dim=1))

        return corr


def dice_loss_softmax(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    true = true.unsqueeze(1)
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def dice_ce(logits, true, eps=1e-7):
    dice_loss = dice_loss_softmax(logits, true, eps)
    ce_loss = nn.CrossEntropyLoss()(logits, true)

    return dice_loss + ce_loss


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def Correlation(fmap1, fmap2, r=3):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht*wd)
    fmap2 = fmap2.view(batch, dim, ht*wd) 
    
    corr = torch.matmul(fmap1.transpose(1,2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr  / torch.sqrt(torch.tensor(dim).float())
    corr = corr.view(-1, 1, ht, wd)
    # corr = F.adaptive_avg_pool2d(corr, (64, 64))
    # corr = corr.view(batch, ht, wd, -1)
    # corr = corr.permute(0, 3, 1, 2).contiguous()

    coords = coords_grid(batch, ht, wd).to(fmap1.device)
    coords = coords.permute(0, 2, 3, 1)
    batch, h1, w1, _ = coords.shape
    dx = torch.linspace(-r, r, 2*r+1)
    dy = torch.linspace(-r, r, 2*r+1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

    centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2)
    delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
    coords_lvl = centroid_lvl + delta_lvl

    corr = bilinear_sampler(corr, coords_lvl)
    corr = corr.view(batch, h1, w1, -1)
    out = corr.permute(0, 3, 1, 2).contiguous().float()

    return out


class RP_Net(nn.Module):
    """
    Fewshot Segmentation model
    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, backbone_cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.backbone_cfg = backbone_cfg
        self.scale = backbone_cfg.get('scale', 4)
        self.num_iter = backbone_cfg['n_iter_refinement']
        self.use_relation_enc = backbone_cfg.get('use_relation_enc', 'relation')

        # Encoder
        if self.config['backbone'] == 'vgg':
            self.encoder = Encoder(in_channels, self.pretrained_path)
            num_feat = 512

        elif self.config['backbone'] == 'UNet':
            self.encoder = U_Net(backbone_cfg)
            num_feat = 256
            if pretrained_path:
                dic = torch.load(self.pretrained_path, map_location='cpu')['state_dict']
                self.load_state_dict(dic)
        elif self.config['backbone'] == 'resnet':
            num_feat = 512
            self.encoder = ResNet18(False)
        else:
            raise NotImplementedError

        self.cre = ContextCorrelationEncoder(backbone_cfg, in_channels=num_feat)

        if self.use_relation_enc == 'concat':
            self.sim_cat = SimpleConcat(backbone_cfg, in_channels=num_feat)

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, registration_field=None, grid=None, query_labels=None, appr_query_labels=None):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = qry_imgs[0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs], dim=0)
        if self.config['backbone'] in ['vgg', 'resnet']:
            imgs_concat = imgs_concat.expand(-1, 3, -1, -1)
        supp_pyramid = self.encoder(imgs_concat, fore_mask[0][0].unsqueeze(1))
        img_fts = supp_pyramid['d4']

        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts.view(n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'

        imgs_concat = torch.cat([torch.cat(qry_imgs, dim=0),], dim=0)
        if self.config['backbone'] in ['vgg', 'resnet']:
            imgs_concat = imgs_concat.expand(-1, 3, -1, -1)
        qry_pyramid = self.encoder(imgs_concat, fore_mask[0][0].unsqueeze(1))
        img_fts = qry_pyramid['d4']


        fts_size = img_fts.shape[-2:]
        qry_fts = img_fts.view(n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W        

        qry_mask = appr_query_labels.unsqueeze(1)
        qry_mask = F.avg_pool2d(qry_mask, self.scale)
        supp_mask = fore_mask[0][0].unsqueeze(1)
        supp_mask = F.avg_pool2d(supp_mask, self.scale)

        if self.use_relation_enc == 'relation':
            supp_fts = self.cre(supp_fts[0][0] * supp_mask, supp_fts[0][0] * (1 - supp_mask))[None, None, ...]
        elif self.use_relation_enc == 'concat':
            supp_fts = self.sim_cat(supp_fts[0][0], supp_mask)[None, None, ...]
        inter_qry_fts = qry_fts

        refinement = {}
        for i in range(self.num_iter):
            if self.use_relation_enc == 'relation': 
                inter_qry_fts = self.cre(qry_fts[0] * qry_mask, qry_fts[0] * (1 - qry_mask))[None, ...]
            elif self.use_relation_enc == 'concat':
                inter_qry_fts = self.sim_cat(qry_fts[0], qry_mask)[None, ...]
            outputs = []
            for epi in range(batch_size):
                supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                                fore_mask[way, shot, [epi]])
                                for shot in range(n_shots)] for way in range(n_ways)]
                supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                                back_mask[way, shot, [epi]])
                                for shot in range(n_shots)] for way in range(n_ways)]


                ###### Obtain the prototypes######
                fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

                ###### Compute the distance ######
                prototypes = [bg_prototype,] + fg_prototypes
                dist = [self.calDist(inter_qry_fts[:, epi], prototype) for prototype in prototypes]
                pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
                outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            outputs = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
            outputs = outputs.view(-1, *outputs.shape[2:])
            output_logits = outputs
            outputs = outputs.softmax(dim=1)[:, 1, ...]
            if self.backbone_cfg['soft_mask'] == False:
                outputs = (outputs > 0.5).float()
            qry_mask = F.avg_pool2d(outputs.unsqueeze(1), self.scale)
            refinement[i] = output_logits
        
        qry_fts = inter_qry_fts

        # ###### Compute loss ######
        align_loss = 0
        outputs = []

        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

         
            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        outputs = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        outputs = outputs.view(-1, *outputs.shape[2:])

        return {
            'output': outputs, 'align_loss': align_loss / batch_size, 'refinement': refinement,
        }


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype
        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch
        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss