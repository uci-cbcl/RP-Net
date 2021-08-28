import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import torch
import os
from torch.utils.data import Dataset
# from config import config
import math
import warnings
from utils.util import annotation2masks, pad2factor, normalize
from utils.util import masks2bboxes_masks
import nrrd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def keep_only_annotation_region(img, mask, margin=20):
    c, d, h, w = mask.shape
    cc, dd, hh, ww = np.where(mask)
    d_max, d_min = dd.max(), dd.min()
    h_max, h_min = hh.max(), hh.min()
    w_max, w_min = ww.max(), ww.min()
    # d_max = min(d_max + 20, d)
    # d_min = max(d_min - 20, 0)
    h_max = min(h_max + margin, h)
    h_min = max(h_min - margin, 0)
    w_max = min(w_max + margin, w)
    w_min = max(w_min - margin, 0)

    if img.ndim == 3:
        return img[d_min:d_max, h_min:h_max, w_min:w_max], mask[:, d_min:d_max, h_min:h_max, w_min:w_max] 
    elif img.ndim == 4:
        return img[:, d_min:d_max, h_min:h_max, w_min:w_max], mask[:, d_min:d_max, h_min:h_max, w_min:w_max]


# Use whole image
class BrainReader(Dataset):
    def __init__(self, data_dir, set_name, config, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.config = config
        self.num_slice = config['num_slice']

        if set_name.endswith('.csv'):
            self.filenames = np.genfromtxt(set_name, dtype=str, delimiter='\n')
        elif set_name.endswith('.npy'):
            self.filenames = np.load(set_name)

        self.crop = Crop(config)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            filename = self.filenames[idx]
            # mask = self.masks[idx].astype(np.float32)
            # m_weight = self.m_weight[idx]
            # mask: [num_class, D, H, W]
            mask = self.load_mask(filename)
            mask = mask.astype(np.float32)

            # imgs: original CT, [D, H, W]
            # Add one more channel dimension, [1, D, H, W]
            imgs, _ = nrrd.read(os.path.join(self.data_dir, '%s_clean.nrrd' % (filename)))
            imgs = self.truncate_image(imgs)
            imgs = imgs[np.newaxis, ...].astype(np.float32)

            imgs, mask = keep_only_annotation_region(imgs, mask)

            # Crop the CT image, according to 
            # 1) the center of the imgs,
            # 2) limit D, H, W, with a maximum size specified by train_max_crop_size
            #
            # TODO: Delete do_scale, do_rotate. The elastic_transform_all has take all these into account
            input, masks, _ = self.crop(imgs, mask, do_jitter=True)

            # Normalize the input
            input = normalize(input, minimum=self.config['HU_range'][0], maximum=self.config['HU_range'][1])

            # In training mode, and if do_elastic, then 50% of the time perform affine and elastic
            # transform to the input image
            if self.mode in ['train'] and self.config['do_elastic'] and np.random.randint(2, size=1).item():
                input, masks = elastic_transform_all(input, masks)

            # # random flip
            # if self.mode in ['train'] and np.random.randint(2, size=1).item():
            #     input = input[:, ::-1, :, :].copy()
            #     masks = masks[:, ::-1, :, :].copy()

            # if self.mode in ['train'] and np.random.randint(2, size=1).item():
            #     input = input[:, :, ::-1, :].copy()
            #     masks = masks[:, :, ::-1, :].copy()

            # if self.mode in ['train'] and np.random.randint(2, size=1).item():
            #     input = input[:, :, :, ::-1].copy()
            #     masks = masks[:, :, :, ::-1].copy()

            # Mask to bounding box, the last column of bboxes is the class
            bboxes, truth_masks = masks2bboxes_masks(masks, border=self.config['bbox_border'])
            truth_masks = np.array(truth_masks).astype(np.uint8)
            bboxes = np.array(bboxes)

            # This should never happen
            if not len(bboxes):
                print(filename, input.shape)

            # class label for each bounding box
            truth_labels = bboxes[:, -1]

            # [z, y, x, d, h, w] for each bounding box
            truth_bboxes = bboxes[:, :-1]

            return [torch.from_numpy(input).float(), truth_bboxes, truth_labels, truth_masks, masks]

        elif self.mode in ['eval']:
            filename = self.filenames[idx]

            # Load OAR masks
            mask = self.load_mask(filename)

            # Load original CT image
            original_img, _ = nrrd.read(os.path.join(self.data_dir, '%s_clean.nrrd' % (filename)))
            original_img = self.truncate_image(original_img)

            original_img, mask = keep_only_annotation_region(original_img, mask)

            imgs = original_img.copy()

            # imgs = imgs[np.newaxis, ...].astype(np.float32)
            # imgs, mask = self.crop(imgs, mask, do_jitter=False)
            # original_img = imgs.copy()[0]

            # pad the CT image, so that it can fit the downsampling
            # imgs = pad2factor(imgs)
            # imgs = imgs[np.newaxis, ...].astype(np.float32)
            mask = self.load_mask(filename)
            imgs, _ = nrrd.read(os.path.join(self.data_dir, '%s_clean.nrrd' % (filename)))
            imgs = self.truncate_image(imgs)
            imgs = imgs[np.newaxis, ...].astype(np.float32)

            imgs, mask = keep_only_annotation_region(imgs, mask)
            imgs, mask, shifts = self.crop(imgs, mask, do_jitter=True)
            original_img = imgs[0].copy()


            input = normalize(imgs, minimum=self.config['HU_range'][0], maximum=self.config['HU_range'][1])
            # original_img = (normalize(original_img, minimum=self.config['HU_range'][0], maximum=self.config['HU_range'][1]) + 1) / 2

            # Mask to bounding box, the last column of bboxes is the class
            bboxes, truth_masks = masks2bboxes_masks(mask, border=self.config['bbox_border'])
            truth_masks = np.array(truth_masks).astype(np.uint8)
            bboxes = np.array(bboxes)
            truth_labels = bboxes[:, -1]
            truth_bboxes = bboxes[:, :-1]

            if self.config.get('pseudo_gt', None) is not None:
                pseudo_gt_mask = np.load(os.path.join(self.config['pseudo_gt'], filename+'.npy'))
                pseudo_gt_mask = pseudo_gt_mask.astype(mask.dtype)
                return [torch.from_numpy(input).float(), truth_bboxes, truth_labels, truth_masks, [mask, pseudo_gt_mask], original_img, shifts]

            return [torch.from_numpy(input).float(), truth_bboxes, truth_labels, truth_masks, mask, original_img, shifts]
        elif self.mode in ['test']:
            filename = self.filenames[idx]
            original_img = np.load(os.path.join(self.data_dir, '%s_clean.npy' % (filename)))

            imgs = original_img.copy()
            imgs = imgs[np.newaxis, ...].astype(np.float32)
            imgs = pad2factor(imgs)

            input = normalize(imgs, minimum=self.config['HU_range'][0], maximum=self.config['HU_range'][1])

            return [torch.from_numpy(input).float(), original_img]

    def __len__(self):
        return len(self.filenames)

    def load_mask(self, filename):
        mask = {}
        for j, roi in enumerate(self.config['roi_names']):
            if os.path.isfile(os.path.join(self.data_dir, '%s_%s.nrrd' % (filename, roi))):
                m, _ = nrrd.read(os.path.join(self.data_dir, '%s_%s.nrrd' % (filename, roi)))
                if self.mode in ['train', 'val', 'eval']:
                    m = self.truncate_image(m)

                mask[roi] = m
        
        mask = annotation2masks(mask, roi_names=self.config['roi_names'])

        return mask

    def truncate_image(self, image):
        # truncate the input image and mask, so it runs faster
        config = self.config
        D, H, W = image.shape
        num_slice = config['num_slice']
        num_x = config['num_x']
        num_y = config['num_y']

        x1 = max(0, W // 2 - num_x // 2)
        x2 = min(W, W // 2 + num_x // 2)
        y1 = max(0, H // 2 - num_y // 2)
        y2 = min(H, H // 2 + num_y // 2)

        return image[:num_slice, y1:y2, x1:x2]


def elastic_transform_all(image,  mask, alpha=1000, sigma=30, alpha_affine=0.04, padding_value=-1., random_state=None):
    """
    Perform affine and elastic transform to an input image and its corresponding mask.

    If this function is called, then 100% perform xy plane affine and elastic transform.
    50% of the time perform zx and zy plane affine and elastic transform respectively.

    Actually, zx and zy plane affine elastic might be too aggressive, but it helps in terms of robustness and performance.
    TODO: Check the influence for detection branch and mask branch respectively. My conjecture is the detection would benefit more.
    If this is the case, then perhapes, if zx and zy plane is transformed, we only train the detection part but not mask part.
    """
    # transform xy plane:
    # if np.random.randint(2, size=1).item():
    image, mask = elastic_transform(image, mask, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, padding_value=padding_value, random_state=None)

    # # transform zx plane:
    # if np.random.randint(2, size=1).item():
    #     # [z, y, x]
    #     image = np.swapaxes(image, 1, 2)
    #     # [num_class, z, y, x]
    #     mask = np.swapaxes(mask, 1, 2)

    #     image, mask = elastic_transform(image, mask, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, padding_value=padding_value, random_state=None)
    #     image = np.swapaxes(image, 1, 2)
    #     mask = np.swapaxes(mask, 1, 2)

    # # transform zy plane:
    # if np.random.randint(2, size=1).item():
    #     # [z, y, x]
    #     image = np.swapaxes(image, 1, 3)
    #     # [num_class, z, y, x]
    #     mask = np.swapaxes(mask, 1, 3)

    #     image, mask = elastic_transform(image, mask, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, padding_value=padding_value, random_state=None)
    #     image = np.swapaxes(image, 1, 3)
    #     mask = np.swapaxes(mask, 1, 3)

    return image, mask


def elastic_transform(image, mask, alpha=1000, sigma=30, alpha_affine=0.04, padding_value=-1., random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    # c, z, y, x
    shape = image.shape 
    shape_size = shape[2:]
    num_class, z, y, x = mask.shape

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    # Random elastic
    dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(x), np.arange(y))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    

    new_img = np.zeros_like(image)
    new_mask = np.zeros_like(mask)

    for i in range(z):
        # Affine transform
        new_img[0, i, :, :] = cv2.warpAffine(image[0, i, :, :], M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=padding_value)

        # Elastic transform
        new_img[0, i, :, :] = map_coordinates(new_img[0, i, :, :], indices, order=1, mode='constant', cval=padding_value).reshape(shape_size)

        for j in range(num_class):
            if np.any(mask[j, i, :, :]):
                new_mask[j, i, :, :] = cv2.warpAffine(mask[j, i, :, :], M, shape_size[::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT, borderValue=0)
                new_mask[j, i, :, :] = map_coordinates(new_mask[j, i, :, :], indices, order=0, mode='constant').reshape(shape_size)

    return new_img, new_mask


class Crop(object):
    """
    Crop the input image and corresponding masks
    """
    def __init__(self, config):
        self.max_crop_size = config['train_max_crop_size']
        self.pad_value = config['pad_value']
        self.jitter = config['jitter_range']
        self.stride = config['stride']

        # order is used for interpolation
        self.order = 1
        self.config = config

    def __call__(self, imgs, mask, mode='train', do_jitter=True):
        order = self.order

        # Maximum size
        max_crop_size = self.max_crop_size

        # Crop size according to the image size
        img_crop_size = [int(math.ceil(d / 16.) * 16) for d in imgs.shape[1:]]

        # Limit the largest crop size
        crop_size = [min(max_crop_size[i], img_crop_size[i]) for i in range(len(img_crop_size))]
        imgs = np.copy(imgs)
        mask = np.copy(mask).astype(np.float32)

        # The center of the imgs
        target = np.array(imgs.shape[1:]) / 2 - np.array(crop_size) / 2

        start = []
        shifts = []
        for i in range(3):
            if do_jitter:
                shift = np.random.random_integers(-self.jitter[i], self.jitter[i])
                s = target[i] + shift
                shifts.append(shift)
            else:
                s = target[i]
            s = min(s, imgs.shape[i + 1] - 1)
            start.append(int(s))

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
            max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
            max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
            max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        mask = mask[:,
            max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
            max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
            max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]

        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        mask = np.pad(mask, pad, 'constant', constant_values=0)
        
        return crop, mask, shifts


def train_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    truth_masks = [batch[b][3] for b in range(batch_size)]
    masks = [batch[b][4] for b in range(batch_size)]

    return [inputs, bboxes, labels, truth_masks, masks]


def eval_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    images = [batch[b][3] for b in range(batch_size)]

    return [inputs, bboxes, labels, images]


def test_collate(batch):
    batch_size = len(batch)
    for b in range(batch_size): 
        inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
        images = [batch[b][1] for b in range(batch_size)]

    return [inputs, images]
