from .brain_reader import BrainReader, elastic_transform_all
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import torch.nn.functional as F
import pandas as pd
import nrrd
import random
import os
from utils.util import normalize, pad2factor
from net.registration import DemonsRegistration, GaussianRegulariser, compute_grid, AffineDemonsRegistration
import nibabel as nib
import torchvision.transforms as transforms


def keep_only_annotation_z_slices(img, mask):
    c, d, h, w = mask.shape
    cc, dd, hh, ww = np.where(mask)
    d_max, d_min = dd.max(), dd.min()
    h_max, h_min = hh.max(), hh.min()
    w_max, w_min = ww.max(), ww.min()

    return img[:, d_min:d_max, :, :], mask[:, d_min:d_max, :, :]


def random_transform(images, labels):
    geo_trans = transforms.Compose([
        transforms.RandomAffine(5, translate=(0.2,0.2), scale=(0.7,1.5), shear=0, fillcolor=None),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(200, pad_if_needed=True)
    ])
    # int_trans = transforms.Compose([
    #     RandomIntensity(0.5, 2)
    # ])
    images = (images + 1) / 2
    # images = int_trans(images)
    image_min = images.min()
    concated = torch.cat([images, labels[None, ...]], dim=1)
    concated = geo_trans(concated)
    images, labels = concated[:,[0],...], concated[:,1,...]
    # torchvision does not support custom padding value for Tensor input
    images[images == 0] = image_min
    images = images * 2 - 1

    return images, labels


def random_label_transform(labels):
    geo_trans = transforms.Compose([
        transforms.RandomAffine(5, translate=(0.02,0.02), scale=(0.5,1.5), shear=5, fillcolor=None),
    ])

    concated = labels[None, None, ...]
    concated = geo_trans(concated)
    labels = concated[:,0,...]
    # torchvision does not support custom padding value for Tensor input

    return labels


def crop(img, mask, crop_size, img_pad_value, mask_pad_value=0):
    c, d, h, w = mask.shape
    ch, cw = crop_size
    rh, rw = min(ch, h), min(cw, w)
    cx, cy = w//2, h//2
    img_crop = img[..., cy-rh//2:cy+rh-rh//2, cx-rw//2:cx+rw-rw//2]
    mask_crop = mask[..., cy-rh//2:cy+rh-rh//2, cx-rw//2:cx+rw-rw//2]
    pad_width = [(0,0), (0,0), 
        ((ch-rh)//2, (ch-rh)-(ch-rh)//2), 
        ((cw-rw)//2, (cw-rw)-(cw-rw)//2)]
    img_pad = np.pad(img_crop, pad_width, mode='constant', constant_values=img_pad_value)
    mask_pad = np.pad(mask_crop, pad_width, mode='constant', constant_values=mask_pad_value)
    return img_pad, mask_pad


def make_support_query_same_size(support_images, support_labels, query_images, query_labels):
    """
    pad the 3D support and query volume to the same size. Only support 1way 1shot and batch size 1 currently

    support_images: n_way * n_shot * batch * C * H * W
    support_labels: n_way * n_shot * batch * C * H * W
    query_images: batch * H * W
    query_labels: batch * H * W
    """
    support_images = support_images[0][0].numpy()
    support_labels = support_labels[0][0].numpy()
    query_images = query_images.numpy()
    query_labels = query_labels.numpy()

    H = max(support_images.shape[2], query_images.shape[2])
    W = max(support_images.shape[3], query_images.shape[3])

    pad = [[0, 0], [0, 0], [0, H - support_images.shape[2]], [0, W - support_images.shape[3]]]
    support_images = np.pad(support_images, pad, 'constant', constant_values=support_images.min())
    pad = [[0, 0], [0, 0], [0, H - query_images.shape[2]], [0, W - query_images.shape[3]]]
    query_images = np.pad(query_images, pad, 'constant', constant_values=query_images.min())

    pad = [[0, 0], [0, H - support_labels.shape[1]], [0, W - support_labels.shape[1]]]
    support_labels = np.pad(support_labels, pad, 'constant', constant_values=support_labels.min())
    pad = [[0, 0], [0, H - query_labels.shape[1]], [0, W - query_labels.shape[1]]]
    query_labels = np.pad(query_labels, pad, 'constant', constant_values=query_labels.min())


    return [[torch.from_numpy(support_images)]], [[torch.from_numpy(support_labels)]], torch.from_numpy(query_images), torch.from_numpy(query_labels)


def get_registration_field(query_images, support_images, support_labels, do_deformable=True):
    support_images_npy = support_images[0][0][:, 0, :, :].numpy()
    support_images_npy = (support_images_npy + 1) / 2.
    support_labels_npy = support_labels[0][0].numpy()

    query_images_npy = query_images[:, 0, :, :].numpy()
    query_images_npy = (query_images_npy + 1) / 2.

    py_reg_pred = []
    registration_field = []
    warped_src_list = []
    py_affine_reg_pred = []
    affine_warped_src_list = []
    for slice_id in range(len(query_images_npy)):
        src = support_images_npy[slice_id]
        dst = query_images_npy[slice_id]
        src_label = support_labels_npy[slice_id]
        
        target_H, target_W = dst.shape
        size = src.shape
        
        src = torch.from_numpy(src).unsqueeze(0).unsqueeze(0)
        dst = torch.from_numpy(dst).unsqueeze(0).unsqueeze(0)
        src_label = torch.from_numpy(src_label).unsqueeze(0).unsqueeze(0)

        registration = AffineDemonsRegistration(size, use_diffeomorphic=True, use_GPU=True, stop_shear=False)

        num_iter_deformable = 0
        if do_deformable:
            num_iter_deformable = 50
            device = "cuda:0"
            src = src.cuda()
            dst = dst.cuda()
            src_label = src_label.cuda()
        else:
            device = "cpu"

        if device != "cpu":
            registration = registration.cuda()
        optimizer_affine = torch.optim.Adam(registration.affine_reg.parameters(), lr=0.01)
        optimizer_demons = torch.optim.Adam(registration.demons.parameters(), lr=0.01)
        regulariser = GaussianRegulariser([1, 1], sigma=[2, 2], dtype=torch.float32, device=device)

        registration.train_registraion(
            # ((normalize(src, 0.20707, 0.3168) + 1) / 2), 
            # ((normalize(dst, 0.20707, 0.3168) + 1) / 2), 
            src, 
            dst, 
            [optimizer_affine, optimizer_demons], 
            regulariser=regulariser, 
            iters=[50, num_iter_deformable], 
            regularise_displacement=False,
            verbose=False
        )                    
        grid = compute_grid(size)
        if device != "cpu":
            grid = grid.cuda()

        warped_label = registration(src_label, grid).cpu()
        warped_label = warped_label[0][0].data.numpy()
        warped_label = (warped_label > 0.1).astype(np.float32)

        affine_warped_label = registration.affine_reg(src_label).cpu()
        affine_warped_label = affine_warped_label[0][0].data.numpy()
        affine_warped_label = (affine_warped_label > 0.1).astype(np.float32)

        warped_src = registration(src, grid)
        warped_src = warped_src[0][0].cpu().data.numpy()

        affine_warped_src = registration.affine_reg(src)
        affine_warped_src = affine_warped_src[0][0].cpu().data.numpy()
        
        warped_src_list.append(warped_src[None, ...])
        py_reg_pred.append(warped_label[None, ...])
        affine_warped_src_list.append(affine_warped_src[None, ...])
        py_affine_reg_pred.append(affine_warped_label[None, ...])
        registration_field.append([registration, grid])

    py_reg_pred = np.concatenate(py_reg_pred, axis=0)[:, None, ...]
    py_reg_pred = torch.from_numpy(py_reg_pred)
    warped_src_list = np.concatenate(warped_src_list, axis=0)
    warped_src_list = warped_src_list * 2 - 1

    py_affine_reg_pred = np.concatenate(py_affine_reg_pred, axis=0)[:, None, ...]
    py_affine_reg_pred = torch.from_numpy(py_affine_reg_pred)
    affine_warped_src_list = np.concatenate(affine_warped_src_list, axis=0)
    affine_warped_src_list = affine_warped_src_list * 2 - 1
        
        
    return registration_field, py_reg_pred, warped_src_list, py_affine_reg_pred, affine_warped_src_list


def gamma_tansform(img, gamma_range):
    img = (img + 1) / 2.
    gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
    cmin = img.min()
    irange = (img.max() - cmin + 1e-5)

    img = img - cmin + 1e-5
    img = irange * np.power(img * 1.0 / irange,  gamma)
    img = img + cmin

    return img * 2 -1


def gamma_tansform_with_label(img, label, gamma_range):
    old_img = img
    img = (img + 1) / 2.
    gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
    cmin = img.min()
    irange = (img.max() - cmin + 1e-5)

    img = img - cmin + 1e-5
    img = irange * np.power(img * 1.0 / irange,  gamma)
    img = img + cmin
    img = img * 2 -1

    # only augment label region
    img = old_img * (1 - label) + img * (label)

    return img


class FewshotVolumeReader(Dataset):
    def __init__(self, data_dir, set_name, config, mode='train'):
        self.data_dir = data_dir
        self.cfg = config
        self.mode = mode
        self.class_csv_dir = config['class_csv_dir']

        if set_name.endswith('.csv'):
            self.filenames = np.genfromtxt(set_name, dtype=str, delimiter='\n')
        elif set_name.endswith('.npy'):
            self.filenames = np.load(set_name)

        if mode in ['train']:
            self.classes = config['train_classes']
        elif mode in ['eval']:
            self.classes = config['eval_classes']
        else:
            raise NotImplementedError
        
        self.read_data_meta()
        self.init_pairs()


    def __getitem__(self, idx, supp_idx=None):
        n_shots = self.cfg['n_shot']
        n_ways = self.cfg['n_way']
        n_elements = n_shots + 1

        qry_class_idx, qry_data_idx = self.indices[idx]
        pid = self.data_info[qry_class_idx][qry_data_idx]['pid']
        n_data_in_class = self.n_data[qry_class_idx]


        support_indicis = list(range(qry_data_idx)) + list(range(qry_data_idx + 1, n_data_in_class))

        if self.mode == 'eval':
            support_indicis = [i  for i in support_indicis]

        rand_support_indicis = random.choices(
            support_indicis, 
            k=n_elements - 1
        )
        support_data_idx = []
        for ind in rand_support_indicis:
            support_data_idx.append((qry_class_idx, ind))

        # if self.mode == 'eval':
        #     support_data_idx = [(qry_class_idx, 6)]

        if supp_idx is not None:
            support_data_idx = []
            support_data_idx.append((qry_class_idx, supp_idx))

        samples = [
            self.load_image_and_mask(
                self.data_info[class_idx][data_idx]['pid'], 
                self.classes[class_idx]
            ) 
            for class_idx, data_idx in support_data_idx
        ]
        support_images = [[
            torch.from_numpy(samples[j]['image']) for j in range(n_shots)]
            for i in range(n_ways)
        ]
        support_labels = [[
            torch.from_numpy(samples[j]['mask']) for j in range(n_shots)]
            for i in range(n_ways)
        ]

        qry_sample = self.load_image_and_mask(
            self.data_info[qry_class_idx][qry_data_idx]['pid'], 
            self.classes[qry_class_idx]
        ) 
        qry_img, qry_mask = qry_sample['image'], qry_sample['mask']
        if self.mode in ['train'] and self.cfg['do_elastic'] and np.random.randint(2, size=1).item():
            qry_img, qry_mask = elastic_transform_all(qry_img, qry_mask)

        query_images = [[torch.from_numpy(qry_img)]]
        query_labels = [[torch.from_numpy(qry_mask)]]


        return {
            'support_images': support_images, 
            'support_labels': support_labels,
            'query_images': query_images, 
            'query_labels': query_labels,
            'class_id': qry_class_idx,
            'pid': pid,
            'supp_pids': support_data_idx,
        }


    def load_image_and_mask(self, filename, roi_name):
        pad_factor = 16
        m, _ = nrrd.read(os.path.join(self.data_dir, '%s_%s.nrrd' % (filename, roi_name)))
        mask = m.astype(np.float32)
        mask = self.truncate_image(mask)
        mask = pad2factor(mask, factor=pad_factor, pad_value=0)
        mask = mask[None, ...]

        # imgs: original CT, [D, H, W]
        # Add one more channel dimension, [1, D, H, W]
        imgs, _ = nrrd.read(os.path.join(self.data_dir, '%s_clean.nrrd' % (filename)))
        imgs = self.truncate_image(imgs)
        imgs = pad2factor(imgs, factor=pad_factor, pad_value=self.cfg['pad_value'])
        imgs = imgs[np.newaxis, ...].astype(np.float32)

        imgs, mask = keep_only_annotation_z_slices(imgs, mask)

        imgs, mask = crop(imgs, mask, self.cfg.get('crop_size', [256, 256]), self.cfg.get('pad_value', -1024), 0)

        imgs = normalize(imgs, minimum=self.cfg['HU_range'][0], maximum=self.cfg['HU_range'][1])

        return {'image': imgs, 'mask': mask}


    def __len__(self):
        return len(self.indices)

    
    def read_data_meta(self):
        self.data_info = []
        self.n_data = []
        filenames = set(self.filenames)

        for roi_name in self.classes:
            df = pd.read_csv(os.path.join(self.class_csv_dir, f'{roi_name}.csv'), dtype=str)
            l = []
            for i, row in df.iterrows():
                if row['pid'] in filenames:
                    l.append({
                        'pid': row['pid'],
                        'z_start': row['z_start'],
                        'z_end': row['z_end'],
                    })
            
            self.data_info.append(l)
            self.n_data.append(len(l))

        print(self.data_info)

    def init_pairs(self):
        cfg = self.cfg
        n_classes = len(self.classes)

        self.indices = []
        for class_idx in range(n_classes):
            for data_idx in range(self.n_data[class_idx]):
                self.indices.append((class_idx, data_idx) )

        return 


    def truncate_image(self, image):
        # truncate the input image and mask, so it runs faster
        config = self.cfg
        D, H, W = image.shape
        num_slice = config['num_slice']
        num_x = config['num_x']
        num_y = config['num_y']

        x1 = max(0, W // 2 - num_x // 2)
        x2 = min(W, W // 2 + num_x // 2)
        y1 = max(0, H // 2 - num_y // 2)
        y2 = min(H, H // 2 + num_y // 2)

        return image[:num_slice, y1:y2, x1:x2]


class Fewshot3DReader(Dataset):
    def __init__(self, data_dir, set_name, config, mode='train'):
        self.cfg = config
        self.k = config['k']
        self.mode = mode
        self.fewshot_volume_reader = FewshotVolumeReader(
            data_dir, 
            set_name, 
            config, 
            mode=mode
        )


    def __getitem__(self, idx):
        samples = self.fewshot_volume_reader[idx]
        # TODO
        # process into k blocks and then random match
        support_images = samples['support_images']
        support_labels = samples['support_labels']
        query_images = samples['query_images']
        query_labels = samples['query_labels']

        if self.cfg.get('use_registration_loss', False):
            registration_field, reg_pred = get_registration_field(query_images, support_images, support_labels)
            if self.cfg.get('use_registration_mask', False):
                support_images[0][0] = torch.cat((support_images[0][0], support_labels[0][0][:, None, ...]), dim=1)
                query_images = torch.cat((query_images, reg_pred), dim=1)

        return {
            'support_images': support_images, 
            'support_labels': support_labels,
            'query_images': query_images, 
            'query_labels': query_labels,
            'class_id': samples['class_id'],
            'registration_field': registration_field,
        }



class FewshotSliceReader(Dataset):
    def __init__(self, data_dir, set_name, config, mode='train'):
        self.cfg = config
        self.k = config['k']
        self.mode = mode
        self.fewshot_volume_reader = FewshotVolumeReader(
            data_dir, 
            set_name, 
            config, 
            mode=mode
        )


    def __getitem__(self, idx):
        samples = self.fewshot_volume_reader[idx]
        # TODO
        # process into k blocks and then random match
        support_images = samples['support_images']
        support_labels = samples['support_labels']
        query_images = samples['query_images']
        query_labels = samples['query_labels']

        assert len(support_images) == 1

        num_support = len(support_images[0])
        num_slices = [img.shape[1] for img in support_images[0]] + [img.shape[1] for img in query_images[0]]

        self.k = min([self.k] + num_slices)
        support_slice_indicis = [
            np.floor(np.arange(n / self.k / 2, n, n / self.k)).astype(np.int32)
            for n in num_slices[:-1]
        ]
        query_slice_indicis = np.arange(0, num_slices[-1], num_slices[-1] / self.k).tolist() + [num_slices[-1]]
        query_slice_indicis = np.floor(np.array(query_slice_indicis)).astype(np.int32)

        new_support_images = []
        new_support_labels = []
        new_query_images = []
        new_query_labels = []
        warped_src = np.zeros(1)
        reg_pred = None

        if self.mode in ['train']:
            new_support_images = [[
                img[:, support_slice_indicis[i], :, :].permute(1, 0, 2, 3).contiguous().expand(-1, 3, -1, -1).clone()
                for i, img in enumerate(support_images[0])
            ]]
            new_support_labels = [[
                mask[0, support_slice_indicis[i], :, :].clone()
                for i, mask in enumerate(support_labels[0])
            ]]
            for i in range(self.k):
                s, e = query_slice_indicis[i], query_slice_indicis[i + 1]
                ind = random.randint(s, e - 1)
                q = query_images[0][0][:, ind, :, :].clone()
                l = query_labels[0][0][:, ind, :, :].clone()

                if self.mode in ['train'] and self.cfg['do_intaug'] and np.random.randint(2, size=1).item():
                    q = torch.from_numpy(gamma_tansform(q.numpy(), self.cfg.get('gamma_range', [0.5, 1.5])))
                    # q = torch.from_numpy(gamma_tansform_with_label(q.numpy(), l.numpy(), self.cfg.get('gamma_range', [0.5, 1.5])))

                q, l = random_transform(q[None, ...], l)
                q = q[0]

                new_query_images.append(q)
                new_query_labels.append(l)

            new_query_images = torch.cat(new_query_images, dim=0).unsqueeze(1).expand(-1, 3, -1, -1)
            new_query_labels = torch.cat(new_query_labels, dim=0)

            shuffle = np.arange(self.k)
            np.random.shuffle(shuffle)
            new_query_images = new_query_images[shuffle, ...]
            new_query_labels = new_query_labels[shuffle, ...]
            new_support_images = [[new_support_images[0][0][shuffle, ...]]]
            new_support_labels = [[new_support_labels[0][0][shuffle, ...]]]
        elif self.mode in ['eval']:
            test_shot = self.cfg.get('test_shot', self.cfg['n_shot'])
            new_query_images = query_images[0][0].permute(1, 0, 2, 3).contiguous().expand(-1, 3, -1, -1)
            new_query_labels = query_labels[0][0][0]

            for i in range(num_support):
                n_shot_images = []
                n_shot_labels = []
                for m in range(test_shot):
                    new_image = []
                    new_label = []
                    for j in range(self.k):
                        s, e = query_slice_indicis[j], query_slice_indicis[j + 1]
                        if j + m >= self.k:
                            offset = 0
                        else:
                            offset = m
                        new_image.append(support_images[0][i][:, [support_slice_indicis[i][j + offset]], :, :].expand(e - s, 3, -1, -1))
                        new_label.append(support_labels[0][i][0, [support_slice_indicis[i][j + offset]], :, :].expand(e - s, -1, -1))

                    new_image = torch.cat(new_image, dim=0)
                    new_label = torch.cat(new_label, dim=0)
                    n_shot_images.append(new_image.unsqueeze(0))
                    n_shot_labels.append(new_label.unsqueeze(0))

                n_shot_images = torch.cat(n_shot_images, dim=0)
                n_shot_labels = torch.cat(n_shot_labels, dim=0)

            new_support_images = [n_shot_images]
            new_support_labels = [n_shot_labels]


        new_support_images, new_support_labels, new_query_images, new_query_labels = make_support_query_same_size(
            new_support_images, 
            new_support_labels, 
            new_query_images, 
            new_query_labels
        )


        if self.cfg.get('use_registration_loss', False):
            registration_field, reg_pred, warped_src, affine_reg_pred, affine_warped_src = get_registration_field(new_query_images, new_support_images, new_support_labels, do_deformable=self.cfg.get('do_deformable', True))
            if self.cfg.get('use_registration_mask', False):
                new_support_images[0][0] = torch.cat((new_support_images[0][0], new_support_labels[0][0][:, None, ...]), dim=1)
                new_query_images = torch.cat((new_query_images, reg_pred), dim=1)
        else:
            registration_field = None
            reg_pred = None
            warped_src = new_support_images[0][0].numpy()
            affine_reg_pred = None
            affine_warped_src = new_support_images[0][0].numpy()

        return {
            'support_images': new_support_images, 
            'support_labels': new_support_labels,
            'query_images': new_query_images, 
            'query_labels': new_query_labels,
            'class_id': samples['class_id'],
            'registration_field': registration_field,
            'support_images_3D': samples['support_images'],
            'support_labels_3D': samples['support_labels'],
            'query_images_3D': samples['query_images'],
            'query_labels_3D': samples['query_labels'],
            'warped_supp': torch.from_numpy(warped_src),
            'warped_supp_label': reg_pred,
            'affine_warped_supp': torch.from_numpy(affine_warped_src),
            'affine_warped_supp_label': affine_reg_pred,
            'pid': samples['pid'],
            'supp_pids': samples['supp_pids'],
        }


    def __len__(self):
        return len(self.fewshot_volume_reader)


class FewshotRegReader(Dataset):
    def __init__(self, data_dir, set_name, config, mode='train'):
        self.config = config
        self.fewshot_reader = FewshotSliceReader(data_dir, set_name, config, mode=mode)
        self.mode = mode


    def __getitem__(self, idx):
        data = self.fewshot_reader[idx]
        registration_field= data['registration_field']
        grids = [reg[1] for reg in registration_field]
        grids = torch.cat(grids, dim=0)

        support_images = [[data['affine_warped_supp'].unsqueeze(1)]]
        support_labels = [[data['affine_warped_supp_label'][:, 0, ...]]]

        appr_query_labels = (data['warped_supp_label'][:, 0, ...] > 0.5).float()

        # if self.mode in ['train']:
        #     new_appr_qry_labels = []

        #     for s in appr_query_labels:
        #         new_appr_qry_labels.append(random_label_transform(s))

        #     new_appr_qry_labels = torch.cat(new_appr_qry_labels, dim=0)
        #     appr_query_labels = new_appr_qry_labels


        # if self.config.get('appr_qry_label_from_deeds', False):
        #     deeds_dir = '/home/htang6/workspace/deedsBCV/'
        #     preprocessed_dir = ''
        #     box = np.load()

        #     query_seg = 
        #     query_seg = query_seg[]


        return {
            'support_images': support_images, 
            'support_labels': support_labels,
            'query_images': data['query_images'][:, [0], ...], 
            'query_labels': data['query_labels'],
            'appr_query_labels': appr_query_labels, 
            'class_id': data['class_id'],
            'registration_field': data['registration_field'],
            'support_images_3D': data['support_images_3D'],
            'support_labels_3D': data['support_labels_3D'],
            'query_images_3D': data['query_images_3D'],
            'query_labels_3D': data['query_labels_3D'],
            'grid': grids,
            'original_support_images': data['support_images'],
            'original_support_labels': data['support_labels'],
            'warped_supp': data['warped_supp'],
            'pid': data['pid'],
            'supp_pids': data['supp_pids'],
        }

    def __len__(self):
        return len(self.fewshot_reader)


def train_collate(batch):
    return batch[0]
