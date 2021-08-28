from __future__ import print_function
import os
import random
import numpy as np
import torch
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

from net.model import model_factory
import time
from collections import defaultdict
from dataset.few_shot_reader import FewshotSliceReader, FewshotVolumeReader, train_collate
from utils.util import Logger, normalize
from config import train_config, data_config, net_config, config
from torch.nn.parallel.data_parallel import data_parallel
import pprint
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np
import argparse
import sys
from tqdm import tqdm
import traceback
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_yaml
from yaml import Loader
from utils.util import dice_score_seperate, pad2same_size, pad2same_size_3d
import ants
import numpy as np
from net.registration import DemonsRegistration, GaussianRegulariser, compute_grid, AffineDemonsRegistration, AffineDEEDSRegistration
import cv2


parser = argparse.ArgumentParser(description='Ua-Net')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--epoch-mask', default=train_config['epoch_mask'], type=int, metavar='NR',
                    help='number of epochs before training mask branch')
parser.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set-name', default=train_config['train_set_name'], type=str, metavar='OUT',
                    help='train set relative path')
parser.add_argument('--val-set-name', default=train_config['val_set_name'], type=str, metavar='OUT',
                    help='val set relative path')
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--yaml', default=None, type=str, metavar='N',
                    help='Training and testing configuration')


def grabCut(img, appr_query_label):
    """
    img: 0 ~ 1
    """
    img = (img * 255).astype(np.uint8)
    img = np.concatenate([img[..., None], img[..., None], img[..., None]], axis=2)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    appr_query_label = appr_query_label.astype(np.uint8)

    yy, xx = np.where(appr_query_label)

    if len(yy) == 0:
        return appr_query_label, appr_query_label
    else:
        ymax, ymin = yy.max(), yy.min()
        xmax, xmin = xx.max(), xx.min()
    s = min(ymax - ymin, xmax - xmin) // 8
    
    dilation_shape = cv2.MORPH_ELLIPSE
    dilatation_size = s
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1))
    sure_fg = cv2.erode(appr_query_label, element)

    dilation_shape = cv2.MORPH_ELLIPSE
    dilatation_size = s * 2
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1))

    pr_fg = cv2.dilate(appr_query_label, element)

    # mask[appr_query_labels == 1] = 3
    mask[pr_fg == 1] = cv2.GC_PR_FGD
    mask[sure_fg == 1] = cv2.GC_FGD
    init_mask = mask.copy()
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    return mask2, init_mask


def main():
    # Load training configuration
    args = parser.parse_args()

    yaml = args.yaml
    
    if not yaml:
        print('No configuration file')
        return
    else:
        config, args = load_yaml(yaml)

    net = args.net
    initial_checkpoint = args.ckpt
    if 'out_dir' in config:
        out_dir = args.out_dir
    else:
        run_name = os.path.splitext(os.path.basename(yaml))[0]
        out_dir = './results/{}/'.format(run_name)
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    batch_size = args.batch_size
    eval_set_name = args.eval_set_name
    num_workers = args.num_workers


    lr_schdule = train_config['lr_schedule']

    # Load data configuration
    data_dir = args.data_dir

    # Initilize data loader
    eval_dataset = FewshotSliceReader(data_dir, eval_set_name, config, mode='eval')
    eval_loader = eval_dataset

    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_eval')
    sys.stdout = Logger(logfile)

    print('[length of train loader %d]' % (len(eval_loader)))

    # Write graph to tensorboard for visualization
    writer = None
    eval_writer = None
    writer = SummaryWriter(tb_out_dir)
    eval_writer = SummaryWriter(os.path.join(tb_out_dir, 'eval'))
    n_run = config.get('n_runs', 1)

    eval_classes = config['eval_classes']
    dsc_reg = defaultdict(list)
    for i in range(n_run):
        print(f'{i + 1} / {n_run}')
        dsc_list = eval(eval_loader, optimizer, eval_writer, config)
        for k in eval_classes:
            dsc_reg[k].append(dsc_list[k])

    for k in eval_classes:
        dsc_reg[k] = np.array(dsc_reg[k])
        
    print('=======Average performance=========')
    for k in eval_classes:
        print(f'{k}, affine {dsc_reg[k].mean(1).mean()} + {dsc_reg[k].mean(1).std()}', end=' ')
        print()
        
    writer.close()
    eval_writer.close()


def eval(eval_loader, optimizer, writer, config):
    s = time.time()
    eval_classes = config['eval_classes']
    dsc_list = defaultdict(list)
    registration_method = config.get('registration_method', '2d')
    print('Using registration method', registration_method)

    with tqdm(enumerate(eval_loader, 0), total=len(eval_loader)) as t:
        for j, (sample_batched) in enumerate(eval_loader):
            support_images_npy = sample_batched['support_images'][0][0][:, 0, :, :].numpy()
            support_images_npy = (support_images_npy + 1) / 2.
            support_labels_npy = sample_batched['support_labels'][0][0].numpy()

            query_images_npy = sample_batched['query_images'][:, 0, :, :].numpy()
            query_images_npy = (query_images_npy + 1) / 2.
            query_labels_npy = sample_batched['query_labels'].numpy()

            class_id = sample_batched['class_id']

            if registration_method == '3d':
                support_images_npy = sample_batched['support_images_3D'][0][0][0].numpy()
                support_images_npy = (support_images_npy + 1) / 2.
                support_labels_npy = sample_batched['support_labels_3D'][0][0][0].numpy()

                query_images_npy = sample_batched['query_images_3D'][0][0][0].numpy()
                query_images_npy = (query_images_npy + 1) / 2.
                query_labels_npy = sample_batched['query_labels_3D'][0][0][0].numpy()

                support_image_ant = ants.from_numpy(support_images_npy)
                query_images_ant = ants.from_numpy(query_images_npy)
                support_mask_ant = ants.from_numpy(support_labels_npy)
                
                reg = ants.registration(support_image_ant, query_images_ant, 'SyN')
                pred_ant = ants.apply_transforms(
                    fixed=query_images_ant, 
                    moving=support_mask_ant, 
                    transformlist=reg['invtransforms'],
                    interpolator='nearestNeighbor',
                    whichtoinvert = [True,False]
                )
                pred = pred_ant.numpy()[None, ...]
            elif registration_method == '2d':
                pred = []

                for slice_id in tqdm(range(len(query_images_npy)), total=len(query_images_npy)):
                    support_image_ant = ants.from_numpy(support_images_npy[slice_id])
                    query_images_ant = ants.from_numpy(query_images_npy[slice_id])

                    support_mask_ant = ants.from_numpy(support_labels_npy[slice_id])
                    
                    reg = ants.registration(support_image_ant, query_images_ant, 'SyN')
                    pred_ant = ants.apply_transforms(
                        fixed=query_images_ant, 
                        moving=support_mask_ant, 
                        transformlist=reg['invtransforms'],
                        interpolator='nearestNeighbor',
                        whichtoinvert = [True,False]
                    )
                    pred.append(pred_ant.numpy()[None, ...])

                pred = np.concatenate(pred, axis=0)[None, ...]
            elif registration_method == 'torch_Demons':
                py_reg_pred = []
                for slice_id in tqdm(range(len(query_images_npy)), total=len(query_images_npy)):
                    src = support_images_npy[slice_id]
                    dst = query_images_npy[slice_id]
                    src_label = support_labels_npy[slice_id]
                    dst_label = query_labels_npy[slice_id]
                    # src = ((normalize(src, 0.20707, 0.3168) + 1) / 2) 
                    # dst = ((normalize(dst, 0.20707, 0.3168) + 1) / 2)
                    
                    target_H, target_W = dst.shape
                    src, dst = pad2same_size([src, dst])
                    src_label, dst_label = pad2same_size([src_label, dst_label])
                    size = src.shape
                    
                    src = torch.from_numpy(src).unsqueeze(0).unsqueeze(0)
                    dst = torch.from_numpy(dst).unsqueeze(0).unsqueeze(0)
                    src_label = torch.from_numpy(src_label).unsqueeze(0).unsqueeze(0)
                    dst_label = torch.from_numpy(dst_label).unsqueeze(0).unsqueeze(0)

                    registration = AffineDemonsRegistration(size, use_diffeomorphic=True, use_GPU=True, stop_shear=False)
                    registration = registration.cuda()
                    optimizer_affine = torch.optim.Adam(registration.affine_reg.parameters(), lr=0.01)
                    optimizer_demons = torch.optim.Adam(registration.demons.parameters(), lr=0.01)
                    regulariser = GaussianRegulariser([1, 1], sigma=[2, 2], dtype=torch.float32, device=torch.device("cuda:0"))

                    registration.train_registraion(
                        ((normalize(src, 0.20707, 0.3168) + 1) / 2).cuda(), 
                        ((normalize(dst, 0.20707, 0.3168) + 1) / 2).cuda(), 
                        [optimizer_affine, optimizer_demons], 
                        regulariser=regulariser, 
                        iters=[50, 50], 
                        regularise_displacement=False
                    )                    
                    
                    grid = compute_grid(size)
                    grid = grid.cuda()
                    warped_label = registration(src_label.cuda(), grid).cpu()
                    warped_label = warped_label[0][0].data.numpy()
                    warped_label = warped_label[:target_H, :target_W]
                    warped_label = (warped_label > 0.1).astype(np.int32)
                    if config.get('use_grabcut', False):
                        # print(support_images_npy.max(), support_images_npy.min())
                        warped_label, _ = grabCut(dst.cpu().data.numpy()[0][0], warped_label)
                        # raise NotImplementedError

                    py_reg_pred.append(warped_label[None, ...])

                py_reg_pred = np.concatenate(py_reg_pred, axis=0)[None, ...]
                pred = py_reg_pred
            elif registration_method == 'torch_DEEDS':
                py_reg_pred = []
                for slice_id in tqdm(range(len(query_images_npy)), total=len(query_images_npy)):
                    src = support_images_npy[slice_id]
                    dst = query_images_npy[slice_id]
                    src_label = support_labels_npy[slice_id]
                    dst_label = query_labels_npy[slice_id]
                    # src = ((normalize(src, 0.20707, 0.3168) + 1) / 2) 
                    # dst = ((normalize(dst, 0.20707, 0.3168) + 1) / 2)
                    
                    target_H, target_W = dst.shape
                    src, dst = pad2same_size([src, dst])
                    src_label, dst_label = pad2same_size([src_label, dst_label])
                    size = src.shape
                    
                    src = torch.from_numpy(src).unsqueeze(0).unsqueeze(0)
                    dst = torch.from_numpy(dst).unsqueeze(0).unsqueeze(0)
                    src_label = torch.from_numpy(src_label).unsqueeze(0).unsqueeze(0)
                    dst_label = torch.from_numpy(dst_label).unsqueeze(0).unsqueeze(0)

                    registration = AffineDEEDSRegistration(size, use_diffeomorphic=True, use_GPU=True, stop_shear=False)
                    registration = registration.cuda()
                    optimizer_affine = torch.optim.Adam(registration.affine_reg.parameters(), lr=0.01)
                    regulariser = GaussianRegulariser([1, 1], sigma=[2, 2], dtype=torch.float32, device=torch.device("cuda:0"))

                    registration.train_registraion(
                        src.cuda() * 255, 
                        dst.cuda() * 255, 
                        # ((normalize(src, 0.20707, 0.3168) + 1) / 2).cuda() * 255, 
                        # ((normalize(dst, 0.20707, 0.3168) + 1) / 2).cuda() * 255, 
                        [optimizer_affine, None], 
                        regulariser=None, 
                        iters=[50, None], 
                        regularise_displacement=False
                    )                    
                    
                    grid = compute_grid(size)
                    grid = grid.cuda()
                    warped_label = registration(src_label.cuda(), grid).cpu()
                    warped_label = warped_label[0][0].data.numpy()
                    warped_label = warped_label[:target_H, :target_W]
                    warped_label = (warped_label > 0.1).astype(np.int32)
                    if config.get('use_grabcut', False):
                        # print(support_images_npy.max(), support_images_npy.min())
                        warped_label, _ = grabCut(dst.cpu().data.numpy()[0][0], warped_label)
                        # raise NotImplementedError

                    py_reg_pred.append(warped_label[None, ...])

                py_reg_pred = np.concatenate(py_reg_pred, axis=0)[None, ...]
                pred = py_reg_pred
            elif registration_method == 'torch_Demons_3d':
                support_images_npy = sample_batched['support_images_3D'][0][0][0].numpy()
                support_images_npy = (support_images_npy + 1) / 2.
                support_labels_npy = sample_batched['support_labels_3D'][0][0][0].numpy()

                query_images_npy = sample_batched['query_images_3D'][0][0][0].numpy()
                query_images_npy = (query_images_npy + 1) / 2.
                query_labels_npy = sample_batched['query_labels_3D'][0][0][0].numpy()

                py_reg_pred = []
                src = support_images_npy
                dst = query_images_npy
                src_label = support_labels_npy
                dst_label = query_labels_npy
                
                target_D, target_H, target_W = dst.shape
                src, dst = pad2same_size_3d([src, dst])
                src_label, dst_label = pad2same_size_3d([src_label, dst_label])
                size = src.shape
                
                src = torch.from_numpy(src).unsqueeze(0).unsqueeze(0)
                dst = torch.from_numpy(dst).unsqueeze(0).unsqueeze(0)
                src_label = torch.from_numpy(src_label).unsqueeze(0).unsqueeze(0)
                dst_label = torch.from_numpy(dst_label).unsqueeze(0).unsqueeze(0)

                registration = AffineDemonsRegistration(size, use_diffeomorphic=True, use_GPU=True, stop_shear=False)
                registration = registration.cuda()
                optimizer_affine = torch.optim.Adam(registration.affine_reg.parameters(), lr=0.01)
                optimizer_demons = torch.optim.Adam(registration.demons.parameters(), lr=0.01)
                regulariser = GaussianRegulariser([1, 1, 1], sigma=[2, 2, 2], dtype=torch.float32, device=torch.device("cuda:0"))

                registration.train_registraion(
                    src.cuda(), 
                    dst.cuda(), 
                    [optimizer_affine, optimizer_demons], 
                    regulariser=regulariser, 
                    iters=[50, 50], 
                    regularise_displacement=False
                )                    
                
                grid = compute_grid(size)
                grid = grid.cuda()
                warped_label = registration(src_label.cuda(), grid).cpu()
                warped_label = warped_label[0][0].data.numpy()
                warped_label = warped_label[:target_D, :target_H, :target_W]
                warped_label = (warped_label > 0.1).astype(np.int32)

                py_reg_pred = warped_label[None, ...]
                pred = py_reg_pred
            else:
                raise NotImplementedError
                
            dsc = dice_score_seperate(pred, query_labels_npy[None, ...], num_class=1)[0]
            dsc_list[eval_classes[class_id]].append(dsc)
            print(f'{class_id}, {dsc}')

            t.update()

    for k in eval_classes:
        v = dsc_list[k]
        print(f'{k}, {np.average(v)}, {np.std(v)}')

    return dsc_list
    

if __name__ == '__main__':
    main()




