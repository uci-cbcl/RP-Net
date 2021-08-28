from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import random
import numpy as np
import torch
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
from net.model import model_factory
import time
from collections import defaultdict
from dataset.few_shot_reader import FewshotRegReader
from utils.util import Logger
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
from utils.util import dice_score_seperate
import torch.nn.functional as F
from net.registration import NCC, MSE


parser = argparse.ArgumentParser(description='RP-Net')
parser.add_argument('--yaml', default=None, type=str, metavar='N',
                    help='Training and testing configuration')

def main():
    # Load training configuration
    args = parser.parse_args()
    old_args = args

    yaml = args.yaml
    
    if not yaml:
        print('No configuration file')
        return
    else:
        config, args = load_yaml(yaml)
        config['n_iter_refinement'] = config['n_test_iter_refinement']
        # args.ckpt = old_args.ckpt

    net = args.net
    initial_checkpoint = args.ckpt
    if 'out_dir' in config:
        out_dir = args.out_dir
    else:
        run_name = os.path.splitext(os.path.basename(yaml))[0]
        out_dir = './results/{}/'.format(run_name)

    optimizer = args.optimizer
    eval_set_name = args.eval_set_name


    # Load data configuration
    data_dir = args.data_dir

    # Initilize data loader
    eval_dataset = FewshotRegReader(data_dir, eval_set_name, config, mode='eval')
    eval_loader = eval_dataset

    # Initilize network
    net = model_factory[net](
        pretrained_path=config['pretrained_path'], 
        cfg={
            'align': True,
            'backbone': config.get('backbone', 'vgg')
        },
        backbone_cfg=config
    )
    net = net.cuda()
    
    start_epoch = 0

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        # optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(state)

    start_epoch = start_epoch + 1

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
    dsc_affine = defaultdict(list)
    dsc_fewshot = defaultdict(list)
    dsc_refinement = defaultdict(lambda: defaultdict(list))
    for i in range(n_run):
        print(f'{i + 1} / {n_run}')
        dsc_affine_list, dsc_fewshot_list, dsc_refinement_list = eval(net, eval_loader, optimizer, eval_writer, config, start_epoch)
        for k in eval_classes:
            dsc_affine[k].append(dsc_affine_list[k])
            dsc_fewshot[k].append(dsc_fewshot_list[k])

            for it, l in dsc_refinement_list[k].items():
                dsc_refinement[k][it].append(l)

    for k in eval_classes:
        dsc_affine[k] = np.array(dsc_affine[k])
        dsc_fewshot[k] = np.array(dsc_fewshot[k])

        for it, _ in dsc_refinement[k].items():
            dsc_refinement[k][it] = np.array(dsc_refinement[k][it])
        
    ref_dsc = []
    print('=======Average performance=========')
    for k in eval_classes:
        print(f'{k}, affine {dsc_affine[k].mean(1).mean()} + {dsc_affine[k].mean(1).std()}, fewshot {dsc_fewshot[k].mean(1).mean()} + {dsc_fewshot[k].mean(1).std()}', end=' ')
        print()

        for ref, l in dsc_refinement[k].items():
            ref_dsc.append(l.mean(1).mean())
            print(f'ref {ref} {l.mean(1).mean()} + {l.mean(1).std()}, ', end=' ')
        print()
    print(ref_dsc)

    writer.close()
    eval_writer.close()


def eval(net, eval_loader, optimizer, writer, config, epoch):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    s = time.time()
    eval_classes = config['eval_classes']
    dsc_list = defaultdict(list)
    dsc_affine_list = defaultdict(list)
    dsc_fewshot_list = defaultdict(list)
    dsc_refinement_list = defaultdict(lambda: defaultdict(list))

    with tqdm(enumerate(eval_loader, 0), total=len(eval_loader)) as t:
        for j, (sample_batched) in t:
            with torch.no_grad():
                batch_size = 2

                support_images = [[shot.float().cuda() for shot in way]
                                for way in sample_batched['support_images']]
                support_fg_mask = [[shot.float().cuda() for shot in way]
                                for way in sample_batched['support_labels']]
                support_bg_mask = [[1 - shot.float().cuda() for shot in way]
                                for way in sample_batched['support_labels']]
                warped_supp = sample_batched['warped_supp'].unsqueeze(1)

                query_images = sample_batched['query_images'].float().cuda()
                query_labels = sample_batched['query_labels'].long().cuda()
                appr_query_labels = sample_batched['appr_query_labels'].cuda()

                grid = sample_batched['grid'].cuda()

                class_id = sample_batched['class_id']
                pid = sample_batched['pid']

                class_idx, supp_idx = sample_batched['supp_pids'][0]
                supp_pid = eval_loader.fewshot_reader.fewshot_volume_reader.data_info[class_idx][supp_idx]['pid']
                pred = []
                fewshot_pred = []
                refinement = defaultdict(list)

                for i in range(int(np.ceil(len(query_images) / batch_size))):
                    support_images_batch = [
                        [shot[i * batch_size:(i + 1) * batch_size] for shot in way]
                        for way in support_images
                    ]
                    support_fg_mask_batch = [
                        [shot[i * batch_size:(i + 1) * batch_size] for shot in way]
                        for way in support_fg_mask
                    ]
                    support_bg_mask_batch = [
                        [shot[i * batch_size:(i + 1) * batch_size] for shot in way]
                        for way in support_bg_mask
                    ]
                    query_images_batch = [query_images[i * batch_size:(i + 1) * batch_size]]
                    query_labels_batch = query_labels[i * batch_size:(i + 1) * batch_size]
                    appr_query_labels_batch = appr_query_labels[i * batch_size:(i + 1) * batch_size]
                    grid_batch = grid[i * batch_size:(i + 1) * batch_size]

                    output = net(
                        support_images_batch, 
                        support_fg_mask_batch, 
                        support_bg_mask_batch,
                        query_images_batch,
                        grid=grid_batch,
                        query_labels=query_labels_batch,
                        appr_query_labels=appr_query_labels_batch
                    )
                    ref = output['refinement']

                    query_pred = output['output']
                    fewshot_pred.append(query_pred.softmax(dim=1)[:, [1], :, :].cpu())
                    for k, v in ref.items():
                        refinement[k].append(ref[k].softmax(dim=1)[:, 1, ...])


                fewshot_pred = torch.cat(fewshot_pred, dim=0).permute(1, 0, 2, 3).contiguous().numpy()
                fewshot_pred = (fewshot_pred > 0.5).astype(np.float32)

                dsc_affine = dice_score_seperate(appr_query_labels.cpu().data.numpy()[None, ...], query_labels.cpu().data.numpy()[None, ...], num_class=1)[0]
                dsc_fewshot = dice_score_seperate(fewshot_pred, query_labels.cpu().data.numpy()[None, ...], num_class=1)[0]
                d = NCC(query_images, warped_supp.cuda()).item()
                d2 = NCC(query_images, support_images[0][0]).item()

                print(f'{j} {pid} {supp_pid} affine ({d}, {d2}) {dsc_affine}, fewshot {dsc_fewshot}', end=' ')

                dsc_affine_list[eval_classes[class_id]].append(dsc_affine)
                dsc_fewshot_list[eval_classes[class_id]].append(dsc_fewshot)

                for k, v in refinement.items():
                    refinement[k] = torch.cat(refinement[k], dim=0)
                    s = dice_score_seperate((refinement[k].cpu().data.numpy() > 0.5).astype(np.int32)[None, ...], query_labels.cpu().data.numpy()[None, ...], num_class=1)[0]
                    dsc_refinement_list[eval_classes[class_id]][k].append(s)
                    print(f'ref {k} {s}, ', end=' ')

                print()
                # t.update()

    for k in eval_classes:
        v = dsc_list[k]
        print(f'{k}, affine {np.average(dsc_affine_list[k])}, voxel morph {np.average(v)}, {np.std(v)}, fewshot {np.average(dsc_fewshot_list[k])}', end=' ')

        for ref, l in dsc_refinement_list[k].items():
            print(f'ref {ref} {np.average(l)}, ', end=' ')
        print()

        # Write to tensorboard
        if writer:
            writer.add_scalar(f'{k}', np.average(dsc_fewshot_list[k]), epoch)

    return dsc_affine_list, dsc_fewshot_list, dsc_refinement_list
    

if __name__ == '__main__':
    main()




