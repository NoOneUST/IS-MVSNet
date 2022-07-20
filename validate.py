"""IS-MVSNet's validation process on TNT dataset"""

import os
import time
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context
from mindspore.ops import operations as P

from src.is_mvsnet import ISMVSNet
from src.tanks import TanksDataset
from src.utils import save_pfm, AverageMeter

from src.model_tnt_intermediate_backbone import MindSporeModel as FeatExt_tnt_intermediate
from src.model_tnt_intermediate_stage1 import MindSporeModel as Stage1_tnt_intermediate
from src.model_tnt_intermediate_stage2 import MindSporeModel as Stage2_tnt_intermediate
from src.model_tnt_intermediate_stage3 import MindSporeModel as Stage3_tnt_intermediate


def get_opts():
    """set options"""
    parser = ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='which gpu used to inference')
    ## data
    parser.add_argument('--root_dir', type=str,
                        default='data/tnt/tankandtemples',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='tanks',
                        choices=['tanks', 'eth3d_highres'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default='intermediate',
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default='Playground', nargs='+',
                        help='specify scan to evaluate (must be in the split)')
    # for depth prediction
    parser.add_argument('--n_views', type=int, default=8,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[32, 16, 8],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[4.0, 2.0, 1.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1920, 1056],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--save_visual', default=True, action='store_true',
                        help='save depth and proba visualization or not')
    parser.add_argument('--conf', type=float, default=0.9,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--depth_range_mode', default='manual', type=str, choices=['auto', 'manual'],
                        help='how to determine depth range of a picture')

    parser.add_argument('--k2', type=int, default=4,
                        help='k in stage 2')
    parser.add_argument('--k3', type=int, default=10,
                        help='k in stage 3')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    context.set_context(mode=1, device_target='GPU', device_id=args.gpu_id, save_graphs=False,
                        enable_graph_kernel=False)

    if args.dataset_name == 'tanks':
        args.root_dir = 'data/tankandtemples'
        args.split = 'intermediate'
        args.conf = 0.9
        args.depth_range_mode = 'manual'
        args.k2 = 4
        args.k3 = 10
        args.img_wh = [1920, 1056]
        dataset = TanksDataset(args.root_dir, args.split, n_views=args.n_views,
                               img_wh=tuple(args.img_wh), depth_range_mode=args.depth_range_mode, scan=args.scan,
                               n_depths=args.n_depths)
        feat_ext = FeatExt_tnt_intermediate()
        stage1 = Stage1_tnt_intermediate()
        stage2 = Stage2_tnt_intermediate()
        stage3 = Stage3_tnt_intermediate()
        ms.load_param_into_net(feat_ext,
                               ms.load_checkpoint('weights/model_tnt_intermediate_backbone.ckpt'))
        ms.load_param_into_net(stage1,
                               ms.load_checkpoint('weights/model_tnt_intermediate_stage1.ckpt'))
        ms.load_param_into_net(stage2,
                               ms.load_checkpoint('weights/model_tnt_intermediate_stage2.ckpt'))
        ms.load_param_into_net(stage3,
                               ms.load_checkpoint('weights/model_tnt_intermediate_stage3.ckpt'))

    img_wh = args.img_wh
    scans = dataset.scans

    print(args.n_depths)
    print(args.interval_ratios)

    # Step 1. Create depth estimation and probability for each scan
    ISMVSNet_eval = ISMVSNet(n_depths=args.n_depths, interval_ratios=args.interval_ratios, feat_ext=feat_ext,
                             stage1=stage1, stage2=stage2, stage3=stage3, height=args.img_wh[1], width=args.img_wh[0],
                             k2=args.k2, k3=args.k3)
    ISMVSNet_eval.set_train(False)

    depth_dir = f'results/{args.dataset_name}/{args.split}/depth'
    print('Creating depth and confidence predictions...')
    if args.scan:
        data_range = [i for i, x in enumerate(dataset.metas) if x[0] == args.scan]
    else:
        data_range = range(len(dataset))
    test_loader = ds.GeneratorDataset(dataset, column_names=['imgs', 'proj_mats', 'init_depth_min', 'depth_interval',
                                                             'scan', 'vid'],
                                      num_parallel_workers=1, shuffle=False)
    test_loader = test_loader.batch(batch_size=1)
    test_data_size = test_loader.get_dataset_size()
    print("train dataset length is:", test_data_size)

    pbar = tqdm(enumerate(test_loader.create_tuple_iterator()), dynamic_ncols=True, total=test_data_size)

    forward_time_avg = AverageMeter()

    scan_list, vid_list = [], []

    depth_folder = f'{img_wh[0]}_{img_wh[1]}_{args.n_views - 1}_{args.k2}_{args.k3}'

    for i, sample in pbar:
        imgs, proj_mats, init_depth_min, depth_interval, scan, vid = sample
        scan = scan.asnumpy().item()
        vid = vid.asnumpy().item()

        depth_file_dir = os.path.join(depth_dir, scan, depth_folder)
        if not os.path.exists(depth_file_dir):
            os.makedirs(depth_file_dir, exist_ok=True)

        begin = time.time()

        depth, proba, est_variance_1, est_variance_2 = ISMVSNet_eval(imgs, proj_mats, init_depth_min, depth_interval)

        forward_time = time.time() - begin
        if i != 0:
            forward_time_avg.update(forward_time)

        depth = P.Squeeze()(depth).asnumpy()
        depth = np.nan_to_num(depth)  # change nan to 0
        proba = P.Squeeze()(proba).asnumpy()
        proba = np.nan_to_num(proba)  # change nan to 0

        save_pfm(os.path.join(depth_dir, f'{scan}/{depth_folder}/depth_{vid:04d}.pfm'), depth)
        save_pfm(os.path.join(depth_dir, f'{scan}/{depth_folder}/proba_{vid:04d}.pfm'), proba)

        if args.save_visual:
            mi = np.min(depth[depth > 0])
            ma = np.max(depth)
            depth = (depth - mi) / (ma - mi + 1e-8)
            depth = (255 * depth).astype(np.uint8)
            depth_img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/{depth_folder}/depth_visual_{vid:04d}.jpg'), depth_img)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/{depth_folder}/proba_visual_{vid:04d}.jpg'),
                        (255 * (proba > args.conf)).astype(np.uint8))
        print(f'step {i} time: {forward_time}s')
    print(f'mean forward time: {forward_time_avg.avg}')

    with open(f'results/{args.dataset_name}/{args.split}/metrics.txt', 'w') as f:
        f.writelines('mean forward time(s/pic):' + str(np.round(forward_time_avg.avg, 4)) + '\n')
        f.close()
    print('Done!')
