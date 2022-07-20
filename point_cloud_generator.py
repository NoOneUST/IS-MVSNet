"""IS-MVSNet's point cloud fusion process on TNT dataset"""

import cv2
import os
import shutil

import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

# for point cloud fusion
from numba import jit
from plyfile import PlyData, PlyElement

import datetime
import matplotlib.pyplot as plt

from src.utils import read_pfm
from src.tanks import TanksDataset


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ismvsnet'], default='ismvsnet',
                        help='which model used to inference')
    parser.add_argument('--root_dir', type=str,
                        default='data/tankandtemples',
                        help='root directory of the dataset')
    parser.add_argument('--dataset_name', type=str, default='tanks',
                        choices=['dtu', 'tanks', 'eth3d_highres'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default='intermediate',
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default='Playground', nargs='+',
                        help='specify scan to evaluate (must be in the split)')
    parser.add_argument('--time', action='store_true', default=False,
                        help='test inference time using third-party tools')
    parser.add_argument('--n_views', type=int, default=8,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[32, 16, 8],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[4.0, 2.0, 1.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1920, 1056],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--depth_range_mode', default='manual', type=str, choices=['auto', 'manual'],
                        help='how to determine depth range of a picture')
    parser.add_argument('--interval_mode', default='normal', type=str, choices=['normal'],
                        help='choose interval mode in testing')
    parser.add_argument('--ld_mode', default='single', type=str, choices=['single'],
                        help='large depth strategy')
    parser.add_argument('--interval_tune_mode', type=str, default='ratio', choices=['ratio'],
                        help='R-MVSNet depth interval mode using in inference')
    parser.add_argument('--sqrt', action='store_true', default=False,
                        help='Use sqrt method while activating ratio interval tuning mode')
    parser.add_argument('--conf', type=float, default=0.9,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--start_num', type=int, default=2,
                        help='start views number using in d2hc strategy')
    parser.add_argument('--max_ref_views', type=int, default=10000,
                        help='max number of ref views (to limit RAM usage)')
    parser.add_argument('--skip', type=int, default=1,
                        help='''how many points to skip when creating the point cloud.
                                Larger = fewer points and smaller file size.
                                Ref: skip=10 creates ~= 3M points = 50MB file
                                     skip=1 creates ~= 30M points = 500MB file
                             ''')
    parser.add_argument('--distance_ratio', type=int, default=4,
                        help='used in geometric distance filtering')
    parser.add_argument('--relative_ratio', type=int, default=1300,
                        help='used in geometric relative depth filtering')

    return parser.parse_args()


# define read_image and read_proj_mat for each dataset
def read_image(dataset_name, root_dir, split, scan, vid, index2name=None):
    if dataset_name == 'dtu':
        return cv2.imread(os.path.join(root_dir, 'TestRectified', scan,
                                       f'images/{vid:08d}.jpg'))
    if dataset_name == 'tanks':
        return cv2.imread(os.path.join(root_dir, split, scan,
                                       f'images/{vid:08d}.jpg'))
    if dataset_name == 'eth3d_highres':
        return cv2.imread(os.path.join(root_dir, f'multi_view_{split}_dslr_undistorted', scan,
                                       f'images/{vid:08d}.jpg'))
    if dataset_name == 'eth3d_lowres':
        if index2name is not None:
            return cv2.imread(os.path.join(root_dir, scan, f'images/{index2name[scan][vid]}'))
        else:
            return cv2.imread(os.path.join(root_dir, scan, f'images/{vid:08d}.jpg'))


def read_refined_image(dataset_name, split, title, vid):
    return cv2.imread(f'results/{dataset_name}/{split}/image_refined/{title}/{vid:08d}.png')


def save_refined_image(image_refined, dataset_name, split, title, vid):
    cv2.imwrite(f'results/{dataset_name}/{split}/image_refined/{title}/{vid:08d}.png',
                image_refined)


def read_proj_mat(dataset_name, dataset, scan, vid):
    if dataset_name == 'dtu':
        return dataset.proj_mats[vid][0][0].astype(np.float32)
    if dataset_name in ['tanks', 'eth3d_highres', 'eth3d_lowres']:
        return dataset.proj_mats[scan][vid][0][0].astype(np.float32)


@jit(nopython=True, fastmath=True)
def xy_ref2src(xy_ref, depth_ref, P_world2ref,
               depth_src, P_world2src, img_wh):
    # create ref grid and project to ref 3d coordinate using depth_ref
    xyz_ref = np.vstack((xy_ref, np.ones_like(xy_ref[:1]))) * depth_ref  # [3, W, H]
    xyz_ref_h = np.vstack((xyz_ref, np.ones_like(xy_ref[:1])))  # [4, W, H]

    P = (P_world2src @ np.ascontiguousarray(np.linalg.inv(P_world2ref)))[:3]
    # project to src 3d coordinate using P_world2ref and P_world2src
    xyz_src_h = P @ xyz_ref_h.reshape(4, -1)
    xy_src = xyz_src_h[:2] / xyz_src_h[2:3]
    xy_src = xy_src.reshape(2, img_wh[1], img_wh[0])

    return xy_src


@jit(nopython=True, fastmath=True)
def xy_src2ref(xy_ref, xy_src, depth_ref, P_world2ref,
               depth_src2ref, P_world2src, img_wh, start_num):
    # project xy_src back to ref view using the sampled depth
    xyz_src = np.vstack((xy_src, np.ones_like(xy_src[:1]))) * depth_src2ref
    xyz_src_h = np.vstack((xyz_src, np.ones_like(xy_src[:1])))
    P = (P_world2ref @ np.ascontiguousarray(np.linalg.inv(P_world2src)))[:3]
    xyz_ref_h = P @ xyz_src_h.reshape(4, -1)
    depth_ref_reproj = xyz_ref_h[2].reshape(img_wh[1], img_wh[0])
    xy_ref_reproj = xyz_ref_h[:2] / xyz_ref_h[2:3]
    xy_ref_reproj = xy_ref_reproj.reshape(2, img_wh[1], img_wh[0])

    # check |p_reproj-p_1| < 1
    pixel_diff = xy_ref_reproj - xy_ref
    pixel_diff = np.sqrt(pixel_diff[0] ** 2 + pixel_diff[1] ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    relative_depth_diff = np.abs((depth_ref_reproj - depth_ref) / depth_ref)
    masks = []
    for i in range(start_num, 11):
        mask_geo = np.logical_and(pixel_diff < i/4, relative_depth_diff < i/1300)
        masks.append(mask_geo)

    return depth_ref_reproj, mask_geo, masks


def check_geo_consistency(depth_ref, P_world2ref,
                          depth_src, P_world2src,
                          image_ref, image_src,
                          img_wh, start_num):
    """
    Check the geometric consistency between ref and src views.
    """
    xy_ref = np.mgrid[:img_wh[1], :img_wh[0]][::-1].astype(np.float32)
    xy_src = xy_ref2src(xy_ref, depth_ref, P_world2ref,
                        depth_src, P_world2src, img_wh)

    # Sample the depth of xy_src using bilinear interpolation
    depth_src2ref = cv2.remap(depth_src,
                              xy_src[0].astype(np.float32),
                              xy_src[1].astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)

    image_src2ref = cv2.remap(image_src,
                              xy_src[0].astype(np.float32),
                              xy_src[1].astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)
    # mask_geo is the pixel location which fulfill the geometric consistence
    depth_ref_reproj, mask_geo, masks = \
        xy_src2ref(xy_ref, xy_src, depth_ref, P_world2ref,
                   depth_src2ref, P_world2src, img_wh, start_num)

    depth_ref_reproj[~mask_geo] = 0
    image_src2ref[~mask_geo] = 0

    return depth_ref_reproj, mask_geo, image_src2ref, masks


args = get_opts()
index2name = None
if args.dataset_name == 'tanks':
    args.root_dir = 'data/tankandtemples'
    args.split = 'intermediate'
    args.conf = 0.9
    args.depth_range_mode = 'manual'
    args.k2 = 4
    args.k3 = 10
    args.img_wh = [1920, 1056]
    args.start_num = 2
    dataset = TanksDataset(args.root_dir, args.split, n_views=args.n_views,
                           img_wh=tuple(args.img_wh), depth_range_mode=args.depth_range_mode, scan=args.scan,
                           n_depths=args.n_depths)
    scans = dataset.scans
point_dir = f'results/{args.dataset_name}/{args.split}/points'
os.makedirs(point_dir, exist_ok=True)
print('Fusing point clouds...')

for scan in scans:
    print(f'Processing {scan} ...')
    img_wh = args.img_wh
    depth_folder = f'{img_wh[0]}_{img_wh[1]}_{args.n_views - 1}'
    point_title = f'{scan}_{img_wh[0]}_{img_wh[1]}_src_num{args.n_views - 1}_conf_th{args.conf}_{args.depth_range_mode}'
    if args.interval_tune_mode is not None:
        point_title += f'_{args.interval_tune_mode}'
    point_file = f'{point_dir}/{point_title}.ply'
    if os.path.exists(point_file):
        continue
    # buffers for the final vertices of this scan
    vs = []
    v_colors = []
    # buffers storing the refined data of each ref view
    os.makedirs(f'results/{args.dataset_name}/{args.split}/image_refined/{point_title}', exist_ok=True)
    image_refined = set()
    depth_refined = {}
    total_ref_list = list(filter(lambda x: x[0] == scan, dataset.metas))[:args.max_ref_views]
    total_used_ref_list = total_ref_list
    for meta in tqdm(total_used_ref_list):
        try:
            ref_vid = meta[2]
            if ref_vid in image_refined: # not yet refined actually
                image_ref = read_refined_image(args.dataset_name, args.split, point_title, ref_vid)
                depth_ref = depth_refined[ref_vid]
            else:
                image_ref = read_image(args.dataset_name, args.root_dir, args.split, scan, ref_vid, index2name)
                image_ref = cv2.resize(image_ref, (img_wh[0] // 2, img_wh[1] // 2),
                                       interpolation=cv2.INTER_LINEAR)[:, :, ::-1]  # to RGB
                depth_ref = read_pfm(f'results/{args.dataset_name}/{args.split}/depth/' \
                                     f'{scan}/{depth_folder}/depth_{ref_vid:04d}.pfm')[0]
            proba_ref = read_pfm(f'results/{args.dataset_name}/{args.split}/depth/' \
                                 f'{scan}/{depth_folder}/proba_{ref_vid:04d}.pfm')[0]
            mask_conf = proba_ref > args.conf  # confidence mask
            # read ref camera's parameters
            P_world2ref = read_proj_mat(args.dataset_name, dataset, scan, ref_vid)

            src_vids = meta[3]
            mask_geos = []
            geo_mask_sums = []
            depth_ref_reprojs = [depth_ref]
            image_src2refs = [image_ref]
            n = len(src_vids)+1
            # for each src view, check the consistency and refine depth
            for ct, src_vid in enumerate(src_vids):
                if src_vid in image_refined:  # use refined data of previous runs
                    image_src = read_refined_image(args.dataset_name, args.split, point_title, src_vid)
                    depth_src = depth_refined[src_vid]
                else:
                    image_src = read_image(args.dataset_name, args.root_dir, args.split, scan, src_vid, index2name)
                    image_src = cv2.resize(image_src, (img_wh[0] // 2, img_wh[1] // 2),
                                           interpolation=cv2.INTER_LINEAR)[:, :, ::-1]  # to RGB
                depth_src = read_pfm(f'results/{args.dataset_name}/{args.split}/depth/{scan}/{depth_folder}'
                                     f'/depth_{src_vid:04d}.pfm')[0]
                depth_refined[src_vid] = depth_src
                # read src camera's parameters
                P_world2src = read_proj_mat(args.dataset_name, dataset, scan, src_vid)
                depth_ref_reproj, mask_geo, image_src2ref, masks = check_geo_consistency(depth_ref, P_world2ref,
                                                                                         depth_src, P_world2src,
                                                                                         image_ref, image_src,
                                                                                         (img_wh[0] // 2, img_wh[1] // 2),
                                                                                         args.start_num)
                depth_ref_reprojs += [depth_ref_reproj]
                image_src2refs += [image_src2ref]
                mask_geos += [mask_geo]
                if ct == 0:
                    for i in range(args.start_num, n):
                        geo_mask_sums.append(masks[i - args.start_num].astype(np.int32))
                else:
                    for i in range(args.start_num, n):
                        geo_mask_sums[i - args.start_num] += masks[i - args.start_num].astype(np.int32)
            mask_geo_sum = np.sum(mask_geos, 0)
            # num of consistent view pairs
            mask_geo_final = mask_geo_sum >= (n - 1)
            print(f'mask_geo_final init value: {mask_geo_final.mean()}')
            for i in range(args.start_num, n):
                mask_geo_final = np.logical_or(mask_geo_final, geo_mask_sums[i - args.start_num] >= i)
                print(f'iter {i}: mask_value: {(geo_mask_sums[i - args.start_num] >= i).mean()}')
            print(f'mask_geo_final end value: {mask_geo_final.mean()}')
            depth_refined[ref_vid] = \
                (np.sum(depth_ref_reprojs, 0) / (mask_geo_sum + 1)).astype(np.float32)
            image_refined_ = \
                np.sum(image_src2refs, 0) / np.expand_dims((mask_geo_sum + 1), -1)

            image_refined.add(ref_vid)
            save_refined_image(image_refined_, args.dataset_name, args.split, point_title, ref_vid)
            mask_final = mask_conf & mask_geo_final

            # create the final points
            xy_ref = np.mgrid[:img_wh[1] // 2, :img_wh[0] // 2][::-1]
            xyz_ref = np.vstack((xy_ref, np.ones_like(xy_ref[:1]))) * depth_refined[ref_vid]
            xyz_ref = xyz_ref.transpose(1, 2, 0)[mask_final].T  # (3, N)
            color = image_refined_[mask_final]  # (N, 3)
            xyz_ref_h = np.vstack((xyz_ref, np.ones_like(xyz_ref[:1])))
            xyz_world = (np.linalg.inv(P_world2ref) @ xyz_ref_h).T  # (N, 4)
            xyz_world = xyz_world[::args.skip, :3]
            color = color[::args.skip]

            # append to buffers
            vs += [xyz_world]
            v_colors += [color]

        except FileNotFoundError:
            # some scenes might not have depth prediction due to too few valid src views
            print(f'Skipping view {ref_vid} due to too few valid source views...')
            continue
    # clear refined buffer
    image_refined.clear()
    depth_refined.clear()
    shutil.rmtree(f'results/{args.dataset_name}/{args.split}/image_refined/{point_title}')

    # process all points in the buffers
    vs = np.ascontiguousarray(np.vstack(vs).astype(np.float32))
    v_colors = np.vstack(v_colors).astype(np.uint8)
    print(f'{scan} contains {len(vs) / 1e6:.2f} M points')
    vs.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_all = np.empty(len(vs), vs.dtype.descr + v_colors.dtype.descr)
    for prop in vs.dtype.names:
        vertex_all[prop] = vs[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(f'{point_dir}/{point_title}.ply')
    del vertex_all, vs, v_colors

print('Done!')
