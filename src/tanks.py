import os
from collections import defaultdict

from PIL import Image
import numpy as np

import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset import py_transforms as transforms


class TanksDataset:
    def __init__(self, root_dir, split='intermediate', n_views=8, levels=3,
                 img_wh=(1920, 1056), depth_range_mode='manual', scan=None, depth_num_ratio=1,
                 interval_tune_mode='ratio', n_depths=None):
        """
        For testing only! You can write training data loader by yourself.
        @depth_interval has no effect. The depth_interval is predefined for each view.
        """
        self.root_dir = root_dir
        self.img_wh = img_wh
        self.depth_range_mode = depth_range_mode
        self.depth_num_ratio = depth_num_ratio
        self.interval_tune_mode = interval_tune_mode
        self.n_depths = n_depths
        assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
            'img_wh must both be multiples of 32!'
        self.split = split
        assert self.split in ['intermediate', 'advanced', 'training']
        self.split_dir = os.path.join(self.root_dir, split)
        self.single_scan = scan
        self.build_metas()
        self.n_views = n_views
        self.levels = levels  # FPN levels
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        if self.single_scan is not None:
            self.scans = self.single_scan if isinstance(self.single_scan, list) else [self.single_scan]

        if self.split == 'intermediate':
            self.image_sizes = {'Playground': (1920, 1080)}
            self.depth_interval = {'Playground': 10.5e-3}
            for key in self.depth_interval.keys():
                self.depth_interval[key] /= self.depth_num_ratio
            # depth interval for each scan (hand tuned)
        self.ref_views_per_scan = defaultdict(list)

        for scan in self.scans:
            with open(os.path.join(self.split_dir, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # pass the exception case
                    if _ == 99 and scan == 'Temple':
                        continue
                    self.metas += [(scan, -1, ref_view, src_views)]
                    self.ref_views_per_scan[scan] += [ref_view]

    def build_proj_mats(self):
        self.proj_mats = {}  # proj mats for each scan
        for scan in self.scans:
            self.proj_mats[scan] = {}
            img_w, img_h = self.image_sizes[scan]
            for vid in self.ref_views_per_scan[scan]:
                # use RMVS interval
                if self.depth_range_mode == 'auto':
                    proj_mat_filename = os.path.join(self.split_dir, scan,
                                                     f'cams/{vid:08d}_cam.txt')
                else:
                    proj_mat_filename = os.path.join(self.split_dir, scan,
                                                     f'rmvs_scan_cams/{vid:08d}_cam.txt')

                intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = \
                    self.read_cam_file(proj_mat_filename)
                intrinsics[0] *= self.img_wh[0] / img_w / 8
                intrinsics[1] *= self.img_wh[1] / img_h / 8
                depth_num *= min(self.img_wh[0] / img_w, self.img_wh[1] / img_h) / 2
                if self.interval_tune_mode == 'num':
                    depth_num -= depth_num % 8
                else:
                    pass

                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat_ls = []
                for l in reversed(range(self.levels)):
                    proj_mat_l = np.eye(4)
                    proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                    intrinsics[:2] *= 2  # 1/4->1/2->1
                    proj_mat_ls += [proj_mat_l]
                # (self.levels, 4, 4) from fine to coarse
                proj_mat_ls = np.stack(proj_mat_ls[::-1])
                self.proj_mats[scan][vid] = (proj_mat_ls, depth_min, depth_interval, depth_num, depth_max)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        mvs_params = lines[11].split()
        depth_min = float(mvs_params[0])
        depth_interval = float(mvs_params[1])
        depth_num = float(mvs_params[2])
        depth_max = float(mvs_params[3])
        return intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max

    def define_transforms(self):
        self.transform = transforms.Compose([py_vision.ToTensor(),
                                             py_vision.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                             ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.split_dir, scan, f'images/{vid:08d}.jpg')

            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min, depth_interval, depth_num, depth_max = self.proj_mats[scan][vid]

            if i == 0:  # reference view
                ref_proj_inv = np.asarray(proj_mat_ls)
                for j in range(proj_mat_ls.shape[0]):
                    ref_proj_inv[j] = np.mat(proj_mat_ls[j]).I

                if self.depth_range_mode == 'auto':
                    depth_interval = (depth_max - depth_min) / (128 * self.depth_num_ratio)
                else:
                    depth_interval = self.depth_interval[scan]
                    depth_interval = depth_interval / (int(self.n_depths[0]) / 32)

                if scan == 'Horse':
                    # expand depth range
                    depth_interval = self.depth_interval[scan] * 1.1
                    depth_min *= 0.9

                sample['init_depth_min'] = [depth_min]
                sample['depth_interval'] = [depth_interval]
                sample['depth_num'] = [depth_num]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

        imgs = np.stack(imgs).squeeze(1)  # (V, 3, H, W)
        proj_mats = np.stack(proj_mats)[:, :, :3]  # (V-1, self.levels, 3, 4) from fine to coarse

        sample['imgs'] = imgs
        sample['proj_mats'] = proj_mats
        sample['scan_vid'] = (scan, ref_view)

        return imgs, proj_mats.astype(np.float32), np.array(sample['init_depth_min'], dtype=np.float32), \
               np.array(sample['depth_interval'], dtype=np.float32), scan, ref_view
