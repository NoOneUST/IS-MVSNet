import mindspore as ms
import mindspore.nn as nn
import mindspore.common as mstype
from mindspore import Tensor, context
import mindspore.ops as op
import mindspore.numpy as np

import torch

from model_tnt_intermediate_backbone_new import MindSporeModel as FeatExt
from model_tnt_intermediate_stage1_new import MindSporeModel as Stage1
from model_tnt_intermediate_stage2_new import MindSporeModel as Stage2
from model_tnt_intermediate_stage3_new import MindSporeModel as Stage3

from src.modules import HomoWarp, get_depth_values, process_depth_input

# context.set_context(mode=context.PYNATIVE_MODE)


class ISMVSNet(nn.Cell):
    """IS-MVSNet"""

    def __init__(self, n_depths=[32, 16, 8], interval_ratios=[4, 2, 1], feat_ext=None, stage1=None,
                 stage2=None, stage3=None, height=1056, width=1920, k2=1, k3=1):
        super(ISMVSNet, self).__init__()

        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.d_scale = 1
        self.feat_ext = feat_ext
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.homo_warp1 = HomoWarp(height // 8, width // 8)
        self.homo_warp2 = HomoWarp(height // 4, width // 4)
        self.homo_warp3 = HomoWarp(height // 2, width // 2)
        self.interpolate2 = op.ResizeBilinear((height // 4, width // 4))
        self.interpolate3 = op.ResizeBilinear((height // 2, width // 2))
        self.k2, self.k3 = k2, k3

    def construct(self, imgs, proj_mats=None, depth_start=None, depth_interval=None):
        """construct function"""

        # feature extraction
        B, V, _, H, W = imgs.shape
        imgs = imgs.reshape(B * V, 3, H, W)
        feat_pack_1, feat_pack_2, feat_pack_3 = self.feat_ext(imgs)
        feat_pack_1 = feat_pack_1.view(B, V, *feat_pack_1.shape[1:])  # (B, V, C, h, w)
        feat_pack_2 = feat_pack_2.view(B, V, *feat_pack_2.shape[1:])  # (B, V, C, h, w)
        feat_pack_3 = feat_pack_3.view(B, V, *feat_pack_3.shape[1:])  # (B, V, C, h, w)

        # stage 1
        ref_feat, srcs_feat = feat_pack_1[:, 0], feat_pack_1[:, 1:]
        srcs_feat = srcs_feat.transpose(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats_now = proj_mats[:, :, 2].transpose(1, 0, 2, 3)  # (V-1, B, 3, 4)
        h_1, w_1 = ref_feat.shape[-2:]
        depth_start_now = depth_start
        depth_interval_now = depth_interval * self.interval_ratios[0]
        depth_start_now, depth_interval_now = process_depth_input(depth_start_now, depth_interval_now, B)
        depth_values = depth_start_now + depth_interval_now * np.arange(0, self.n_depths[0], dtype=ref_feat.dtype)  # (D)
        depth_values = np.tile(depth_values.reshape(1, self.n_depths[0], 1, 1), (B, 1, h_1, w_1))
        warped_srcs = []
        for idx in range(srcs_feat.shape[0]):
            warped_srcs.append(self.homo_warp1(srcs_feat[idx], proj_mats_now[idx], depth_values))  # (1,32,32,64,80)
        warped_srcs = np.stack(warped_srcs, axis=0).transpose(0, 1, 5, 2, 3, 4)
        ref_ncdhw = np.tile(ref_feat.expand_dims(2), (1, 1, self.n_depths[0] // self.d_scale, 1, 1))
        pair_1_1, pair_1_2_1, pair_1_2_2, est_depth_1, prob_map_1 = self.stage1(ref_ncdhw, warped_srcs, depth_values)
        # photometric consistency for stage 1
        ref_volume = ref_feat.expand_dims(2)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        for idx in range(srcs_feat.shape[0]):
            warped_volume = self.homo_warp1(srcs_feat[idx], proj_mats_now[idx], est_depth_1).transpose(0, 4, 1, 2, 3)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        est_variance_1 = (volume_sq_sum / V - (volume_sum / V) ** 2).mean()

        # stage 2
        ref_feat, srcs_feat = feat_pack_2[:, 0], feat_pack_2[:, 1:]
        depth_start_now = self.interpolate2(est_depth_1)
        srcs_feat = srcs_feat.transpose(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats_now = proj_mats[:, :, 1].transpose(1, 0, 2, 3)  # (V-1, B, 3, 4)
        depth_interval_now = depth_interval * self.interval_ratios[1]
        depth_start_now, depth_interval_now = process_depth_input(depth_start_now, depth_interval_now, B)
        depth_values = get_depth_values(depth_start_now, self.n_depths[1], depth_interval_now, k=self.k2)
        warped_srcs = []
        for idx in range(srcs_feat.shape[0]):
            warped_srcs.append(self.homo_warp2(srcs_feat[idx], proj_mats_now[idx], depth_values))  # (1,32,32,64,80)
        warped_srcs = np.stack(warped_srcs, axis=0).transpose(0, 1, 5, 2, 3, 4)
        ref_ncdhw = np.tile(ref_feat.expand_dims(2), (1, 1, self.n_depths[1] // self.d_scale, 1, 1))
        pair_2_1, pair_2_2_1, pair_2_2_2, est_depth_2, prob_map_2 = self.stage2(ref_ncdhw, warped_srcs, depth_values)
        # photometric consistency for stage 2
        ref_volume = ref_feat.expand_dims(2)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        for idx in range(srcs_feat.shape[0]):
            warped_volume = self.homo_warp2(srcs_feat[idx], proj_mats_now[idx], est_depth_2).transpose(0, 4, 1, 2, 3)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        est_variance_2 = (volume_sq_sum / V - (volume_sum / V) ** 2).mean()

        # stage 3
        ref_feat, srcs_feat = feat_pack_3[:, 0], feat_pack_3[:, 1:]
        depth_start_now = self.interpolate3(est_depth_2)
        srcs_feat = srcs_feat.transpose(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats_now = proj_mats[:, :, 0].transpose(1, 0, 2, 3)  # (V-1, B, 3, 4)
        depth_interval_now = depth_interval * self.interval_ratios[2]
        depth_start_now, depth_interval_now = process_depth_input(depth_start_now, depth_interval_now, B)
        depth_values = get_depth_values(depth_start_now, self.n_depths[2], depth_interval_now, k=self.k3)
        warped_srcs = []
        for idx in range(srcs_feat.shape[0]):
            warped_srcs.append(self.homo_warp3(srcs_feat[idx], proj_mats_now[idx], depth_values))  # (1,32,32,64,80)
        warped_srcs = np.stack(warped_srcs, axis=0).transpose(0, 1, 5, 2, 3, 4)
        ref_ncdhw = np.tile(ref_feat.expand_dims(2), (1, 1, self.n_depths[2] // self.d_scale, 1, 1))
        pair_3_1, pair_3_2_1, pair_3_2_2, est_depth_3, prob_map_3 = self.stage3(ref_ncdhw, warped_srcs, depth_values)

        return est_depth_3, prob_map_3, est_variance_1, est_variance_2


def main():
    inputs = torch.load('ISMVSNet_pl/raw_input/input.pth')
    imgs, proj_mats, init_depth_min, depth_interval = [Tensor(x.numpy()) for x in inputs]
    feat_ext = FeatExt()
    stage1 = Stage1()
    stage2 = Stage2()
    stage3 = Stage3()
    ms.load_param_into_net(feat_ext,
                           ms.load_checkpoint('model_tnt_intermediate_backbone_new.ckpt'))
    ms.load_param_into_net(stage1,
                           ms.load_checkpoint('model_tnt_intermediate_stage1_new.ckpt'))
    ms.load_param_into_net(stage2,
                           ms.load_checkpoint('model_tnt_intermediate_stage2_new.ckpt'))
    ms.load_param_into_net(stage3,
                           ms.load_checkpoint('model_tnt_intermediate_stage3_new.ckpt'))
    net = ISMVSNet(feat_ext=feat_ext, stage1=stage1, stage2=stage2, stage3=stage3)
    est_depth_3, prob_map_3, est_variance_1, est_variance_2 = net(imgs, proj_mats, init_depth_min, depth_interval)


if __name__ == "__main__":
    main()
