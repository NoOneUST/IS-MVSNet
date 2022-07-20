import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module7(nn.Cell):
    def __init__(self, reducesum_1_keep_dims, reducesum_1_axis):
        super(Module7, self).__init__()
        self.reducesum_1 = P.ReduceSum(keep_dims=reducesum_1_keep_dims)
        self.reducesum_1_axis = reducesum_1_axis

    def construct(self, x, x0):
        opt_mul_0 = P.Mul()(x, x0)
        opt_reducesum_1 = self.reducesum_1(opt_mul_0, self.reducesum_1_axis)
        return opt_reducesum_1


class Module47(nn.Cell):
    def __init__(self, gather_0_weight_value, module7_0_reducesum_1_keep_dims, module7_0_reducesum_1_axis):
        super(Module47, self).__init__()
        self.gather_0_input_weight = Tensor(np.array(gather_0_weight_value))
        self.gather_0_axis = 0
        self.gather_0 = P.Gather()
        self.reshape_1 = P.Reshape()
        self.reshape_1_shape = tuple([1, 8, 4, 32, 132, 240])
        self.reshape_2 = P.Reshape()
        self.reshape_2_shape = tuple([1, 8, 4, 32, 132, 240])
        self.module7_0 = Module7(reducesum_1_keep_dims=module7_0_reducesum_1_keep_dims,
                                 reducesum_1_axis=module7_0_reducesum_1_axis)

    def construct(self, x, x0):
        opt_gather_0_axis = self.gather_0_axis
        opt_gather_0 = self.gather_0(x, self.gather_0_input_weight, opt_gather_0_axis)
        opt_reshape_1 = self.reshape_1(x0, self.reshape_1_shape)
        opt_reshape_2 = self.reshape_2(opt_gather_0, self.reshape_2_shape)
        module7_0_opt = self.module7_0(opt_reshape_1, opt_reshape_2)
        return module7_0_opt


class Module0(nn.Cell):
    def __init__(self):
        super(Module0, self).__init__()
        self.conv3d_0 = nn.Conv3d(in_channels=8,
                                  out_channels=8,
                                  kernel_size=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(1, 1, 1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv3d_2 = nn.Conv3d(in_channels=8,
                                  out_channels=8,
                                  kernel_size=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(1, 1, 1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_4 = nn.ReLU()
        self.conv3d_5 = nn.Conv3d(in_channels=8,
                                  out_channels=16,
                                  kernel_size=(3, 3, 3),
                                  stride=(2, 2, 2),
                                  padding=(1, 1, 1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=True)
        self.conv3d_6 = nn.Conv3d(in_channels=8,
                                  out_channels=16,
                                  kernel_size=(1, 1, 1),
                                  stride=(2, 2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()
        self.conv3d_8 = nn.Conv3d(in_channels=16,
                                  out_channels=16,
                                  kernel_size=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(1, 1, 1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_10 = nn.ReLU()
        self.conv3dtranspose_11 = nn.Conv3dTranspose(in_channels=16,
                                                     out_channels=8,
                                                     kernel_size=(3, 3, 3),
                                                     stride=(2, 2, 2),
                                                     padding=(1, 1, 1, 1, 1, 1),
                                                     pad_mode="pad",
                                                     output_padding=(1, 1, 1),
                                                     dilation=(1, 1, 1),
                                                     group=1,
                                                     has_bias=False)
        self.concat_12 = P.Concat(axis=1)

    def construct(self, x):
        opt_conv3d_0 = self.conv3d_0(x)
        opt_relu_1 = self.relu_1(opt_conv3d_0)
        opt_conv3d_2 = self.conv3d_2(opt_relu_1)
        opt_add_3 = P.Add()(opt_conv3d_2, x)
        opt_relu_4 = self.relu_4(opt_add_3)
        opt_conv3d_5 = self.conv3d_5(opt_relu_4)
        opt_conv3d_6 = self.conv3d_6(opt_relu_4)
        opt_relu_7 = self.relu_7(opt_conv3d_5)
        opt_conv3d_8 = self.conv3d_8(opt_relu_7)
        opt_add_9 = P.Add()(opt_conv3d_8, opt_conv3d_6)
        opt_relu_10 = self.relu_10(opt_add_9)
        opt_conv3dtranspose_11 = self.conv3dtranspose_11(opt_relu_10)
        opt_concat_12 = self.concat_12((opt_conv3dtranspose_11, opt_relu_4, ))
        return opt_concat_12


class Module17(nn.Cell):
    def __init__(self):
        super(Module17, self).__init__()
        self.module0_0 = Module0()
        self.conv3d_0 = nn.Conv3d(in_channels=16,
                                  out_channels=8,
                                  kernel_size=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(1, 1, 1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=False)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv3d_0 = self.conv3d_0(module0_0_opt)
        return opt_conv3d_0


class Module22(nn.Cell):
    def __init__(self):
        super(Module22, self).__init__()
        self.conv3d_0 = nn.Conv3d(in_channels=8,
                                  out_channels=1,
                                  kernel_size=(3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(1, 1, 1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1, 1),
                                  group=1,
                                  has_bias=False)
        self.squeeze_1 = P.Squeeze(axis=1)

    def construct(self, x):
        opt_conv3d_0 = self.conv3d_0(x)
        opt_squeeze_1 = self.squeeze_1(opt_conv3d_0)
        return opt_squeeze_1


class Module60(nn.Cell):
    def __init__(self):
        super(Module60, self).__init__()
        self.module22_0 = Module22()
        self.transpose_0 = P.Transpose()
        self.softmax_1 = nn.Softmax(axis=3)
        self.transpose_2 = P.Transpose()

    def construct(self, x):
        module22_0_opt = self.module22_0(x)
        opt_transpose_0 = self.transpose_0(module22_0_opt, (0, 3, 2, 1))
        opt_softmax_1 = self.softmax_1(opt_transpose_0)
        opt_transpose_2 = self.transpose_2(opt_softmax_1, (0, 3, 2, 1))
        return opt_transpose_2


class Module11(nn.Cell):
    def __init__(self):
        super(Module11, self).__init__()
        self.reducesum_1 = P.ReduceSum(keep_dims=True)
        self.reducesum_1_axis = 1

    def construct(self, x, x0):
        opt_mul_0 = P.Mul()(x, x0)
        opt_reducesum_1 = self.reducesum_1(opt_mul_0, self.reducesum_1_axis)
        return opt_reducesum_1


class Module68(nn.Cell):
    def __init__(self):
        super(Module68, self).__init__()
        self.module11_0 = Module11()
        self.cast_0 = P.Cast()
        self.cast_0_to = mindspore.float32

    def construct(self, x, x0):
        module11_0_opt = self.module11_0(x, x0)
        opt_cast_0 = self.cast_0(module11_0_opt, self.cast_0_to)
        return opt_cast_0


class Module74(nn.Cell):
    def __init__(self, module7_0_reducesum_1_keep_dims, module7_0_reducesum_1_axis):
        super(Module74, self).__init__()
        self.module7_0 = Module7(reducesum_1_keep_dims=module7_0_reducesum_1_keep_dims,
                                 reducesum_1_axis=module7_0_reducesum_1_axis)
        self.conv2d_1 = nn.Conv2d(in_channels=1,
                                  out_channels=8,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=8,
                                  out_channels=8,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_4 = nn.ReLU()

    def construct(self, x, x0):
        opt_log_0 = P.Log()(x)
        module7_0_opt = self.module7_0(x0, opt_log_0)
        opt_conv2d_1 = self.conv2d_1(module7_0_opt)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_add_5 = P.Add()(opt_relu_4, module7_0_opt)
        return opt_add_5


class Module79(nn.Cell):
    def __init__(self):
        super(Module79, self).__init__()
        self.neg_0 = P.Neg()

    def construct(self, x):
        opt_neg_0 = self.neg_0(x)
        opt_exp_1 = P.Exp()(opt_neg_0)
        return opt_exp_1


class Module83(nn.Cell):
    def __init__(self):
        super(Module83, self).__init__()
        self.module79_0 = Module79()
        self.expanddims_0 = P.ExpandDims()
        self.expanddims_0_axis = 2

    def construct(self, x):
        module79_0_opt = self.module79_0(x)
        opt_expanddims_0 = self.expanddims_0(module79_0_opt, self.expanddims_0_axis)
        return opt_expanddims_0


class Module86(nn.Cell):
    def __init__(self):
        super(Module86, self).__init__()

    def construct(self, x, x0, x1):
        opt_mul_0 = P.Mul()(x, x0)
        opt_add_1 = P.Add()(x1, opt_mul_0)
        return opt_add_1


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.module47_0 = Module47(gather_0_weight_value=0,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_0 = Module17()
        self.module60_0 = Module60()
        self.module68_0 = Module68()
        self.neg_169 = P.Neg()
        self.clip_by_value_170_min = 9.999999717180685e-10
        self.clip_by_value_170_max = 1.0
        self.module74_0 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_267 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_268 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_0 = Module83()
        self.add_318_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 132, 240)).astype(np.float32)),
                                      name=None)
        self.add_327_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 8, 32, 132, 240)).astype(np.float32)),
                                      name=None)
        self.module47_1 = Module47(gather_0_weight_value=1,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_1 = Module17()
        self.module60_1 = Module60()
        self.module68_1 = Module68()
        self.neg_172 = P.Neg()
        self.clip_by_value_173_min = 9.999999717180685e-10
        self.clip_by_value_173_max = 1.0
        self.module74_1 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_269 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_270 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_1 = Module83()
        self.module86_0 = Module86()
        self.module47_2 = Module47(gather_0_weight_value=2,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_2 = Module17()
        self.module60_2 = Module60()
        self.module68_2 = Module68()
        self.neg_175 = P.Neg()
        self.clip_by_value_176_min = 9.999999717180685e-10
        self.clip_by_value_176_max = 1.0
        self.module74_2 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_271 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_272 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_2 = Module83()
        self.module86_1 = Module86()
        self.module47_3 = Module47(gather_0_weight_value=3,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_3 = Module17()
        self.module60_3 = Module60()
        self.module68_3 = Module68()
        self.neg_178 = P.Neg()
        self.clip_by_value_179_min = 9.999999717180685e-10
        self.clip_by_value_179_max = 1.0
        self.module74_3 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_273 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_274 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_3 = Module83()
        self.module86_2 = Module86()
        self.module47_4 = Module47(gather_0_weight_value=4,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_4 = Module17()
        self.module60_4 = Module60()
        self.module68_4 = Module68()
        self.neg_181 = P.Neg()
        self.clip_by_value_182_min = 9.999999717180685e-10
        self.clip_by_value_182_max = 1.0
        self.module74_4 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_275 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_276 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_4 = Module83()
        self.module86_3 = Module86()
        self.module47_5 = Module47(gather_0_weight_value=5,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_5 = Module17()
        self.module60_5 = Module60()
        self.module68_5 = Module68()
        self.neg_184 = P.Neg()
        self.clip_by_value_185_min = 9.999999717180685e-10
        self.clip_by_value_185_max = 1.0
        self.module74_5 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_277 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_278 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_5 = Module83()
        self.module86_4 = Module86()
        self.module47_6 = Module47(gather_0_weight_value=6,
                                   module7_0_reducesum_1_keep_dims=False,
                                   module7_0_reducesum_1_axis=2)
        self.module17_6 = Module17()
        self.module60_6 = Module60()
        self.module68_6 = Module68()
        self.neg_187 = P.Neg()
        self.clip_by_value_188_min = 9.999999717180685e-10
        self.clip_by_value_188_max = 1.0
        self.module74_6 = Module74(module7_0_reducesum_1_keep_dims=True, module7_0_reducesum_1_axis=1)
        self.conv2d_279 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_280 = nn.Conv2d(in_channels=8,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.module83_6 = Module83()
        self.module86_5 = Module86()
        self.expanddims_217 = P.ExpandDims()
        self.expanddims_217_axis = 0
        self.expanddims_219 = P.ExpandDims()
        self.expanddims_219_axis = 0
        self.expanddims_221 = P.ExpandDims()
        self.expanddims_221_axis = 0
        self.expanddims_223 = P.ExpandDims()
        self.expanddims_223_axis = 0
        self.expanddims_225 = P.ExpandDims()
        self.expanddims_225_axis = 0
        self.expanddims_227 = P.ExpandDims()
        self.expanddims_227_axis = 0
        self.expanddims_229 = P.ExpandDims()
        self.expanddims_229_axis = 0
        self.concat_237 = P.Concat(axis=0)
        self.expanddims_282 = P.ExpandDims()
        self.expanddims_282_axis = 0
        self.expanddims_285 = P.ExpandDims()
        self.expanddims_285_axis = 0
        self.expanddims_288 = P.ExpandDims()
        self.expanddims_288_axis = 0
        self.expanddims_291 = P.ExpandDims()
        self.expanddims_291_axis = 0
        self.expanddims_294 = P.ExpandDims()
        self.expanddims_294_axis = 0
        self.expanddims_297 = P.ExpandDims()
        self.expanddims_297_axis = 0
        self.expanddims_300 = P.ExpandDims()
        self.expanddims_300_axis = 0
        self.concat_309 = P.Concat(axis=0)
        self.expanddims_283 = P.ExpandDims()
        self.expanddims_283_axis = 0
        self.expanddims_286 = P.ExpandDims()
        self.expanddims_286_axis = 0
        self.expanddims_289 = P.ExpandDims()
        self.expanddims_289_axis = 0
        self.expanddims_292 = P.ExpandDims()
        self.expanddims_292_axis = 0
        self.expanddims_295 = P.ExpandDims()
        self.expanddims_295_axis = 0
        self.expanddims_298 = P.ExpandDims()
        self.expanddims_298_axis = 0
        self.expanddims_301 = P.ExpandDims()
        self.expanddims_301_axis = 0
        self.concat_310 = P.Concat(axis=0)
        self.module17_7 = Module17()
        self.module60_7 = Module60()
        self.module11_0 = Module11()
        self.sub_363_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 32, 1, 1)).astype(np.float32)), name=None)
        self.greater_366_cmp_to = 2.0
        self.cast_368 = P.Cast()
        self.cast_368_to = mindspore.float32
        self.reducesum_370 = P.ReduceSum(keep_dims=True)
        self.reducesum_370_axis = 1
        self.module11_1 = Module11()
        self.cast_364 = P.Cast()
        self.cast_364_to = mindspore.float32

    def construct(self, ref_ncdhw, warped_srcs, x1):
        module47_0_opt = self.module47_0(warped_srcs, ref_ncdhw)
        module17_0_opt = self.module17_0(module47_0_opt)
        module60_0_opt = self.module60_0(module17_0_opt)
        module68_0_opt = self.module68_0(module60_0_opt, x1)
        opt_neg_169 = self.neg_169(module60_0_opt)
        opt_clip_by_value_170 = P.clip_by_value(module60_0_opt, self.clip_by_value_170_min, self.clip_by_value_170_max)
        module74_0_opt = self.module74_0(opt_clip_by_value_170, opt_neg_169)
        opt_conv2d_267 = self.conv2d_267(module74_0_opt)
        opt_conv2d_268 = self.conv2d_268(module74_0_opt)
        module83_0_opt = self.module83_0(opt_conv2d_267)
        opt_add_318 = self.add_318_bias + module83_0_opt
        opt_mul_319 = P.Mul()(module17_0_opt, module83_0_opt)
        opt_add_327 = self.add_327_bias + opt_mul_319
        module47_1_opt = self.module47_1(warped_srcs, ref_ncdhw)
        module17_1_opt = self.module17_1(module47_1_opt)
        module60_1_opt = self.module60_1(module17_1_opt)
        module68_1_opt = self.module68_1(module60_1_opt, x1)
        opt_neg_172 = self.neg_172(module60_1_opt)
        opt_clip_by_value_173 = P.clip_by_value(module60_1_opt, self.clip_by_value_173_min, self.clip_by_value_173_max)
        module74_1_opt = self.module74_1(opt_clip_by_value_173, opt_neg_172)
        opt_conv2d_269 = self.conv2d_269(module74_1_opt)
        opt_conv2d_270 = self.conv2d_270(module74_1_opt)
        module83_1_opt = self.module83_1(opt_conv2d_269)
        opt_add_326 = P.Add()(opt_add_318, module83_1_opt)
        module86_0_opt = self.module86_0(module17_1_opt, module83_1_opt, opt_add_327)
        module47_2_opt = self.module47_2(warped_srcs, ref_ncdhw)
        module17_2_opt = self.module17_2(module47_2_opt)
        module60_2_opt = self.module60_2(module17_2_opt)
        module68_2_opt = self.module68_2(module60_2_opt, x1)
        opt_neg_175 = self.neg_175(module60_2_opt)
        opt_clip_by_value_176 = P.clip_by_value(module60_2_opt, self.clip_by_value_176_min, self.clip_by_value_176_max)
        module74_2_opt = self.module74_2(opt_clip_by_value_176, opt_neg_175)
        opt_conv2d_271 = self.conv2d_271(module74_2_opt)
        opt_conv2d_272 = self.conv2d_272(module74_2_opt)
        module83_2_opt = self.module83_2(opt_conv2d_271)
        opt_add_328 = P.Add()(opt_add_326, module83_2_opt)
        module86_1_opt = self.module86_1(module17_2_opt, module83_2_opt, module86_0_opt)
        module47_3_opt = self.module47_3(warped_srcs, ref_ncdhw)
        module17_3_opt = self.module17_3(module47_3_opt)
        module60_3_opt = self.module60_3(module17_3_opt)
        module68_3_opt = self.module68_3(module60_3_opt, x1)
        opt_neg_178 = self.neg_178(module60_3_opt)
        opt_clip_by_value_179 = P.clip_by_value(module60_3_opt, self.clip_by_value_179_min, self.clip_by_value_179_max)
        module74_3_opt = self.module74_3(opt_clip_by_value_179, opt_neg_178)
        opt_conv2d_273 = self.conv2d_273(module74_3_opt)
        opt_conv2d_274 = self.conv2d_274(module74_3_opt)
        module83_3_opt = self.module83_3(opt_conv2d_273)
        opt_add_330 = P.Add()(opt_add_328, module83_3_opt)
        module86_2_opt = self.module86_2(module17_3_opt, module83_3_opt, module86_1_opt)
        module47_4_opt = self.module47_4(warped_srcs, ref_ncdhw)
        module17_4_opt = self.module17_4(module47_4_opt)
        module60_4_opt = self.module60_4(module17_4_opt)
        module68_4_opt = self.module68_4(module60_4_opt, x1)
        opt_neg_181 = self.neg_181(module60_4_opt)
        opt_clip_by_value_182 = P.clip_by_value(module60_4_opt, self.clip_by_value_182_min, self.clip_by_value_182_max)
        module74_4_opt = self.module74_4(opt_clip_by_value_182, opt_neg_181)
        opt_conv2d_275 = self.conv2d_275(module74_4_opt)
        opt_conv2d_276 = self.conv2d_276(module74_4_opt)
        module83_4_opt = self.module83_4(opt_conv2d_275)
        opt_add_332 = P.Add()(opt_add_330, module83_4_opt)
        module86_3_opt = self.module86_3(module17_4_opt, module83_4_opt, module86_2_opt)
        module47_5_opt = self.module47_5(warped_srcs, ref_ncdhw)
        module17_5_opt = self.module17_5(module47_5_opt)
        module60_5_opt = self.module60_5(module17_5_opt)
        module68_5_opt = self.module68_5(module60_5_opt, x1)
        opt_neg_184 = self.neg_184(module60_5_opt)
        opt_clip_by_value_185 = P.clip_by_value(module60_5_opt, self.clip_by_value_185_min, self.clip_by_value_185_max)
        module74_5_opt = self.module74_5(opt_clip_by_value_185, opt_neg_184)
        opt_conv2d_277 = self.conv2d_277(module74_5_opt)
        opt_conv2d_278 = self.conv2d_278(module74_5_opt)
        module83_5_opt = self.module83_5(opt_conv2d_277)
        opt_add_334 = P.Add()(opt_add_332, module83_5_opt)
        module86_4_opt = self.module86_4(module17_5_opt, module83_5_opt, module86_3_opt)
        module47_6_opt = self.module47_6(warped_srcs, ref_ncdhw)
        module17_6_opt = self.module17_6(module47_6_opt)
        module60_6_opt = self.module60_6(module17_6_opt)
        module68_6_opt = self.module68_6(module60_6_opt, x1)
        opt_neg_187 = self.neg_187(module60_6_opt)
        opt_clip_by_value_188 = P.clip_by_value(module60_6_opt, self.clip_by_value_188_min, self.clip_by_value_188_max)
        module74_6_opt = self.module74_6(opt_clip_by_value_188, opt_neg_187)
        opt_conv2d_279 = self.conv2d_279(module74_6_opt)
        opt_conv2d_280 = self.conv2d_280(module74_6_opt)
        module83_6_opt = self.module83_6(opt_conv2d_279)
        opt_add_336 = P.Add()(opt_add_334, module83_6_opt)
        module86_5_opt = self.module86_5(module17_6_opt, module83_6_opt, module86_4_opt)
        opt_expanddims_217 = self.expanddims_217(module68_0_opt, self.expanddims_217_axis)
        opt_expanddims_219 = self.expanddims_219(module68_1_opt, self.expanddims_219_axis)
        opt_expanddims_221 = self.expanddims_221(module68_2_opt, self.expanddims_221_axis)
        opt_expanddims_223 = self.expanddims_223(module68_3_opt, self.expanddims_223_axis)
        opt_expanddims_225 = self.expanddims_225(module68_4_opt, self.expanddims_225_axis)
        opt_expanddims_227 = self.expanddims_227(module68_5_opt, self.expanddims_227_axis)
        opt_expanddims_229 = self.expanddims_229(module68_6_opt, self.expanddims_229_axis)
        opt_concat_237 = self.concat_237(
            (opt_expanddims_217, opt_expanddims_219, opt_expanddims_221, opt_expanddims_223, opt_expanddims_225,
             opt_expanddims_227, opt_expanddims_229,
             ))
        opt_expanddims_282 = self.expanddims_282(opt_conv2d_267, self.expanddims_282_axis)
        opt_expanddims_285 = self.expanddims_285(opt_conv2d_269, self.expanddims_285_axis)
        opt_expanddims_288 = self.expanddims_288(opt_conv2d_271, self.expanddims_288_axis)
        opt_expanddims_291 = self.expanddims_291(opt_conv2d_273, self.expanddims_291_axis)
        opt_expanddims_294 = self.expanddims_294(opt_conv2d_275, self.expanddims_294_axis)
        opt_expanddims_297 = self.expanddims_297(opt_conv2d_277, self.expanddims_297_axis)
        opt_expanddims_300 = self.expanddims_300(opt_conv2d_279, self.expanddims_300_axis)
        opt_concat_309 = self.concat_309(
            (opt_expanddims_282, opt_expanddims_285, opt_expanddims_288, opt_expanddims_291, opt_expanddims_294,
             opt_expanddims_297, opt_expanddims_300,
             ))
        opt_expanddims_283 = self.expanddims_283(opt_conv2d_268, self.expanddims_283_axis)
        opt_expanddims_286 = self.expanddims_286(opt_conv2d_270, self.expanddims_286_axis)
        opt_expanddims_289 = self.expanddims_289(opt_conv2d_272, self.expanddims_289_axis)
        opt_expanddims_292 = self.expanddims_292(opt_conv2d_274, self.expanddims_292_axis)
        opt_expanddims_295 = self.expanddims_295(opt_conv2d_276, self.expanddims_295_axis)
        opt_expanddims_298 = self.expanddims_298(opt_conv2d_278, self.expanddims_298_axis)
        opt_expanddims_301 = self.expanddims_301(opt_conv2d_280, self.expanddims_301_axis)
        opt_concat_310 = self.concat_310(
            (opt_expanddims_283, opt_expanddims_286, opt_expanddims_289, opt_expanddims_292, opt_expanddims_295,
             opt_expanddims_298, opt_expanddims_301,
             ))
        opt_div_339 = P.Div()(module86_5_opt, opt_add_336)
        module17_7_opt = self.module17_7(opt_div_339)
        module60_7_opt = self.module60_7(module17_7_opt)
        module11_0_opt = self.module11_0(module60_7_opt, self.sub_363_bias)
        opt_sub_363 = self.sub_363_bias - module11_0_opt
        opt_abs_365 = P.Abs()(opt_sub_363)
        opt_greater_366 = P.Greater()(opt_abs_365, self.greater_366_cmp_to)
        opt_logicalnot_367 = P.LogicalNot()(opt_greater_366)
        opt_cast_368 = self.cast_368(opt_logicalnot_367, self.cast_368_to)
        opt_mul_369 = P.Mul()(module60_7_opt, opt_cast_368)
        opt_reducesum_370 = self.reducesum_370(opt_mul_369, self.reducesum_370_axis)
        module11_1_opt = self.module11_1(module60_7_opt, x1)
        opt_cast_364 = self.cast_364(module11_1_opt, self.cast_364_to)
        return opt_concat_237, opt_concat_309, opt_concat_310, opt_cast_364, opt_reducesum_370
