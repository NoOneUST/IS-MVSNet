import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module6(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_stride):
        super(Module6, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=conv2d_0_stride,
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module7(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module6_0_conv2d_0_in_channels,
                 module6_0_conv2d_0_out_channels, module6_0_conv2d_0_stride):
        super(Module7, self).__init__()
        self.module6_0 = Module6(conv2d_0_in_channels=module6_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module6_0_conv2d_0_out_channels,
                                 conv2d_0_stride=module6_0_conv2d_0_stride)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module6_0_opt = self.module6_0(x)
        opt_conv2d_0 = self.conv2d_0(module6_0_opt)
        return opt_conv2d_0


class Module8(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module6_0_conv2d_0_in_channels,
                 module6_0_conv2d_0_out_channels, module6_0_conv2d_0_stride):
        super(Module8, self).__init__()
        self.module6_0 = Module6(conv2d_0_in_channels=module6_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module6_0_conv2d_0_out_channels,
                                 conv2d_0_stride=module6_0_conv2d_0_stride)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()

    def construct(self, x):
        module6_0_opt = self.module6_0(x)
        opt_conv2d_0 = self.conv2d_0(module6_0_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, x)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_1_in_channels, conv2d_1_out_channels,
                 conv2d_3_in_channels, conv2d_3_out_channels):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_conv2d_1 = self.conv2d_1(opt_conv2d_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_add_4 = P.Add()(opt_conv2d_3, opt_conv2d_0)
        opt_relu_5 = self.relu_5(opt_add_4)
        return opt_relu_5


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=16,
                                  kernel_size=(5, 5),
                                  stride=(2, 2),
                                  padding=(2, 2, 2, 2),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module7_0 = Module7(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=32,
                                 module6_0_conv2d_0_in_channels=16,
                                 module6_0_conv2d_0_out_channels=32,
                                 module6_0_conv2d_0_stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(in_channels=16,
                                  out_channels=32,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()
        self.module8_0 = Module8(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=32,
                                 module6_0_conv2d_0_in_channels=32,
                                 module6_0_conv2d_0_out_channels=32,
                                 module6_0_conv2d_0_stride=(1, 1))
        self.module7_1 = Module7(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 module6_0_conv2d_0_in_channels=32,
                                 module6_0_conv2d_0_out_channels=64,
                                 module6_0_conv2d_0_stride=(2, 2))
        self.conv2d_14 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_18 = nn.ReLU()
        self.module8_1 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 module6_0_conv2d_0_in_channels=64,
                                 module6_0_conv2d_0_out_channels=64,
                                 module6_0_conv2d_0_stride=(1, 1))
        self.module7_2 = Module7(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 module6_0_conv2d_0_in_channels=64,
                                 module6_0_conv2d_0_out_channels=128,
                                 module6_0_conv2d_0_stride=(2, 2))
        self.conv2d_25 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_29 = nn.ReLU()
        self.module8_2 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 module6_0_conv2d_0_in_channels=128,
                                 module6_0_conv2d_0_out_channels=128,
                                 module6_0_conv2d_0_stride=(1, 1))
        self.conv2dbackpropinput_35 = P.Conv2DBackpropInput(out_channel=64,
                                                            kernel_size=(3, 3),
                                                            stride=(2, 2),
                                                            pad=(1, 1, 1, 1),
                                                            pad_mode="pad",
                                                            dilation=(1, 1),
                                                            group=1)
        self.conv2dbackpropinput_35_weight = Parameter(Tensor(
            np.random.uniform(0, 1, (128, 64, 3, 3)).astype(np.float32)),
                                                       name=None)
        self.conv2dbackpropinput_35_out_channel = 64
        self.conv2dbackpropinput_35_h_out = 264
        self.conv2dbackpropinput_35_w_out = 480
        self.concat_37 = P.Concat(axis=1)
        self.module1_0 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=64,
                                 conv2d_1_in_channels=64,
                                 conv2d_1_out_channels=64,
                                 conv2d_3_in_channels=64,
                                 conv2d_3_out_channels=64)
        self.conv2dbackpropinput_44 = P.Conv2DBackpropInput(out_channel=32,
                                                            kernel_size=(3, 3),
                                                            stride=(2, 2),
                                                            pad=(1, 1, 1, 1),
                                                            pad_mode="pad",
                                                            dilation=(1, 1),
                                                            group=1)
        self.conv2dbackpropinput_44_weight = Parameter(Tensor(
            np.random.uniform(0, 1, (64, 32, 3, 3)).astype(np.float32)),
                                                       name=None)
        self.conv2dbackpropinput_44_out_channel = 32
        self.conv2dbackpropinput_44_h_out = 528
        self.conv2dbackpropinput_44_w_out = 960
        self.concat_46 = P.Concat(axis=1)
        self.module1_1 = Module1(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=32,
                                 conv2d_1_in_channels=32,
                                 conv2d_1_out_channels=32,
                                 conv2d_3_in_channels=32,
                                 conv2d_3_out_channels=32)
        self.conv2d_36 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.conv2d_45 = nn.Conv2d(in_channels=64,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.conv2d_53 = nn.Conv2d(in_channels=32,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)

    def construct(self, imgs):
        opt_conv2d_0 = self.conv2d_0(imgs)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module7_0_opt = self.module7_0(opt_relu_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_1)
        opt_add_6 = P.Add()(module7_0_opt, opt_conv2d_3)
        opt_relu_7 = self.relu_7(opt_add_6)
        module8_0_opt = self.module8_0(opt_relu_7)
        module7_1_opt = self.module7_1(module8_0_opt)
        opt_conv2d_14 = self.conv2d_14(module8_0_opt)
        opt_add_17 = P.Add()(module7_1_opt, opt_conv2d_14)
        opt_relu_18 = self.relu_18(opt_add_17)
        module8_1_opt = self.module8_1(opt_relu_18)
        module7_2_opt = self.module7_2(module8_1_opt)
        opt_conv2d_25 = self.conv2d_25(module8_1_opt)
        opt_add_28 = P.Add()(module7_2_opt, opt_conv2d_25)
        opt_relu_29 = self.relu_29(opt_add_28)
        module8_2_opt = self.module8_2(opt_relu_29)
        opt_conv2dbackpropinput_35 = self.conv2dbackpropinput_35(
            module8_2_opt, self.conv2dbackpropinput_35_weight,
            (module8_2_opt.shape[0], self.conv2dbackpropinput_35_out_channel, self.conv2dbackpropinput_35_h_out,
             self.conv2dbackpropinput_35_w_out))
        opt_concat_37 = self.concat_37((opt_conv2dbackpropinput_35, module8_1_opt, ))
        module1_0_opt = self.module1_0(opt_concat_37)
        opt_conv2dbackpropinput_44 = self.conv2dbackpropinput_44(
            module1_0_opt, self.conv2dbackpropinput_44_weight,
            (module1_0_opt.shape[0], self.conv2dbackpropinput_44_out_channel, self.conv2dbackpropinput_44_h_out,
             self.conv2dbackpropinput_44_w_out))
        opt_concat_46 = self.concat_46((opt_conv2dbackpropinput_44, module8_0_opt, ))
        module1_1_opt = self.module1_1(opt_concat_46)
        opt_conv2d_36 = self.conv2d_36(module8_2_opt)
        opt_conv2d_45 = self.conv2d_45(module1_0_opt)
        opt_conv2d_53 = self.conv2d_53(module1_1_opt)
        return opt_conv2d_36, opt_conv2d_45, opt_conv2d_53
