import torch
import torch.nn as nn
import torch.nn.functional as F
#import FReLU
#from dyrelu import DyReluB

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


class MSDFN(nn.Module):
    def __init__(self):
        super(MSDFN, self).__init__()
        self.relu64 = DyReLUB(64, conv_type='2d')
        self.relu128 = DyReLUB(128, conv_type='2d')
        self.relu256 = DyReLUB(256, conv_type='2d')
        self.relu512 = DyReLUB(512, conv_type='2d')
        self.relu1024 = DyReLUB(1024, conv_type='2d')
        self.relu3 = DyReLUB(3,reduction=1,conv_type='2d')
        #TODO: ---------------------------------------对深度图像进行降采样（maxpooling），供上卷积使用，以下定义所需计算单元--------------------------------------------------------------------
        self.conv1_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
        )
        self.pool1_f = nn.MaxPool2d(2, stride=2, padding=0 )

        self.conv3_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(64, 128, 3, 1, 1),                                                                                # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.pool2_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv5_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(128, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )

        self.pool3_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv7_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(256, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.pool4_f = nn.MaxPool2d(2, stride=2, padding=0)
        # TODO: ---------------------------------------对输入彩色图像进行降采样（maxpooling），供下卷积提取特征使用，以下定义所需计算单元--------------------------------------------------------
        self.conv1 = nn.Sequential(                                                                                     # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
            )
        self.conv2 = nn.Sequential(                                                                                     # conv1入，64进64出,---> pool1
            nn.Conv2d(64, 64, 3, 1, 1),                                                                                 # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
        )
        self.pool1 = nn.MaxPool2d(2,stride=2,padding=0 )                                                                # 原图入，1进1出,---> concate_input1   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool1_input = nn.MaxPool2d(2,stride=2,padding=0 )                                                          # conv_2入，1进64出,---> conv_input1
        self.conv_input1 = nn.Sequential(                                                                               # pool_input入，1进64出,---> concate_input1
            nn.Conv2d(3, 64, 3, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        #TODO: concate_input1 -----------------------------------------channel --> 128--------------------------------------
        self.conv3 = nn.Sequential(                                                                                     # concat入，---> conv4
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.conv4 = nn.Sequential(                                                                                     # conv1入，--> pool2
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv4入 ---> concate_input2   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool2_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input入 ---> conv_input2
        self.conv_input2 = nn.Sequential(                                                                               # pool_input2入，1进64出,---> concate_input2
            nn.Conv2d(3, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        # TODO: concate_input2 ----------------------------------------channel-->256----------------------------------------
        self.conv5 = nn.Sequential(                                                                                     # concate_input2入，---> conv5
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )
        self.conv6 = nn.Sequential(                                                                                     # conv1入，---> pool3
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv6入，---> concate_input3   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool3_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input2入，,---> conv_input3
        self.conv_input3 = nn.Sequential(                                                                               # pool3_input入，---> concate_input3
            nn.Conv2d(3, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        # TODO: concate_input3 ----------------------------------------channel-->512----------------------------------------
        self.conv7 = nn.Sequential(                                                                                     # concate_input3入，--->conv6
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.conv8 = nn.Sequential(                                                                                     # conv7入, ---> pool4
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv8入，1进1出,---> concate_input4   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool4_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # ool_input3入，1进1出,---> conv_input4
        self.conv_input4 = nn.Sequential(                                                                               # pool4_input入，1进64出,---> concate_input4
            nn.Conv2d(3, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        # TODO:concate_input4 ----------------------------------------channel-->1024----------------------------------------

        self.conv9 = nn.Sequential(                                                                                     # concat4入，1024进1024出,---> conv10
            nn.Conv2d(1024, 1024, 3, 1, 1),
            #DyReLUB(1024, conv_type='2d')
        )
        self.conv10 = nn.Sequential(                                                                                    # conv9t入，1024进1024出,---> concate_input2
            nn.Conv2d(1024, 1024, 3, 1, 1),
            #DyReLUB(1024, conv_type='2d')
        )

        # TODO: ---------------------------------------decoder上卷积，以下定义所需计算单元--------------------------------------------------------
        #上卷积+concat+relu（卷积）+relu(卷积)
        self.deconv1 =nn.Sequential(
            nn.ConvTranspose2d(1024,512,2,2,0),
            #DyReLUB(512, conv_type='2d')
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512+512+512, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        #---------------------------------------------------------------------------------------------------------------
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2, 0),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256+256+256, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(128 + 128 + 128, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64+64+64, 64, 3, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.conv19= nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            #DyReLUB(3, conv_type='2d')
        )
        #----------------------------------------------------------------------------------------------------
        self.conv1_input = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv2_input = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv3_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv4_input = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1),
            #DyReLUB(512,conv_type='2d')
        )
    def forward(self,image,depth):
        """
        具体计算过程：
        编码时：有雾彩色图像作主线，深度图作副线
        解码时：解码为主线，编码和有雾图作副线
        计算结果：直接生成图像
        :param depth:深度图
        :param image:彩色有雾图像
        :return:
        """
        #----------彩色图下卷积---------------------------------------------------------------------------------------------
        conv1_f = self.relu64(self.conv1_f(image))
        pool1_f = self.pool1_f(conv1_f)

        conv3_f = self.relu128(self.conv3_f(pool1_f))
        pool2_f = self.pool2_f(conv3_f)

        conv5_f = self.relu256(self.conv5_f(pool2_f))
        pool3_f = self.pool3_f(conv5_f)

        conv7_f = self.relu512(self.conv7_f(pool3_f))
        pool4_f = self.pool4_f(conv7_f)
        #------------深度图下卷积----------------------------------------------------------------------------------------------
        # conv1_input = self.relu64(self.conv1_input(depth))
        # pool1_input = self.pool2_input(conv1_input)
        #
        # conv2_input = self.relu128(self.conv2_input(pool1_input))
        # pool2_input = self.pool2_input(conv2_input)
        #
        # conv3_input = self.relu256(self.conv3_input(pool2_input))
        # pool3_input = self.pool3_input(conv3_input)
        #
        # conv4_input = self.relu512(self.conv4_input(pool3_input))

        #------------编码主线-----------------------------------------------------------------------------------------------------
        conv1 = self.relu64(self.conv1(depth))
        conv2 = self.relu64(self.conv2(conv1))
        pool1 = self.pool1(conv2)

        #pool1_input = self.pool2_input(image)

        concate_input1 = torch.cat((pool1, pool1_f), 1)
        #--------------------------128------------------------
        conv3 = self.relu128(self.conv3(concate_input1))
        conv4 = self.relu128(self.conv4(conv3))
        pool2 = self.pool2(conv4)

        #pool2_input = self.pool2_input(pool1_input)

        concate_input2 = torch.cat((pool2, pool2_f), 1)
        #--------------------------256------------------------
        conv5 = self.relu256(self.conv5(concate_input2))
        conv6 = self.relu256(self.conv6(conv5))
        pool3 = self.pool3(conv6)



        concate_input3 = torch.cat((pool3, pool3_f), 1)
        # --------------------------512------------------------
        conv7 = self.relu512(self.conv7(concate_input3))
        conv8 = self.relu512(self.conv8(conv7))
        pool4 = self.pool4(conv8)

        #pool4_input = self.pool3_input(pool3_input)

        concate_input4 = torch.cat((pool4, pool4_f), 1)

        conv9 = self.relu1024(self.conv9(concate_input4))
        conv10 = self.relu1024(self.conv10(conv9))
        #----------------解码主线------------------------------------------------------------------------------------------
        #-------------decoder step1：1024 -> 512 ------------------------------------------------------
        
        deconv1 = self.relu512(self.deconv1(conv10))
        conb1 = torch.cat((deconv1,conv8,conv7_f),1)
        conv11 =self.relu512(self.conv11(conb1))
        conv12 = self.relu512(self.conv12(conv11))
        #-------------decoder step2：512 -> 256------------------------------------------------------
        deconv2 = self.relu256(self.deconv2(conv12))
        conb2 = torch.cat((deconv2, conv6, conv5_f),1)
        conv13 = self.relu256(self.conv13(conb2))
        conv14 = self.relu256(self.conv14(conv13))
        # -------------decoder step3：256 -> 128------------------------------------------------------
        deconv3 = self.relu128(self.deconv3(conv14))
        conb3 = torch.cat((deconv3, conv4, conv3_f), 1)
        conv15 = self.relu128(self.conv15(conb3))
        conv16 = self.relu128(self.conv16(conv15))
        # -------------decoder step4：128 -> 64------------------------------------------------------
        deconv4 = self.relu64(self.deconv4(conv16))
        conb3 = torch.cat((deconv4, conv2, conv1_f),1)
        conv17 = self.relu64(self.conv17(conb3))
        conv18 = self.relu64(self.conv18(conv17))
        # -------------decoder step5：64 -> 3------------------------------------------------------
        conv19 = self.relu3(self.conv19(conv18))
        #conv19 = F.relu(self.conv19(conv18))


        return conv19


class MSDFN_A(nn.Module):
    def __init__(self):
        super(MSDFN_A, self).__init__()
        self.relu64 = DyReLUA(64, conv_type='2d')
        self.relu128 = DyReLUA(128, conv_type='2d')
        self.relu256 = DyReLUA(256, conv_type='2d')
        self.relu512 = DyReLUA(512, conv_type='2d')
        self.relu1024 = DyReLUA(1024, conv_type='2d')
        self.relu3 = DyReLUA(3,reduction=1,conv_type='2d')
        #TODO: ---------------------------------------对深度图像进行降采样（maxpooling），供上卷积使用，以下定义所需计算单元--------------------------------------------------------------------
        self.conv1_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
        )
        self.pool1_f = nn.MaxPool2d(2, stride=2, padding=0 )

        self.conv3_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(64, 128, 3, 1, 1),                                                                                # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.pool2_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv5_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(128, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )

        self.pool3_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv7_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(256, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.pool4_f = nn.MaxPool2d(2, stride=2, padding=0)
        # TODO: ---------------------------------------对输入彩色图像进行降采样（maxpooling），供下卷积提取特征使用，以下定义所需计算单元--------------------------------------------------------
        self.conv1 = nn.Sequential(                                                                                     # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
            )
        self.conv2 = nn.Sequential(                                                                                     # conv1入，64进64出,---> pool1
            nn.Conv2d(64, 64, 3, 1, 1),                                                                                 # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
        )
        self.pool1 = nn.MaxPool2d(2,stride=2,padding=0 )                                                                # 原图入，1进1出,---> concate_input1   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool1_input = nn.MaxPool2d(2,stride=2,padding=0 )                                                          # conv_2入，1进64出,---> conv_input1
        self.conv_input1 = nn.Sequential(                                                                               # pool_input入，1进64出,---> concate_input1
            nn.Conv2d(3, 64, 3, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        #TODO: concate_input1 -----------------------------------------channel --> 128--------------------------------------
        self.conv3 = nn.Sequential(                                                                                     # concat入，---> conv4
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.conv4 = nn.Sequential(                                                                                     # conv1入，--> pool2
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv4入 ---> concate_input2   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool2_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input入 ---> conv_input2
        self.conv_input2 = nn.Sequential(                                                                               # pool_input2入，1进64出,---> concate_input2
            nn.Conv2d(3, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        # TODO: concate_input2 ----------------------------------------channel-->256----------------------------------------
        self.conv5 = nn.Sequential(                                                                                     # concate_input2入，---> conv5
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )
        self.conv6 = nn.Sequential(                                                                                     # conv1入，---> pool3
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv6入，---> concate_input3   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool3_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input2入，,---> conv_input3
        self.conv_input3 = nn.Sequential(                                                                               # pool3_input入，---> concate_input3
            nn.Conv2d(3, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        # TODO: concate_input3 ----------------------------------------channel-->512----------------------------------------
        self.conv7 = nn.Sequential(                                                                                     # concate_input3入，--->conv6
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.conv8 = nn.Sequential(                                                                                     # conv7入, ---> pool4
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv8入，1进1出,---> concate_input4   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool4_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # ool_input3入，1进1出,---> conv_input4
        self.conv_input4 = nn.Sequential(                                                                               # pool4_input入，1进64出,---> concate_input4
            nn.Conv2d(3, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        # TODO:concate_input4 ----------------------------------------channel-->1024----------------------------------------

        self.conv9 = nn.Sequential(                                                                                     # concat4入，1024进1024出,---> conv10
            nn.Conv2d(1024, 1024, 3, 1, 1),
            #DyReLUB(1024, conv_type='2d')
        )
        self.conv10 = nn.Sequential(                                                                                    # conv9t入，1024进1024出,---> concate_input2
            nn.Conv2d(1024, 1024, 3, 1, 1),
            #DyReLUB(1024, conv_type='2d')
        )

        # TODO: ---------------------------------------decoder上卷积，以下定义所需计算单元--------------------------------------------------------
        #上卷积+concat+relu（卷积）+relu(卷积)
        self.deconv1 =nn.Sequential(
            nn.ConvTranspose2d(1024,512,2,2,0),
            #DyReLUB(512, conv_type='2d')
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512+512+512, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        #---------------------------------------------------------------------------------------------------------------
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2, 0),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256+256+256, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(128 + 128 + 128, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64+64+64, 64, 3, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.conv19= nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            #DyReLUB(3, conv_type='2d')
        )
        #----------------------------------------------------------------------------------------------------
        self.conv1_input = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv2_input = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv3_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv4_input = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1),
            #DyReLUB(512,conv_type='2d')
        )
    def forward(self,image,depth):
        """
        具体计算过程：
        编码时：有雾彩色图像作主线，深度图作副线
        解码时：解码为主线，编码和有雾图作副线
        计算结果：直接生成图像
        :param depth:深度图
        :param image:彩色有雾图像
        :return:
        """
        #----------彩色图下卷积---------------------------------------------------------------------------------------------
        conv1_f = self.relu64(self.conv1_f(image))
        pool1_f = self.pool1_f(conv1_f)

        conv3_f = self.relu128(self.conv3_f(pool1_f))
        pool2_f = self.pool2_f(conv3_f)

        conv5_f = self.relu256(self.conv5_f(pool2_f))
        pool3_f = self.pool3_f(conv5_f)

        conv7_f = self.relu512(self.conv7_f(pool3_f))
        pool4_f = self.pool4_f(conv7_f)
        #------------深度图下卷积----------------------------------------------------------------------------------------------
        # conv1_input = self.relu64(self.conv1_input(depth))
        # pool1_input = self.pool2_input(conv1_input)
        #
        # conv2_input = self.relu128(self.conv2_input(pool1_input))
        # pool2_input = self.pool2_input(conv2_input)
        #
        # conv3_input = self.relu256(self.conv3_input(pool2_input))
        # pool3_input = self.pool3_input(conv3_input)
        #
        # conv4_input = self.relu512(self.conv4_input(pool3_input))

        #------------编码主线-----------------------------------------------------------------------------------------------------
        conv1 = self.relu64(self.conv1(depth))
        conv2 = self.relu64(self.conv2(conv1))
        pool1 = self.pool1(conv2)

        #pool1_input = self.pool2_input(image)

        concate_input1 = torch.cat((pool1, pool1_f), 1)
        #--------------------------128------------------------
        conv3 = self.relu128(self.conv3(concate_input1))
        conv4 = self.relu128(self.conv4(conv3))
        pool2 = self.pool2(conv4)

        #pool2_input = self.pool2_input(pool1_input)

        concate_input2 = torch.cat((pool2, pool2_f), 1)
        #--------------------------256------------------------
        conv5 = self.relu256(self.conv5(concate_input2))
        conv6 = self.relu256(self.conv6(conv5))
        pool3 = self.pool3(conv6)



        concate_input3 = torch.cat((pool3, pool3_f), 1)
        # --------------------------512------------------------
        conv7 = self.relu512(self.conv7(concate_input3))
        conv8 = self.relu512(self.conv8(conv7))
        pool4 = self.pool4(conv8)

        #pool4_input = self.pool3_input(pool3_input)

        concate_input4 = torch.cat((pool4, pool4_f), 1)

        conv9 = self.relu1024(self.conv9(concate_input4))
        conv10 = self.relu1024(self.conv10(conv9))
        #----------------解码主线------------------------------------------------------------------------------------------
        #-------------decoder step1：1024 -> 512 ------------------------------------------------------
        
        deconv1 = self.relu512(self.deconv1(conv10))
        conb1 = torch.cat((deconv1,conv8,conv7_f),1)
        conv11 =self.relu512(self.conv11(conb1))
        conv12 = self.relu512(self.conv12(conv11))
        #-------------decoder step2：512 -> 256------------------------------------------------------
        deconv2 = self.relu256(self.deconv2(conv12))
        conb2 = torch.cat((deconv2, conv6, conv5_f),1)
        conv13 = self.relu256(self.conv13(conb2))
        conv14 = self.relu256(self.conv14(conv13))
        # -------------decoder step3：256 -> 128------------------------------------------------------
        deconv3 = self.relu128(self.deconv3(conv14))
        conb3 = torch.cat((deconv3, conv4, conv3_f), 1)
        conv15 = self.relu128(self.conv15(conb3))
        conv16 = self.relu128(self.conv16(conv15))
        # -------------decoder step4：128 -> 64------------------------------------------------------
        deconv4 = self.relu64(self.deconv4(conv16))
        conb3 = torch.cat((deconv4, conv2, conv1_f),1)
        conv17 = self.relu64(self.conv17(conb3))
        conv18 = self.relu64(self.conv18(conv17))
        # -------------decoder step5：64 -> 3------------------------------------------------------
        conv19 = self.relu3(self.conv19(conv18))
        #conv19 = F.relu(self.conv19(conv18))


        return conv19


class MSDFN_CUT_D(nn.Module):
    def __init__(self):
        super(MSDFN_CUT_D, self).__init__()
        self.relu64 = DyReLUA(64, conv_type='2d')
        self.relu128 = DyReLUA(128, conv_type='2d')
        self.relu256 = DyReLUA(256, conv_type='2d')
        self.relu512 = DyReLUA(512, conv_type='2d')
        self.relu1024 = DyReLUA(1024, conv_type='2d')
        self.relu3 = DyReLUA(3,reduction=1,conv_type='2d')
        #TODO: ---------------------------------------对深度图像进行降采样（maxpooling），供上卷积使用，以下定义所需计算单元--------------------------------------------------------------------
        self.conv1_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
        )
        self.pool1_f = nn.MaxPool2d(2, stride=2, padding=0 )

        self.conv3_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(64, 128, 3, 1, 1),                                                                                # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.pool2_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv5_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(128, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )

        self.pool3_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv7_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(256, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.pool4_f = nn.MaxPool2d(2, stride=2, padding=0)
        # TODO: ---------------------------------------对输入彩色图像进行降采样（maxpooling），供下卷积提取特征使用，以下定义所需计算单元--------------------------------------------------------
        self.conv1 = nn.Sequential(                                                                                     # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
            )
        self.conv2 = nn.Sequential(                                                                                     # conv1入，64进64出,---> pool1
            nn.Conv2d(64, 64, 3, 1, 1),                                                                                 # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(64, conv_type='2d')
        )
        self.pool1 = nn.MaxPool2d(2,stride=2,padding=0 )                                                                # 原图入，1进1出,---> concate_input1   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool1_input = nn.MaxPool2d(2,stride=2,padding=0 )                                                          # conv_2入，1进64出,---> conv_input1
        self.conv_input1 = nn.Sequential(                                                                               # pool_input入，1进64出,---> concate_input1
            nn.Conv2d(3, 64, 3, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        #TODO: concate_input1 -----------------------------------------channel --> 128--------------------------------------
        self.conv3 = nn.Sequential(                                                                                     # concat入，---> conv4
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.conv4 = nn.Sequential(                                                                                     # conv1入，--> pool2
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(128, conv_type='2d')
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv4入 ---> concate_input2   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool2_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input入 ---> conv_input2
        self.conv_input2 = nn.Sequential(                                                                               # pool_input2入，1进64出,---> concate_input2
            nn.Conv2d(3, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        # TODO: concate_input2 ----------------------------------------channel-->256----------------------------------------
        self.conv5 = nn.Sequential(                                                                                     # concate_input2入，---> conv5
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )
        self.conv6 = nn.Sequential(                                                                                     # conv1入，---> pool3
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(256, conv_type='2d')
        )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv6入，---> concate_input3   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool3_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input2入，,---> conv_input3
        self.conv_input3 = nn.Sequential(                                                                               # pool3_input入，---> concate_input3
            nn.Conv2d(3, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        # TODO: concate_input3 ----------------------------------------channel-->512----------------------------------------
        self.conv7 = nn.Sequential(                                                                                     # concate_input3入，--->conv6
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.conv8 = nn.Sequential(                                                                                     # conv7入, ---> pool4
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            #DyReLUB(512, conv_type='2d')
        )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv8入，1进1出,---> concate_input4   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool4_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # ool_input3入，1进1出,---> conv_input4
        self.conv_input4 = nn.Sequential(                                                                               # pool4_input入，1进64出,---> concate_input4
            nn.Conv2d(3, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        # TODO:concate_input4 ----------------------------------------channel-->1024----------------------------------------

        self.conv9 = nn.Sequential(                                                                                     # concat4入，1024进1024出,---> conv10
            nn.Conv2d(1024, 1024, 3, 1, 1),
            #DyReLUB(1024, conv_type='2d')
        )
        self.conv10 = nn.Sequential(                                                                                    # conv9t入，1024进1024出,---> concate_input2
            nn.Conv2d(1024, 1024, 3, 1, 1),
            #DyReLUB(1024, conv_type='2d')
        )

        # TODO: ---------------------------------------decoder上卷积，以下定义所需计算单元--------------------------------------------------------
        #上卷积+concat+relu（卷积）+relu(卷积)
        self.deconv1 =nn.Sequential(
            nn.ConvTranspose2d(1024,512,2,2,0),
            #DyReLUB(512, conv_type='2d')
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512+512, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            #DyReLUB(512, conv_type='2d')
        )
        #---------------------------------------------------------------------------------------------------------------
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2, 0),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.conv19= nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            #DyReLUB(3, conv_type='2d')
        )
        #----------------------------------------------------------------------------------------------------
        self.conv1_input = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            #DyReLUB(64, conv_type='2d')
        )
        self.conv2_input = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1),
            #DyReLUB(128, conv_type='2d')
        )
        self.conv3_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            #DyReLUB(256, conv_type='2d')
        )
        self.conv4_input = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1),
            #DyReLUB(512,conv_type='2d')
        )
    def forward(self,image,depth):
        """
        具体计算过程：
        编码时：有雾彩色图像作主线，深度图作副线
        解码时：解码为主线，编码和有雾图作副线
        计算结果：直接生成图像
        :param depth:深度图
        :param image:彩色有雾图像
        :return:
        """
        #----------彩色图下卷积---------------------------------------------------------------------------------------------
        conv1_f = self.relu64(self.conv1_f(image))
        pool1_f = self.pool1_f(conv1_f)

        conv3_f = self.relu128(self.conv3_f(pool1_f))
        pool2_f = self.pool2_f(conv3_f)

        conv5_f = self.relu256(self.conv5_f(pool2_f))
        pool3_f = self.pool3_f(conv5_f)

        conv7_f = self.relu512(self.conv7_f(pool3_f))
        pool4_f = self.pool4_f(conv7_f)
        #------------深度图下卷积----------------------------------------------------------------------------------------------
        # conv1_input = self.relu64(self.conv1_input(depth))
        # pool1_input = self.pool2_input(conv1_input)
        #
        # conv2_input = self.relu128(self.conv2_input(pool1_input))
        # pool2_input = self.pool2_input(conv2_input)
        #
        # conv3_input = self.relu256(self.conv3_input(pool2_input))
        # pool3_input = self.pool3_input(conv3_input)
        #
        # conv4_input = self.relu512(self.conv4_input(pool3_input))

        #------------编码主线-----------------------------------------------------------------------------------------------------
        conv1 = self.relu64(self.conv1(depth))
        conv2 = self.relu64(self.conv2(conv1))
        pool1 = self.pool1(conv2)

        #pool1_input = self.pool2_input(image)

        concate_input1 = torch.cat((pool1, pool1_f), 1)
        #--------------------------128------------------------
        conv3 = self.relu128(self.conv3(concate_input1))
        conv4 = self.relu128(self.conv4(conv3))
        pool2 = self.pool2(conv4)

        #pool2_input = self.pool2_input(pool1_input)

        concate_input2 = torch.cat((pool2, pool2_f), 1)
        #--------------------------256------------------------
        conv5 = self.relu256(self.conv5(concate_input2))
        conv6 = self.relu256(self.conv6(conv5))
        pool3 = self.pool3(conv6)



        concate_input3 = torch.cat((pool3, pool3_f), 1)
        # --------------------------512------------------------
        conv7 = self.relu512(self.conv7(concate_input3))
        conv8 = self.relu512(self.conv8(conv7))
        pool4 = self.pool4(conv8)

        #pool4_input = self.pool3_input(pool3_input)

        concate_input4 = torch.cat((pool4, pool4_f), 1)

        conv9 = self.relu1024(self.conv9(concate_input4))
        conv10 = self.relu1024(self.conv10(conv9))
        #----------------解码主线------------------------------------------------------------------------------------------
        #-------------decoder step1：1024 -> 512 ------------------------------------------------------
        
        deconv1 = self.relu512(self.deconv1(conv10))
        conb1 = torch.cat((deconv1,conv8),1)
        conv11 =self.relu512(self.conv11(conb1))
        conv12 = self.relu512(self.conv12(conv11))
        #-------------decoder step2：512 -> 256------------------------------------------------------
        deconv2 = self.relu256(self.deconv2(conv12))
        conb2 = torch.cat((deconv2, conv6),1)
        conv13 = self.relu256(self.conv13(conb2))
        conv14 = self.relu256(self.conv14(conv13))
        # -------------decoder step3：256 -> 128------------------------------------------------------
        deconv3 = self.relu128(self.deconv3(conv14))
        conb3 = torch.cat((deconv3, conv4), 1)
        conv15 = self.relu128(self.conv15(conb3))
        conv16 = self.relu128(self.conv16(conv15))
        # -------------decoder step4：128 -> 64------------------------------------------------------
        deconv4 = self.relu64(self.deconv4(conv16))
        conb3 = torch.cat((deconv4, conv2),1)
        conv17 = self.relu64(self.conv17(conb3))
        conv18 = self.relu64(self.conv18(conv17))
        # -------------decoder step5：64 -> 3------------------------------------------------------
        conv19 = self.relu3(self.conv19(conv18))
        #conv19 = F.relu(self.conv19(conv18))


        return conv19


class MSDFN_CUT_E(nn.Module):
    def __init__(self):
        super(MSDFN_CUT_E, self).__init__()
        self.relu64 = DyReLUA(64, conv_type='2d')
        self.relu128 = DyReLUA(128, conv_type='2d')
        self.relu256 = DyReLUA(256, conv_type='2d')
        self.relu512 = DyReLUA(512, conv_type='2d')
        self.relu1024 = DyReLUA(1024, conv_type='2d')
        self.relu3 = DyReLUA(3, reduction=1, conv_type='2d')
        # TODO: ---------------------------------------对深度图像进行降采样（maxpooling），供上卷积使用，以下定义所需计算单元--------------------------------------------------------------------
        self.conv1_f = nn.Sequential(  # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(64, conv_type='2d')
        )
        self.pool1_f = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3_f = nn.Sequential(  # 原图入，1进64出,---> conv2
            nn.Conv2d(64, 128, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(128, conv_type='2d')
        )
        self.pool2_f = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv5_f = nn.Sequential(  # 原图入，1进64出,---> conv2
            nn.Conv2d(128, 256, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(256, conv_type='2d')
        )

        self.pool3_f = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv7_f = nn.Sequential(  # 原图入，1进64出,---> conv2
            nn.Conv2d(256, 512, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(512, conv_type='2d')
        )
        self.pool4_f = nn.MaxPool2d(2, stride=2, padding=0)
        # TODO: ---------------------------------------对输入彩色图像进行降采样（maxpooling），供下卷积提取特征使用，以下定义所需计算单元--------------------------------------------------------
        self.conv1 = nn.Sequential(  # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(64, conv_type='2d')
        )
        self.conv2 = nn.Sequential(  # conv1入，64进64出,---> pool1
            nn.Conv2d(64, 64, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(64, conv_type='2d')
        )
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)  # 原图入，1进1出,---> concate_input1   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool1_input = nn.MaxPool2d(2, stride=2, padding=0)  # conv_2入，1进64出,---> conv_input1
        self.conv_input1 = nn.Sequential(  # pool_input入，1进64出,---> concate_input1
            nn.Conv2d(3, 64, 3, 1, 1),
            # DyReLUB(64, conv_type='2d')
        )
        # TODO: concate_input1 -----------------------------------------channel --> 128--------------------------------------
        self.conv3 = nn.Sequential(  # concat入，---> conv4
            nn.Conv2d(64, 64, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(128, conv_type='2d')
        )
        self.conv4 = nn.Sequential(  # conv1入，--> pool2
            nn.Conv2d(64, 128, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(128, conv_type='2d')
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)  # conv4入 ---> concate_input2   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool2_input = nn.MaxPool2d(2, stride=2, padding=0)  # pool_input入 ---> conv_input2
        self.conv_input2 = nn.Sequential(  # pool_input2入，1进64出,---> concate_input2
            nn.Conv2d(3, 128, 3, 1, 1),
            # DyReLUB(128, conv_type='2d')
        )
        # TODO: concate_input2 ----------------------------------------channel-->256----------------------------------------
        self.conv5 = nn.Sequential(  # concate_input2入，---> conv5
            nn.Conv2d(128, 128, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(256, conv_type='2d')
        )
        self.conv6 = nn.Sequential(  # conv1入，---> pool3
            nn.Conv2d(128, 256, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(256, conv_type='2d')
        )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)  # conv6入，---> concate_input3   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool3_input = nn.MaxPool2d(2, stride=2, padding=0)  # pool_input2入，,---> conv_input3
        self.conv_input3 = nn.Sequential(  # pool3_input入，---> concate_input3
            nn.Conv2d(3, 256, 3, 1, 1),
            # DyReLUB(256, conv_type='2d')
        )
        # TODO: concate_input3 ----------------------------------------channel-->512----------------------------------------
        self.conv7 = nn.Sequential(  # concate_input3入，--->conv6
            nn.Conv2d(256, 256, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(512, conv_type='2d')
        )
        self.conv8 = nn.Sequential(  # conv7入, ---> pool4
            nn.Conv2d(256, 512, 3, 1, 1),  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            # DyReLUB(512, conv_type='2d')
        )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0)  # conv8入，1进1出,---> concate_input4   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool4_input = nn.MaxPool2d(2, stride=2, padding=0)  # ool_input3入，1进1出,---> conv_input4
        self.conv_input4 = nn.Sequential(  # pool4_input入，1进64出,---> concate_input4
            nn.Conv2d(3, 512, 3, 1, 1),
            # DyReLUB(512, conv_type='2d')
        )
        # TODO:concate_input4 ----------------------------------------channel-->1024----------------------------------------

        self.conv9 = nn.Sequential(  # concat4入，1024进1024出,---> conv10
            nn.Conv2d(512, 512, 3, 1, 1),
            # DyReLUB(1024, conv_type='2d')
        )
        self.conv10 = nn.Sequential(  # conv9t入，1024进1024出,---> concate_input2
            nn.Conv2d(512, 1024, 3, 1, 1),
            # DyReLUB(1024, conv_type='2d')
        )

        # TODO: ---------------------------------------decoder上卷积，以下定义所需计算单元--------------------------------------------------------
        # 上卷积+concat+relu（卷积）+relu(卷积)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, 2, 0),
            # DyReLUB(512, conv_type='2d')
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512 + 512 + 512, 512, 3, 1, 1),
            # DyReLUB(512, conv_type='2d')
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            # DyReLUB(512, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2, 0),
            # DyReLUB(256, conv_type='2d')
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256 + 256 + 256, 256, 3, 1, 1),
            # DyReLUB(256, conv_type='2d')
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            # DyReLUB(256, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            # DyReLUB(128, conv_type='2d')
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(128 + 128 + 128, 128, 3, 1, 1),
            # DyReLUB(128, conv_type='2d')
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            # DyReLUB(128, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            # DyReLUB(64, conv_type='2d')
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, 64, 3, 1, 1),
            # DyReLUB(64, conv_type='2d')
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            # DyReLUB(64, conv_type='2d')
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.conv19 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            # DyReLUB(3, conv_type='2d')
        )
        # ----------------------------------------------------------------------------------------------------
        self.conv1_input = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            # DyReLUB(64, conv_type='2d')
        )
        self.conv2_input = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1),
            # DyReLUB(128, conv_type='2d')
        )
        self.conv3_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            # DyReLUB(256, conv_type='2d')
        )
        self.conv4_input = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1),
            # DyReLUB(512,conv_type='2d')
        )

    def forward(self, image, depth):
        """
        具体计算过程：
        编码时：有雾彩色图像作主线，深度图作副线
        解码时：解码为主线，编码和有雾图作副线
        计算结果：直接生成图像
        :param depth:深度图
        :param image:彩色有雾图像
        :return:
        """
        # ----------彩色图下卷积---------------------------------------------------------------------------------------------
        conv1_f = self.relu64(self.conv1_f(image))
        pool1_f = self.pool1_f(conv1_f)

        conv3_f = self.relu128(self.conv3_f(pool1_f))
        pool2_f = self.pool2_f(conv3_f)

        conv5_f = self.relu256(self.conv5_f(pool2_f))
        pool3_f = self.pool3_f(conv5_f)

        conv7_f = self.relu512(self.conv7_f(pool3_f))
        pool4_f = self.pool4_f(conv7_f)
        # ------------编码主线-----------------------------------------------------------------------------------------------------
        conv1 = self.relu64(self.conv1(depth))
        conv2 = self.relu64(self.conv2(conv1))
        pool1 = self.pool1(conv2)

        # --------------------------128------------------------
        conv3 = self.relu64(self.conv3(pool1))
        conv4 = self.relu128(self.conv4(conv3))
        pool2 = self.pool2(conv4)

        # --------------------------256------------------------
        conv5 = self.relu128(self.conv5(pool2))
        conv6 = self.relu256(self.conv6(conv5))
        pool3 = self.pool3(conv6)


        # --------------------------512------------------------
        conv7 = self.relu256(self.conv7(pool3))
        conv8 = self.relu512(self.conv8(conv7))
        pool4 = self.pool4(conv8)

        conv9 = self.relu512(self.conv9(pool4))
        conv10 = self.relu1024(self.conv10(conv9))
        # ----------------解码主线------------------------------------------------------------------------------------------
        # -------------decoder step1：1024 -> 512 ------------------------------------------------------

        deconv1 = self.relu512(self.deconv1(conv10))
        conb1 = torch.cat((deconv1, conv8, conv7_f), 1)
        conv11 = self.relu512(self.conv11(conb1))
        conv12 = self.relu512(self.conv12(conv11))
        # -------------decoder step2：512 -> 256------------------------------------------------------
        deconv2 = self.relu256(self.deconv2(conv12))
        conb2 = torch.cat((deconv2, conv6, conv5_f), 1)
        conv13 = self.relu256(self.conv13(conb2))
        conv14 = self.relu256(self.conv14(conv13))
        # -------------decoder step3：256 -> 128------------------------------------------------------
        deconv3 = self.relu128(self.deconv3(conv14))
        conb3 = torch.cat((deconv3, conv4, conv3_f), 1)
        conv15 = self.relu128(self.conv15(conb3))
        conv16 = self.relu128(self.conv16(conv15))
        # -------------decoder step4：128 -> 64------------------------------------------------------
        deconv4 = self.relu64(self.deconv4(conv16))
        conb3 = torch.cat((deconv4, conv2, conv1_f), 1)
        conv17 = self.relu64(self.conv17(conb3))
        conv18 = self.relu64(self.conv18(conv17))
        # -------------decoder step5：64 -> 3------------------------------------------------------
        conv19 = self.relu3(self.conv19(conv18))
        # conv19 = F.relu(self.conv19(conv18))

        return conv19