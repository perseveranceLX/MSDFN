import torch
import torch.nn as nn
import torch.nn.functional as F
#import FReLU

class DEPTHV2(nn.Module):
    def __init__(self):
        super(DEPTHV2, self).__init__()
        #TODO: ---------------------------------------对深度图像进行降采样（maxpooling），供上卷积使用，以下定义所需计算单元--------------------------------------------------------------------
        self.conv1_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.pool1_f = nn.MaxPool2d(2, stride=2, padding=0 )

        self.conv3_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(64, 128, 3, 1, 1),                                                                                # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.pool2_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv5_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(128, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )

        self.pool3_f = nn.MaxPool2d(2, stride=2, padding=0 )
        self.conv7_f = nn.Sequential(                                                                                   # 原图入，1进64出,---> conv2
            nn.Conv2d(256, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )

        # TODO: ---------------------------------------对输入彩色图像进行降采样（maxpooling），供下卷积提取特征使用，以下定义所需计算单元--------------------------------------------------------
        self.conv1 = nn.Sequential(                                                                                     # 原图入，1进64出,---> conv2
            nn.Conv2d(3, 64, 3, 1, 1),                                                                                  # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(                                                                                     # conv1入，64进64出,---> pool1
            nn.Conv2d(64, 64, 3, 1, 1),                                                                                 # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2,stride=2,padding=0 )                                                                # 原图入，1进1出,---> concate_input1   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool1_input = nn.MaxPool2d(2,stride=2,padding=0 )                                                          # conv_2入，1进64出,---> conv_input1
        self.conv_input1 = nn.Sequential(                                                                               # pool_input入，1进64出,---> concate_input1
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU()
        )
        #TODO: concate_input1 -----------------------------------------dim --> 128--------------------------------------
        self.conv3 = nn.Sequential(                                                                                     # concat入，---> conv4
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(                                                                                     # conv1入，--> pool2
            nn.Conv2d(128, 128, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv4入 ---> concate_input2   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool2_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input入 ---> conv_input2
        self.conv_input2 = nn.Sequential(                                                                               # pool_input2入，1进64出,---> concate_input2
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.ReLU()
        )
        # TODO: concate_input2 ----------------------------------------dim-->256----------------------------------------
        self.conv5 = nn.Sequential(                                                                                     # concate_input2入，---> conv5
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(                                                                                     # conv1入，---> pool3
            nn.Conv2d(256, 256, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv6入，---> concate_input3   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool3_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # pool_input2入，,---> conv_input3
        self.conv_input3 = nn.Sequential(                                                                               # pool3_input入，---> concate_input3
            nn.Conv2d(3, 256, 3, 1, 1),
            nn.ReLU()
        )
        # TODO: concate_input3 ----------------------------------------dim-->512----------------------------------------
        self.conv7 = nn.Sequential(                                                                                     # concate_input3入，--->conv6
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(                                                                                     # conv7入, ---> pool4
            nn.Conv2d(512, 512, 3, 1, 1),                                                                               # 输入通道数，输出通道数，卷积核大小，步长，边界，卷积核之间的间距
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0 )                                                              # conv8入，1进1出,---> concate_input4   ：进入自动编码器后直接卷积作为后面concat使用
        self.pool4_input = nn.MaxPool2d(2, stride=2, padding=0 )                                                        # ool_input3入，1进1出,---> conv_input4
        self.conv_input4 = nn.Sequential(                                                                               # pool4_input入，1进64出,---> concate_input4
            nn.Conv2d(3, 512, 3, 1, 1),
            nn.ReLU()
        )
        # TODO:concate_input4 ----------------------------------------dim-->1024----------------------------------------

        self.conv9 = nn.Sequential(                                                                                     # concat4入，1024进1024出,---> conv10
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(                                                                                    # conv9t入，1024进1024出,---> concate_input2
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.ReLU()
        )

        # TODO: ---------------------------------------decoder上卷积，以下定义所需计算单元--------------------------------------------------------
        #上卷积+concat+relu（卷积）+relu(卷积)
        self.deconv1 =nn.Sequential(
            nn.ConvTranspose2d(1024,512,2,2,0),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512+512+512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        #---------------------------------------------------------------------------------------------------------------
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2, 0),
            nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256+256+256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            nn.ReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(128 + 128 + 128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2, 0),
            nn.ReLU()
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64+64+64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU()
        )
        # ---------------------------------------------------------------------------------------------------------------
        self.conv19= nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            nn.ReLU()
        )
    # def forward(self,depth,image):
    #     """
    #     具体计算过程：
    #     编码时：有雾彩色图像作主线，有雾彩色图像降采样为副线；
    #     解码时：解码为主线，编码和深度图作副线
    #     计算结果：直接生成图像
    #     :param depth:深度图
    #     :param image:彩色有雾图像
    #     :return:
    #     """
    #     #----------深度图下卷积---------------------------------------------------------------------------------------------
    #     conv1_f = self.conv1_f(depth)
    #     pool1_f = self.pool1_f(conv1_f)
    #     conv3_f = self.conv3_f(pool1_f)
    #     pool2_f = self.pool2_f(conv3_f)
    #     conv5_f = self.conv5_f(pool2_f)
    #     pool3_f = self.pool3_f(conv5_f)
    #     conv7_f = self.conv7_f(pool3_f)
    #     #------------有雾彩色图----------------------------------------------------------------------------------------------
    #     conv1 = self.conv1(image)
    #     conv2 = self.conv2(conv1)
    #     pool1 = self.pool1(conv2)

    #     pool1_input = self.pool2_input(image)
    #     conv_input1 = self.conv_input1(pool1_input)
    #     concate_input1 = torch.cat((pool1, conv_input1), 1)
    #     #--------------------------128------------------------
    #     conv3 = self.conv3(concate_input1)
    #     conv4 = self.conv4(conv3)
    #     pool2 = self.pool2(conv4)

    #     pool2_input = self.pool2_input(pool1_input)
    #     conv_input2 = self.conv_input2(pool2_input)
    #     concate_input2 = torch.cat((pool2, conv_input2), 1)
    #     #--------------------------256------------------------
    #     conv5 = self.conv5(concate_input2)
    #     conv6 = self.conv6(conv5)
    #     pool3 = self.pool3(conv6)

    #     pool3_input = self.pool3_input(pool2_input)
    #     conv_input3 = self.conv_input3(pool3_input)
    #     concate_input3 = torch.cat((pool3, conv_input3), 1)
    #     # --------------------------512------------------------
    #     conv7 = self.conv7(concate_input3)
    #     conv8 = self.conv8(conv7)
    #     pool4 = self.pool4(conv8)

    #     pool4_input = self.pool3_input(pool3_input)
    #     conv_input4 = self.conv_input4(pool4_input)
    #     concate_input4 = torch.cat((pool4, conv_input4), 1)

    #     conv9 = self.conv9(concate_input4)
    #     conv10 = self.conv10(conv9)
    #     #----------------上卷积------------------------------------------------------------------------------------------
    #     deconv1 = self.deconv1(conv10)
    #     conb1 = torch.cat((deconv1,conv8,conv7_f),1)
    #     conv11 =self.conv11(conb1)
    #     conv12 = self.conv12(conv11)

    #     deconv2 = self.deconv2(conv12)
    #     conb2 = torch.cat((deconv2, conv6, conv5_f),1)
    #     conv13 = self.conv13(conb2)
    #     conv14 = self.conv14(conv13)

    #     deconv3 = self.deconv3(conv14)
    #     conb3 = torch.cat((deconv3, conv4, conv3_f), 1)
    #     conv15 = self.conv15(conb3)
    #     conv16 = self.conv16(conv15)

    #     deconv4 = self.deconv4(conv16)
    #     conb3 = torch.cat((deconv4, conv2, conv1_f), 1)
    #     conv17 = self.conv17(conb3)
    #     conv18 = self.conv18(conv17)
    #     conv19 = self.conv19(conv18)
    #     #output = (conv19 * image) - conv19 + 1
    #     output = conv19+image
    #     return output
    def forward(self,depth,image):
        """
        具体计算过程：
        编码时：有雾彩色图像作主线，深度图作副线
        解码时：解码为主线，编码和有雾图作副线
        计算结果：直接生成图像
        :param depth:深度图
        :param image:彩色有雾图像
        :return:
        """
        #----------深度图下卷积---------------------------------------------------------------------------------------------
        conv1_f = self.conv1_f(depth)
        pool1_f = self.pool1_f(conv1_f)
        conv3_f = self.conv3_f(pool1_f)
        pool2_f = self.pool2_f(conv3_f)
        conv5_f = self.conv5_f(pool2_f)
        pool3_f = self.pool3_f(conv5_f)
        conv7_f = self.conv7_f(pool3_f)
        #------------有雾彩色图----------------------------------------------------------------------------------------------
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        pool1 = self.pool1(conv2)

        pool1_input = self.pool2_input(image)
        conv_input1 = self.conv_input1(pool1_input)
        concate_input1 = torch.cat((pool1, conv_input1), 1)
        #--------------------------128------------------------
        conv3 = self.conv3(concate_input1)
        conv4 = self.conv4(conv3)
        pool2 = self.pool2(conv4)

        pool2_input = self.pool2_input(pool1_input)
        conv_input2 = self.conv_input2(pool2_input)
        concate_input2 = torch.cat((pool2, conv_input2), 1)
        #--------------------------256------------------------
        conv5 = self.conv5(concate_input2)
        conv6 = self.conv6(conv5)
        pool3 = self.pool3(conv6)

        pool3_input = self.pool3_input(pool2_input)
        conv_input3 = self.conv_input3(pool3_input)
        concate_input3 = torch.cat((pool3, conv_input3), 1)
        # --------------------------512------------------------
        conv7 = self.conv7(concate_input3)
        conv8 = self.conv8(conv7)
        pool4 = self.pool4(conv8)

        pool4_input = self.pool3_input(pool3_input)
        conv_input4 = self.conv_input4(pool4_input)
        concate_input4 = torch.cat((pool4, conv_input4), 1)

        conv9 = self.conv9(concate_input4)
        conv10 = self.conv10(conv9)
        #----------------上卷积------------------------------------------------------------------------------------------
        deconv1 = self.deconv1(conv10)
        conb1 = torch.cat((deconv1,conv8,conv7_f),1)
        conv11 =self.conv11(conb1)
        conv12 = self.conv12(conv11)

        deconv2 = self.deconv2(conv12)
        conb2 = torch.cat((deconv2, conv6, conv5_f),1)
        conv13 = self.conv13(conb2)
        conv14 = self.conv14(conv13)

        deconv3 = self.deconv3(conv14)
        conb3 = torch.cat((deconv3, conv4, conv3_f), 1)
        conv15 = self.conv15(conb3)
        conv16 = self.conv16(conv15)

        deconv4 = self.deconv4(conv16)
        conb3 = torch.cat((deconv4, conv2, conv1_f), 1)
        conv17 = self.conv17(conb3)
        conv18 = self.conv18(conv17)
        conv19 = self.conv19(conv18)
        #output = (conv19 * image) - conv19 + 1
        output = conv19+image
        return output