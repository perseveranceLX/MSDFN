from PIL import Image
import glob
import cv2
import numpy as np
import os
import math
from skimage.measure import compare_psnr
import time
def Psnr(imageA,imageB):
    """
    :param imageA: 预测图像
    :param imageB: 清晰图像
    :return:
    """
    im1 = cv2.imread(imageA)
    im2 = cv2.imread(imageB)
    im1 = cv2.resize(im1,(640,480))
    im2 = cv2.resize(im2,(640,480))
    mse = np.mean((im1 - im2) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

def SSIMnp(y_pred,y_true):
    y_pred = cv2.imread(y_pred)
    y_true = cv2.imread(y_true)
    y_pred = cv2.resize(y_pred, (640, 480))
    y_true = cv2.resize(y_true, (640, 480))
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
def IRDN_comput():
    gt_List = glob.glob("gt/*")
    i1_List = glob.glob("i1/outdoor/-SSIM/outdoor sym/*")
    i2_List = glob.glob("i2/outdoor/-SSIM/outdoor sym/*")
    i3_List = glob.glob("i3/outdoor/-SSIM/outdoor sym/*")
    i4_List = glob.glob("i4/outdoor/-SSIM/outdoor sym/*")
    i5_List = glob.glob("i5/outdoor/-SSIM/outdoor sym/*")
    i6_List = glob.glob("i6/outdoor/-SSIM/outdoor sym/*")
    fileList = glob.glob("haze image/*")
    i1_ssim, i2_ssim, i3_ssim, i4_ssim, i5_ssim, i6_ssim = [], [], [], [], [], []

    i1_psnr, i2_psnr, i3_psnr, i4_psnr, i5_psnr, i6_psnr = [], [], [], [], [], []
    for i in range(6):
        i1_ssim.append(SSIMnp(i1_List[i], gt_List[i]))
        i1_psnr.append(Psnr(i1_List[i], gt_List[i]))

        i2_ssim.append(SSIMnp(i2_List[i], gt_List[i]))
        i2_psnr.append(Psnr(i2_List[i], gt_List[i]))

        i3_ssim.append(SSIMnp(i3_List[i], gt_List[i]))
        i3_psnr.append(Psnr(i3_List[i], gt_List[i]))

        i4_ssim.append(SSIMnp(i4_List[i], gt_List[i]))
        i4_psnr.append(Psnr(i4_List[i], gt_List[i]))

        i5_ssim.append(SSIMnp(i5_List[i], gt_List[i]))
        i5_psnr.append(Psnr(i5_List[i], gt_List[i]))

        i6_ssim.append(SSIMnp(i6_List[i], gt_List[i]))
        i6_psnr.append(Psnr(i6_List[i], gt_List[i]))
        # print(SSIMnp(i1_List[i],gt_List[i])
    ssim_list = [np.mean(i1_ssim), np.mean(i2_ssim), np.mean(i3_ssim), np.mean(i4_ssim), np.mean(i5_ssim),
                 np.mean(i6_ssim)]
    psnr_list = [np.mean(i1_psnr), np.mean(i2_psnr), np.mean(i3_psnr), np.mean(i4_psnr), np.mean(i5_psnr),
                 np.mean(i6_psnr)]
    #print(psnr_list)
    #print(ssim_list)
    return psnr_list,ssim_list

def IRDN_computall():
    gt_List = glob.glob(r"D:\PROJECT\test dataset\indoor sym clear\*")
    i1_List = glob.glob("D:\PROJECT\IRDNresult\i1\indoor\-SSIM\*")
    i2_List = glob.glob("D:\PROJECT\IRDNresult\i2\indoor\-SSIM\*")
    i3_List = glob.glob("D:\PROJECT\IRDNresult\i3\indoor\-SSIM\*")
    i4_List = glob.glob("D:\PROJECT\IRDNresult\i4\indoor\-SSIM\*")
    #i5_List = glob.glob("D:\PROJECT\IRDNresult\i5\outdoor\MSE\outdoor sym\*")
    #i6_List = glob.glob("D:\PROJECT\IRDNresult\i6\outdoor\MSE\outdoor sym\*")

    i1_ssim, i2_ssim, i3_ssim, i4_ssim, i5_ssim, i6_ssim = [], [], [], [], [], []

    i1_psnr, i2_psnr, i3_psnr, i4_psnr, i5_psnr, i6_psnr = [], [], [], [], [], []
    for i in range(len(i1_List)):
        i1_ssim.append(SSIMnp(i1_List[i], gt_List[i]))
        i1_psnr.append(Psnr(i1_List[i], gt_List[i]))

        i2_ssim.append(SSIMnp(i2_List[i], gt_List[i]))
        i2_psnr.append(Psnr(i2_List[i], gt_List[i]))

        i3_ssim.append(SSIMnp(i3_List[i], gt_List[i]))
        i3_psnr.append(Psnr(i3_List[i], gt_List[i]))

        i4_ssim.append(SSIMnp(i4_List[i], gt_List[i]))
        i4_psnr.append(Psnr(i4_List[i], gt_List[i]))

        #i5_ssim.append(SSIMnp(i5_List[i], gt_List[i]))
        #i5_psnr.append(Psnr(i5_List[i], gt_List[i]))

        #i6_ssim.append(SSIMnp(i6_List[i], gt_List[i]))
        #i6_psnr.append(Psnr(i6_List[i], gt_List[i]))
        # print(SSIMnp(i1_List[i],gt_List[i])
    ssim_list = [np.mean(i1_ssim), np.mean(i2_ssim), np.mean(i3_ssim), np.mean(i4_ssim), np.mean(i5_ssim),
                 np.mean(i6_ssim)]
    psnr_list = [np.mean(i1_psnr), np.mean(i2_psnr), np.mean(i3_psnr), np.mean(i4_psnr), np.mean(i5_psnr),
                 np.mean(i6_psnr)]
    #print(psnr_list)
    #print(ssim_list)
    return psnr_list,ssim_list

def comput_other(method,c):
    """
    计算选中图像的ssim.psnr
    :param method:
    :param c:
    :return:
    """
    gt_List = glob.glob("gt/*")
    _list = glob.glob("other method/%s/%s/*" % (method,c))
    _ssim,_psnr=[],[]
    for i in range(6):
        _ssim.append(SSIMnp(_list[i], gt_List[i]))
        _psnr.append(Psnr(_list[i], gt_List[i]))
    print(np.mean(_ssim))
    print(np.mean(_psnr))
    return np.mean(_ssim),np.mean(_psnr)


def comput_other2(path1,path2):
    """
    计算全部图像的ssim,psnr
    :param path1:
    :param path2:
    :return:
    """
    gt_List = glob.glob(path1)
    _list = glob.glob(path2)
    
    _ssim,_psnr=[],[]
   # p = r"D:\PROJECT\DEHAZE METHOD\non-local-dehazing-master\imageish\\"

    for i in range(len(_list)):
        #index = p+"\\"+_list[i].split("\\")[-1]
        #print(index)
        # print(_list[i])
        # index = path1 + _list[i].split("/")[-1][:4]+'.png'
        #print(index)
        print(_list[i],gt_List[i])
        print("-"*50)
        _ssim.append(SSIMnp(_list[i],gt_List[i]))
        _psnr.append(Psnr(_list[i], gt_List[i]))
    #print(_ssim,np.mean(_ssim))
    #print(_psnr,np.mean(_psnr))
    return np.mean(_ssim),np.mean(_psnr)

if __name__ == "__main__":
    #test_list = glob.glob("D:/数据/桌面数据/对比/室外合成/*")
    # irdn_ssim,irdn_psnr = IRDN_computall()
    # print(irdn_ssim,"\n",irdn_psnr)
    #dcp_ssim,dcp_psnr=comput_other("dcp","outdoor sym")
    #grid_ssim,grid_psnr=comput_other("dcp","outdoor sym")
    #print("GridDehazeNet:")
    #grid_ssim,grid_psnr=comput_other2(r"D:\PROJECT\test dataset\indoor sym clear\*",r"D:\PROJECT\IRDNresult\i6\indoor\-SSIM\i6indoor-ssim\*")
    #grid_ssim,grid_psnr=comput_other2(r"D:\PROJECT\MSBDN-DFF\TEST\HR\*",r"D:\PROJECT\MSBDN-DFF\TEST\Results\*")
    #print(grid_psnr,grid_ssim)
    #print("DCP:")
    # dcp_ssim,dcp_psnr=comput_other2(r"D:\PROJECT\test dataset\outdoor sym clear\*",r"D:\PROJECT\other method result\DCP\outdoor sym\*")
    # print(dcp_ssim,dcp_psnr)
    # #dcp_ssim,dcp_psnr=comput_other2(r"D:\PROJECT\test dataset\outdoor sym clear\*",r"D:\PROJECT\other method result\DCP\outdoor sym\*")
    # print("aodnetNet")
    # aodnetNet_ssim,aodnetNet_psnr = comput_other2(r"D:\PROJECT\test dataset\indoor sym clear\*",r"D:\PROJECT\other method result\AOD-Net\indoor sym\*")
    # print(aodnetNet_ssim,aodnetNet_psnr)
    #psnr_list,ssim_list=IRDN_computall()
    print("en")
    aodnetNet_ssim,aodnetNet_psnr=comput_other2(r"dataset/testdataset/outdoor/gt/*",r"test_result/outdoor/D-IPB/*")
    print(aodnetNet_ssim,aodnetNet_psnr)