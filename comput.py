import glob
import cv2
import numpy as np
import math

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


def comput(path1,path2):
    """
    计算全部图像的ssim,psnr
    :param path1:
    :param path2:
    :return:
    """
    gt_List = glob.glob(path1)
    _list = glob.glob(path2)
    
    _ssim,_psnr=[],[]


    for i in range(len(_list)):

        print(_list[i],gt_List[i])
        print("-"*50)
        _ssim.append(SSIMnp(_list[i],gt_List[i]))
        _psnr.append(Psnr(_list[i], gt_List[i]))

    return np.mean(_ssim),np.mean(_psnr)

if __name__ == "__main__":
    print("computing!")
    MSDFN_ssim,MSDFN_psnr=comput(r"dataset/testdataset/outdoor/gt/*",r"test_result/outdoor/*")
    print(MSDFN_ssim,MSDFN_psnr)