import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import math
from math import exp


def PSNR(img1, img2):
    """
    img1:pred_img
    img2:gt
    """


    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    mse = numpy.mean((img1 - img2) ** 2)
    #rmse = numpy.mean((im1 - im2) ** 2)
    if mse < 1.0e-10:
        return 100
    # PIXEL_MAX = 1
    psnr = 10 * math.log10(255.0 ** 2 / mse)
    return psnr
