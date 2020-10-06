import torch
import torchvision
import torch.optim
import model_depth_Drelu as model
import numpy as np

from PIL import Image
import glob
import time,os

def dehaze_image( image_depth_path,image_hazy_path,Id):
    """
    当前，输入为有雾图像和无雾图像，有雾图像和无雾图像都转为灰度图进人网络
    :param image_down_path:深度图下采样，当前用有雾图像的灰度图代替
    :param image_label_path:深度图，用无雾图像灰度图代替
    :param image_add_path:有雾彩色图像
    :param Id:
    :return:
    """
    print(image_hazy_path,image_depth_path)
    img_hazy = Image.open(image_hazy_path)
    img_depth = Image.open(image_depth_path)

    img_hazy = img_hazy.resize((640,480), Image.ANTIALIAS)
    img_depth = img_depth.resize((640,480), Image.ANTIALIAS)
    img_depth = Image.merge('RGB',[img_depth,img_depth,img_depth])

    img_hazy = (np.asarray(img_hazy) / 255.0)
    img_depth = (np.asarray(img_depth) / 255.0)

    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)
    img_depth = torch.from_numpy(img_depth).float().permute(2, 0, 1).cuda().unsqueeze(0)

    dehaze_net = model.MSDFN().cuda()
    dehaze_net.load_state_dict(torch.load('trained_model/outdoor/MSDFN.pth'))

    clean_image = dehaze_net(img_hazy, img_depth)
    temp_tensor = (clean_image,0)
    index = image_depth_path.split('\\')[-1]
    #torchvision.utils.save_image(torch.cat((img_hazy,clean_image),0), "test_result/outdoor/%s" % index)
    torchvision.utils.save_image(clean_image, "test_result/outdoor/%s" % index)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    depth_list = glob.glob(r"testdataset/outdoor/depth/*")
    hazy_liat = glob.glob(r"testdataset/outdoor/hazy/*")

    for Id in range(len(depth_list)):
        dehaze_image(depth_list[Id],hazy_liat[Id],Id)
        print(depth_list[Id], "done!")