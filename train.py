import torch
import torch.optim
import os
import argparse
import time
import dataloader
import model_depth_Drelu as model
from SSIM import SSIM
import torchvision
import matplotlib.pyplot as plt

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):

    dehaze_net = model.MSDFN().cuda()
    dehaze_net.apply(weights_init)
    train_dataset = dataloader.dehazing_loader(config.orig_images_path,config.hazy_images_path,config.label_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path_val,config.hazy_images_path_val,config.label_images_path_val, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)


    criterion = SSIM()
    comput_ssim = SSIM()
    
    dehaze_net.train()

    zt = 1
    Iters = 0                                                                                                           
    indexX = []                                                                                                    
    indexY = []
    for epoch in range(1,config.num_epochs):
        if epoch == 0:
            config.lr = 0.0001
        elif epoch == 1:
            config.lr = 0.00009
        elif epoch >1 and epoch <=3:
            config.lr = 0.00006
        elif epoch >3 and epoch <=5:
            config.lr = 0.00003
        elif epoch >5 and epoch <=7:
            config.lr = 0.00001
        elif epoch >7 and epoch <=9:
            config.lr = 0.000009
        elif epoch >9 and epoch <=11:
            config.lr = 0.000006
        elif epoch >11 and epoch <=13:
            config.lr = 0.000003
        elif epoch >13:
            config.lr = 0.000001
        optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr)
        print("now lr == %f"%config.lr)
        print("*" * 80 + "第%i轮" % epoch + "*" * 80)
        
        for iteration, (img_clean, img_haze,img_depth) in enumerate(train_loader):

            img_clean = img_clean.cuda()
            img_haze = img_haze.cuda()
            img_depth = img_depth.cuda()

            try:
                clean_image = dehaze_net(img_haze,img_depth)
                if config.lossfunc == "MSE":
                    loss = criterion(clean_image, img_clean)                                                             # MSE损失
                else:
                    loss = criterion(img_clean, clean_image)                                                            # -SSIM损失
                    loss = -loss
                # indexX.append(loss.item())
                # indexY.append(iteration)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
                optimizer.step()
                Iters += 1
                if ((iteration + 1) % config.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % config.snapshot_iter) == 0:
                    torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e



        _ssim=[]

        print("start Val!")
        #Validation Stage
        with torch.no_grad():
            for iteration1, (img_clean, img_haze,img_depth) in enumerate(val_loader):
                print("va1 : %s"%str(iteration1))
                img_clean = img_clean.cuda()
                img_haze = img_haze.cuda()
                img_depth = img_depth.cuda()

                clean_image = dehaze_net(img_haze, img_depth)
                
                _s = comput_ssim(img_clean,clean_image)
                _ssim.append(_s.item())
                torchvision.utils.save_image(torch.cat((img_haze,img_clean,clean_image), 0),
                                             config.sample_output_folder + "/epoch%s"%epoch +"/"+ str(iteration1 + 1) + ".jpg")
                torchvision.utils.save_image(clean_image,config.sample_output_folder + "/epoch%s" % epoch + "/" + str(iteration1 + 1) + ".jpg")
            _ssim = np.array(_ssim)
            
            print("-----The %i Epoch mean-ssim is :%f-----" %(epoch,np.mean(_ssim)))

            with open("trainlog/%s%s.log" % (config.lossfunc,config.actfuntion), "a+", encoding="utf-8") as f:
                s = "[%i,%f]" %(epoch,np.mean(_ssim))+ "\n"
                f.write(s)
            indexX.append(epoch+1)
            indexY.append(np.mean(_ssim))
        print(indexX,indexY)
        plt.plot(indexX,indexY,linewidth=2)
        plt.pause(0.1)
    plt.savefig("trainlog/%s%s.png" % (config.lossfunc,config.actfuntion))
    torch.save(dehaze_net.state_dict(), config.snapshots_folder + "MSDFN.pth")

if __name__ == "__main__":
    """
    	:param orig_images_path:无雾图像
    	:param hazy_images_path:有雾彩色图像
    	:param label_images_path:深度图
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="dataset/outdoor/gt/")
    parser.add_argument('--hazy_images_path', type=str, default="dataset/outdoor/hazy/")
    parser.add_argument('--label_images_path', type=str, default="dataset/outdoor/depth/")

    parser.add_argument('--orig_images_path_val', type=str, default="dataset/testdataset/outdoor/gt/")
    parser.add_argument('--hazy_images_path_val', type=str, default="dataset/testdataset/outdoor/hazy/")
    parser.add_argument('--label_images_path_val', type=str, default="dataset/testdataset/outdoor/depth/")

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=100)
    parser.add_argument('--snapshots_folder', type=str, default="trained_model/outdoor/drelu-B/")
    parser.add_argument('--sample_output_folder', type=str, default="sample/outdoor/drelu-B")
    parser.add_argument('--in_or_out', type=str, default="outdoor")
    parser.add_argument('--lossfunc', type=str, default="SSIM",help="choose Loss Function(MSE or -SSIM-b1-n10-aodnet).")
    parser.add_argument('--actfuntion', type=str, default="drelu-B",help="drelu or relu")
    parser.add_argument('--cudaid', type=str, default="0",help="choose cuda device id 0-7).")

    config = parser.parse_args()



    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    for i in range(config.num_epochs):

        path = config.sample_output_folder + "/epoch%s" % str(i)
        if not os.path.exists(path):
            os.mkdir(path)

    s = time.time()
    train(config)
    e = time.time()
    print(str(e-s))
