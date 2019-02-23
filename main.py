import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.utils as utils
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
import vgg
import argparse

import neural_style_transfer as N
import transform_net as T
import misc

def main():
    parser = argparse.ArgumentParser(description='Pytorch implementation of Neural Artistic Style Transfer')
    parser.add_argument('--w_content', default=80.0, type=float, help='Weight for content loss')
    parser.add_argument('--w_style', default=1.0, type=float, help='Weight for style loss')
    parser.add_argument('--img_content', default='content.jpg', help='Image name for content')
    parser.add_argument('--img_style', default='style.jpg', help='Image name for style')
    parser.add_argument('--iteration', '-i', default=50, type=int, help='Total iteration')
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=int, help='Learning Rate')
    parser.add_argument('--batch_size', '-bs', default=1, type=int, help='Batch size')
    parser.add_argument('--image_size', '-is', default=256, type=int, help='Image size')
    args = parser.parse_args()

    ### Setting parameters ###
    w_content = args.w_content
    w_style = args.w_style
    iteration = args.iteration
    lr = args.learning_rate
    batch_s = args.batch_size
    image_s = args.image_size

    ### Load Model ###
    vggnet = vgg.vgg19(pretrained=True).cuda().eval()
    resnet = T.TransformNet().cuda().train()

    ### Load Images ###
    image_style, image_content = N.image_loader(args.img_style, args.img_content, batch_s, image_s)
    #image_modify = image_content.clone() 
    #image_modify.requires_grad = True
    train_loader, _, _ = misc.load_lsun(batch_s, image_s)

    ### Iteration ###
    optimi = optim.Adam(resnet.parameters(), lr=lr)
    print('entering epoch')
    for epoch in range(iteration):
        for batch_idx, batch_data in enumerate(train_loader): 
            optimi.zero_grad()
            batch_img, _ = batch_data
            batch_img = batch_img.cuda()

            image_resnet = resnet(batch_img)
            net_m, content_losses, style_losses = N.get_layer_out(vggnet, batch_img, image_style)
            net_m(image_resnet)

            content_loss_sum = 0.0
            style_loss_sum = 0.0
            for c in content_losses:
                content_loss_sum += c.loss
            for s in style_losses:
                style_loss_sum += s.loss
            loss = style_loss_sum * w_style + content_loss_sum * w_content
            loss.backward()
            if True: 
                print('epoch: {}, batch: {},  loss: {} / {} / {}'.format(epoch, batch_idx, loss.data, style_loss_sum.data*w_style, content_loss_sum.data*w_content))
            optimi.step()
            
            if batch_idx % 100 == 0:
                utils.save_image(torch.squeeze(image_resnet[0]), 'output_train_e{}b{}.jpg'.format(epoch, batch_idx))
                utils.save_image(torch.squeeze(batch_img[0]), 'output_train_gt_e{}b{}.jpg'.format(epoch, batch_idx))
                image_test = resnet(image_content)
                utils.save_image(torch.squeeze(image_test[0]), 'output_test_e{}b{}.jpg'.format(epoch, batch_idx))
                print(torch.max(batch_img), torch.max(image_test), torch.max(image_content))
                print(torch.min(batch_img), torch.min(image_test), torch.min(image_content))
            if batch_idx % 5000 == 0:
                torch.save(resnet.state_dict(), './saved_model/model_e{}b{}.pt'.format(epoch, batch_idx))

if __name__ == '__main__':
    main()
