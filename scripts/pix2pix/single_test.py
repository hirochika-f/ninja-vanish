from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from util import load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--input', required=True, help='facades')
parser.add_argument('--model', type=str, default='checkpoint/facades/netG_model_epoch_200.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

netG = torch.load(opt.model)

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

img = load_img(opt.input)
img = transform(img)
input = Variable(img, volatile=True).view(1, -1, 256, 256)

if opt.cuda:
    netG = netG.cuda()
    input = input.cuda()

out = netG(input)
out = out.cpu()
out_img = out.data[0]
filename, ext = os.path.splitext(opt.input)
print(filename, ext)
output_name = filename + '_result' + ext
save_img(out_img, output_name)
