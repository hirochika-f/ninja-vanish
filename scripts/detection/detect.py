from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, help='path to dataset')
parser.add_argument('--image_path', type=str, default='data/samples', help='path to image')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

def single_load(img_path, img_size=416):
    # Extract image
    img = np.array(Image.open(img_path))
    img_shape = (img_size, img_size)
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
    # Resize and normalize
    input_img = resize(input_img, (*img_shape, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.array([input_img])
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()

    return img_path, input_img

img_path, input_img = single_load(opt.image_path, img_size=opt.img_size) 

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()

# Configure input
input_imgs = Variable(input_img.type(Tensor))

# Get detections
with torch.no_grad():
    detections = model(input_imgs)
    detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

# Log progress
current_time = time.time()
inference_time = datetime.timedelta(seconds=current_time - prev_time)
prev_time = current_time
print ('\t+ Inference Time: %s' % (inference_time))

# Save image and detections
imgs.append(img_path)
img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print ('\nSaving images:')
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    person_center_x = 0
    person_center_y = 0

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        max_bbox_area = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            # Find the biggest person and calculate center of the bbox
            if classes[int(cls_pred)] == 'person':
                bbox_area = box_h * box_w
                if max_bbox_area < bbox_area:
                    person_center_x = int(x1 + box_w / 2)
                    person_center_y = int(y1 + box_h / 2) 
                    max_bbox_area = bbox_area

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    output_filename, ext = os.path.splitext(opt.image_path)
    output_filename += '_detect'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # Print position of the most biggest person
    print(person_center_x, person_center_y)

# Crop biggest person
import equi

im = cv2.imread(opt.image_path)
img_height, img_width = im.shape[:2]
latitude = np.linspace(np.pi/2, -np.pi/2, num=img_height)
longtitude = np.linspace(-np.pi, np.pi, num=img_width)
equ = equi.Equirectangular(im)
fov = 120
theta = np.rad2deg(longtitude[person_center_x])
phi = np.rad2deg(latitude[person_center_y])
height = 1500
width = 1500

with open('/root/position.txt', 'w') as f:
    print(theta, file=f)
    print(phi, file=f)
perspective, lat, lon = equ.get_perspective_image(fov, theta, phi, height, width)
pers_name, ext = os.path.splitext(opt.image_path)
pers_name = pers_name + '_pers' + ext
cv2.imwrite(pers_name, perspective)
print('Save Perspective Image as ' + pers_name)
