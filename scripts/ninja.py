import os
import subprocess as sp
import argparse
import numpy as np
import cv2
import equi


def main(args):
    filename, ext = os.path.splitext(args.img)
    resize_name = filename + '_resize'
    pers_name = resize_name + '_pers'
    hollow_name = pers_name + '_hollow'
    interpolated_name = hollow_name + '_result'
    ninja_name = filename + '_ninja'
    resize_name += ext
    pers_name += ext
    hollow_name += ext
    interpolated_name += ext
    ninja_name += ext


    # Resize
    im = cv2.imread(args.img)
    im = cv2.resize(im, (1920, 1080))
    cv2.imwrite(resize_name, im)

    # Detection
    os.chdir('/root/PyTorch-YOLOv3')
    cmd = 'pipenv run python3 detect.py --image_path /root/test_theta.jpg'
    cmd = cmd.split()
    cmd[-1] = resize_name
    sp.call(cmd)

    # Segmentation
    os.chdir('/root/semantic-segmentation-pytorch')
    cmd = 'pipenv run python3 -u test.py --model_path baseline-resnet50dilated-ppm_deepsup --result /root/ --arch_encoder resnet50dilated --test_imgs /root/test_theta_pers.jpg'
    cmd = cmd.split()
    cmd[-1] = pers_name
    sp.call(cmd)

    # Interpolation
    os.chdir('/root/pytorch-pix2pix')
    cmd = 'pipenv run python3 single_test.py  --model checkpoint/ninja_vanish/netG_model_epoch_50.pth --cuda --input /root/test_theta_pers_hollow.jpg'
    cmd = cmd.split()
    cmd[-1] = hollow_name
    sp.call(cmd)

    # Back projection
    with open('/root/position.txt', 'r') as f:
        theta, phi = f.readlines()
        theta = float(theta[:-2])
        phi = float(phi[-2])

    im = cv2.imread(resize_name)
    img_height, img_width = im.shape[:2]

    equ = equi.Equirectangular(im)
    fov = 130
    height = 256
    width = 256

    _, lat, lon = equ.get_perspective_image(fov, theta, phi, height, width)
    interpolated = cv2.imread(interpolated_name)
    back_img = equ.back_perspective_image(interpolated)
    cv2.imwrite(ninja_name, back_img)
    print('Save Perspective Image as ' + ninja_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img')
    args = parser.parse_args()
    main(args)

