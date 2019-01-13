import os
import subprocess as sp
import argparse


def main(args):
    filename, ext = os.path.splitext(args.img)
    pers_name = filename + '_pers'
    hollow_name = pers_name + '_hollow'
    interpolated_name = hollow_name + '_result'
    pers_name += ext
    hollow_name += ext
    interpolated_name += ext

    # Detection
    os.chdir('/root/PyTorch-YOLOv3')
    cmd = 'pipenv run python3 detect.py --image_path /root/test_theta.jpg'
    cmd = cmd.split()
    cmd[-1] = pers_name
    sp.call(cmd)

    # Segmentation
    os.chdir('/root/semantic-segmentation-pytorch')
    cmd = 'pipenv run python3 -u test.py --model_path baseline-resnet50dilated-ppm_deepsup --result /root/ --arch_encoder resnet50dilated --test_imgs /root/test_theta_pers.jpg'
    cmd = cmd.split()
    cmd[-1] = hollow_name
    sp.call(cmd)

    # Interpolation
    os.chdir('/root/pytorch-pix2pix')
    cmd = 'pipenv run python3 single_test.py  --model checkpoint/ninja_vanish/netG_model_epoch_50.pth --cuda --input /root/test_theta_pers_hollow.jpg'
    cmd = cmd.split()
    cmd[-1] = interpolated_name
    sp.call(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img')
    args = parser.parse_args()
    main(args)

