import os
import argparse
import numpy as np
import cv2


def shadowing(im):
    index = np.where(im[:, :, 3] != 0)
    im[index[0], index[1], :3] = 0
    return im


def limited_resize(im):
    height, width = im.shape[:2]
    if height >= width:
        scale = 224 / height
    else:
        scale = 224 / width
    im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()

    filenames = os.listdir(args.dir)
    for filename in filenames:
        full_path = os.path.join(args.dir, filename)
        new_name = os.path.join('images/shadow/', filename)
        im = cv2.imread(full_path, -1)
        shadow_im = shadowing(im)
        shadow_im = limited_resize(shadow_im)
        cv2.imwrite(new_name, shadow_im)


if __name__ == '__main__':
    main()
