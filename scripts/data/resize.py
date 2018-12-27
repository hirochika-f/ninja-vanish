import os
import argparse
import cv2


def uniform_resize(dir_name):
    filenames = os.listdir(dir_name)
    for filename in filenames:
        full_path = os.path.join(dir_name, filename)
        im = cv2.imread(full_path)
        im = cv2.resize(im, (224, 224))
        cv2.imwrite(full_path, im)


def limited_resize(dir_name):
    filenames = os.listdir(dir_name)
    for filename in filenames:
        full_path = os.path.join(dir_name, filename)
        im = cv2.imread(full_path, -1)
        height, width = im.shape[:2]
        if height >= width:
            scale = 224 / height
        else:
            scale = 224 / width
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        cv2.imwrite(full_path, im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()

    uniform_resize(args.dir)
    # limited_resize(args.dir)

    
if __name__ == '__main__':
    main()
