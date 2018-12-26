import os
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()

    filenames = os.listdir(args.dir)
    for filename in filenames:
        full_path = os.path.join(args.dir, filename)
        im = cv2.imread(full_path)
        im = cv2.resize(im, (224, 224))
        cv2.imwrite(full_path, im)


if __name__ == '__main__':
    main()
