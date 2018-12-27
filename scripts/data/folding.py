import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()
    symbols = ['a', 'b', 'c', 'd']

    filenames = os.listdir(args.dir)
    for filename in filenames:
        full_path = os.path.join(args.dir, filename)
        filename, ext = os.path.splitext(filename)
        for i in range(4):
            new_filename = filename + symbols[i] + ext
            new_fullpath = os.path.join(args.dir, new_filename)
            shutil.copy(full_path, new_fullpath)
 

if __name__ == '__main__':
     main()
