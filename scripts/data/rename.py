import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    args = parser.parse_args()

    filenames = os.listdir(args.dir)
    i = 0
    for filename in filenames:
        full_path = os.path.join(args.dir, filename)
        new_name = 'b_' + str(i) + '.jpg'
        new_name = os.path.join(args.dir, new_name)
        os.rename(full_path, new_name)
        i = i + 1


if __name__ == '__main__':
    main()
