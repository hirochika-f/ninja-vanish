import os
import argparse
import numpy as np
import cv2
import shadowing


def masking(shadow):
    mask = shadow[:, :, 3]
    index = np.where(mask != 0)
    mask[index] = 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = np.array(mask / 255.0, dtype=np.uint8)
    return mask


def random_resize(im):
    max_scale = 1
    min_scale = 0.4
    scale = (max_scale - min_scale) * np.random.rand() + min_scale
    im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
    return im


def random_paste(scene, mask):
    height, width = mask.shape[:2]
    if width == 224:
        x = 0
    else:
        x = np.random.randint(0, 224-width)
    if height == 224:
        y = 0
    else:
        y = np.random.randint(0, 224-height)
    scene[y:y+height, x:x+width] *= 1 - mask
    return scene


def main():
    scene_dir = 'images/scene/'
    shadow_dir = 'images/shadow/'
    hollow_dir = 'images/hollow/'

    scenes = os.listdir(scene_dir)
    shadows = os.listdir(shadow_dir)
    
    for scene_name in scenes:
        answer_name = scene_name
        scene_name = os.path.join(scene_dir, scene_name)
        scene = cv2.imread(scene_name)
        shadow_name = np.random.choice(shadows)
        shadow_name = os.path.join(shadow_dir, shadow_name)
        person = cv2.imread(shadow_name, -1)
        shadow = shadowing.shadowing(person)
        shadow = random_resize(shadow)
        mask = masking(shadow)
        scene = random_paste(scene, mask)
        hollow_name = os.path.join(hollow_dir, answer_name)
        cv2.imwrite(hollow_name, scene)


if __name__ == '__main__':
    main()
