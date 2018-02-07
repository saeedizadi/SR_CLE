import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy import misc
from skimage.measure import block_reduce
from tqdm import tqdm


def downsample(image, mag_ratio):
    image_ds = block_reduce(image, block_size=(mag_ratio, mag_ratio, 1), func=np.mean)
    return image_ds


def upsample(image, size):
    im_up = misc.imresize(image, size=size, interp='bicubic')
    return im_up


def main(args):
    root = args.indir.rsplit('/', 1)[0]

    if os.path.exists(os.path.join(root, 'lowres')):
        shutil.rmtree(os.path.join(root, 'lowres'))
    shutil.copytree(args.indir, os.path.join(root, 'lowres'))

    kernel = np.ones((5, 5), np.float32) / 25

    for (dirpath, _, _) in os.walk(os.path.join(root, 'lowres')):
        filenames = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(dirpath, '*.jpg'))]
        for f in tqdm(filenames):
            im = np.array(Image.open(os.path.join(dirpath, f + args.ext)).convert('RGB'))
            im = cv2.filter2D(im, -1, kernel)
            original_size = im.shape
            im_ds = downsample(im, args.magnif)
            im_us = upsample(im_ds, size=original_size)
            temp = Image.fromarray(im_us, mode='RGB')
            new_filename = f + args.ext
            temp.save(os.path.join(dirpath, new_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='../data')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--magnif', type=int, default=4)
    args = parser.parse_args()
    main(args)
