import numpy as np
from skimage.measure import block_reduce
from scipy import misc
import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import cv2



def downsample(image, mag_ratio):
    image_ds = block_reduce(image, block_size=(mag_ratio, mag_ratio, 1), func=np.mean)
    return image_ds

def upsample(image, size):
    im_up = misc.imresize(image, size=size, interp='bicubic')
    return im_up

def main(args):


    args.outdir = args.indir
    kernel = np.ones((5,5), np.float32)/25
    filenames = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(args.indir,'*.jpg'))]



    for f in tqdm(filenames):

        im = np.array(Image.open(os.path.join(args.indir, f + args.ext)))
        im = cv2.filter2D(im, -1, kernel)

        original_size = im.shape

        im_ds = downsample(im, 8)
        im_us = upsample(im_ds, size=original_size)
        temp = Image.fromarray(im_us)
        new_filename = f + '_lowres' + args.ext

        temp.save(os.path.join(args.outdir, new_filename))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='../data')
    parser.add_argument('--outdir', type=str, default='../data/out')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--magnif', type=int, default=4)
    args = parser.parse_args()
    main(args)
