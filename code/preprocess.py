import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy import misc
#from skimage.measure import block_reducea
from tqdm import tqdm


def downsample(image, mag_ratio):
    #image_ds = block_reduce(image, block_size=(mag_ratio, mag_ratio, 1), func=np.mean)
    image_ds = misc.imresize(image, 1./mag_ratio, interp='bicubic')
    return image_ds


def upsample(image, size):
    im_up = misc.imresize(image, size=size, interp='bicubic')
    return im_up

def crop(im, height, width):
    im_width, im_height = im.size
    for i in range(im_height//height):
        for j in range(im_width//width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def partiton_images_blockwise(args):

    imgdir = os.path.join(args.indir, 'train')
    filenames = glob.glob(os.path.join(imgdir, "*" + args.ext))
    for ind, currfile in enumerate(filenames):

        im = Image.open(currfile).convert('L')
        imgwidth, imgheight = im.size

        height = imgheight / args.part_ratio
        width = imgwidth / args.part_ratio

        start_num = 0
        for k, piece in enumerate(crop(im, height, width), start_num):

            img = Image.new('L', (width, height), 255)
            img.paste(piece)
            currfile = currfile.split('/')[-1].split('.')[0]
            path = os.path.join(imgdir, currfile + '_crop_{0}.bmp'.format(k + 1))
            img.save(path, 'BMP')


def generate_lowres_dataset(args):
    root = args.indir.rsplit('/', 1)[0]

    if os.path.exists(os.path.join(root, 'lowres')):
        shutil.rmtree(os.path.join(root, 'lowres'))
    shutil.copytree(args.indir, os.path.join(root, 'lowres'))

    kernel = np.ones((5, 5), np.float32) / 25

    for (dirpath, _, _) in os.walk(os.path.join(root, 'lowres')):
        filenames = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(dirpath, '*' + args.ext))]
        for f in tqdm(filenames):
            im = np.array(Image.open(os.path.join(dirpath, f + args.ext)).convert('L'))
            #im = cv2.filter2D(im, -1, kernel)
            original_size = im.shape
            im_ds = downsample(im, args.magnif)

            #im_us = im_ds
            im_us = upsample(im_ds, size=original_size)
            temp = Image.fromarray(im_us, mode='L')
            new_filename = f + args.ext
            temp.save(os.path.join(dirpath, new_filename))



def main(args):

    if args.split:
        partiton_images_blockwise(args)
    if args.downscale:
        generate_lowres_dataset(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='../data')
    parser.add_argument('--ext', type=str, default='.bmp')
    parser.add_argument('--magnif', type=int, default=8)
    parser.add_argument('--part-ratio', type=int, default=16)
    parser.add_argument('--downscale', action='store_true')
    parser.add_argument('--split', action='store_true')

    args = parser.parse_args()
    main(args)
