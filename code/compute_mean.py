import argparse
import torch
from dataset import SRDataset
import numpy as np

def compute_mean(args):
    dsetTrain  = SRDataset(args.highresdir, args.lowresdir).train_data
    dsetTrain = dsetTrain.astype(np.float32)/255

    print dsetTrain.shape
    mean = []
    std = []

    for i in range(1):
        pixels = dsetTrain[:, :, i].ravel()
        mean.append(np.mean(pixels))
        std.append(np.std(pixels))
    print("means: {}".format(mean))
    print("stdevs: {}".format(std))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--highresdir', type=str, required=True)
    parser.add_argument('--lowresdir', type=str, required=True)
    
    args = parser.parse_args()
    compute_mean(args)
