import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class SRDataset(Dataset):
    def __init__(self, highres_root, lowres_root, ext='bmp', transform=None):
        super(SRDataset, self).__init__()

        self.highres_root = highres_root
        self.lowres_root = lowres_root
        self.ext = ext
        self.transform = transform

        self.filenames = [k.split('/')[-1].split('.')[0] for k in glob(os.path.join(self.highres_root, '*.' + ext))]
        self.train_data = []
        for fname in self.filenames:
            im = Image.open(os.path.join(self.lowres_root, fname + '.' + self.ext)).convert('L')
            self.train_data.append(im)

        self.train_data = np.stack(self.train_data, axis=0)

    def __getitem__(self, index):
        SRfilename = os.path.join(self.highres_root, self.filenames[index] + '.' + self.ext)
        LRfilename = os.path.join(self.lowres_root, self.filenames[index] + '.' + self.ext)

        im = Image.open(SRfilename).convert('RGB')
        target = Image.open(LRfilename).convert('RGB')

        if self.transform is not None:
            im, target = self.transform(im, target)

        return im, target

    def __len__(self):
        return len(self.filenames)
