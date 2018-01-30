from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import numpy as np

class SRDataset(Dataset):
    def __init__(self, highres_root, lowres_root, ext = 'jpg', transform=None):
        super(SRDataset, self).__init__()

        self.highres_root = highres_root
        self.lowres_root = lowres_root
        self.transform = transform

        self.SRfilenames = [k for k in glob(os.path.join(self.highres_root, '*.' + ext))]
        self.LRfilenames = [k for k in glob(os.path.join(self.lowres_root, '*.' + ext))]


    def __getitem__(self, index):
        im = Image.open(self.SRfilenames[index]).convert('RGB')
        target = Image.open(self.LRfilenames[index]).convert('RGB')


        if self.transform is not None:
            im, target = self.transform(im, target)


        return im, target



    def __len__(self):
        return len(self.SRfilenames)

