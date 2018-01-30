
import transforms as transforms
import torch.utils.data as data
from conf import get_arguments
from dataset import SRDataset
import cv2
import numpy as np

def prepare_data(sr_dir,lr_dir, patch_size, batch_size):

    # --- transform the input data ---
    transform = transforms.Compose([transforms.RandomCrop(patch_size, patch_size), transforms.ToTensor()])

    dset = SRDataset(highres_root=sr_dir, lowres_root=lr_dir, transform=transform)

    # --- contstruct the data loader
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True)

    return dloader


def main(args):

    trLoader = prepare_data(sr_dir=args.srimgdir, lr_dir=args.lrimgdir, patch_size=args.patch_size, batch_size=args.batch_size)
    for i, (image,target) in enumerate(trLoader):

        # mat1 = np.transpose(image[0].numpy(), (1, 2, 0))
        # mat2 = np.transpose(target[0].numpy(), (1, 2, 0))
        # final_frame = cv2.hconcat((mat1, mat2))
        # cv2.imshow("", final_frame )
        # cv2.waitKey(0)









if __name__ == '__main__':
    args = get_arguments()
    main(args)
