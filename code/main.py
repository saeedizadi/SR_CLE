
import torchvision.datasets as datasets
import transforms as transforms
# import torchvision.transforms as transforms
import torch.utils.data as data
from conf import get_arguments
from dataset import SRDataset

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
        print i
        print image.size()
        print target.size()








if __name__ == '__main__':
    args = get_arguments()
    main(args)
