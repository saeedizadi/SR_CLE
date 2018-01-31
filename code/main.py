import torch
import co_transforms as co_transforms
import torchvision.transforms as transforms
import torch.utils.data as data
from conf import get_arguments
from dataset import SRDataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn


from generator import Generator
import cv2
import numpy as np

def prepare_data(sr_dir,lr_dir, patch_size, batch_size):

    # --- transform the input data ---
    transform = co_transforms.Compose([co_transforms.RandomCrop(patch_size, patch_size), co_transforms.ToTensor()])

    dset = SRDataset(highres_root=sr_dir, lowres_root=lr_dir, transform=transform)

    # --- contstruct the data loader
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True)

    return dloader


def train(model, trData, optimizer, lossfn, epoch, scale_transform, batch_size, lowres_dim, cuda=True):

    for step, (image , target) in enumerate(trData):

        mat1 = np.transpose(image[0].numpy(), (1, 2, 0))
        mat2 = np.transpose(target[0].numpy(), (1, 2, 0))
        final_frame = cv2.hconcat((mat1, mat2))
        cv2.imshow("", final_frame )
        cv2.waitKey(0)

	lowres_img = torch.FloatTensor(batch_size, 3, lowres_dim , lowres_dim)
        for j in range(args.batch_size):
            lowres_img[j] = scale_transform(target[j])


	
#        mat1 = np.transpose(lowres_img[0].numpy(), (1, 2, 0))
#        mat2 = np.transpose(target[0].numpy(), (1, 2, 0))
#        #final_frame = cv2.hconcat((mat1, mat2))
#        cv2.imshow("1", mat1 )
#        cv2.imshow("2", mat2 )
#        cv2.waitKey(0)

        if cuda:
            lowres_img = lowres_img.cuda()
            target = target.cuda()

	lowres_img = Variable(lowres_img)
	target = Variable(target)

        optimizer.zero_grad()


        output = model(lowres_img)
	print output.size()
        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()



def main(args):

    trLoader = prepare_data(sr_dir=args.srimgdir, lr_dir=args.lrimgdir, patch_size=args.patch_size,
                            batch_size=args.batch_size)

    scale = transforms.Compose([transforms.ToPILImage(), transforms.Resize(args.patch_size/args.downscale_ratio), transforms.ToTensor()])

    model = Generator(5, 2)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    criterion= nn.MSELoss()

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()




    for epoch in range(args.num_epochs):
        # --- add some visualization here ---
        train(model=model, trData=trLoader, optimizer=optimizer, lossfn = criterion, epoch = epoch, scale_transform=scale, batch_size=args.batch_size, lowres_dim=args.patch_size/args.downscale_ratio)












if __name__ == '__main__':
    args = get_arguments()
    main(args)
