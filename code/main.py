import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

import co_transforms as co_transforms
from conf import get_arguments
from dataset import SRDataset
from generator import Generator


def prepare_data(sr_dir, lr_dir, patch_size, batch_size, val=False):
    if val:
        transform = co_transforms.Compose([co_transforms.RandomCrop(patch_size, patch_size), co_transforms.ToTensor()])
    else:
        transform = co_transforms.Compose(
            [co_transforms.RandomCrop(patch_size, patch_size), co_transforms.RandomHorizontalFlip(),
             co_transforms.RandomVerticalFlip(), co_transforms.RandomRotation((0, 90)), co_transforms.ToTensor()])

    dset = SRDataset(highres_root=sr_dir, lowres_root=lr_dir, transform=transform)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True)
    return dloader


def train(model, trData, optimizer, lossfn, batch_size, lowres_dim, cuda=True):
    train_loss = 0.
    model.train()
    downsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(lowres_dim), transforms.ToTensor()])
    for step, (high, _) in enumerate(trData):

        #        mat1 = np.transpose(image[0].numpy(), (1, 2, 0))
        #        mat2 = np.transpose(target[0].numpy(), (1, 2, 0))
        #        final_frame = cv2.hconcat((mat1, mat2))
        #        cv2.imshow("", final_frame )
        #        cv2.waitKey(0)
        low = torch.FloatTensor(high.size()[0], 3, lowres_dim, lowres_dim)
        for j in range(high.size()[0]):
            low[j] = downsample(high[j])

        if cuda:
            low = low.cuda()
            high = high.cuda()

        low = Variable(low)
        high = Variable(high)

        optimizer.zero_grad()
        output = model(low)
        loss = lossfn(output, high)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy()

    return float(train_loss) / len(trData)


def validate(model, vlData, lossfn, batch_size, lowres_dim, cuda=True):
    val_loss = 0.
    model.eval()
    downsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(lowres_dim), transforms.ToTensor()])
    for step, (high, _) in enumerate(vlData):

        low = torch.FloatTensor(high.size()[0], 3, lowres_dim, lowres_dim)
        for j in range(high.size()[0]):
            low[j] = downsample(high[j])

        if cuda:
            low = low.cuda()
            high = high.cuda()

        low = Variable(low, volatile=True)
        high = Variable(high, volatile=True)

        output = model(low)
        loss = lossfn(output, high)
        val_loss += loss.data.cpu().numpy()

    return float(val_loss) / len(vlData)


def save_snapshot(state, filename='checkpoint.pth.tar', savedir='./checkpoints', is_best=False):
    fullname = os.path.join(savedir, filename)
    torch.save(state, fullname)
    if is_best:
        best_fullname = os.path.join(savedir, 'checkpoint_best.pth.tar')
        shutil.copyfile(fullname, best_fullname)

    print('\t[Snapshot]')


def main(args):
    # --- load data ---
    trLoader = prepare_data(sr_dir=args.srtraindir, lr_dir=args.lrtraindir, patch_size=args.patch_size,
                            batch_size=args.batch_size)
    valLoader = prepare_data(sr_dir=args.srvaldir, lr_dir=args.lrvaldir, patch_size=args.patch_size,
                             batch_size=args.batch_size, val=True)

    # --- define the model and NN settings ---
    model = Generator(5, 2)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    criterion = nn.MSELoss()


    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # --- start training the network
    best_loss = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(model=model, trData=trLoader, optimizer=optimizer, lossfn=criterion,
                           batch_size=args.batch_size, lowres_dim=args.patch_size / args.downscale_ratio)

        val_loss = validate(model=model, vlData=valLoader, lossfn=criterion,
                            batch_size=args.batch_size, lowres_dim=args.patch_size / args.downscale_ratio)

        if epoch % args.log_step == 0:

            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss:{2:.4f}]'
                  '\t[ValLoss:{3:.4f}]').format(epoch, args.num_epochs, train_loss, val_loss),
            filename = 'checkpoint_{0:02}.pth.tar'.format(epoch)
            if val_loss < best_loss:
                save_snapshot({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                              filename=filename, savedir=args.savedir, is_best=True)
                best_loss = train_loss
                continue

            save_snapshot({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                          filename=filename, savedir=args.savedir)

        print('[Epoch: {0:02}/{1:02}]'
              '\t[TrainLoss:{2:.4f}]'
              '\t[ValLoss:{3:.4f}]').format(epoch, args.num_epochs, train_loss, val_loss)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
