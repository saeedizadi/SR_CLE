import glob
import os
import shutil
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

import co_transforms as co_transforms
from conf import get_arguments
from dataset import SRDataset
from evaluation import PSNR
from generator import SRResNet
from visualize import Dashboard
from SRDenseNet import SRDenseNet


def prepare_data(sr_dir, lr_dir, patch_size, batch_size, mode='train', shuffle=True):
    if mode is 'val':
        transform = co_transforms.Compose(
            [co_transforms.Grayscale(),
             co_transforms.RandomCrop(patch_size, patch_size),
             co_transforms.ToTensor()])

    elif mode is 'train':
        transform = co_transforms.Compose(
            [co_transforms.Grayscale(),
             co_transforms.RandomCrop(patch_size, patch_size),
             co_transforms.RandomHorizontalFlip(),
             co_transforms.RandomVerticalFlip(),
             co_transforms.ToTensor()])
    else:
        transform = co_transforms.Compose(
            [co_transforms.Grayscale(),
             co_transforms.ToTensor()])

    dset = SRDataset(highres_root=sr_dir, lowres_root=lr_dir, transform=transform)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    return dloader


def train(model, trData, optimizer, lossfn, batch_size, lowres_dim, cuda=True):
    train_loss = 0.
    psnr_sum = 0.

    psnr = PSNR()

    model.train()
    downsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(lowres_dim), transforms.ToTensor()])
    for step, (high, _) in enumerate(trData):

        # mat1 = np.transpose(high[0].numpy(), (1, 2, 0))
        # mat2 = np.transpose(low[0].numpy(), (1, 2, 0))
        # final_frame = cv2.hconcat((mat1, mat2))
        # cv2.imshow("", final_frame)
        # cv2.waitKey(0)

        # -- Removes online downsampling ---
        low = torch.FloatTensor(high.size()[0], 1, lowres_dim, lowres_dim)
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
        psnr_sum += psnr(high.data.cpu().numpy(), output.data.cpu().numpy())

    return float(train_loss) / len(trData.dataset), float(psnr_sum) / len(trData.dataset)


def validate(model, vlData, lossfn, batch_size, lowres_dim, cuda=True):
    val_loss = 0.
    model.eval()
    # downsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(lowres_dim), transforms.ToTensor()])
    for step, (high, _) in enumerate(vlData):

        # --- removes online downsampling
        low = torch.FloatTensor(high.size()[0], 1, lowres_dim, lowres_dim)
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


def test(model, testData, savedir, lowres_dim, cuda=True):
    psnr = PSNR()

    psnr_sum = 0.
    ssim_sum = 0.

    model.eval()
    # downsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(lowres_dim), transforms.ToTensor()])
    toImage = transforms.ToPILImage()
    batch_size = 5
    for step, (high, low) in enumerate(testData):

        # --- removes online downsampling ---
        # low = torch.FloatTensor(high.size()[0], 3, lowres_dim, lowres_dim)
        # for j in range(high.size()[0]):
        #     low[j] = downsample(high[j])

        if cuda:
            low = low.cuda()
            high = high.cuda()
        low = Variable(low, volatile=True)
        high = Variable(high, volatile=True)
        output = model(low)
        filenames = testData.dataset.filenames
        for j in range(output.size()[0]):
            index = step * batch_size + j
            output_img = toImage(output[j].data.cpu())
            res_filename = os.path.join(args.savedir, filenames[index] + '_result.bmp')
            output_img.save(res_filename, 'BMP')

            # --- compute the required scores ---
            psnr_sum += psnr(high.data.cpu().numpy(), output.data.cpu().numpy())
            #ssim_sum += ssim(high.data.cpu().numpy(), output.data.cpu().numpy())

    return psnr_sum / len(testData.dataset), #ssim_sum / len(testData.dataset)


def save_snapshot(state, filename='checkpoint.pth.tar', savedir='./checkpoints'):
    fullname = os.path.join(savedir, filename)
    torch.save(state, fullname)


def show_results(highdir, lowdir, resdir, port=8097):
    dashboard = Dashboard(port=port)
    filenames = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(highdir, '*.jpg'))]
    shuffle(filenames)



    batch = np.empty((0, 3, 1024, 1024))

    for i in range(20):
        currfile = filenames[i]
        im = np.array(Image.open(os.path.join(highdir, currfile + ".jpg")).convert('RGB'))
        im = im.transpose((2, 0, 1))
        batch = np.append(batch, im[np.newaxis, :, :, :], axis=0)

        im = np.array(Image.open(os.path.join(lowdir, currfile + ".jpg")).convert('RGB'))
        im = im.transpose((2, 0, 1))
        batch = np.append(batch, im[np.newaxis, :, :, :], axis=0)

        im = np.array(Image.open(os.path.join(resdir, currfile + "_result.bmp")).convert('RGB'))
        im = im.transpose((2, 0, 1))
        batch = np.append(batch, im[np.newaxis, :, :, :], axis=0)

    dashboard.grid_plot(batch, nrow=3)


def main(args):
    # --- load data ---

    # --- define the model and NN settings ---
    # model = SRResNet(16, 1)
    model = SRDenseNet(8)
    criterion = nn.MSELoss()

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # --- start training the network
    if args.mode == "train":
        trLoader = prepare_data(sr_dir=args.srtraindir, lr_dir=args.lrtraindir, patch_size=args.patch_size,
                                batch_size=args.batch_size, mode='train')
        valLoader = prepare_data(sr_dir=args.srvaldir, lr_dir=args.lrvaldir, patch_size=args.patch_size,
                                 batch_size=args.batch_size, mode='val')

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=1e-08)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

        best_loss = float('inf')
        for epoch in range(1, args.num_epochs + 1):
            scheduler.step()

            train_loss, train_psnr = train(model=model, trData=trLoader, optimizer=optimizer, lossfn=criterion,
                               batch_size=args.batch_size, lowres_dim=args.patch_size / args.downscale_ratio,cuda=args.cuda)

            val_loss = validate(model=model, vlData=valLoader, lossfn=criterion,
                                batch_size=args.batch_size, lowres_dim=args.patch_size / args.downscale_ratio, cuda=args.cuda)

            if epoch % args.log_step == 0:

                filename = 'checkpoint_{0:02}.pth.tar'.format(epoch)
                save_snapshot({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                              filename=filename, savedir=args.savedir)

                print('[Epoch: {0:02}/{1:02}]'
                      '\t[TrainLoss:{2:.4f}]'
                      '\t[TrainPSNR:{3:.4f}]'
                      '\t[ValLoss:{4:.4f}]').format(epoch, args.num_epochs, train_loss, train_psnr, val_loss),
                print('\t [Snapshot]')

                if val_loss < best_loss:
                    best_fullname = os.path.join(args.savedir, 'checkpoint_best.pth.tar')
                    shutil.copyfile(os.path.join(args.savedir, filename), best_fullname)
                    best_loss = train_loss
                continue

            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss:{2:.4f}]'
                  '\t[TrainPSNR:{3:.4f}]'
                  '\t[ValLoss:{4:.4f}]').format(epoch, args.num_epochs, train_loss, train_psnr, val_loss)

    elif args.mode == 'test':
    
        testLoader = prepare_data(sr_dir=args.srtestdir, lr_dir=args.lrtestdir, patch_size='',
                                  batch_size=args.batch_size, mode='test', shuffle=False)
    
        filename = 'checkpoint_{0:02}.pth.tar'.format(args.state)
        checkpoint = torch.load(os.path.join(args.weightdir, filename))
    
        model.load_state_dict(checkpoint['state_dict'])
    
        psnr = test(model, testLoader, args.savedir, lowres_dim=args.image_size / args.downscale_ratio, cuda=args.cuda)
        print('[PSNR: {0:.4f}]'.format(float(psnr[0])))
    
    elif args.mode == "show":
        show_results(args.hrdir, args.lrdir, args.resdir, port=args.visdom_port)


if __name__ == '__main__':
    args = get_arguments()
    main(args)

