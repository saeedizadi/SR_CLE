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
from discriminator import Discrimantor
from evaluation import PSNR
from srdensenet import SRDenseNet_ALL
from visualize import Dashboard


def prepare_data(sr_dir, lr_dir, patch_size, batch_size, mode='train', shuffle=True):
    if mode is 'val':
        transform = co_transforms.Compose(
            [co_transforms.Grayscale(),
             # co_transforms.RandomCrop(patch_size, patch_size),
             co_transforms.ToTensor(), ])
    #             co_transforms.Normalize(mean=[0.3787], std=[0.2464])])

    elif mode is 'train':
        transform = co_transforms.Compose(
            [co_transforms.Grayscale(),
             # co_transforms.RandomCrop(patch_size, patch_size),
             co_transforms.RandomHorizontalFlip(),
             co_transforms.RandomVerticalFlip(),
             co_transforms.ToTensor(), ])
        #            co_transforms.Normalize(mean=[0.3787], std=[0.2464])])
    else:
        transform = co_transforms.Compose(
            [co_transforms.Grayscale(),
             co_transforms.ToTensor(), ])
    #            co_transforms.Normalize(mean=[0.3787], std=[0.2464])])

    dset = SRDataset(highres_root=sr_dir, lowres_root=lr_dir, transform=transform)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    return dloader


def train(model, trData, optimizer, lossfn, batch_size, lowres_dim, cuda=True):
    train_loss = 0.
    psnr_sum = 0.

    psnr = PSNR()

    model.train()
    downsample = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(lowres_dim, Image.BICUBIC), transforms.ToTensor()])
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
        psnr_sum += (low.size()[0] * psnr(high.data.cpu().numpy(), output.data.cpu().numpy()))

    return float(train_loss) / len(trData.dataset), float(psnr_sum) / len(trData.dataset)

def gan_train(model, trData, optimizer, lossfn, batch_size, lowres_dim, cuda=True):
    G_train_loss = 0.
    D_train_loss = 0.
    psnr_sum = 0.

    psnr = PSNR()
    downsample = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(lowres_dim, Image.BICUBIC), transforms.ToTensor()])
    for step, (high, _) in enumerate(trData):

        low = torch.FloatTensor(high.size()[0], 1, lowres_dim, lowres_dim)
        for j in range(high.size()[0]):
            low[j] = downsample(high[j])

        if cuda:
            low = low.cuda()
            high = high.cuda()


        # --- Train the discriminator Real ---
        for p in model['discriminator'].parameters():
            p.requires_grad = True

        model['discriminator'].zero_grad()

        model['discriminator'].train()
        model['generator'].eval()

        D_r_targets = Variable(torch.ones(high.size()[0]).cuda())
        D_r_output = model['discriminator'](high)
        D_r_loss = lossfn['discriminator'](D_r_output, D_r_targets)

        # --- Train the discriminator Fake---
        G_outputs = model['generator'](low)
        G_outputs = G_outputs .detach()

        D_f_targets = Variable(torch.zeros(high.size()[0]).cuda())
        D_f_output = model['discriminator'](G_outputs)
        D_f_loss = lossfn['discriminator'](D_f_output, D_f_targets)

        D_loss = (D_f_loss + D_r_loss)

        D_loss.backward()
        optimizer['discriminator'].step()
        D_train_loss += D_loss.data.cpu().numpy()


        # --- Train the generator ---
        for p in model['discriminator'].parameters():
            p.requires_grad = False

        model['generator'].zero_grad()

        model['discriminator'].eval()
        model['generator'].train()

        G_outputs = model['generator'](low)
        loss = lossfn['generator'](G_outputs, high)

        D_r_targets = Variable(torch.ones(high.size()[0], 1).cuda())
        output = model['discriminator'](G_outputs)
        G_loss = lossfn['discriminator'](output, D_r_targets)

        G_loss = G_loss + 100 * loss
        G_loss.backward()
        optimizer['discriminator'].step()

        G_train_loss += loss.data.cpu().numpy()
        psnr_sum += (low.size()[0] * psnr(high.data.cpu().numpy(), output.data.cpu().numpy()))

    return float(G_train_loss) / len(trData.dataset), float(psnr_sum) / len(trData.dataset), float(D_train_loss) / len(trData.dataset)





def validate(model, vlData, lossfn, batch_size, lowres_dim, cuda=True):
    val_loss = 0.
    model.eval()
    downsample = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(lowres_dim, Image.BICUBIC), transforms.ToTensor()])
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
    downsample = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(lowres_dim, Image.BICUBIC), transforms.ToTensor()])
    toImage = transforms.ToPILImage()
    batch_size = 4
    for step, (high, _) in enumerate(testData):

        # --- removes online downsampling ---
        low = torch.FloatTensor(high.size()[0], 1, lowres_dim, lowres_dim)
        for j in range(high.size()[0]):
            low[j] = downsample(high[j])

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
            res_filename = os.path.join(args.savedir, filenames[index] + '_srdensenet.bmp')
            output_img.save(res_filename, 'BMP')

            # --- compute the required scores ---
            psnr_sum += psnr(high.data.cpu().numpy(), output.data.cpu().numpy())
            # ssim_sum += ssim(high.data.cpu().numpy(), output.data.cpu().numpy())

    return psnr_sum / len(testData.dataset),  # ssim_sum / len(testData.dataset)


def save_snapshot(state, filename='checkpoint.pth.tar', savedir='./checkpoints'):
    fullname = os.path.join(savedir, filename)
    torch.save(state, fullname)


def show_results(highdir, lowdir, resdir, methods, port=8097):
    dashboard = Dashboard(port=port)
    filenames = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(highdir, '*.bmp'))]
    shuffle(filenames)

    batch = np.empty((0, 1024, 1024))
    for i in range(5):
        currfile = filenames[i]
        im = np.array(Image.open(os.path.join(highdir, currfile + ".bmp")).convert('L'))
        # im = im.transpose((2, 0, 1))
        batch = np.append(batch, im[np.newaxis, :, :], axis=0)
        for m in methods:
            im = Image.open(os.path.join(resdir, currfile + '_' + m + ".bmp")).convert('L')
            # background = Image.new('RGB', (1024,1024), (255,255,255))
            # bg_w, bg_h = background.size
            # offset = ((bg_w - im_w) / 2, (bg_h - im_h) / 2)
            # background.paste(im, offset)
            # im = np.array(background)
            #     im = im.transpose((2, 0, 1))
            batch = np.append(batch, im[np.newaxis, :, :], axis=0)

            # im = np.array(Image.open(os.path.join(resdir, currfile + "_result.bmp")).convert('RGB'))
            # im = im.transpose((2, 0, 1))
            # batch = np.append(batch, im[np.newaxis, :, :, :], axis=0)
    dashboard.grid_plot(batch, nrow=len(m) + 1)


def quantitative_evaluate(hrdir, resdir, methods):
    psnr = PSNR()
    scores = dict()

    filenames = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(hrdir, '*.bmp'))]
    for m in methods:
        images = []
        results = []
        for f in filenames:
            im = np.array(Image.open(os.path.join(hrdir, '.bmp')).convert('L'))
            images.append(im)
            res = np.array(Image.open(os.path.join(resdir, '_' + m + '.bmp')).convert('L'))
            results.append(res)

        images = np.stack(images, axis=0)
        results = np.stack(results, axis=0)
        print images.shape
        print results.shape

        scores[m] = PSNR(images, res)

    print scores


def main(args):
    # --- start training the network
    if args.mode == "train":

        # model = SRResNet(16, 1)
        model = SRDenseNet_ALL(8)
        criterion = nn.L1Loss()

        if args.cuda:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(2)).cuda()
            criterion = criterion.cuda()

        trLoader = prepare_data(sr_dir=args.srtraindir, lr_dir=args.lrtraindir, patch_size=args.patch_size,
                                batch_size=args.batch_size, mode='train')
        # valLoader = prepare_data(sr_dir=args.srvaldir, lr_dir=args.lrvaldir, patch_size=args.patch_size,
        #                         batch_size=args.batch_size/4, mode='val')

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

        epoch_start = 1
        if args.resume:
            filename = 'checkpoint_{0:02}.pth.tar'.format(args.state)
            checkpoint = torch.load(os.path.join(args.savedir, filename))

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']

        best_loss = float('inf')
        for epoch in range(epoch_start, args.num_epochs + 1):
            scheduler.step()

            train_loss, train_psnr = train(model=model, trData=trLoader, optimizer=optimizer, lossfn=criterion,
                                           batch_size=args.batch_size,
                                           lowres_dim=args.patch_size / args.downscale_ratio, cuda=args.cuda)

            # --- commented since due to error ---
            # val_loss = validate(model=model, vlData=valLoader, lossfn=criterion,
            #                   batch_size=args.batch_size, lowres_dim=args.image_size / args.downscale_ratio, cuda=args.cuda)
            val_loss = 0.0

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

    if args.mode == "gan_train":

        # model = SRResNet(16, 1)
        model['generator'] = SRDenseNet_ALL(8).cuda()
        model['discriminator'] = Discrimantor().cuda()

        criterion['generator'] = nn.L1Loss().cuda()
        criterion['discriminator'] = nn.BCELoss().cuda()

        model = torch.nn.DataParallel(model, device_ids=range(2)).cuda()

        trLoader = prepare_data(sr_dir=args.srtraindir, lr_dir=args.lrtraindir, patch_size=args.patch_size,
                                batch_size=args.batch_size, mode='train')
        # valLoader = prepare_data(sr_dir=args.srvaldir, lr_dir=args.lrvaldir, patch_size=args.patch_size,
        #                         batch_size=args.batch_size/4, mode='val')

        optimizer['generator'] = optim.SGD(model['generator'].parameters(), lr=args.learning_rate,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)
        optimizer['discriminator'] = optim.SGD(model['discriminator'].parameters(), lr=args.learning_rate,
                                               momentum=args.momentum,
                                               weight_decay=args.weight_decay)

        best_loss = float('inf')
        for epoch in range(1, args.num_epochs + 1):
            G_train_loss, train_psnr, D_train_loss = gan_train(model=model, trData=trLoader, optimizer=optimizer,
                                                               lossfn=criterion,
                                                               batch_size=args.batch_size,
                                                               lowres_dim=args.patch_size / args.downscale_ratio,
                                                               cuda=args.cuda)

            val_loss = 0.0
            if epoch % args.log_step == 0:

                filename = 'checkpoint_generator_{0:02}.pth.tar'.format(epoch)
                save_snapshot(
                    {'epoch': epoch, 'state_dict': model['generator'].state_dict(),
                     'optimizer': optimizer['generator'].state_dict()},
                    filename=filename, savedir=args.savedir)

                filename = 'checkpoint_discriminator_{0:02}.pth.tar'.format(epoch)
                save_snapshot(
                    {'epoch': epoch, 'state_dict': model['discriminator'].state_dict(),
                     'optimizer': optimizer['discriminator'].state_dict()},
                    filename=filename, savedir=args.savedir)

                print('[Epoch: {0:02}/{1:02}]'
                      '\t[GTrainLoss:{2:.4f}]'
                      '\t[GTrainPSNR:{3:.4f}]'
                      '\t[DTrainPSNR:{4:.4f}]'
                      '\t[ValLoss:{5:.4f}]').format(epoch, args.num_epochs, G_train_loss, train_psnr, D_train_loss,
                                                    val_loss),
                print('\t [Snapshot]')

                if val_loss < best_loss:
                    best_fullname = os.path.join(args.savedir, 'checkpoint_best.pth.tar')
                    shutil.copyfile(os.path.join(args.savedir, filename), best_fullname)
                    best_loss = G_train_loss
                continue

            print('[Epoch: {0:02}/{1:02}]'
                  '\t[GTrainLoss:{2:.4f}]'
                  '\t[GTrainPSNR:{3:.4f}]'
                  '\t[DTrainPSNR:{4:.4f}]'
                  '\t[ValLoss:{5:.4f}]').format(epoch, args.num_epochs, G_train_loss, train_psnr, D_train_loss,
                                                val_loss)


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
