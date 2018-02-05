import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

import co_transforms as co_transforms
from conf import get_arguments
from dataset import SRDataset
from generator import Generator


def prepare_data(sr_dir, lr_dir, patch_size, batch_size, mode='train'):
    if mode is 'val':
        print "I'm here"
        transform = co_transforms.Compose([co_transforms.RandomCrop(patch_size, patch_size), co_transforms.ToTensor()])
    elif mode is 'train':
        transform = co_transforms.Compose(
            [co_transforms.RandomCrop(patch_size, patch_size), co_transforms.RandomHorizontalFlip(),
             co_transforms.RandomVerticalFlip(), co_transforms.RandomRotation((0, 90)), co_transforms.ToTensor()])
    else:
        transform = co_transforms.ToTensor()

    dset = SRDataset(highres_root=sr_dir, lowres_root=lr_dir, transform=transform)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False)
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

def test(model, testData, savedir, lowres_dim, cuda=True):

    model.eval()
    downsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(lowres_dim), transforms.ToTensor()])
    toImage = transforms.ToPILImage()

    for step, (high, _) in enumerate(testData):
        low = torch.FloatTensor(high.size()[0], 3, lowres_dim, lowres_dim)
        for j in range(high.size()[0]):
            low[j] = downsample(high[j])

        if cuda:
            low = low.cuda()
#
        low = Variable(low, volatile=True)
#
        output = model(low)
        for j in range(output.size()[0]):
            output_img = toImage(output[j].data.cpu())
            filename = os.path.join(args.savedir, 'result_{0:03}.bmp'.format(step*output.size()[0]+ j))
            print filename
            output_img.save(filename,'BMP')



def save_snapshot(state, filename='checkpoint.pth.tar', savedir='./checkpoints'):
    fullname = os.path.join(savedir, filename)
    torch.save(state, fullname)


def main(args):
    # --- load data ---

    # --- define the model and NN settings ---
    model = Generator(5, 2)
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

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500], gamma=0.1)

        best_loss = float('inf')
        for epoch in range(1, args.num_epochs + 1):
            scheduler.step()
            train_loss = train(model=model, trData=trLoader, optimizer=optimizer, lossfn=criterion,
                               batch_size=args.batch_size, lowres_dim=args.patch_size / args.downscale_ratio)

            val_loss = validate(model=model, vlData=valLoader, lossfn=criterion,
                                batch_size=args.batch_size, lowres_dim=args.patch_size / args.downscale_ratio)

            if epoch % args.log_step == 0:

                filename = 'checkpoint_{0:02}.pth.tar'.format(epoch)
                save_snapshot({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                              filename=filename, savedir=args.savedir)


                print('[Epoch: {0:02}/{1:02}]'
                      '\t[TrainLoss:{2:.4f}]'
                      '\t[ValLoss:{3:.4f}]').format(epoch, args.num_epochs, train_loss, val_loss),
                print('\t [Snapshot]')

                if val_loss < best_loss:
                    best_fullname = os.path.join(args.savedir, 'checkpoint_best.pth.tar')
                    shutil.copyfile(os.path.join(args.savedir, filename), best_fullname)
                    best_loss = train_loss
                continue


            print('[Epoch: {0:02}/{1:02}]'
                  '\t[TrainLoss:{2:.4f}]'
                  '\t[ValLoss:{3:.4f}]').format(epoch, args.num_epochs, train_loss, val_loss)

    elif args.mode == 'test':

        testLoader = prepare_data(sr_dir=args.srtestdir, lr_dir=args.lrtestdir, patch_size='',
                                batch_size=args.batch_size, mode='test')

        filename = 'checkpoint_{0:02}.pth.tar'.format(args.state)
        checkpoint = torch.load(os.path.join(args.weightdir, filename))

        model.load_state_dict(checkpoint['state_dict'])

        test(model,testLoader, args.savedir, lowres_dim=args.image_size / args.downscale_ratio)



if __name__ == '__main__':
    args = get_arguments()
    main(args)
