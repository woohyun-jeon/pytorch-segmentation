import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from datasets import *
import models

import torch
import torch.nn as nn
import albumentations as A



def get_arguments():
    parser = argparse.ArgumentParser(description='Parameter setting for training segmentation model')
    parser.add_argument('--model_name', type=str, default='unet', help='model name')
    parser.add_argument('--root_dir', type=str, default='E:/study/segmentation/data', help='root directory')
    parser.add_argument('--out_dir', type=str, default='E:/study/segmentation/outputs', help='output directory')
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    args = parser.parse_args()

    return args


def train(model, epoch, device, train_dataloader, optimizer, criterion):
    model.train()

    train_loss = 0.
    for data, target in tqdm(train_dataloader, desc='Epoch {} Train'.format(epoch), total=len(train_dataloader)):
        data, target = data.to(device), target.to(device)

        # feed forward
        output = model(data)

        # get loss
        loss = criterion(output, target)
        train_loss += loss.item()

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    return train_loss


def valid(model, epoch, device, valid_dataloader, criterion):
    model.eval()

    valid_loss = 0.
    with torch.no_grad():
        for data, target in tqdm(valid_dataloader, desc='Epoch {} Validation'.format(epoch), total=len(valid_dataloader)):
            data, target = data.to(device), target.to(device)

            # get predicted
            output = model(data)

            # get loss
            loss = criterion(output, target)
            valid_loss += loss.item()

    valid_loss /= len(valid_dataloader)

    return valid_loss


def main():
    # get arguments
    args = get_arguments()

    # confirm device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device type -- {}'.format(device))

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # generate dataset
    train_aug = A.Compose([
        A.RandomCrop(args.img_size, args.img_size),
        A.OneOf([
          A.GaussNoise(),
          A.HorizontalFlip(p=0.5),
          A.RandomRotate90(p=0.5),
          A.VerticalFlip(p=0.5)
        ], p=0.5)
    ])

    val_aug = A.Compose([
        A.RandomCrop(args.img_size, args.img_size)
    ])

    train_dataset = CityScapeDataset(data_dir=args.root_dir, split='train', transform=train_aug)
    valid_dataset = CityScapeDataset(data_dir=args.root_dir, split='val', transform=val_aug)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2*args.num_workers,
                                                   shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=2*args.num_workers,
                                                   shuffle=True)

    # set model
    model = models.__dict__[args.model_name](in_channels=3, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)

    # train
    train_losses = []
    valid_losses = []
    min_loss = 9999.
    for epoch in range(args.epochs):
        train_loss = train(model, epoch, device, train_dataloader, optimizer, criterion)
        valid_loss = valid(model, epoch, device, valid_dataloader, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('epoch: {}/{} -- '.format(epoch+1, args.epochs),
              'train loss: {:.4f}'.format(train_loss),
              'val loss: {:.4f}'.format(valid_loss)
              )

        scheduler.step()

        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, args.model_name.lower() + '_cityscape_512.pth'))

    # save loss changes to image file
    plt.figure(figsize=(7.5, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='valid')
    plt.title('Train-Valid Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xticks(range(1, len(train_losses) + 1), fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(args.out_dir, args.model_name.lower() + '_loss.png'))


if __name__ == '__main__':
    main()