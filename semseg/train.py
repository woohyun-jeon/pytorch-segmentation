import os
import argparse
from tqdm import tqdm
from datasets.ai4space import AI4SpaceDataset
from models import unet
from utils import metrics
import torch
import torch.nn as nn


def get_arguments():
    parser = argparse.ArgumentParser(description='Parameter setting for training segmentation model')
    parser.add_argument('--root_dir', type=str, help='root directory', required=True)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--img_size', type=int, default=1024, help='image size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    args = parser.parse_args()

    return args


def main():
    # get arguments
    args = get_arguments()

    # get device information
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataset
    train_dataset = AI4SpaceDataset(data_dir=args.root_dir, split='train')
    val_dataset = AI4SpaceDataset(data_dir=args.root_dir, split='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True, drop_last=True)

    # define model
    model = unet.UNet(in_channels=3, num_classes=args.num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # train model
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_accs, val_accs = [], []
    best_iou = 0
    for epoch in range(args.epochs):
        # do training
        epoch_trloss, epoch_triou, epoch_tracc = 0, 0, 0
        for train_img, train_msk in tqdm(train_loader):
            model.train()
            # set device
            train_img, train_msk = train_img.to(DEVICE), train_msk.to(DEVICE)

            # predict
            train_out = model(train_img)
            train_loss = criterion(train_out, train_msk)

            # evaluate
            train_iou = metrics.estimate_miou(pred_mask=train_out, gt_mask=train_msk, num_classes=args.num_classes)
            train_acc = metrics.estimate_accuracy(pred_mask=train_out, gt_mask=train_msk)

            # update loss, iou, accuracy
            epoch_trloss += train_loss.item()
            epoch_triou += train_iou
            epoch_tracc += train_acc

            # update model weight
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # do validation
        epoch_valloss, epoch_valiou, epoch_valacc = 0, 0, 0
        for val_img, val_msk in tqdm(val_loader):
            with torch.no_grad():
                model.eval()

                # set device
                val_img, val_msk = val_img.to(DEVICE), val_msk.to(DEVICE)

                # predict
                val_out = model(val_img)
                val_loss = criterion(val_out, val_msk)

                # evaluate
                val_iou = metrics.estimate_miou(pred_mask=val_out, gt_mask=val_msk, num_classes=args.num_classes)
                val_acc = metrics.estimate_accuracy(pred_mask=val_out, gt_mask=val_msk)

                # update loss, iou, accuracy
                epoch_valloss += val_loss.item()
                epoch_valiou += val_iou
                epoch_valacc += val_acc

        # update history
        train_losses.append(epoch_trloss/len(train_loader))
        val_losses.append(epoch_valloss/len(val_loader))
        train_ious.append(epoch_triou/len(train_loader))
        val_ious.append(epoch_valiou/len(val_loader))
        train_accs.append(epoch_tracc/len(train_loader))
        val_accs.append(epoch_valacc/len(val_loader))

        print('epoch: {}/{} -- '.format(epoch+1, args.epochs),
              'train loss: {:.4f}'.format(epoch_trloss/len(train_loader)),
              'val loss: {:.4f}'.format(epoch_valloss/len(val_loader)),
              'train iou: {:.4f}'.format(epoch_triou/len(train_loader)),
              'val iou: {:.4f}'.format(epoch_valiou/len(val_loader)),
              'train acc: {:.4f}'.format(epoch_tracc/len(train_loader)),
              'val acc: {:.4f}'.format(epoch_valacc/len(val_loader)))

        check_iou = epoch_valiou/len(val_loader)
        if best_iou < check_iou:
            # update best iou value
            best_iou = check_iou

            # save model
            torch.save(model.state_dict(), os.path.join(args.root_dir, 'model.pth'))


if __name__ == '__main__':
    main()