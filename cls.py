import argparse
import os
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from utils import *

warnings.filterwarnings("ignore")


# train or val for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Val', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classification Model')
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='/home/data/imagenet/ILSVRC2012',
                        help='Path to dataset, only works for ImageNet')
    parser.add_argument('--backbone_type', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50'], help='Backbone type')
    parser.add_argument('--norm_type', type=str, default='bn', choices=['bn', 'in', 'wn'], help='Norm type')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=90, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    data_name, data_path, batch_size, epochs = args.data_name, args.data_path, args.batch_size, args.epochs
    backbone_type, norm_type = args.backbone_type, args.norm_type
    train_data = get_dataset(data_name, 'train', data_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_data = get_dataset(data_name, 'val', data_path)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Model(backbone_type, num_classes=len(train_data.class_to_idx), norm_type=norm_type).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, crop_size[data_name][-1],
                                                       crop_size[data_name][-1]).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [], 'val_loss': [], 'val_acc@1': [], 'val_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(data_name, backbone_type, norm_type, batch_size)
    if not os.path.exists('results'):
        os.mkdir('results')

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        val_loss, val_acc_1, val_acc_5 = train_val(model, val_loader, None)
        lr_scheduler.step()
        results['val_loss'].append(val_loss)
        results['val_acc@1'].append(val_acc_1)
        results['val_acc@5'].append(val_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if val_acc_1 > best_acc:
            best_acc = val_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
