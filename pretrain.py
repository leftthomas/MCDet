import argparse
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import utils
from backbone import FastSCNN

warnings.filterwarnings("ignore")


# train or val for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(), target.cuda()
            out = net(data)
            prediction = torch.argmax(out, dim=-1)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            total_correct += torch.sum(prediction == target).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC: {:.2f}%'
                                     .format('Train' if is_train else 'Val', epoch, epochs, total_loss / total_num,
                                             total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--data_path', type=str, default='/home/data/imagenet/ILSVRC2012', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    data_path, batch_size, epochs = args.data_path, args.batch_size, args.epochs
    train_data = datasets.ImageFolder(root='{}/{}'.format(data_path, 'train'), transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_data = datasets.ImageFolder(root='{}/{}'.format(data_path, 'val'), transform=utils.val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    model = FastSCNN(in_channels=3, num_classes=1000).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.3), int(epochs * 0.6)], gamma=0.1)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        val_loss, val_acc = train_val(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/backbone_statistics.csv', index_label='epoch')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'results/backbone.pth')
