import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import ImageReader, recall


def train(net, optim):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        out = net(inputs)
        loss = cel_criterion(out.permute(0, 2, 1).contiguous(), labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(out, dim=-1)
        total_loss += loss.item()
        total_correct += torch.sum(pred == labels).item() / ENSEMBLE_SIZE
        total_num += inputs.size(0)
        data_bar.set_description('Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, NUM_EPOCHS, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def eval(net, recalls):
    net.eval()
    with torch.no_grad():
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in eval_dict[key]['data_loader']:
                inputs, labels = inputs.cuda(), labels.cuda()
                out = net(inputs)
                out = F.normalize(out, dim=-1)
                eval_dict[key]['features'].append(out.cpu())
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

    if DATA_NAME == 'isc':
        acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recalls,
                          eval_dict['gallery']['features'], gallery_data_set.labels)
    else:
        acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recalls)
    desc = ''
    for index, id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(id, acc_list[index] * 100)
        results['test_recall@{}'.format(recall_ids[index])].append(acc_list[index] * 100)
    print(desc)
    return acc_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--model_type', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'], help='backbone type')
    parser.add_argument('--load_ids', action='store_true', help='load already generated ids or not')
    parser.add_argument('--batch_size', default=10, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=48, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='meta class size')

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCHS = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    ENSEMBLE_SIZE, META_CLASS_SIZE, CROP_TYPE = opt.ensemble_size, opt.meta_class_size, opt.crop_type
    MODEL_TYPE, LOAD_IDS = opt.model_type, opt.load_ids
    save_name_pre = '{}_{}_{}_{}_{}'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE, ENSEMBLE_SIZE, META_CLASS_SIZE)
    recall_ids = [int(k) for k in RECALLS.split(',')]

    results = {'train_loss': [], 'train_accuracy': []}
    for index, id in enumerate(recall_ids):
        results['test_recall@{}'.format(recall_ids[index])] = []
    if not os.path.exists('results'):
        os.mkdir('results')

    train_data_set = ImageReader(DATA_NAME, 'train', CROP_TYPE, ENSEMBLE_SIZE, META_CLASS_SIZE, LOAD_IDS)
    train_data_loader = DataLoader(train_data_set, BATCH_SIZE, shuffle=True, num_workers=8)

    train_ext_data_set = ImageReader(DATA_NAME, 'train_ext', CROP_TYPE)
    train_ext_data_loader = DataLoader(train_ext_data_set, BATCH_SIZE, shuffle=False, num_workers=8)
    test_data_set = ImageReader(DATA_NAME, 'query' if DATA_NAME == 'isc' else 'test', CROP_TYPE)
    test_data_loader = DataLoader(test_data_set, BATCH_SIZE, shuffle=False, num_workers=8)
    eval_dict = {'train': {'data_loader': train_ext_data_loader}, 'test': {'data_loader': test_data_loader}}
    if DATA_NAME == 'isc':
        gallery_data_set = ImageReader(DATA_NAME, 'gallery', CROP_TYPE)
        gallery_data_loader = DataLoader(gallery_data_set, BATCH_SIZE, shuffle=False, num_workers=8)
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    model = Model(ENSEMBLE_SIZE, META_CLASS_SIZE, MODEL_TYPE).cuda()
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)
    cel_criterion = CrossEntropyLoss()

    best_recall = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train(model, optimizer)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        lr_scheduler.step()
        rank = eval(model, recall_ids)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['train_images'] = train_ext_data_set.images
            data_base['train_labels'] = train_ext_data_set.labels
            data_base['train_features'] = eval_dict['train']['features']
            data_base['test_images'] = test_data_set.images
            data_base['test_labels'] = test_data_set.labels
            data_base['test_features'] = eval_dict['test']['features']
            data_base['gallery_images'] = gallery_data_set.images if DATA_NAME == 'isc' else test_data_set.images
            data_base['gallery_labels'] = gallery_data_set.labels if DATA_NAME == 'isc' else test_data_set.labels
            data_base['gallery_features'] = eval_dict['gallery']['features'] \
                if DATA_NAME == 'isc' else eval_dict['test']['features']
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))
