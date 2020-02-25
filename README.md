# WN
A PyTorch implementation of WN based on the paper [Weight Normalization]().

<div align="center">
  <img src="architecture.png"/>
</div>

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

## Datasets
[CIFAR10](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [CIFAR100](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), 
and [ImageNet(ILSVRC2012)](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) are used in this repo.

## Usage
### Train Classification Model
```
python cls.py --data_name cifar100 --batch_size 2
optional arguments:
--data_name                   Dataset name [default value is 'cifar10'](choices=['cifar10', 'cifar100', 'imagenet'])
--data_path                   Path to dataset, only works for ImageNet [default value is '/home/data/imagenet/ILSVRC2012']
--backbone_type               Backbone type [default value is 'resnet18'](choices=['resnet18', 'resnet34', 'resnet50', 'resnext50'])
--norm_type                   Norm type [default value is 'bn'](choices=['bn', 'in', 'wn'])
--batch_size                  Number of images in each mini-batch [default value is 32]
--epochs                      Number of sweeps over the dataset to train [default value is 90]
```

## Results
Adam optimizer is used with learning rate scheduling. The models are trained with batch size `32` on one 
NVIDIA Tesla V100 (32G) GPUs.

### CIFAR10
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>Norm Type</th>
      <th>Batch Size</th>
      <th>Params (M)</th>
      <th>FLOPs (M)</th>
      <th>Top1 Acc (%)</th>
      <th>Top5 Acc (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">1</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">2</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">4</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">8</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">16</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">32</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">64</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">128</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">256</td>
      <td align="center">11.18</td>
      <td align="center">37.12</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>





