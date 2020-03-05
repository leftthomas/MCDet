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
      <th>FLOPs</th>
      <th>Top1 Acc (%)</th>
      <th>Top5 Acc (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">1</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">2</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">4</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">88.91</td>
      <td align="center">99.73</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">8</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">91.36</td>
      <td align="center">99.77</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">16</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">92.57</td>
      <td align="center">99.78</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">32</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">93.17</td>
      <td align="center">99.82</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">64</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">92.58</td>
      <td align="center">99.75</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">128</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">92.26</td>
      <td align="center">99.73</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">BN</td>
      <td align="center">256</td>
      <td align="center">11.17</td>
      <td align="center">556.67M</td>
      <td align="center">91.23</td>
      <td align="center">99.74</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">1</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">85.92</td>
      <td align="center">99.30</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">2</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">87.35</td>
      <td align="center">99.43</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">4</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">89.18</td>
      <td align="center">99.62</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">8</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">90.23</td>
      <td align="center">99.67</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">16</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">90.49</td>
      <td align="center">99.70</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">32</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">90.11</td>
      <td align="center">99.66</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">64</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">90.82</td>
      <td align="center">99.74</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">128</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">90.12</td>
      <td align="center">99.60</td>
    </tr>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">IN</td>
      <td align="center">256</td>
      <td align="center">11.16</td>
      <td align="center">555.44M</td>
      <td align="center">89.25</td>
      <td align="center">99.50</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">1</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">2</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">4</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">89.50</td>
      <td align="center">99.65</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">8</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">92.49</td>
      <td align="center">99.82</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">16</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">93.02</td>
      <td align="center">99.83</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">32</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">93.54</td>
      <td align="center">99.86</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">64</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">93.48</td>
      <td align="center">99.89</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">128</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">91.94</td>
      <td align="center">99.81</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">BN</td>
      <td align="center">256</td>
      <td align="center">21.28</td>
      <td align="center">1.16G</td>
      <td align="center">88.06</td>
      <td align="center">99.63</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">1</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">84.94</td>
      <td align="center">99.33</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">2</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">4</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">89.33</td>
      <td align="center">99.53</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">8</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">90.86</td>
      <td align="center">99.68</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">16</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">89.87</td>
      <td align="center">99.62</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">32</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">91.64</td>
      <td align="center">99.79</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">64</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">91.16</td>
      <td align="center">99.75</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">128</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">90.72</td>
      <td align="center">99.69</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">IN</td>
      <td align="center">256</td>
      <td align="center">21.27</td>
      <td align="center">1.16G</td>
      <td align="center">89.59</td>
      <td align="center">99.68</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">1</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">2</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">4</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">79.01</td>
      <td align="center">98.88</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">8</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">91.09</td>
      <td align="center">99.77</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">16</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">90.33</td>
      <td align="center">99.72</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">32</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">92.15</td>
      <td align="center">99.76</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">64</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">89.11</td>
      <td align="center">99.57</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">128</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">87.20</td>
      <td align="center">99.55</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">BN</td>
      <td align="center">256</td>
      <td align="center">23.52</td>
      <td align="center">1.30G</td>
      <td align="center">89.27</td>
      <td align="center">99.55</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">1</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">52.46</td>
      <td align="center">93.54</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">2</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">55.58</td>
      <td align="center">94.14</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">4</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">65.56</td>
      <td align="center">97.00</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">8</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">52.34</td>
      <td align="center">93.60</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">16</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">61.67</td>
      <td align="center">95.86</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">32</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">64.85</td>
      <td align="center">96.79</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">64</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">71.21</td>
      <td align="center">97.88</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">128</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">69.78</td>
      <td align="center">97.58</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">IN</td>
      <td align="center">256</td>
      <td align="center">23.47</td>
      <td align="center">1.30G</td>
      <td align="center">63.63</td>
      <td align="center">96.23</td>
    </tr>   
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">1</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">2</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">10.00</td>
      <td align="center">50.00</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">4</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">69.99</td>
      <td align="center">97.75</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">8</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">91.01</td>
      <td align="center">99.73</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">16</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">93.35</td>
      <td align="center">99.82</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">32</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">93.18</td>
      <td align="center">99.80</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">64</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">91.12</td>
      <td align="center">99.70</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">128</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">90.98</td>
      <td align="center">99.74</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">BN</td>
      <td align="center">256</td>
      <td align="center">22.99</td>
      <td align="center">1.35G</td>
      <td align="center">85.53</td>
      <td align="center">99.39</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">1</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">2</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">87.33</td>
      <td align="center">99.37</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">4</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">83.03</td>
      <td align="center">99.19</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">8</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">89.51</td>
      <td align="center">99.62</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">16</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">84.38</td>
      <td align="center">99.14</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">32</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">65.92</td>
      <td align="center">96.83</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">64</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">85.39</td>
      <td align="center">99.28</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">128</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">82.93</td>
      <td align="center">99.17</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">IN</td>
      <td align="center">256</td>
      <td align="center">22.92</td>
      <td align="center">1.34G</td>
      <td align="center">78.16</td>
      <td align="center">98.6</td>
    </tr>
  </tbody>
</table>





