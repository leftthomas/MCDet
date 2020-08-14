# MCDet
PyTorch Multiple Channel MMDetection Library, support multiple channel image loading and processing.

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```

## Examples
### RepPointsV2
#### Install
```
git clone https://github.com/Scalsol/RepPointsV2.git
cd RepPointsV2
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"
```
#### Config
Add these files in this repo to `mmdet` dir, and use these files in `configs` dir to train or test.

#### Train
```shell
./tools/dist_train.sh configs/reppoints_v2/reppoints_v2_r50_fpn_1x_bdd100k.py 8
```
#### Test
```shell
./tools/dist_test.sh configs/reppoints_v2/reppoints_v2_r50_fpn_1x_bdd100k.py work_dirs/reppoints_v2_r50_fpn_1x_bdd100k/epoch_12.pth 8 --eval bbox
```

## Contribution
Any contributions to Multiple Channel MMDetection Library are welcome!

2.mmdet/datasets下添加一个`bdd100k.py`的实现文件；

3.mmdet/datasets/__init__.py添加`BDD100KDataset`;

4.mmdet/datasets/pipelines下添加一个`multi_channel_transforms.py`的实现文件;

5.mmdet/datasets/pipelines/__init__.py添加以下`transforms`:
* `LoadImageFromFile`--->`MultiChannelLoadImageFromFile`;
* `Resize`--->`MultiChannelResize`;
* `RandomFlip`--->`MultiChannelRandomFlip`;
* `Normalize`--->`MultiChannelNormalize`;
* `Pad`--->`MultiChannelPad`;
* `RPDV2FormatBundle`--->`MultiChannelRPDV2FormatBundle`;
* `DefaultFormatBundle`--->`MultiChannelDefaultFormatBundle`;
* `ImageToTensor`--->`MultiChannelImageToTensor`;

6.修改原有config下的配置文件
* 修改数据路径`data_root`;
* `model-->backbone-->in_channels=6`