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
Copy these files in this repo to corresponding dirs, and use these files in `configs` dir to train or test.
Pay attention to the config option:
* `model-->backbone-->in_channels=6`

#### Train
```shell
./tools/dist_train.sh configs/reppoints_v2/reppoints_v2_r50_fpn_1x_bdd100k.py 8
```
#### Test
```shell
./tools/dist_test.sh configs/reppoints_v2/reppoints_v2_r50_fpn_1x_bdd100k.py work_dirs/reppoints_v2_r50_fpn_1x_bdd100k/epoch_12.pth 8 --eval bbox
```

## Features
* `LoadImageFromFile`--->`MultiChannelLoadImageFromFile`;
* `Resize`--->`MultiChannelResize`;
* `RandomFlip`--->`MultiChannelRandomFlip`;
* `Normalize`--->`MultiChannelNormalize`;
* `Pad`--->`MultiChannelPad`;
* `RPDV2FormatBundle`--->`MultiChannelRPDV2FormatBundle`;
* `DefaultFormatBundle`--->`MultiChannelDefaultFormatBundle`;
* `ImageToTensor`--->`MultiChannelImageToTensor`;

## Contribution
Any contributions to Multiple Channel MMDetection Library are welcome!