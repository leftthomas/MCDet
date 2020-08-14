# RepPoints
1.config下添加一个`bdd100k.py`的配置文件；

2.mmdet/datasets下添加一个`bdd100k.py`的实现文件；

3.mmdet/datasets/__init__.py添加`BDD100KDataset`;

4.mmdet/datasets/pipelines下添加一个`multi_channel_transforms.py`的实现文件;

5.mmdet/datasets/pipelines/__init__.py添加以下`transforms`:

* `LoadImageFromFile`--->`LoadMultiChannelImageFromFiles`;
* `Resize`--->`MultiChannelResize`;
* `RandomFlip`--->`MultiChannelRandomFlip`;
* `Normalize`--->`MultiChannelNormalize`;
* `Pad`--->`MultiChannelPad`;
* `RPDV2FormatBundle`--->`MultiChannelRPDV2FormatBundle`;
* `DefaultFormatBundle`--->`MultiChannelDefaultFormatBundle`;

6.修改原有config下的配置文件
* 修改数据路径`data_root`;
* `model-->backbone-->in_channels=6`