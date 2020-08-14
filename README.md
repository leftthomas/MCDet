# RepPoints
1.config下添加一个bdd100k.py的配置文件；
2.mmdet/datasets下添加一个`bdd100k.py`的实现文件；
3.mmdet/datasets/__init__.py添加`BDD100KDataset`;
4.mmdet/datasets/pipelines下添加一个`multi_channel_transforms.py`的实现文件;
5.mmdet/datasets/pipelines/__init__.py添加`LoadMultiChannelImageFromFiles`, `MultiChannelResize`;
6.修改原有config下的配置文件,修改数据路径`data_root`;
7.`LoadImageFromFile`--->`LoadMultiChannelImageFromFiles`;
8.`Resize`--->`MultiChannelResize`;
