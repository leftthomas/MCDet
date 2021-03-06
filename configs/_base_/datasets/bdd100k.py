dataset_type = 'BDD100KDataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='MultiChannelLoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiChannelResize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='MultiChannelRandomFlip', flip_ratio=0.5),
    dict(type='MultiChannelNormalize', **img_norm_cfg),
    dict(type='MultiChannelPad', size_divisor=32),
    dict(type='MultiChannelDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='MultiChannelLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='MultiChannelResize', keep_ratio=True),
            dict(type='MultiChannelRandomFlip'),
            dict(type='MultiChannelNormalize', **img_norm_cfg),
            dict(type='MultiChannelPad', size_divisor=32),
            dict(type='MultiChannelImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'bdd100k_labels_images_det_coco_train_night.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'bdd100k_labels_images_det_coco_val_night.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'bdd100k_labels_images_det_coco_test_night.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
