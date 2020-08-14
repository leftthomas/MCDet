_base_ = './reppoints_minmax_r50_fpn_1x_bdd100k.py'
model = dict(bbox_head=dict(transform_method='exact_minmax'))
