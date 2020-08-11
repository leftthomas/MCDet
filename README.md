# RepPoints
A PyTorch implementation of RepPoints. 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
- mmcv-full
```
pip install mmcv-full
```
- mmdetection
```
pip install -r requirements/build.txt
pip install -v -e .
```

### Train
```shell
./tools/dist_train.sh configs/reppoints_v2/reppoints_v2_r50_fpn_giou_1x_coco.py 8
```

### Inference
```shell
./tools/dist_test.sh configs/reppoints_v2/reppoints_v2_r50_fpn_giou_1x_coco.py work_dirs/reppoints_v2_r50_fpn_giou_1x_coco/epoch_12.pth 8 --eval bbox
```

## Main Results

### RepPoints V2

**ResNe(X)ts:**

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link 
--- |:---:|:---:|:---:|:---:
RepPoints_V2_R_50_FPN_1x | No | 40.9 | --- | [Google](https://drive.google.com/file/d/1QBYTLITOJG5dSjU35YewE9efSCH_VGg2/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1ZvJ3gk_FVOVHmvmy87cr_w) / [Log](https://drive.google.com/file/d/1Ra2XC-Zjfpx6YG91ZRY_8qI_XnDe_Txu/view?usp=sharing)
RepPoints_V2_R_50_FPN_GIoU_1x | No  | 41.1 | 41.3 | [Google](https://drive.google.com/file/d/1lbYUpvA33GHaEImKRhbR7H5S36Dxubcf/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1kyt5YNWO-gg_W4iUuwZxiw) / [Log](https://drive.google.com/file/d/1yDwNToYAZdPWTs4vxzbBNqRCLIrRxx_H/view?usp=sharing)
RepPoints_V2_R_50_FPN_GIoU_2x | Yes  | 43.9 | 44.4 | [Google](https://drive.google.com/file/d/13FfoXOfTsO-eLTcO__WXUxQRRWNzAdrL/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1QAvjzGI1zrnockXX6cZrZA) / [Log](https://drive.google.com/file/d/1yDwNToYAZdPWTs4vxzbBNqRCLIrRxx_H/view?usp=sharing)
RepPoints_V2_R_101_FPN_GIoU_2x | Yes  | 45.8 | 46 | [Google](https://drive.google.com/file/d/1MUb1Y1_OoqhwFkvdyE6QthbUc1l2ixYS/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1YNxbnmeq20mVef5ZAlMTxQ) / [Log](https://drive.google.com/file/d/1m5BM1PWXWKwfsvEc54b0fCcMIlXKPFG_/view?usp=sharing)
RepPoints_V2_R_101_FPN_dcnv2_GIoU_2x | Yes  | 47.7 | 48.1 | [Google](https://drive.google.com/file/d/1VaBAPWOzku0tfpUWa5FmOsu_Fn_3UWaY/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/14V2hz6VrXJv_acQ-SQlBUg) / [Log](https://drive.google.com/file/d/1f_WwyNFlqvCSU-P723b-LHG1kpRhddns/view?usp=sharing)
RepPoints_V2_X_101_FPN_GIoU_2x | Yes  | 47.3 | 47.8 | [Google](https://drive.google.com/file/d/1rThw_7yXi185-VXfeXY81iJNwoAywQVA/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1Vp4vtkSSfAbkI1_--jDQ5g) / [Log](https://drive.google.com/file/d/13Nj__4nvZEEJtwNhvIcKjDwbmu5eQZoo/view?usp=sharing)
RepPoints_V2_X_101_FPN_dcnv2_GIoU_2x | Yes  | 49.3 | 49.4 | [Google](https://drive.google.com/file/d/1db6cK7pEjRgN8QjaGV8OnZYp35nuxd7G/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1idrD8kmgYTP_q5_mSSlpTQ) / [Log](https://drive.google.com/file/d/1DlCYQiWUanPVwyowjFLrgKCYz57EJE2S/view?usp=sharing)

**MobileNets**:

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
RepPoints_V2_MNV2_c128_FPN_2x | Yes | 36.8 | --- | [Google](https://drive.google.com/file/d/1mnoXOyzp6dCYbQx7rKbzjKlc0RzitOpV/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1_sEMkhDjYjhJwMSiNNoYdg) / [Log](https://drive.google.com/file/d/11UQGvOuOykFD0iW3xz1DsqJSwJaw2tmw/view?usp=sharing)
RepPoints_V2_MNV2_FPN_2x | Yes | 39.4 | --- | [Google](https://drive.google.com/file/d/1xk8jGZiRs2iskywf3hB6tINMK_3lDL3u/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1OdiEtxWe5f45GaaprITdrA) / [Log](https://drive.google.com/file/d/1A1ldy4HzPStKjz0Xm96sPnyB-NS2wzMk/view?usp=sharing)

### RepPoints V1

**ResNe(X)ts:**

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
RepPoints_V1_R_50_FPN_1x | No | 38.8 | --- | [Google](https://drive.google.com/file/d/1DMoTicyL5FejCL3042rwZWPojTC2-qRH/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1t4zaVFCH0A35xEDXo3RUMQ) / [Log](https://drive.google.com/file/d/1Oq-3DFnfbJ6F5doJobuhZU9y7BkRC0NH/view?usp=sharing)
RepPoints_V1_R_50_FPN_GIoU_1x | No  | 39.9 | ---| [Google](https://drive.google.com/file/d/1IJp3bBCrRuDDQcxjwoUenBuA-KSzYWkf/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1swvcTxgiUWSRCOSKOJ7Mjg) / [Log](https://drive.google.com/file/d/1bRCfSQPYFjxXIari9F8AWXOYUUS4VVJg/view?usp=sharing)
RepPoints_V1_R_50_FPN_GIoU_2x | Yes  | 42.7 | --- | [Google](https://drive.google.com/file/d/1tZpfOGmxzToaikaFdpyhjCjm8ltOJaYY/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1pFvleWnZjVsFeHJjiX3dpw) / [Log](https://drive.google.com/file/d/1iBSrXp4ngug9jaWb1gndSj4NQQQzbpAC/view?usp=sharing)
RepPoints_V1_R_101_FPN_GIoU_2x | Yes  | 44.4 | --- | [Google](https://drive.google.com/file/d/1YiR4m8GNWQ472tgGXOdeaWbnS2KiAR4z/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1BhVjPvBJaWM3Okq1Ytk8Iw) / [Log](https://drive.google.com/file/d/1QhN3vedurGiRAl6aiSwtGNwxN3TAOXuP/view?usp=sharing)
RepPoints_V1_R_101_FPN_dcnv2_GIoU_2x | Yes  | 46.6 | --- | [Google](https://drive.google.com/file/d/112jG1a2TUnABqCR1ccKrIzAE6_8P3LAe/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/18e0zGQah6aqCY2qIXe88Ew) / [Log](https://drive.google.com/file/d/1ut23n60vRY0f8VF97bgh2KWN05rmto9N/view?usp=sharing)
RepPoints_V1_X_101_FPN_GIoU_2x | Yes  | 46.3 | --- | [Google](https://drive.google.com/file/d/1UohtogF-znE0NnqXHSrcGBuE9B3pNaDr/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1qbCyHBZksS_l-eXH8Wh5Ug) / [Log](https://drive.google.com/file/d/1wOBP8-oBg53llJOfspddecA5NNpdcRix/view?usp=sharing)
RepPoints_V1_X_101_FPN_dcnv2_GIoU_2x | Yes  | 48.3 | --- | [Google](https://drive.google.com/file/d/14oSaFilmT6EMTkAuP23OyQtLbWRB0BiN/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1Xl5VUG7z7IQlz6F8267f6A) / [Log](https://drive.google.com/file/d/14u0m5eFdLFRGJT--mx7wzsdIVNb8Atu8/view?usp=sharing)

**MobileNets**:

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
RepPoints_V1_MNV2_c128_FPN_2x | Yes | 35.7 | --- | [Google](https://drive.google.com/file/d/14l_m3lLacfw7mTafv7cuvczhkWeF2cn4/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1hHeV8KZLdtLNvVYRl-najw) / [Log](https://drive.google.com/file/d/1pg1gW3ajOqgB4oLmEsiKJ2cReAIdxRkp/view?usp=sharing)
RepPoints_V1_MNV2_FPN_2x | Yes | 37.8 | --- | [Google](https://drive.google.com/file/d/1Ex6us97waqWP25H20xBZZEfQXfI8mfjW/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1AyibFbRgSc8bug0KZDO1Yw) / [Log](https://drive.google.com/file/d/1BanM614yhFaxtLeSI_vYbtTjsarj-Xxl/view?usp=sharing)

[1] *GIoU means using GIoU loss instead of smooth-l1 loss for the regression branch, which we find improves the final performance.* \
[2] *X-101 denotes ResNeXt-101-64x4d.* \
[3] *1x, 2x, 3x mean the model is trained for 12, 24 and 36 epochs, respectively.* \
[4] *For multi-scale training, the shorter side of images is randomly chosen from [480, 960].* \
[5] *`dcnv2` denotes deformable convolutional networks v2.* \
[6] *c128 denotes the model has 128 (instead of 256) channels in towers.*\
[7] *We use syncbn, GIoU loss for the regression branch to train mobilenet v2 models by default.*

## Citation
```
@article{chen2020reppointsv2,
  title={RepPoints V2: Verification Meets Regression for Object Detection},
  author={Chen, Yihong and Zhang, Zheng and Cao, Yue and Wang, Liwei and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2007.08508},
  year={2020}
}

@inproceedings{yang2019reppoints,
  title={RepPoints: Point Set Representation for Object Detection},
  author={Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019}
}
```

