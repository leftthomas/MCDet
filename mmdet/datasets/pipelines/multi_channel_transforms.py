import mmcv
import numpy as np

from .transforms import Resize, RandomFlip, Normalize, Pad
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None
try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class MultiChannelResize(Resize):

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            images, img_num = [], results[key].shape[-1]
            if self.keep_ratio:
                for index in range(img_num):
                    img, scale_factor = mmcv.imrescale(
                        results[key][:, :, :, index], results['scale'], return_scale=True)
                    images.append(img)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                for index in range(img_num):
                    img, w_scale, h_scale = mmcv.imresize(
                        results[key][:, :, :, index], results['scale'], return_scale=True)
                    images.append(img)
            images = np.stack(images, axis=-1)
            results[key] = images

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = images.shape
            # in case that there is no padding
            results['pad_shape'] = images.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class MultiChannelRandomFlip(RandomFlip):

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                images, img_num = [], results[key].shape[-1]
                for index in range(img_num):
                    images.append(mmcv.imflip(
                        results[key][:, :, :, index], direction=results['flip_direction']))
                results[key] = np.stack(images, axis=-1)
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results


@PIPELINES.register_module()
class MultiChannelNormalize(Normalize):

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            images, img_num = [], results[key].shape[-1]
            for index in range(img_num):
                images.append(mmcv.imnormalize(results[key][:, :, :, index], self.mean, self.std, self.to_rgb))
            results[key] = np.stack(images, axis=-1)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class MultiChannelPad(Pad):

    def _pad_img(self, results):
        for key in results.get('img_fields', ['img']):
            images, img_num = [], results[key].shape[-1]
            if self.size is not None:
                for index in range(img_num):
                    images.append(mmcv.impad(results[key][:, :, :, index], self.size, self.pad_val))
            elif self.size_divisor is not None:
                for index in range(img_num):
                    images.append(mmcv.impad_to_multiple(
                        results[key][:, :, :, index], self.size_divisor, pad_val=self.pad_val))
            images = np.stack(images, axis=-1)
            results[key] = images
        results['pad_shape'] = images.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
