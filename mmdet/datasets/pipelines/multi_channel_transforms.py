import mmcv
import numpy as np

from .transforms import Resize
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
            results[key] = np.stack(images, axis=-1)

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio
