from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class BDD100KDataset(CocoDataset):
    CLASSES = ('person', 'rider', 'car', 'bus', 'truck', 'bike',
               'motor', 'traffic light', 'traffic sign', 'train')

    def pre_pipeline(self, results):
        """Load real and gen images"""
        results['img_info']['filename'] = ['real/' + results['img_info']['filename'],
                                           'gen/' + results['img_info']['filename']]
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
