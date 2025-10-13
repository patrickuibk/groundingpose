import numpy as np
import copy  # added

from .coco_dataset import CocoStylePoseDataset
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO


@DATASETS.register_module()
class CocoStyleTopDownPoseDataset(CocoStylePoseDataset):
    """Top-down variant: one sample per annotation (bbox) with same format as CocoStylePoseDataset plus 'bbox'."""

    def load_data_list(self):
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco: COCO = self.COCOAPI(local_path)

        # Initialize category-related attributes required by BaseDetDataset.filter_data
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo.get('classes', []))  # may be empty for keypoint-only usage
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []

        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            # Align with pose parser (expects 'id')
            raw_img_info['id'] = raw_img_info.get('id', raw_img_info.get('img_id', img_id))
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            anns = self.coco.load_anns(ann_ids)
            for ann in anns:
                total_ann_ids.append(ann['id'])
                # Build single-annotation raw_data_info
                raw_data_info = {
                    'raw_img_info': raw_img_info,
                    'raw_ann_info': [ann]
                }
                parsed = self.parse_data_info(raw_data_info)
                if not parsed['instances']:
                    continue
                # Attach tight bbox (x1,y1,x2,y2) for cropping transform
                x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
                if w <= 0 or h <= 0:
                    continue
                parsed['bbox'] = np.array([x, y, x + w, y + h], dtype=np.float32)
                data_list.append(parsed)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), \
                f"Annotation ids in '{self.ann_file}' are not unique!"
        del self.coco
        return data_list