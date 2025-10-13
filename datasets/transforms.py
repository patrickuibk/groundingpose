import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmcv.transforms import to_tensor



@TRANSFORMS.register_module()
class LoadKeypointGraphAnnotationsAsBbox(BaseTransform):
    """Load and process the ``instances`` provided
    by dataset.

    Required Keys:

    - height
    - width
    - instances

      - keypoint_id: Unique ID for each keypoint instance
      - keypoint_coords: [x, y] coordinates of the keypoint
      - keypoint_label: Label index for the keypoint
      - ignore_flag: Whether to ignore this keypoint during training
      - keypoint_relations: Dictionary mapping relation names to lists of related keypoint IDs

    Added Keys:

    - gt_keypoint_coords (np.ndarray): Keypoint coordinates with shape (-1, 1, 2)
    - gt_bboxes (BaseBoxes[torch.float32]): Bounding boxes with shape (-1, 4)
    - gt_bboxes_labels (np.int64): Label indices for keypoints with shape (-1,)
    - gt_keypoint_ids (np.int64): Instance IDs for keypoints with shape (-1,)
    - gt_ignore_flags (bool): Ignore flags for keypoints with shape (-1,)
    - gt_keypoint_relations (dict): Dictionary mapping relation types to lists of [source_id, target_id] pairs

    Args:
        box_type (str): The box type used to wrap the bboxes. If ``box_type``
            is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
    """

    def __init__(
            self,
            box_type: str = 'hbox',
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.box_type = box_type

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        gt_bboxes = []
        gt_ignore_flags = []
        gt_keypoint_ids = []
        gt_keypoint_labels = []
        gt_keypoint_coords = []
        gt_keypoint_relations = {}
        
        instances = results.get('instances', [])
        
        gt_keypoint_relations = {}
        for instance in instances:
            bbox = [
                instance['keypoint_coords'][0],
                instance['keypoint_coords'][1],
                instance['keypoint_coords'][0],
                instance['keypoint_coords'][1]
            ]

            gt_bboxes.append(bbox)
            gt_ignore_flags.append(instance['ignore_flag'])
            gt_keypoint_ids.append(instance['keypoint_id'])
            gt_keypoint_coords.append([instance['keypoint_coords']])
            gt_keypoint_labels.append(instance['keypoint_label'])
            
            # Process keypoint_relations by type
            if 'keypoint_relations' in instance:
                for rel_type, related_ids in instance['keypoint_relations'].items():
                    if rel_type not in gt_keypoint_relations:
                        gt_keypoint_relations[rel_type] = []
                    for related_id in related_ids:
                        gt_keypoint_relations[rel_type].append([instance['keypoint_id'], related_id])

        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        results['gt_keypoint_ids'] = np.array(gt_keypoint_ids, dtype=np.int64)
        results['gt_keypoint_coords'] = np.array(gt_keypoint_coords, dtype=np.float32).reshape((-1, 1, 2))
        results['gt_keypoint_labels'] = np.array(gt_keypoint_labels, dtype=np.int64)
        results['gt_keypoint_relations'] = gt_keypoint_relations

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(box_size={self.box_size}, '
        repr_str += f'box_type={self.box_type})'
        return repr_str


@TRANSFORMS.register_module()
class ConvertRelationsToMatrix(BaseTransform):
    """Convert relation dictionary to adjacency matrices.
    
    Required Keys:
    - gt_keypoint_ids
    - gt_keypoint_relations (dict): Dictionary mapping relation types to lists of [source_id, target_id] pairs
    
    Added Keys:
    - gt_relation_matrices (ndarray): Adjacency matrices with shape (num_keypoints, num_keypoints, num_keypoint_relations)
    """
    
    def __init__(self, enforce_symmetry=True, **kwargs):
        super().__init__(**kwargs)
        self.enforce_symmetry = enforce_symmetry
    
    def transform(self, results: dict) -> dict:
        keypoint_ids = results['gt_keypoint_ids']
        keypoint_relations = results['gt_keypoint_relations']
        relation_types = list(keypoint_relations.keys())
        
        num_keypoints = len(keypoint_ids)
        num_keypoint_relations = len(relation_types)
        
        # Create a mapping from keypoint_id to index
        id_to_idx = {kid: idx for idx, kid in enumerate(keypoint_ids)}
        
        # Create a mapping from relation type to index
        rel_type_to_idx = {rel_type: idx for idx, rel_type in enumerate(relation_types)}
        
        # Initialize relation matrices with shape (num_keypoints, num_keypoints, num_keypoint_relations)
        relation_matrices = np.zeros((num_keypoints, num_keypoints, num_keypoint_relations), dtype=np.int64)
        
        # Fill the adjacency matrix for each relation type
        for rel_type, pairs in keypoint_relations.items():
            if rel_type not in rel_type_to_idx:
                continue
                
            rel_idx = rel_type_to_idx[rel_type]
            
            for src_id, dst_id in pairs:
                if src_id in id_to_idx and dst_id in id_to_idx:
                    src_idx = id_to_idx[src_id]
                    dst_idx = id_to_idx[dst_id]
                    relation_matrices[src_idx, dst_idx, rel_idx] = 1

        # Enforce symmetry if requested
        if self.enforce_symmetry:
            for rel_idx in range(num_keypoint_relations):
                mat = relation_matrices[:, :, rel_idx]
                relation_matrices[:, :, rel_idx] = np.logical_or(mat, mat.T).astype(np.int64)
        
        # Store relation matrices with shape (num_keypoints, num_keypoints, num_keypoint_relations)
        results['gt_relation_matrices'] = relation_matrices
        
        return results


@TRANSFORMS.register_module()
class TransformKeypoints(BaseTransform):
    def __init__(self, **kwargs):
        """Resize keypoints and set bboxes to a fixed size after transform.

        Required Keys:
        - homography_matrix
        - gt_keypoint_coords

        Modified Keys:
        - gt_keypoint_coords
        - gt_bboxes (if keypoint_box_size is not None)
        """
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        homography_matrix = results.get('homography_matrix', None)

        if homography_matrix is None:
            return results

        gt_keypoint_coords = results.get('gt_keypoint_coords', None)
        if gt_keypoint_coords is not None:
            # gt_keypoint_coords shape: (N, 1, 2)
            N = gt_keypoint_coords.shape[0]
            keypoints_flat = gt_keypoint_coords.reshape(-1, 2)  # (N, 2)
            # Convert to homogeneous coordinates
            keypoints_h = np.concatenate([keypoints_flat, np.ones((N, 1), dtype=keypoints_flat.dtype)], axis=1)  # (N, 3)
            # Apply homography
            keypoints_transformed = (homography_matrix @ keypoints_h.T).T  # (N, 3)
            # Normalize
            keypoints_transformed = keypoints_transformed[:, :2] / keypoints_transformed[:, 2:3]
            results['gt_keypoint_coords'] = keypoints_transformed.reshape((-1, 1, 2))

        return results
    
    

@TRANSFORMS.register_module()
class PackKeypointGraphInputs(PackDetInputs):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_keypoint_labels': 'labels',
        'gt_keypoint_ids': 'keypoint_ids',
        'gt_keypoint_coords': 'keypoint_coords',
    }
    
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the sample.
        """
        packed_results = super().transform(results)
        
        if 'gt_relation_matrices' in results:
            relation_matrices = results['gt_relation_matrices']
            
            if 'gt_ignore_flags' in results:
                valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
                
                # Filter relation matrices in both dimensions (rows and columns)
                relation_matrices = relation_matrices[valid_idx][:, valid_idx, :]
            
            packed_results['data_samples'].gt_instances.relation_matrices = to_tensor(relation_matrices)
        
        return packed_results


@TRANSFORMS.register_module()
class TopDownBBoxCrop(BaseTransform):
    """Crop image tightly to bbox (x1,y1,x2,y2). Adjust keypoint coordinates by translation only.
    
    Also records a 3x3 homography_matrix (pure translation) so later transforms
    (e.g. TransformKeypoints) can update keypoint coordinates consistently.
    The new translation homography is left-multiplied onto any existing one.
    
    Additionally updates gt_bboxes (if present) by applying the same translation and clamping
    to the new image size.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        if 'bbox' not in results or 'img' not in results:
            return results
        img = results['img']
        h, w = img.shape[:2]
        x1, y1, x2, y2 = np.array(results['bbox'], dtype=np.float32)
        x1c = max(0, int(np.floor(x1)))
        y1c = max(0, int(np.floor(y1)))
        x2c = min(w, int(np.ceil(x2)))
        y2c = min(h, int(np.ceil(y2)))
        if x2c <= x1c or y2c <= y1c:
            return results

        crop = img[y1c:y2c, x1c:x2c].copy()

        # Record the homography matrix for this crop (pure translation)
        offset_w = x1c
        offset_h = y1c
        homography_matrix = np.array(
            [[1, 0, -offset_w],
             [0, 1, -offset_h],
             [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results['homography_matrix']

        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            new_w = crop.shape[1]
            new_h = crop.shape[0]
            if hasattr(gt_bboxes, 'tensor'):  # BaseBoxes instance
                b = gt_bboxes.tensor.clone()
                # subtract translation
                b[:, 0] -= offset_w  # x1
                b[:, 2] -= offset_w  # x2
                b[:, 1] -= offset_h  # y1
                b[:, 3] -= offset_h  # y2
                # clamp
                b[:, 0].clamp_(0, new_w)
                b[:, 2].clamp_(0, new_w)
                b[:, 1].clamp_(0, new_h)
                b[:, 3].clamp_(0, new_h)
                gt_bboxes.tensor = b
            else:  # numpy array
                b = gt_bboxes
                b[:, [0, 2]] -= offset_w
                b[:, [1, 3]] -= offset_h
                b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, new_w)
                b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, new_h)
                results['gt_bboxes'] = b

        results['img'] = crop
        results['bbox'] = np.array([0, 0, crop.shape[1], crop.shape[0]], dtype=np.float32)
        results['img_shape'] = (crop.shape[0], crop.shape[1])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"
