import numpy as np
from typing import Sequence, Optional

from mmpose.evaluation.metrics.keypoint_2d_metrics import PCKAccuracy
from mmdet.registry import METRICS
from tools.graph_matching import sequential_matching
from collections import Counter


@METRICS.register_module()
class TopDownPCKAccuracy(PCKAccuracy):
    """
    Top-down PCK metric for single-instance pose estimation per sample.

    This metric evaluates keypoint detection accuracy using Percentage of Correct Keypoints (PCK)
    in a top-down setting, where each data sample contains a single ground truth instance (already cropped).

    For each data sample:
      - Predicted keypoints (with labels and scores) are matched to ground truth keypoints using
        graph matching (RRWM + Hungarian), leveraging skeleton relations for assignment.
      - Predicted keypoint coordinates (relative to crop) are mapped back to global image coordinates
        using the original bounding box.
      - Both prediction and ground truth arrays are constructed in a fixed shape (num_keypoints, 3).
      - PCK is computed with normalization by the bounding box size (max(width, height)).

    Required data_sample fields:
      pred_instances:
        - label_names: list[str]
        - scores: Tensor[N]
        - keypoints: Tensor[N,2] (local coords relative to cropped bbox origin)
        - relation_scores: Tensor[N,N,C] (optional, for graph matching)
      raw_ann_info: list with a single COCO-style annotation dict (bbox, keypoints, category_id)
    """

    def __init__(self,
                 thr: float = 0.05,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 max_keypoints: int = 100,
                 **kwargs):
        """
        Args:
            thr (float): PCK threshold (as a fraction of bbox size).
            collect_device (str): Device for result collection.
            prefix (str, optional): Prefix for metric name.
            max_keypoints (int): Maximum number of keypoints (for fixed output shape).
            **kwargs: Additional arguments for base class.
        """
        super().__init__(thr=thr,
                         norm_item='bbox',
                         collect_device=collect_device,
                         prefix=prefix)
        self.max_keypoints = int(max_keypoints)

    def _build_gt(self, ann):
        """
        Build ground truth keypoint array and relation matrix for a given annotation.

        Args:
            ann (dict): Annotation dictionary containing at least 'category_id' and 'keypoints' fields.

        Returns:
            tuple:
                - gt_kpts (np.ndarray): Array of visible keypoints (K, 3).
                - gt_label_names (list): List of keypoint label names for the given category.
                - gt_rel (np.ndarray): Relation matrix (K, K, 1) for visible keypoints.
        """
        meta = self.dataset_meta
        cat_id = ann['category_id']
        gt_label_names = meta['category_keypoints'][cat_id]

        gt_kpts = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)

        # Ensure gt_label_names and gt_kpts have the same length
        min_len = min(len(gt_label_names), gt_kpts.shape[0])
        gt_label_names = gt_label_names[:min_len]
        gt_kpts = gt_kpts[:min_len]

        # Build relation matrix among visible keypoints using category skeleton
        gt_rel = np.zeros((gt_kpts.shape[0], gt_kpts.shape[0], 1), dtype=np.float32)
        skel = self.dataset_meta['category_skeleton'][cat_id]
        for u, v in skel:
            if u == v:
                continue
            if u < gt_kpts.shape[0] and v < gt_kpts.shape[0]:
                gt_rel[u, v, 0] = 1.0
                gt_rel[v, u, 0] = 1.0

        # Filter to only visible keypoints
        visible = gt_kpts[:, 2] > 0
        gt_kpts = gt_kpts[visible]
        gt_rel = gt_rel[visible][:, visible, :]
        gt_label_names = [n for i, n in enumerate(gt_label_names) if visible[i]]

        return gt_kpts, gt_label_names, gt_rel

    def _get_preds(self, pred_inst, gt_label_names):
        """
        Select and align predicted keypoints to ground truth label names.

        Args:
            pred_inst (dict): Predicted instance dictionary with keys 'label_names', 'scores', 'keypoints', 'relation_scores'.
            gt_label_names (list): List of ground truth keypoint label names.

        Returns:
            tuple:
                - pred_labels (list): Selected predicted label names.
                - pred_scores (np.ndarray): Selected predicted keypoint scores.
                - pred_keypoints (np.ndarray): Selected predicted keypoint coordinates.
                - rel (np.ndarray): Selected and symmetrized relation matrix.
        """
        pred_label_names = pred_inst['label_names']
        pred_keypoint_scores = pred_inst['scores'].cpu().numpy().astype(np.float32)
        pred_keypoints = pred_inst['keypoints'].cpu().numpy().astype(np.float32)
        pred_relation_scores = pred_inst['relation_scores'].cpu().numpy().astype(np.float32)

        label_counter = Counter(gt_label_names)
        selected_indices = []
        for name, count in label_counter.items():
            indices = [i for i, pred_name in enumerate(pred_label_names) if name == pred_name]
            # Select top-scoring predictions for this label
            count = min(count, len(indices))
            top_indices = np.argsort(pred_keypoint_scores[indices])[-count:]
            selected_indices.extend([indices[i] for i in top_indices])

        selected_indices = np.array(selected_indices, dtype=int)
        pred_labels = [pred_label_names[i] for i in selected_indices]
        pred_scores = pred_keypoint_scores[selected_indices]
        pred_keypoints = pred_keypoints[selected_indices]
        rel = pred_relation_scores[np.ix_(selected_indices, selected_indices, range(pred_relation_scores.shape[2]))]

        # Symmetrize and zero diagonal
        rel = (rel + np.transpose(rel, (1, 0, 2))) / 2.0
        for i in range(rel.shape[2]):
            np.fill_diagonal(rel[..., i], 0)

        return pred_labels, pred_scores, pred_keypoints, rel

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        """
        Process a batch of data samples and accumulate results for metric computation.

        Args:
            data_batch (Sequence[dict]): Input data batch (not used).
            data_samples (Sequence[dict]): List of data samples, each containing predictions and ground truth.
        """
        for data_sample in data_samples:
            ann = data_sample['raw_ann_info'][0]

            gt_keypoints, gt_label_names, gt_rel = self._build_gt(ann)

            if len(gt_label_names) == 0:
                continue

            pred_label_names, pred_scores, pred_keypoints, pred_rel = self._get_preds(data_sample['pred_instances'], gt_label_names)

            # Local->global coords
            bx, by, bw, bh = ann['bbox']
            pred_keypoints = pred_keypoints.copy()
            pred_keypoints[:, 0] += bx
            pred_keypoints[:, 1] += by

            # Label name -> int
            unique_names = set(gt_label_names) | set(pred_label_names)
            name_to_idx = {name: i for i, name in enumerate(sorted(unique_names))}
            gt_labels = np.array([name_to_idx[name] for name in gt_label_names])
            pred_labels = np.array([name_to_idx[name] for name in pred_label_names])

            match = sequential_matching(
                algorithm_sequence=['rrwm', 'our2opt'],
                gt_labels=gt_labels,
                gt_relation_matrices=gt_rel,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                pred_relation_matrices=pred_rel,
            )

            # Build gt and prediction array with fixed global shape (max_keypoints, 3
            gt_full = np.zeros((self.max_keypoints, 3), dtype=np.float32)
            gt_full[:gt_keypoints.shape[0], :] = gt_keypoints

            pred_full = np.zeros((self.max_keypoints, 3), dtype=np.float32)
            for gt_pos, pred_pos in match.items():
                pred_full[gt_pos, 0] = pred_keypoints[pred_pos, 0]
                pred_full[gt_pos, 1] = pred_keypoints[pred_pos, 1]
                pred_full[gt_pos, 2] = pred_scores[pred_pos]

            # Mask: visible GT entries are the first K rows we filled
            mask = gt_full[:, 2] > 0

            side = max(bw, bh)
            self.results.append({
                'pred_coords': pred_full[:, :2].reshape(1, self.max_keypoints, 2),
                'gt_coords': gt_full[:, :2].reshape(1, self.max_keypoints, 2),
                'mask': mask.reshape(1, self.max_keypoints),
                'bbox_size': np.array([[side, side]], dtype=np.float32),
                'gt_instance_id': ann.get('id', -1),
                'category_id': ann['category_id']
            })