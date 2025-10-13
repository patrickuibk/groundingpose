import os.path as osp
from typing import List, Union, Optional
import numpy as np
import os
import json

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CocoStylePoseDataset(CocoDataset):
    """COCO-style dataset with per-category keypoints and optional relations.
    
    Features:
    - Supports per-category keypoint name mappings.
    - Optional fully-connected relation between an instance's keypoints, named by `relation_name`.
    - Adds a 'skeleton' relation if categories define `skeleton` in the COCO categories.
    - Can restrict text vocabulary to only visible keypoints in an image.
    - Relations are attached per keypoint instance as neighbor lists of global keypoint IDs
      (e.g., {'fc': [ids...], 'skeleton': [ids...]}), excluding self.
    """
    METAINFO = {}

    def __init__(self,
                 data_root,
                 ann_file,
                 sigmas: str = None,
                 relation_name: Optional[str] = None,
                 only_visible_keypoints: bool = False,
                 consistent_keypoints_per_category: bool = True,
                 **kwargs):
        """Initialize dataset.
        
        Args:
            data_root (str): Dataset root path.
            ann_file (str): COCO-style annotation file (relative to data_root or absolute).
            sigmas (str, optional): Comma-separated list of floats for keypoint sigmas (len must match num_keypoints).
            relation_name (str, optional): If set, adds a fully-connected relation for each instance's local keypoints
                under this name (e.g., 'fc').
            only_visible_keypoints (bool): If True, limits `data_info['text']` to keypoints visible across the image.
            consistent_keypoints_per_category (bool): If True, all categories share the same keypoint order/names
                defined by the first category; otherwise, uses per-category lists.
            **kwargs: Passed to parent CocoDataset.
        """
        self.relation_name = relation_name
        self.sigmas = sigmas
        self.only_visible_keypoints = only_visible_keypoints
        self.consistent_keypoints_per_category = consistent_keypoints_per_category
        metainfo = self.load_metainfo(data_root, ann_file)
        super().__init__(data_root=data_root, ann_file=ann_file, metainfo=metainfo, **kwargs)

    def load_metainfo(self, data_root, ann_file: str) -> dict:
        """Load dataset metainfo from COCO categories and optional sigmas.
        
        Returns:
            dict: Metainfo with fields:
                - classes (Tuple[str, ...]): Global keypoint names.
                - relation_names (Tuple[str, ...]): Includes `relation_name` if provided and
                  'skeleton' if any category defines a skeleton.
                - num_keypoints (int): Number of global keypoints.
                - category_keypoints (Dict[int, Tuple[str, ...]]): Per-category keypoint names.
                - category_skeleton (Dict[int, List[Tuple[int, int]]]): Edges per category using
                  category-local keypoint indices.
                - category_to_kpt_idx (Dict[int, List[int]]): Map category to indices in `classes`.
                - kpt_name_to_idx (Dict[str, int]): Name-to-index in `classes`.
                - sigmas (np.ndarray, optional): If provided via `sigmas`.
        """
        if not os.path.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)

        with open(ann_file, 'r', encoding='utf-8') as f:
            raw_data_info = json.load(f)

        metainfo = {}

        categories = raw_data_info.get('categories', [])
        # Build per-category keypoint name mapping and global classes
        category_keypoints = {}
        if self.consistent_keypoints_per_category:
            unique_keypoint_names = categories[0]['keypoints'] if categories else []
            for cat in categories:
                category_keypoints[cat['id']] = tuple(unique_keypoint_names)
        else:
            # Union of all keypoint names across categories (deduplicated, order-preserving)
            seen = set()
            unique_keypoint_names = []
            for cat in categories:
                kps = cat.get('keypoints', [])
                category_keypoints[cat['id']] = tuple(kps)
                for kp in kps:
                    if kp not in seen:
                        seen.add(kp)
                        unique_keypoint_names.append(kp)

        metainfo['classes'] = tuple(unique_keypoint_names)
        # Build relation names: fully-connected (if requested) and skeleton (if any category provides it)
        relation_names = []
        if self.relation_name:
            relation_names.append(self.relation_name)
        # Per-category skeleton
        category_skeleton = {}
        has_skeleton = False
        for cat in categories:
            skel = cat.get('skeleton', []) or []
            edges = []
            for pair in skel:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    u = int(pair[0])
                    v = int(pair[1])
                    if u >= 0 and v >= 0:
                        edges.append((u, v))
            if edges:
                has_skeleton = True
            category_skeleton[cat['id']] = edges
        if has_skeleton:
            relation_names.append('skeleton')
        metainfo['relation_names'] = tuple(relation_names)
        metainfo['num_keypoints'] = len(unique_keypoint_names)
        metainfo['category_keypoints'] = category_keypoints
        metainfo['category_skeleton'] = category_skeleton
        # Added: precomputed mapping from category to global keypoint indices
        name_to_idx = {n: i for i, n in enumerate(unique_keypoint_names)}
        metainfo['category_to_kpt_idx'] = {
            cid: [name_to_idx[n] for n in kp_names if n in name_to_idx]
            for cid, kp_names in category_keypoints.items()
        }
        # Added mapping for metric expansion
        metainfo['kpt_name_to_idx'] = name_to_idx

        if self.sigmas is not None:
            metainfo['sigmas'] = np.array(
                [float(sigma) for sigma in self.sigmas.split(',')],
                dtype=np.float32)
            if len(metainfo['sigmas']) != metainfo['num_keypoints']:
                raise ValueError(
                    f'Length of sigmas ({len(metainfo["sigmas"])}) does not match '
                    f'number of keypoints ({metainfo["num_keypoints"]}).')
        return metainfo

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse one image's annotations into detector-friendly format.
        
        Args:
            raw_data_info (dict): Must contain 'raw_img_info' and 'raw_ann_info' from COCO.
        
        Returns:
            dict: Parsed fields:
                - img_path, img_id, height, width, crowd_index? (if present), custom_entities=True
                - text (Tuple[str, ...]): Keypoint vocabulary for this image (visible-only if configured).
                - relation_text (Tuple[str, ...]): Relation names present (e.g., ('fc', 'skeleton')).
                - instances (List[dict]): One entry per visible keypoint, with:
                    - keypoint_id (int): Global running id within the image.
                    - keypoint_coords (List[float, float]): [x, y].
                    - keypoint_label (int): Index into `data_info['text']`.
                    - keypoint_name (str): Name in `data_info['text']`.
                    - ignore_flag (int): 0 by default.
                    - keypoint_relations (Dict[str, List[int]]): For each relation type,
                      a list of neighbor global keypoint IDs for this keypoint. For example:
                      - For fully-connected (`relation_name`): all other kept keypoints of the same instance.
                      - For 'skeleton': category-skeleton-adjacent kept keypoints of the same instance.
                - raw_ann_info: Original annotations for reference.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        if 'crowdIndex' in img_info:
            data_info['crowd_index'] = img_info['crowdIndex']
        data_info['custom_entities'] = True

        if self.only_visible_keypoints:
            # Collect visible keypoint names using per-category mappings
            visible_names = []
            seen_names = set()
            cat_kp_map = self.metainfo.get('category_keypoints', {})
            for ann in ann_info:
                keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                kp_names = cat_kp_map.get(ann.get('category_id'), self.metainfo['classes'])
                for i, (_, _, v) in enumerate(keypoints):
                    if i >= len(kp_names):
                        break
                    if v > 0:
                        name = kp_names[i]
                        if name not in seen_names:
                            seen_names.add(name)
                            visible_names.append(name)
            data_info['text'] = tuple(visible_names) if visible_names else tuple(self.metainfo['classes'])
        else:
            data_info['text'] = tuple(self.metainfo['classes'])

        text_to_label = {text: i for i, text in enumerate(data_info['text'])}
        # Always set (may be empty tuple)
        data_info['relation_text'] = self.metainfo['relation_names']

        instances = []
        keypoint_id = 0
        cat_kp_map = self.metainfo.get('category_keypoints', {})
        cat_skel_map = self.metainfo.get('category_skeleton', {})

        for ann in ann_info:
            if ann.get('ignore', False):
                continue

            keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            kp_names = cat_kp_map.get(ann.get('category_id'), self.metainfo['classes'])

            # Determine which local keypoints will be kept (visible and in current text vocabulary)
            visible_local_idxs = []
            for i, (_, _, v) in enumerate(keypoints):
                if i >= len(kp_names):
                    break
                name = kp_names[i]
                if v > 0 and name in text_to_label:
                    visible_local_idxs.append(i)

            if not visible_local_idxs:
                continue

            # Pre-assign global keypoint ids for only the kept keypoints
            start_kp_id = keypoint_id
            local_to_global = {li: start_kp_id + j for j, li in enumerate(visible_local_idxs)}
            # Prepare adjacency for relations on local indices
            # Fully connected neighbors (exclude self)
            fc_neighbors_local = None
            if self.relation_name:
                fc_neighbors_local = {
                    li: [lj for lj in visible_local_idxs if lj != li]
                    for li in visible_local_idxs
                }

            # Skeleton neighbors (use zero-based category skeleton, undirected)
            skel_edges = cat_skel_map.get(ann.get('category_id')) or []
            skel_adj = {li: set() for li in range(len(kp_names))}
            for u, v in skel_edges:
                if 0 <= u < len(kp_names) and 0 <= v < len(kp_names):
                    skel_adj[u].add(v)
                    skel_adj[v].add(u)
            skel_neighbors_local = {
                li: [lj for lj in skel_adj.get(li, []) if lj in local_to_global]
                for li in visible_local_idxs
            }

            # Now emit instances and attach per-instance neighbor lists (global ids)
            for j, li in enumerate(visible_local_idxs):
                x, y, _ = keypoints[li]
                text = kp_names[li]
                kid = local_to_global[li]

                relation_dict_i = {}
                if self.relation_name:
                    # Map local neighbors to global ids, exclude self
                    nbrs = [local_to_global[lj] for lj in fc_neighbors_local[li]]
                    if nbrs:
                        relation_dict_i[self.relation_name] = nbrs
                # Skeleton neighbors to global ids
                skel_nbrs = [local_to_global[lj] for lj in skel_neighbors_local.get(li, [])]
                if skel_nbrs:
                    relation_dict_i['skeleton'] = skel_nbrs

                instances.append({
                    'keypoint_id': kid,
                    'keypoint_coords': [x, y],
                    'keypoint_label': text_to_label[text],
                    'keypoint_name': text,
                    'ignore_flag': 0,
                    'keypoint_relations': relation_dict_i
                })
                keypoint_id += 1

        data_info['instances'] = instances
        data_info['raw_ann_info'] = ann_info
        return data_info