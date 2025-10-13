import numpy as np
from scipy.spatial.distance import cdist


def compute_pairwise_distances(gt_coords, pred_coords):
    """Compute pairwise Euclidean distances between GT and prediction keypoints."""
    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return np.zeros((len(gt_coords), len(pred_coords)))
    return cdist(gt_coords, pred_coords)

def _as_1d_array(x):
    """Return x as a 1D numpy array (float64)."""
    return np.atleast_1d(np.asarray(x, dtype=np.float64))

def _format_vectorized_output(arr_ds, n_d, n_s):
    """
    Convert a (D,S) array into:
      - float if D==1 and S==1,
      - (D,) if D>1 and S==1,
      - (S,) if D==1 and S>1,
      - (D,S) otherwise.
    """
    if n_d == 1 and n_s == 1:
        return float(arr_ds[0, 0])
    if n_d > 1 and n_s == 1:
        return arr_ds[:, 0]
    if n_d == 1 and n_s > 1:
        return arr_ds[0, :]
    return arr_ds


def compute_recall(
    gt_coords_list,
    gt_labels_list,
    pred_coords_list,
    pred_labels_list,
    pred_scores_list,
    max_dim_list,
    distance_threshold_norm,
    score_threshold,
    distances_list=None
):
    """
    Vectorized recall over distance and score thresholds.

    Inputs:
      - distance_threshold_norm: float or 1D array of normalized distance thresholds.
      - score_threshold: float or 1D array of score thresholds.
    Returns:
      - float if both thresholds are scalars,
      - 1D array if one of them is a vector,
      - 2D array (D, S) if both are vectors,
        where D=len(distance_threshold_norm), S=len(score_threshold).
    """
    dist_vec = _as_1d_array(distance_threshold_norm)
    score_vec = _as_1d_array(score_threshold)
    n_d, n_s = len(dist_vec), len(score_vec)

    if distances_list is None:
        distances_list = [
            compute_pairwise_distances(gt_coords, pred_coords)
            for gt_coords, pred_coords in zip(gt_coords_list, pred_coords_list)
        ]

    total_gt = 0
    covered_counts = np.zeros((n_d, n_s), dtype=np.float64)

    for idx, (gt_coords, gt_labels, pred_coords, pred_labels, pred_scores, max_dim) in enumerate(zip(
        gt_coords_list, gt_labels_list, pred_coords_list, pred_labels_list, pred_scores_list, max_dim_list
    )):
        if len(gt_coords) == 0:
            continue

        total_gt += len(gt_coords)

        if len(pred_coords) == 0:
            # No predictions -> contributes zeros to covered_counts for all thresholds
            continue

        distances = distances_list[idx]  # [G, P]
        # [G, P] label agreement
        label_match_gp = (gt_labels[:, None] == pred_labels[None, :])
        # [S, P] score mask
        valid_pred_sp = (pred_scores[None, :] >= score_vec[:, None])

        # [S, G, P] valid & label
        valid_and_label_sgp = valid_pred_sp[:, None, :] & label_match_gp[None, :, :]

        # Distances with invalid entries masked to inf -> min across P robust
        masked_distances_sgp = np.where(valid_and_label_sgp, distances[None, :, :], np.inf)
        # [S, G]
        min_dists_sg = np.min(masked_distances_sgp, axis=2)

        # Convert normalized thresholds to pixels for this image
        dist_px_d = dist_vec * float(max_dim)  # [D]

        # [D, S, G] coverage boolean
        covered_dsg = (min_dists_sg[None, :, :] <= dist_px_d[:, None, None])
        # Sum over G -> [D, S]
        covered_counts += covered_dsg.sum(axis=2)

    recalls_ds = (covered_counts / total_gt) if total_gt > 0 else np.zeros((n_d, n_s), dtype=np.float64)
    return _format_vectorized_output(recalls_ds, n_d, n_s)


def compute_precision(
    gt_coords_list,
    gt_labels_list,
    pred_coords_list,
    pred_labels_list,
    pred_scores_list,
    max_dim_list,
    distance_threshold_norm,
    score_threshold,
    distances_list=None
):
    """
    Vectorized precision over distance and score thresholds.

    Inputs:
      - distance_threshold_norm: float or 1D array of normalized distance thresholds.
      - score_threshold: float or 1D array of score thresholds.
    Returns:
      - float if both thresholds are scalars,
      - 1D array if one of them is a vector,
      - 2D array (D, S) if both are vectors,
        where D=len(distance_threshold_norm), S=len(score_threshold).

    Definition kept identical to the original:
      For each GT, the first matching prediction within the distance threshold
      and above the score threshold counts as TP, additional matches for that GT
      count as FP.
    """
    dist_vec = _as_1d_array(distance_threshold_norm)
    score_vec = _as_1d_array(score_threshold)
    n_d, n_s = len(dist_vec), len(score_vec)

    if distances_list is None:
        distances_list = [
            compute_pairwise_distances(gt_coords, pred_coords)
            for gt_coords, pred_coords in zip(gt_coords_list, pred_coords_list)
        ]

    tp_ds = np.zeros((n_d, n_s), dtype=np.float64)
    fp_ds = np.zeros((n_d, n_s), dtype=np.float64)

    for idx, (gt_coords, gt_labels, pred_coords, pred_labels, pred_scores, max_dim) in enumerate(zip(
        gt_coords_list, gt_labels_list, pred_coords_list, pred_labels_list, 
        pred_scores_list, max_dim_list
    )):
        if len(gt_coords) == 0 or len(pred_coords) == 0:
            continue

        distances_pg = distances_list[idx].T  # [P, G]
        label_match_pg = (pred_labels[:, None] == gt_labels[None, :])  # [P, G]
        valid_pred_sp = (pred_scores[None, :] >= score_vec[:, None])   # [S, P]

        # [S, P, G] score+label base mask
        base_spg = valid_pred_sp[:, :, None] & label_match_pg[None, :, :]

        # Distance thresholds in pixels for this image -> [D]
        dist_px_d = dist_vec * float(max_dim)

        # Distance condition [D, 1, P, G]
        dist_cond_d1pg = (distances_pg[None, None, :, :] <= dist_px_d[:, None, None, None])
        # Combine with base -> [D, S, P, G]
        pos_mask_dspg = dist_cond_d1pg & base_spg[None, :, :, :]

        # Positives per GT -> [D, S, G]
        positives_per_gt_dsg = pos_mask_dspg.sum(axis=2)

        # FP: extra positives per GT beyond the first
        fp_per_image_ds = np.maximum(0, positives_per_gt_dsg - 1).sum(axis=2)  # [D, S]
        # TP: at least one positive per GT
        tp_per_image_ds = (positives_per_gt_dsg > 0).sum(axis=2)               # [D, S]

        fp_ds += fp_per_image_ds
        tp_ds += tp_per_image_ds

    denom = tp_ds + fp_ds
    precision_ds = np.divide(tp_ds, denom, out=np.ones_like(tp_ds, dtype=np.float64), where=denom > 0)
    return _format_vectorized_output(precision_ds, n_d, n_s)



def compute_precision_recall_relation(
    gt_coords_list,
    gt_labels_list,
    gt_relations_list,
    relation_names_list,
    pred_coords_list,
    pred_labels_list,
    pred_scores_list,
    pred_relations_list,
    max_dim_list,
    keypoint_score_threshold,
    keypoint_distance_threshold,
    relation_score_thresholds=None,
    distances_list=None
):
    """
    Compute precision-recall curves for relations over a list of images by varying relation score threshold.
    Vectorized over relation_score_thresholds with numpy broadcasting.
    """
    if relation_score_thresholds is None:
        relation_score_thresholds = np.arange(0, 1.06, 0.05)

    thr_vec = _as_1d_array(relation_score_thresholds)  # [T]
    T = len(thr_vec)

    # Collect union of relation names across images
    relation_names = set()
    for names in relation_names_list:
        if names:
            relation_names.update(names)

    if distances_list is None:
        distances_list = [
            compute_pairwise_distances(gt_coords, pred_coords)
            for gt_coords, pred_coords in zip(gt_coords_list, pred_coords_list)
        ]

    # Initialize global TP/FP/FN accumulators per relation (arrays of shape [T])
    tp_counts = {relation: np.zeros(T, dtype=np.float64) for relation in relation_names}
    fp_counts = {relation: np.zeros(T, dtype=np.float64) for relation in relation_names}
    fn_counts = {relation: np.zeros(T, dtype=np.float64) for relation in relation_names}

    for idx, (gt_coords, gt_labels, gt_relations, rel_names, pred_coords, pred_labels, pred_scores, pred_relations, max_dim) in enumerate(zip(
        gt_coords_list, gt_labels_list, gt_relations_list, relation_names_list,
        pred_coords_list, pred_labels_list, pred_scores_list, pred_relations_list, max_dim_list
    )):
        # Skip images without necessary data
        if len(gt_coords) == 0 or gt_relations is None or len(gt_relations) == 0 or pred_relations is None:
            continue

        distance_threshold_px = float(keypoint_distance_threshold) * float(max_dim)

        # Filter predictions by keypoint score threshold
        keypoint_mask = pred_scores >= keypoint_score_threshold
        if np.sum(keypoint_mask) < 2:
            continue

        filtered_indices = np.where(keypoint_mask)[0]
        orig_to_filt = {orig_idx: i for i, orig_idx in enumerate(filtered_indices)}

        distances = distances_list[idx]

        # For each GT keypoint, find the best matching prediction of the same label within distance
        gt_to_best_pred = {}
        for gt_i in range(len(gt_coords)):
            gt_label = gt_labels[gt_i]
            # valid preds: same label and pass score
            matching_pred_mask = (pred_labels == gt_label) & keypoint_mask
            if not np.any(matching_pred_mask):
                continue
            matching_pred_indices = np.where(matching_pred_mask)[0]
            pred_distances = distances[gt_i, matching_pred_indices]
            min_j = np.argmin(pred_distances)
            min_dist = pred_distances[min_j]
            if min_dist <= distance_threshold_px:
                gt_to_best_pred[gt_i] = matching_pred_indices[min_j]

        if len(gt_to_best_pred) < 2:
            continue

        assigned_gt = sorted(gt_to_best_pred.keys())
        # map matched original pred indices -> filtered indices
        pred_idx_orig = [gt_to_best_pred[i] for i in assigned_gt]
        pred_idx_filt = [orig_to_filt[o] for o in pred_idx_orig]

        # Subset predicted relations to filtered/matched nodes
        filtered_pred_relations = pred_relations[keypoint_mask][:, keypoint_mask, :]  # [P', P', R]
        K = len(assigned_gt)
        if K < 2:
            continue

        # Off-diagonal mask for K x K matrices
        off_diag = ~np.eye(K, dtype=bool)

        # For each relation channel present in this image, accumulate counts vectorized over thresholds
        for rel_idx, rel_name in enumerate(rel_names or []):

            # Ground-truth submatrix for assigned GT nodes -> boolean
            gt_rel_matrix = np.array(gt_relations)[:, :, rel_idx]  # [G, G]
            gt_sub = gt_rel_matrix[np.ix_(assigned_gt, assigned_gt)].astype(bool)  # [K, K]

            # Predicted relation scores submatrix (matched predicted nodes) for this relation
            pred_sub = filtered_pred_relations[np.ix_(pred_idx_filt, pred_idx_filt, [rel_idx])][:, :, 0]  # [K, K]

            # Flatten off-diagonal pairs
            gt_flat = gt_sub[off_diag]                    # [N_pairs]
            pred_scores_flat = pred_sub[off_diag]         # [N_pairs]

            # Broadcast thresholds: [T, N_pairs]
            pred_has_T = (pred_scores_flat[None, :] >= thr_vec[:, None])
            gt_flat_row = gt_flat[None, :]  # [1, N_pairs]

            # Accumulate TP/FP/FN across thresholds
            tp_counts[rel_name] += np.sum(pred_has_T & gt_flat_row, axis=1)
            fp_counts[rel_name] += np.sum(pred_has_T & (~gt_flat_row), axis=1)
            fn_counts[rel_name] += np.sum((~pred_has_T) & gt_flat_row, axis=1)

    # Build curves
    relation_curves = {}
    for relation in relation_names:
        tp = tp_counts[relation]
        fp = fp_counts[relation]
        fn = fn_counts[relation]

        denom_p = tp + fp
        precision = np.divide(tp, denom_p, out=np.ones_like(tp), where=denom_p > 0)

        denom_r = tp + fn
        recall = np.divide(tp, denom_r, out=np.zeros_like(tp), where=denom_r > 0)

        denom_f = precision + recall
        f1 = np.divide(2 * precision * recall, denom_f, out=np.zeros_like(denom_f), where=denom_f > 0)

        relation_curves[relation] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'thresholds': thr_vec.tolist()
        }

    return relation_curves


def compute_pck(gt_coords, pred_coords, matchings, img_width, img_height, pck_threshold=0.05):
    """
    Compute Percentage of Correct Keypoints (PCK) for a single image.

    Args:
        gt_coords (array-like of shape [N, 2]): Ground-truth keypoint coordinates (x, y).
        pred_coords (array-like of shape [M, 2]): Predicted keypoint coordinates (x, y).
        matchings (dict[int, int]): Mapping from ground-truth index -> predicted index for matched keypoints.
        img_width (int | float): Image width in pixels.
        img_height (int | float): Image height in pixels.
        pck_threshold (float, optional): Normalized distance threshold; a match is correct if
            ||gt - pred|| / max(img_width, img_height) < pck_threshold. Defaults to 0.05.

    Returns:
        float: PCK in [0, 1], counting unmatched GT keypoints as incorrect.

    Notes:
        - Only matched GT indices present in 'matchings' are used for distance computation.
        - Unmatched GT keypoints are included in the denominator and treated as misses.
    """
    n_gt = len(gt_coords)
    gt_to_pred = np.full(n_gt, -1, dtype=np.int32)
    for gt_idx, pred_idx in matchings.items():
        gt_to_pred[gt_idx] = pred_idx

    matched_mask = gt_to_pred >= 0
    matched_gt_indices = np.where(matched_mask)[0]
    matched_pred_indices = gt_to_pred[matched_mask]

    if len(matched_gt_indices) == 0:
        return 0.0

    gt_points = np.asarray(gt_coords, dtype=np.float32)[matched_gt_indices]
    pred_points = np.asarray(pred_coords, dtype=np.float32)[matched_pred_indices]

    dists = np.linalg.norm(gt_points - pred_points, axis=1)
    dists = dists / max(img_width, img_height)
    correct = np.sum(dists < pck_threshold)

    return correct / n_gt if n_gt > 0 else 0.0