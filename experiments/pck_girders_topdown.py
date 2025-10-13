import os
import sys
import json
import glob
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
print(f"Added {ROOT} to PYTHONPATH")

from datasets.metrics.utils import compute_pck
from tools.graph_matching import sequential_matching


def parse_threshold_spec(spec: Optional[str], default=None):
    """
    Accept forms:
      "0.0,0.05,0.1"
      "start:stop:step"  (inclusive stop if (stop-start)/step is integral within float tol)
    """
    if spec is None:
        return default
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("Range spec must be start:stop:step")
        start, stop, step = map(float, parts)
        vals = []
        v = start
        # Guard against float drift
        while v <= stop + 1e-9:
            vals.append(round(v, 10))
            v += step
        return vals
    else:
        return [float(x) for x in spec.split(",") if x.strip() != ""]


def _index_predictions_by_image_id(
    pred_dir: str,
    pred_suffix: str = "_pred.json",
) -> Tuple[Dict[int, dict], Dict[str, int]]:
    """
    Scan pred_dir, load all prediction JSONs, and index them by image_id taken from inside each JSON.
    Does not rely on filenames.
    """
    preds_by_id: Dict[int, dict] = {}
    scanned = 0
    indexed = 0
    duplicates = 0

    def _extract_image_id(d: dict) -> Optional[int]:
        # Common key variants
        for k in ("image_id", "img_id", "imageId", "imageID"):
            if k in d and d[k] is not None:
                return int(d[k])
        # Nested locations
        for parent in ("image", "meta", "metadata"):
            if parent in d and isinstance(d[parent], dict):
                for k in ("id", "image_id", "img_id"):
                    if k in d[parent] and d[parent][k] is not None:
                        return int(d[parent][k])
        return None

    # Load every JSON in pred_dir, optionally filter by suffix if set
    pattern = os.path.join(pred_dir, f"*{pred_suffix}") if pred_suffix else os.path.join(pred_dir, "*.json")
    for path in glob.glob(pattern):
        scanned += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        img_id = _extract_image_id(d)
        if img_id is None:
            continue
        if img_id in preds_by_id:
            duplicates += 1
            # Keep the first; skip subsequent duplicates
            continue
        preds_by_id[img_id] = d
        indexed += 1

    return preds_by_id, {"scanned": scanned, "indexed": indexed, "duplicates": duplicates}


def process_coco_annotations(
    coco_ann_path: str,
    pred_dir: str,
    distance_thresholds: List[float],
    score_threshold: float = 0.3,
    pred_suffix: str = "_pred.json",
) -> Dict[str, float]:
    """
    Iterate over a COCO annotation file. For each image, fetch the prediction by image_id from a preloaded index.
    For each annotation (single group), filter predicted keypoints to the bbox and above score, run graph matching
    (LA + FAQ + Our 2OPT) using expected bbox-based coordinates, then compute PCK vs thresholds.
    """
    with open(coco_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_per_img: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        anns_per_img.setdefault(ann["image_id"], []).append(ann)
    categories = {c["id"]: c for c in coco.get("categories", [])}

    preds_by_id, pred_index_info = _index_predictions_by_image_id(pred_dir, pred_suffix=pred_suffix)

    all_pcks_per_thr: Dict[float, List[float]] = {thr: [] for thr in distance_thresholds}
    # Weighted aggregation accumulators
    weighted_sum_per_thr: Dict[float, float] = {thr: 0.0 for thr in distance_thresholds}
    weight_count_per_thr: Dict[float, int] = {thr: 0 for thr in distance_thresholds}

    num_images = 0
    num_images_with_preds = 0
    num_annotations = 0
    total_pairs = 0

    for image_id, img in images.items():
        num_images += 1
        img_w = int(img["width"])
        img_h = int(img["height"])

        pred_data_full = preds_by_id.get(image_id)
        if pred_data_full is None:
            raise ValueError(f"No prediction JSON found for image_id={image_id} in pred_dir={pred_dir}.")

        num_images_with_preds += 1

        pred_coords_full = np.asarray(pred_data_full["keypoint_coords"], dtype=float)
        pred_label_names_full = np.asarray(pred_data_full["keypoint_label_names"])
        pred_scores_full = np.asarray(pred_data_full["keypoint_scores"], dtype=float)
        pred_rel_full = np.asarray(pred_data_full["keypoint_relation_scores"], dtype=float)

        for ann in anns_per_img.get(image_id, []):
            num_annotations += 1
            bx, by, bw, bh = ann["bbox"]  # [x, y, w, h]

            # Parse COCO keypoints [x, y, v]
            gt_kps = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
            if gt_kps.size == 0:
                raise ValueError(f"No GT keypoints for image_id={image_id}.")
            gt_coords = gt_kps[:, :2]
            n_gt = gt_coords.shape[0]

            # Derive GT label names for keypoints from category keypoints
            cat = categories[ann["category_id"]]
            cat_kp_names = cat["keypoints"]
            gt_label_names = [str(cat_kp_names[i]) for i in range(gt_kps.shape[0])]

            # Create GT relation matrices from skeleton with same third-dim R
            R = pred_rel_full.shape[2]
            skeleton_edges = cat["skeleton"]
            gt_relation_matrices = np.zeros((n_gt, n_gt, R), dtype=float)
            # For each edge in skeleton, set relation to 1 for all R channels
            for edge in skeleton_edges:
                i, j = edge
                # COCO skeleton is 1-indexed; convert to 0-indexed
                i -= 1
                j -= 1
                if 0 <= i < n_gt and 0 <= j < n_gt:
                    gt_relation_matrices[i, j, :] = 1.0
                    gt_relation_matrices[j, i, :] = 1.0

            # Filter predicted keypoints to those inside bbox and above score threshold
            xs = pred_coords_full[:, 0]
            ys = pred_coords_full[:, 1]
            in_box = (xs >= bx) & (xs <= bx + bw) & (ys >= by) & (ys <= by + bh)
            good_kp_score = pred_scores_full >= score_threshold
            keep_mask_full = in_box & good_kp_score

            keep_idx = np.where(keep_mask_full)[0]
            if keep_idx.size == 0:
                raise ValueError(f"No predicted keypoints inside bbox for image_id={image_id}.")

            pred_coords = pred_coords_full[keep_idx]
            pred_label_names = pred_label_names_full[keep_idx]
            pred_scores = pred_scores_full[keep_idx] if pred_scores_full.size > 0 else np.ones(keep_idx.size, dtype=float)
            pred_rel = pred_rel_full[np.ix_(keep_idx, keep_idx, np.arange(R))]

            # Prepare labels (union of GT and predicted label names)
            all_label_names = sorted(set(gt_label_names) | set(pred_label_names))
            label_name_to_int = {name: idx for idx, name in enumerate(all_label_names)}
            gt_label_ids = np.asarray([label_name_to_int[name] for name in gt_label_names], dtype=int)
            pred_label_ids = np.asarray([label_name_to_int[name] for name in pred_label_names], dtype=int)

            # Build expected GT coordinates along the longer bbox side, both forward and backward
            if bw >= bh:
                # Horizontal: spread along x, centered in y
                xs_line_fwd = np.linspace(bx, bx + bw, num=n_gt, dtype=float)
                xs_line_bwd = np.linspace(bx + bw, bx, num=n_gt, dtype=float)
                ys_line = np.full(n_gt, by + bh / 2.0, dtype=float)
                expected_gt_coords_fwd = np.stack([xs_line_fwd, ys_line], axis=1)
                expected_gt_coords_bwd = np.stack([xs_line_bwd, ys_line], axis=1)
            else:
                # Vertical: spread along y, centered in x
                ys_line_fwd = np.linspace(by, by + bh, num=n_gt, dtype=float)
                ys_line_bwd = np.linspace(by + bh, by, num=n_gt, dtype=float)
                xs_line = np.full(n_gt, bx + bw / 2.0, dtype=float)
                expected_gt_coords_fwd = np.stack([xs_line, ys_line_fwd], axis=1)
                expected_gt_coords_bwd = np.stack([xs_line, ys_line_bwd], axis=1)

            # Normalize coordinates for matching
            norm = np.array([img_w, img_h], dtype=float)
            pred_coords_norm = pred_coords / norm
            expected_gt_coords_fwd_norm = expected_gt_coords_fwd / norm
            expected_gt_coords_bwd_norm = expected_gt_coords_bwd / norm

            # Run graph matching (LA + FAQ + Our 2OPT) for both directions
            pipeline = ['la', 'faq', 'our2opt']
            try:
                result_fwd = sequential_matching(
                    gt_label_ids,
                    gt_relation_matrices,
                    pred_label_ids,
                    pred_scores,
                    pred_rel,
                    algorithm_sequence=pipeline,
                    initial_matching=None,
                    gt_coords=expected_gt_coords_fwd_norm,
                    pred_coords=pred_coords_norm,
                    return_intermediate=False
                )

                result_bwd = sequential_matching(
                    gt_label_ids,
                    gt_relation_matrices,
                    pred_label_ids,
                    pred_scores,
                    pred_rel,
                    algorithm_sequence=pipeline,
                    initial_matching=None,
                    gt_coords=expected_gt_coords_bwd_norm,
                    pred_coords=pred_coords_norm,
                    return_intermediate=False
                )
            except Exception as e:
                raise RuntimeError(f"Graph matching failed for image_id={image_id}: {e}")
            

            # Compute PCK for each threshold using true GT coords and both matchings, take best
            for thr in distance_thresholds:
                pck_fwd = compute_pck(gt_coords, pred_coords, result_fwd, img_w, img_h, pck_threshold=float(thr))
                pck_bwd = compute_pck(gt_coords, pred_coords, result_bwd, img_w, img_h, pck_threshold=float(thr))
                pck_val = max(pck_fwd, pck_bwd)
                print(f"Image {image_id}, Ann {ann['id']}, Thr {thr:.3f}: PCK={pck_val:.3f}")
                all_pcks_per_thr[thr].append(pck_val)
                # Weighted aggregation by number of GT keypoints
                weighted_sum_per_thr[thr] += pck_val * n_gt
                weight_count_per_thr[thr] += n_gt

            total_pairs += 1

    # Aggregate weighted means
    results = {
        str(thr): (
            float(weighted_sum_per_thr[thr] / weight_count_per_thr[thr])
            if weight_count_per_thr[thr] > 0 else 0.0
        )
        for thr in distance_thresholds
    }
    info = {
        "num_matched_pairs": total_pairs,
        "score_threshold": float(score_threshold),
        "thresholds": [float(t) for t in distance_thresholds],
        "matching_pipeline": "la + faq + our2opt",
        "expected_coords": "n_gt points along longer bbox side, centered on shorter side",
        "aggregation": "weighted mean by number of GT keypoints"
    }
    return {"info": info, "pck_vs_threshold": results}


def main():
    ap = argparse.ArgumentParser(description="Compute PCK for girders using graph matching and expected bbox-based coordinates over a COCO annotation file")
    ap.add_argument("--coco-ann", required=True, help="Path to COCO annotations JSON")
    ap.add_argument("--pred-dir", required=True, help="Directory containing per-image prediction files")
    ap.add_argument("--pred-suffix", default="_pred.json", help="Suffix to filter prediction files (default: _pred.json)")
    ap.add_argument("--distance-thresholds", default="0.05,0.01,0.005,0.001", help="List or range start:stop:step (normalized)")
    ap.add_argument("--score-threshold", type=float, default=0.3, help="Keypoint score threshold to filter predictions")
    ap.add_argument("--output", default=None, help="Optional output JSON path (default: alongside pred-dir as pck_girders_coco.json)")
    args = ap.parse_args()

    thresholds = parse_threshold_spec(args.distance_thresholds)
    if thresholds is None or len(thresholds) == 0:
        raise ValueError("No distance thresholds provided.")

    out = process_coco_annotations(
        coco_ann_path=args.coco_ann,
        pred_dir=args.pred_dir,
        distance_thresholds=thresholds,
        score_threshold=args.score_threshold,
        pred_suffix=args.pred_suffix,
    )

    default_out = os.path.join(os.path.dirname(os.path.abspath(args.pred_dir)), "pck_girders_coco.json")
    out_path = args.output or default_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved PCK vs thresholds to {out_path}")


if __name__ == "__main__":
    main()