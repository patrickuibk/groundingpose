import os
import sys
import glob
import json
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow importing project utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.metrics.utils import (
    compute_pairwise_distances,
)

def discover_categories(work_dir_base: str, model_type: str) -> List[str]:
    """
    List subfolders under <work_dir_base>/<model_type> as categories.
    """
    root = os.path.join(work_dir_base, model_type)
    if not os.path.isdir(root):
        return []
    cats = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(cats)

def load_category_errors(metrics_dir: str, score_threshold: float = 0.0) -> List[float]:
    """
    Collect normalized error distances for the category found in metrics_dir.
    For each GT keypoint, take the closest prediction with the same label above score_threshold
    (predictions can be reused across GTs). If labels are unavailable, fall back to closest overall.
    Distances are normalized by the longest side of the corresponding image.
    """
    pred_files = glob.glob(os.path.join(metrics_dir, '*_pred.json'))
    errors_px: List[float] = []

    for pred_file in tqdm(pred_files, desc=f"Loading {os.path.basename(metrics_dir)}"):
        img_name = os.path.basename(pred_file).replace('_pred.json', '')
        gt_file = os.path.join(os.path.dirname(pred_file), f'{img_name}_gt.json')

        try:
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
        except Exception:
            continue

        gt_coords = np.array(gt_data.get('keypoint_coords', []), dtype=float)
        gt_labels = np.array(gt_data.get('keypoint_label_names', []))
        pred_coords = np.array(pred_data.get('keypoint_coords', []), dtype=float)
        pred_labels = np.array(pred_data.get('keypoint_label_names', []))
        pred_scores = np.array(pred_data.get('keypoint_scores', []), dtype=float)

        if gt_coords.size == 0 or pred_coords.size == 0:
            continue
        if pred_scores.size == 0:
            pred_scores = np.zeros(len(pred_coords), dtype=float)

        # Determine normalization factor (longest side). 
        img_width, img_height = gt_data['img_width'], gt_data['img_height']
        longest_side = max(img_width, img_height)

        # Pairwise distances in pixel coordinates, normalized by longest side
        distances = compute_pairwise_distances(gt_coords, pred_coords) / float(longest_side)

        # For each GT, take the closest prediction with the same label (allow reuse), respect score threshold.
        # If labels are missing or lengths mismatch, fall back to using all predictions.
        use_labels = (gt_labels.size == gt_coords.shape[0]) and (pred_labels.size == pred_coords.shape[0])
        for gt_idx in range(gt_coords.shape[0]):
            if use_labels:
                label = gt_labels[gt_idx]
                cand_idx = np.where(pred_labels == label)[0]
            else:
                cand_idx = np.arange(pred_coords.shape[0], dtype=int)
            if cand_idx.size == 0:
                continue
            local_dists = distances[gt_idx, cand_idx]
            j = int(np.argmin(local_dists))
            pred_idx = int(cand_idx[j])
            if pred_scores[pred_idx] < score_threshold:
                continue
            errors_px.append(float(local_dists[j]))

    return errors_px

def plot_error_histograms(
    work_dir_base: str,
    model_type: str,
    bins: int = 50,
    score_threshold: float = 0.0,
    out_path: str = None,
    show_ci: bool = False,
):
    categories = discover_categories(work_dir_base, model_type)
    if not categories:
        print(f"No categories found under: {os.path.join(work_dir_base, model_type)}")
        return

    per_category_errors: Dict[str, List[float]] = {}
    for cat in categories:
        metrics_dir = os.path.join(work_dir_base, model_type, cat)
        errs = load_category_errors(metrics_dir, score_threshold=score_threshold)
        per_category_errors[cat] = errs

    # Build shared bin edges across all categories (unified x-axis)
    all_errs = np.concatenate(
        [np.asarray(v, dtype=float) for v in per_category_errors.values() if len(v) > 0]
    ) if any(len(v) > 0 for v in per_category_errors.values()) else np.array([], dtype=float)

    if all_errs.size == 0:
        print("No normalized errors found. Nothing to plot.")
        return

    bin_count = int(bins)
    # Distribute bins between 0 and 0.015
    x_min = 0.0
    x_max = 0.015
    bin_edges = np.linspace(x_min, x_max, bin_count + 1)

    # Plot 4x2 figure
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(categories):
            cat = categories[i]
            errs = np.array(per_category_errors.get(cat, []), dtype=float)
            if errs.size > 0:
                ax.hist(errs, bins=bin_edges, color='tab:blue', alpha=0.8, edgecolor='white')
                ax.set_xlim(x_min, x_max)
                # Set 5 x-ticks evenly spaced between x_min and x_max
                ax.set_xticks(np.linspace(x_min, x_max, 4))
                # 95% CI of the mean (normal approximation), optional
                if show_ci and errs.size > 1:
                    mean = float(np.mean(errs))
                    se = float(np.std(errs, ddof=1) / np.sqrt(errs.size))
                    ci_low, ci_high = mean - 1.96 * se, mean + 1.96 * se
                    # ax.axvspan(ci_low, ci_high, color='tab:orange', alpha=0.2, label='95% CI (mean)')
                    ax.axvline(mean, color='tab:orange', linestyle='--', label='Mean')
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f'Dataset {i+1}')
            ax.set_xlabel('Error (normalized by longest image side)')
            ax.set_ylabel('Count')
            ax.grid(True, linestyle='--', alpha=0.3)
        else:
            ax.axis('off')

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(work_dir_base, f'{model_type}_error_hist_4x2.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram figure: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot 2x4 histograms of pixel error distributions for up to 8 datasets.")
    parser.add_argument('work_dir_base', type=str, help='Path to test results directory (root containing model types).')
    parser.add_argument('--bins', type=int, default=30, help='Number of histogram bins.')
    parser.add_argument('--score-threshold', type=float, default=0.3, help='Minimum keypoint score to include a prediction.')
    parser.add_argument('--out', type=str, default=None, help='Output image path. Defaults to <work_dir_base>/<model_type>_error_hist_2x4.png')
    parser.add_argument('--show-ci', action='store_true', help='Shade 95% CI of the mean error per dataset.')
    args = parser.parse_args()

    if not os.path.exists(args.work_dir_base):
        print(f"Base directory not found: {args.work_dir_base}")
        sys.exit(1)

    model_types = discover_categories(args.work_dir_base, '.')
    for mt in model_types:
        print(f"Found model type: {mt}")
        plot_error_histograms(
            work_dir_base=args.work_dir_base,
            model_type=mt,
            bins=args.bins,
            score_threshold=args.score_threshold,
            out_path=args.out,
            show_ci=args.show_ci,
        )

if __name__ == "__main__":
    main()