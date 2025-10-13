import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.metrics.utils import (
    compute_pairwise_distances,
    compute_precision,
    compute_recall,
    compute_precision_recall_relation
)


def discover_categories(work_dir_base):
    """Discover categories by listing subfolders under each model type."""
    categories = set()
    for model_type in ['shared', 'finetuned', 'individual']:
        model_dir = os.path.join(work_dir_base, model_type)
        if os.path.isdir(model_dir):
            for entry in os.listdir(model_dir):
                category_path = os.path.join(model_dir, entry)
                if os.path.isdir(category_path):
                    categories.add(entry)
    return sorted(list(categories))


def load_and_preprocess_image(args):
    pred_file, metrics_dir = args
    img_name = os.path.basename(pred_file).replace('_pred.json', '')
    gt_file = os.path.join(os.path.dirname(pred_file), f'{img_name}_gt.json')
    if not os.path.exists(gt_file):
        print(f"Warning: GT file not found for {pred_file}")
        return None
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    img_width, img_height = gt_data['img_width'], gt_data['img_height']
    max_dim = max(img_width, img_height)

    gt_coords = np.array(gt_data.get('keypoint_coords', []))
    gt_labels = np.array(gt_data.get('keypoint_label_names', []))

    pred_coords = np.array(pred_data.get('keypoint_coords', []))
    pred_labels = np.array(pred_data.get('keypoint_label_names', []))
    pred_scores = np.array(pred_data.get('keypoint_scores', []))

    distances = compute_pairwise_distances(gt_coords, pred_coords)

    # Optional relation data
    gt_relations = np.array(gt_data.get('relation_matrices', []))
    relation_names = gt_data.get('relation_names', [])
    pred_relations = np.array(pred_data.get('keypoint_relation_scores', []))

    return {
        'img_width': img_width,
        'img_height': img_height,
        'max_dim': max_dim,
        'gt_coords': gt_coords,
        'gt_labels': gt_labels,
        'gt_relations': gt_relations,
        'relation_names': relation_names,
        'pred_coords': pred_coords,
        'pred_labels': pred_labels,
        'pred_scores': pred_scores,
        'pred_relations': pred_relations,
        'distances': distances,
    }

def load_and_preprocess_image_pairs(metrics_dir):
    """Load prediction/GT files and preprocess all image pairs for a category using processes."""
    pred_files = glob.glob(os.path.join(metrics_dir, '*_pred.json'))
    args_list = [(pred_file, metrics_dir) for pred_file in pred_files]
    precomputed_pairs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_and_preprocess_image, args_list), total=len(pred_files), desc=f"Loading and preprocessing images"))
        precomputed_pairs = [r for r in results if r is not None]
    return precomputed_pairs

def compute_all_metrics(work_dir_base, categories, model_type):
    """
    Compute all metrics (keypoint PR, relation PR, keypoint PR-distance) for a specific model type
    across all categories.
    """
    print(f"-- Computing metrics for model type: {model_type}")
    
    # Define thresholds
    distance_thresholds = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0]
    keypoint_score_threshold = 0.3  # Fixed threshold for keypoint PR and relation PR
    keypoint_distance_threshold = 0.005  # Fixed threshold for keypoint and relation PR curves
    score_thresholds = np.arange(0, 1.06, 0.05)  # match utils default

    # Initialize result structures
    all_keypoint_pr_data = {}
    all_keypoint_pr_dist_data = {}
    all_relation_pr_data = {}
    
    for category in categories:
        print(f"- Processing category: {category}")
        metrics_dir = os.path.join(work_dir_base, model_type, category)
        if not os.path.exists(metrics_dir):
            print(f"Skipping {category} - directory not found: {metrics_dir}")
            continue
            
        # Load and preprocess all image pairs
        precomputed_pairs = load_and_preprocess_image_pairs(metrics_dir)
        if not precomputed_pairs:
            print(f"No prediction/GT pairs found for {category}")
            continue

        # Build per-image lists required by generalized utils
        gt_coords_list = [d['gt_coords'] for d in precomputed_pairs]
        gt_labels_list = [d['gt_labels'] for d in precomputed_pairs]
        pred_coords_list = [d['pred_coords'] for d in precomputed_pairs]
        pred_labels_list = [d['pred_labels'] for d in precomputed_pairs]
        pred_scores_list = [d['pred_scores'] for d in precomputed_pairs]
        max_dim_list = [d['max_dim'] for d in precomputed_pairs]

        gt_relations_list = [d.get('gt_relations') for d in precomputed_pairs]
        relation_names_list = [d.get('relation_names') for d in precomputed_pairs]
        pred_relations_list = [d.get('pred_relations') for d in precomputed_pairs]

        distances_list = [d.get('distances') for d in precomputed_pairs] if 'distances' in precomputed_pairs[0] else None

        # Keypoint PR (vary score thresholds at fixed distance)
        precisions = compute_precision(
            gt_coords_list, gt_labels_list,
            pred_coords_list, pred_labels_list, pred_scores_list,
            max_dim_list, keypoint_distance_threshold, score_thresholds, distances_list
        )
        recalls = compute_recall(
            gt_coords_list, gt_labels_list,
            pred_coords_list, pred_labels_list, pred_scores_list,
            max_dim_list, keypoint_distance_threshold, score_thresholds, distances_list
        )
        all_keypoint_pr_data[category] = (np.asarray(precisions).tolist(), np.asarray(recalls).tolist(), score_thresholds)

        # Keypoint PR (vary distance thresholds at fixed score)
        dist_precisions = compute_precision(
            gt_coords_list, gt_labels_list,
            pred_coords_list, pred_labels_list, pred_scores_list,
            max_dim_list, distance_thresholds, keypoint_score_threshold, distances_list
        )
        dist_recalls = compute_recall(
            gt_coords_list, gt_labels_list,
            pred_coords_list, pred_labels_list, pred_scores_list,
            max_dim_list, distance_thresholds, keypoint_score_threshold, distances_list
        )
        all_keypoint_pr_dist_data[category] = (np.asarray(dist_precisions).tolist(), np.asarray(dist_recalls).tolist(), distance_thresholds)

        # Relation PR
        relation_curves = compute_precision_recall_relation(
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
            None,
            distances_list
        )
        all_relation_pr_data[category] = relation_curves
    
    return (
        all_keypoint_pr_data,
        all_relation_pr_data,
        all_keypoint_pr_dist_data,
        keypoint_score_threshold,
        keypoint_distance_threshold
    )

# Helper to format thresholds without trailing zeros (e.g., 0.3000 -> 0.3)
def _fmt_thr(value, max_decimals=4):
    try:
        s = f"{float(value):.{max_decimals}f}"
        s = s.rstrip('0').rstrip('.')
        return '0' if s in ('-0', '') else s
    except Exception:
        return str(value)

def plot_pr_series(
    ax,
    series,
    title,
    xlabel='Recall',
    ylabel='Precision',
    legend_loc='lower left',
    f1_at=None,
    highlight_interval=None,
    legend_interval_label=None,
    highlight_point=None,
    legend_point_label=None
):
    """
    Generic PR plotting utility.
    - series: list of dicts with keys:
        { 'recall': np.ndarray, 'precision': np.ndarray, 'thresholds': np.ndarray, 'label': str, 'color': optional }
    - f1_at: scalar threshold value at which to append F1 to legend labels (if present in thresholds)
    - highlight_interval: (low, high) to draw a thick segment of the PR curve over that thresholds range
    - highlight_point: scalar threshold to mark as a single point on the PR curve
    """
    colors = plt.cm.tab10.colors

    for i, s in enumerate(series):
        rec = np.asarray(s['recall'])
        prec = np.asarray(s['precision'])
        thr = np.asarray(s['thresholds'])
        color = s.get('color', colors[i % len(colors)])

        # Label with F1@value if requested
        label = s['label']
        if f1_at is not None and np.any(np.isclose(thr, f1_at)):
            idx = np.where(np.isclose(thr, f1_at))[0][0]
            p, r = prec[idx], rec[idx]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            # Compact threshold formatting without trailing zeros
            label = f"{label} (F1@{_fmt_thr(f1_at)}={f1:.2f})"

        ax.plot(rec, prec, label=label, color=color)

        # Highlight continuous interval
        if highlight_interval is not None:
            low, high = highlight_interval
            mask = (thr >= low) & (thr <= high)
            if mask.sum() > 1:
                ax.plot(rec[mask], prec[mask], color=color, linewidth=8, alpha=0.6, solid_capstyle='butt')

        # Highlight a single point (e.g., distance=0.005)
        if highlight_point is not None and np.any(np.isclose(thr, highlight_point)):
            idx = np.where(np.isclose(thr, highlight_point))[0][0]
            ax.plot(
                rec[idx], prec[idx],
                marker='o', markersize=8, color=color,
                alpha=0.8, markeredgecolor='black', markeredgewidth=1.5
            )

    # Legend entries for highlights
    if highlight_interval is not None and legend_interval_label:
        ax.plot([], [], color='gray', linewidth=8, alpha=0.3, label=legend_interval_label, solid_capstyle='butt')
    if highlight_point is not None and legend_point_label:
        ax.plot([], [], marker='o', markersize=8, color='gray', alpha=0.3,
                markeredgecolor='black', markeredgewidth=1.5, linestyle='None', label=legend_point_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc=legend_loc)

def plot_keypoint_pr_curve(ax, all_keypoint_pr_data, categories, dataset_names, keypoint_score_threshold, keypoint_distance_threshold):
    """
    Plot Keypoint Precision-Recall curves on the given axis.
    Legend shows F1 at fixed keypoint_score_threshold.
    Highlights the threshold interval [0.2, 0.8].
    """
    series = []
    for idx, category in enumerate(categories):
        if category not in all_keypoint_pr_data:
            continue
        precisions, recalls, score_thresholds = all_keypoint_pr_data[category]
        series.append({
            'recall': np.asarray(recalls),
            'precision': np.asarray(precisions),
            'thresholds': np.asarray(score_thresholds),
            'label': dataset_names[idx],
        })

    plot_pr_series(
        ax,
        series=series,
        title=f'Keypoint Precision-Recall Curve\n(Score Threshold in [0, 1], Distance Threshold: {keypoint_distance_threshold})',
        f1_at=keypoint_score_threshold,
        highlight_interval=(0.2, 0.8),
        legend_interval_label='Score interval [0.2, 0.8]'
    )

def plot_relation_pr_curve(ax, all_relation_pr_data, categories, dataset_names, keypoint_score_threshold, keypoint_distance_threshold):
    """
    Plot Relation Precision-Recall curves averaged over relation types per category.
    Legend shows F1 at fixed keypoint_score_threshold.
    Highlights interval [0.2, 0.8].
    """
    series = []
    for idx, category in enumerate(categories):
        if category not in all_relation_pr_data:
            continue
        relation_curves = all_relation_pr_data[category]
        if not relation_curves:
            continue

        thresholds = relation_curves[list(relation_curves.keys())[0]]['thresholds']
        thresholds_arr = np.asarray(thresholds)

        avg_precision = []
        avg_recall = []
        for t_idx in range(len(thresholds_arr)):
            precisions = [curves['precision'][t_idx] for curves in relation_curves.values()]
            recalls = [curves['recall'][t_idx] for curves in relation_curves.values()]
            avg_precision.append(np.mean(precisions))
            avg_recall.append(np.mean(recalls))

        series.append({
            'recall': np.asarray(avg_recall),
            'precision': np.asarray(avg_precision),
            'thresholds': thresholds_arr,
            'label': dataset_names[idx],
        })

    plot_pr_series(
        ax,
        series=series,
        title=f'Relation Precision-Recall Curve\n(Score Threshold: {keypoint_score_threshold}, Distance Threshold: {keypoint_distance_threshold})',
        f1_at=keypoint_score_threshold,
        highlight_interval=(0.2, 0.8),
        legend_interval_label='Score interval [0.2, 0.8]'
    )

def plot_keypoint_pr_distance_curve(ax, all_keypoint_pr_dist_data, categories, dataset_names, keypoint_score_threshold):
    """
    Plot Keypoint Precision-Recall curves by varying distance thresholds on the given axis.
    Highlights only the single threshold 0.005 and reports F1@0.005.
    """
    f1_dist = 0.005
    series = []
    for idx, category in enumerate(categories):
        if category not in all_keypoint_pr_dist_data:
            continue
        precisions, recalls, distance_thresholds = all_keypoint_pr_dist_data[category]
        series.append({
            'recall': np.asarray(recalls),
            'precision': np.asarray(precisions),
            'thresholds': np.asarray(distance_thresholds),
            'label': dataset_names[idx],
        })

    plot_pr_series(
        ax,
        series=series,
        title=f'Keypoint Precision-Recall Curve\n(Distance Threshold in [0, 1], Score Threshold: {keypoint_score_threshold})',
        f1_at=f1_dist,
        highlight_point=f1_dist,
        legend_point_label='Distance Threshold 0.005'
    )

def plot_model_metrics(
    all_keypoint_pr_data, all_relation_pr_data, all_keypoint_pr_dist_data,
    categories, model_type, plots_dir, keypoint_score_threshold, keypoint_distance_threshold
):
    """
    Plot metrics for a specific model type in a single figure with 3 subplots.
    """
    print(f"Plotting metrics for model type: {model_type}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    dataset_names = [f'Dataset {i+1}' for i in range(len(categories))]

    # Plot 1: Keypoint Precision-Recall curves (by score)
    plot_keypoint_pr_curve(axes[0], all_keypoint_pr_data, categories, dataset_names, 
                           keypoint_score_threshold, keypoint_distance_threshold)

    # Plot 2: Keypoint Precision-Recall by distance threshold
    plot_keypoint_pr_distance_curve(axes[1], all_keypoint_pr_dist_data, categories, dataset_names,
                                   keypoint_score_threshold)

    # Plot 3: Relation Precision-Recall curves
    plot_relation_pr_curve(axes[2], all_relation_pr_data, categories, dataset_names, 
                           keypoint_score_threshold, keypoint_distance_threshold)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_type}_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_model_comparison(
    all_model_relation_pr_data,
    all_model_keypoint_pr_data,
    all_model_keypoint_pr_dist_data,
    categories,
    plots_dir,
    keypoint_score_threshold,
    keypoint_distance_threshold,
    model_types=None,
    legend_name_map=None
):
    """
    Create plots comparing different model types by aggregating data across datasets.
    Uses generalized PR plotting to avoid duplication.
    """
    print(f"Plotting model comparison...")

    if model_types is None:
        model_types = ['individual', 'shared', 'finetuned']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Helper: average keypoint PR (score) across categories for a model
    def avg_keypoint_score(model_data):
        all_prec, all_rec, thresholds = [], [], None
        for category in categories:
            if category in model_data:
                p, r, th = model_data[category]
                if thresholds is None:
                    thresholds = th
                all_prec.append(np.asarray(p))
                all_rec.append(np.asarray(r))
        if not all_prec or thresholds is None:
            return None
        return np.mean(np.vstack(all_prec), axis=0), np.mean(np.vstack(all_rec), axis=0), np.asarray(thresholds)

    # Helper: average keypoint PR (distance) across categories for a model
    def avg_keypoint_distance(model_data):
        all_prec, all_rec, thresholds = [], [], None
        for category in categories:
            if category in model_data:
                p, r, th = model_data[category]
                if thresholds is None:
                    thresholds = th
                all_prec.append(np.asarray(p))
                all_rec.append(np.asarray(r))
        if not all_prec or thresholds is None:
            return None
        return np.mean(np.vstack(all_prec), axis=0), np.mean(np.vstack(all_rec), axis=0), np.asarray(thresholds)

    # Helper: average relation PR across relations per category, then across categories
    def avg_relation(model_rel_data):
        cat_curves = []
        thresholds = None
        for category in categories:
            rel_dict = model_rel_data.get(category)
            if not rel_dict:
                continue
            if thresholds is None:
                first_key = list(rel_dict.keys())[0]
                thresholds = rel_dict[first_key]['thresholds']
            th_arr = np.asarray(thresholds)
            cat_prec, cat_rec = [], []
            for t_idx in range(len(th_arr)):
                precisions = [curves['precision'][t_idx] for curves in rel_dict.values()]
                recalls = [curves['recall'][t_idx] for curves in rel_dict.values()]
                cat_prec.append(np.mean(precisions))
                cat_rec.append(np.mean(recalls))
            cat_curves.append((np.asarray(cat_prec), np.asarray(cat_rec)))
        if not cat_curves or thresholds is None:
            return None
        prec_stack = np.vstack([c[0] for c in cat_curves])
        rec_stack = np.vstack([c[1] for c in cat_curves])
        return np.mean(prec_stack, axis=0), np.mean(rec_stack, axis=0), np.asarray(thresholds)

    # 1) Keypoint PR (score)
    series = []
    for model_type in model_types:
        if model_type not in all_model_keypoint_pr_data:
            continue
        avg = avg_keypoint_score(all_model_keypoint_pr_data[model_type])
        if avg is None:
            continue
        p, r, th = avg
        disp = legend_name_map.get(model_type, model_type.capitalize()) if legend_name_map else model_type.capitalize()
        series.append({'recall': r, 'precision': p, 'thresholds': th, 'label': disp})
    plot_pr_series(
        axes[0],
        series=series,
        title=f'Keypoint Precision-Recall Curve\n(Score Threshold in [0, 1], Distance Threshold: {keypoint_distance_threshold})',
        f1_at=keypoint_score_threshold,
        highlight_interval=(0.2, 0.8),
        legend_interval_label='Score interval [0.2, 0.8]'
    )

    # 2) Keypoint PR (distance)
    f1_dist = 0.005
    series = []
    for model_type in model_types:
        if model_type not in all_model_keypoint_pr_dist_data:
            continue
        avg = avg_keypoint_distance(all_model_keypoint_pr_dist_data[model_type])
        if avg is None:
            continue
        p, r, th = avg
        disp = legend_name_map.get(model_type, model_type.capitalize()) if legend_name_map else model_type.capitalize()
        series.append({'recall': r, 'precision': p, 'thresholds': th, 'label': f'{disp}'})
    plot_pr_series(
        axes[1],
        series=series,
        title=f'Keypoint Precision-Recall Curve\n(Distance Threshold in [0, 1], Score Threshold: {keypoint_score_threshold})',
        f1_at=f1_dist,
        highlight_point=f1_dist,
        legend_point_label='Distance Threshold 0.005'
    )

    # 3) Relation PR (score)
    series = []
    for model_type in model_types:
        if model_type not in all_model_relation_pr_data:
            continue
        avg = avg_relation(all_model_relation_pr_data[model_type])
        if avg is None:
            continue
        p, r, th = avg
        disp = legend_name_map.get(model_type, model_type.capitalize()) if legend_name_map else model_type.capitalize()
        series.append({'recall': r, 'precision': p, 'thresholds': th, 'label': disp})
    plot_pr_series(
        axes[2],
        series=series,
        title=f'Relation Precision-Recall Curve\n(Score Threshold: {keypoint_score_threshold}, Distance Threshold: {keypoint_distance_threshold})',
        f1_at=keypoint_score_threshold,
        highlight_interval=(0.2, 0.8),
        legend_interval_label='Score interval [0.2, 0.8]'
    )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def main():
    if len(sys.argv) > 1:
        work_dir_base = sys.argv[1]
    else:
        print("Usage: python psd_train_test_eval.py <path_to_test_results_dir>")
        return

    # Ensure the path exists
    if not os.path.exists(work_dir_base):
        print(f"The specified path does not exist: {work_dir_base}")
        return

    print(f"Using directory: {work_dir_base}")

    # Discover categories
    categories = discover_categories(work_dir_base)
    print(f"Discovered categories: {', '.join(categories)}")

    # Define model types
    model_types = ['individual', 'shared', 'finetuned']
    
    # Compute all metrics for each model type
    all_model_keypoint_pr_data = {}
    all_model_relation_pr_data = {}
    all_model_keypoint_pr_dist_data = {}

    for model_type in model_types:
        print(f"\nProcessing {model_type} model...")
        (
            keypoint_pr_data,
            relation_pr_data,
            keypoint_pr_dist_data,
            keypoint_score_threshold,
            keypoint_distance_threshold
        ) = compute_all_metrics(
            work_dir_base, categories, model_type
        )
        
        # Store data for later comparison
        all_model_keypoint_pr_data[model_type] = keypoint_pr_data
        all_model_relation_pr_data[model_type] = relation_pr_data
        all_model_keypoint_pr_dist_data[model_type] = keypoint_pr_dist_data

        # Plot model-specific metrics
        plot_model_metrics(
            keypoint_pr_data, relation_pr_data, keypoint_pr_dist_data,
            categories, model_type, work_dir_base,
            keypoint_score_threshold, keypoint_distance_threshold
        )
    
    # Create aggregated model comparison plot
    plot_model_comparison(
        all_model_relation_pr_data, all_model_keypoint_pr_data, all_model_keypoint_pr_dist_data,
        categories, work_dir_base, keypoint_score_threshold, keypoint_distance_threshold
    )

    print(f"\nAll visualizations saved to: {work_dir_base}")

if __name__ == "__main__":
    main()