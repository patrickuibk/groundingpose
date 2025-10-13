import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from collections import defaultdict
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.graph_matching import sequential_matching
from datasets.metrics.utils import compute_pck
import concurrent.futures
from functools import partial
from tqdm import tqdm  
import itertools

# Update pipelines to match algorithm names expected by sequential_matching
# All possible intermediate algorithms (excluding 'la' and 'our2opt')
intermediate_algorithms = ['faq', 'sm', 'rrwm', 'ipfp']

# Generate all partial permutations of length 3 of the intermediate algorithms
partial_permutations = list(itertools.permutations(intermediate_algorithms, 1))

# Build pipelines: always start with 'la', end with 'our2opt'
METHOD_PIPELINES = []
for perm in partial_permutations:
    pipeline = ['la'] + list(perm) + ['our2opt']
    METHOD_PIPELINES.append(pipeline)

# Also include the baseline 'la' only
METHOD_PIPELINES.insert(0, ['la'])
# Keep the visualization labels the same
PIPELINE_LABELS = ['+ '.join([method.upper().replace('OUR2OPT', 'Our 2OPT') for method in pipeline]) for pipeline in METHOD_PIPELINES]

# Helper: consistent label rendering for methods and pipeline prefixes (used for intermediates too)
def _method_to_label(m):
    # Render like existing labels: uppercase, but map OUR2OPT -> 'Our 2OPT'
    up = m.upper()
    return up.replace('OUR2OPT', 'Our 2OPT')

def _pipeline_to_label(pipeline):
    return '+ '.join(_method_to_label(m) for m in pipeline)

# Ordered list of all prefix labels across all pipelines (e.g., 'LA', 'LA+ FAQ', ...)
PREFIX_LABELS_ORDER = []
_seen = set()
for pipe in METHOD_PIPELINES:
    for i in range(1, len(pipe) + 1):
        lbl = _pipeline_to_label(pipe[:i])
        if lbl not in _seen:
            PREFIX_LABELS_ORDER.append(lbl)
            _seen.add(lbl)

def extract_pck_from_metrics(metrics_file, threshold="0.05"):
    """Extract AVG/PCK@thresh from metrics.json."""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    key = f"AVG/PCK@{threshold}"
    return float(metrics[key]) if key in metrics else None

def add_noise(coords, noise_level, relation_matrices=None):
    coords += np.random.normal(loc=0.0, scale=noise_level, size=coords.shape)
    return coords


def plot_category_results(category, results, noise_levels, output_dir):
    """Plot PCK vs noise level for a single category with multiple pipelines and their intermediate prefixes."""
    # Prepare data for plotting from whatever method labels are present (final and intermediate)
    df_rows = []
    # Only consider method labels (exclude any metadata like 'greedy_pck', 'gt_pck')
    present_method_labels = [k for k in results.keys() if isinstance(results[k], dict) and all(nl in results[k] for nl in noise_levels)]
    # Preserve a stable plotting order based on PREFIX_LABELS_ORDER then any extras
    ordered_labels = [l for l in PREFIX_LABELS_ORDER if l in present_method_labels]
    ordered_labels += [l for l in present_method_labels if l not in ordered_labels]
    for label in ordered_labels:
        for noise_level in noise_levels:
            pcks = results.get(label, {}).get(noise_level, [])
            for pck in pcks:
                df_rows.append({
                    'Noise Level': noise_level,
                    'PCK': pck,
                    'Method': label,
                })
    df = pd.DataFrame(df_rows)
    if df.empty:
        return
    plt.figure(figsize=(6, 6))
    la_label = _pipeline_to_label(['la'])
    for label in ordered_labels:
        method_df = df[df['Method'] == label]
        linestyle = '--' if label == la_label else '-'
        sns.lineplot(
            data=method_df,
            x='Noise Level',
            y='PCK',
            label=label,
            linestyle=linestyle,
            errorbar=None
        )
    plt.xlabel('Noise Level')
    plt.ylabel('PCK')
    plt.title(f'PCK vs Noise Level - {category}')
    plt.legend(bbox_to_anchor=(0.05, 0.05), loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'pck_vs_noise_{category}.png'), dpi=300)
    plt.ylim(0, 1.05)
    plt.close()

def plot_best_method_vs_noise(category, results, noise_levels, output_dir, pck_threshold, categories=None):
    """
    Plot PCK vs noise level for the best-performing final method (highest average PCK across all noise levels),
    plus linear assignment (dashed), and all prefixes of the best pipeline (except linear assignment alone).
    """
    # Compute average PCK for each final pipeline label
    avg_pck_per_label = {}
    for label in PIPELINE_LABELS:
        all_pcks = []
        for noise_level in noise_levels:
            pcks = results.get(label, {}).get(noise_level, [])
            all_pcks.extend(pcks)
        if all_pcks:
            avg_pck_per_label[label] = np.mean(all_pcks)
    if not avg_pck_per_label:
        return

    # Find best final label (highest average PCK) and its pipeline
    best_final_label = max(avg_pck_per_label, key=avg_pck_per_label.get)
    best_idx = PIPELINE_LABELS.index(best_final_label)
    best_pipeline = METHOD_PIPELINES[best_idx]
    best_label = _pipeline_to_label(best_pipeline)

    # Build all prefixes of the best pipeline (exclude LA baseline and exclude the full label here; it's plotted separately)
    la_label = _pipeline_to_label(['la'])
    prefix_labels = []
    for i in range(2, len(best_pipeline)):  # start at length 2 to skip 'LA' baseline
        lbl = _pipeline_to_label(best_pipeline[:i])
        prefix_labels.append(lbl)

    # Plot
    plt.figure(figsize=(6, 6))
    x = noise_levels

    # Plot linear assignment baseline (dashed)
    la_means, la_stds = [], []
    for noise_level in noise_levels:
        la_pcks = results.get(la_label, {}).get(noise_level, [])
        la_means.append(np.mean(la_pcks) if la_pcks else np.nan)
        la_stds.append(np.std(la_pcks) if la_pcks else 0)
    plt.errorbar(x, la_means, yerr=la_stds, label=la_label, fmt='--o', capsize=5)

    # Plot prefixes of best pipeline (solid lines)
    for lbl in prefix_labels:
        means, stds = [], []
        for noise_level in noise_levels:
            pcks = results.get(lbl, {}).get(noise_level, [])
            means.append(np.mean(pcks) if pcks else np.nan)
            stds.append(np.std(pcks) if pcks else 0)
        if any(np.isfinite(means)):
            plt.errorbar(x, means, yerr=stds, label=lbl, fmt='-o', capsize=5)

    # Plot best pipeline (final) with thicker line
    best_means, best_stds = [], []
    for noise_level in noise_levels:
        best_pcks = results.get(best_label, {}).get(noise_level, [])
        best_means.append(np.mean(best_pcks) if best_pcks else np.nan)
        best_stds.append(np.std(best_pcks) if best_pcks else 0)
    plt.errorbar(x, best_means, yerr=best_stds, label=best_label, fmt='-o', capsize=5, linewidth=3)

    plt.xlabel('Standard Deviation of Expected Keypoint Coordinates')
    plt.ylabel(f'PCK@{pck_threshold}')
    plt.ylim(0, 1.05)

    dataset_idx = categories.index(category) + 1
    plt.title(f'Dataset {dataset_idx}: PCK@{pck_threshold} vs Noise')
    plt.legend(bbox_to_anchor=(0.05, 0.05), loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'pck_vs_noise_{category}_best.png'), dpi=300)
    plt.close()

def plot_summary_results(results, categories, noise_levels, output_dir, gt_pck_per_category=None):
    """Plot summary of PCK vs noise level for all completed categories using pipelines."""
    if not categories:
        return
    df_rows = []
    for category in categories:
        for noise_level in noise_levels:
            for label in PIPELINE_LABELS:
                pcks = results[category].get(label, {}).get(noise_level, [])
                for pck in pcks:
                    df_rows.append({
                        'Category': category,
                        'Noise Level': noise_level,
                        'Method': label,
                        'PCK': pck
                    })
    df = pd.DataFrame(df_rows)
    if len(df) == 0:
        return
    plt.figure(figsize=(14, 8))
    avg_by_method = df.groupby(['Method', 'Noise Level'])['PCK'].mean().reset_index()
    sns.lineplot(
        data=avg_by_method,
        x='Noise Level',
        y='PCK',
        hue='Method',
        markers=True,
        dashes=False
    )
    
    plt.title('Average PCK by Method Across All Categories')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_pck_by_pipeline.png'), dpi=300)
    plt.close()

def process_image_file(pred_file, category, noise_levels, num_repeats, pck_threshold, score_threshold, method_pipelines=None):
    """Process a single image file across all noise levels and repeats and collect PCK for all pipeline prefixes."""
    base_filename = os.path.basename(pred_file).replace("_pred.json", "")
    
    # Load prediction and ground truth data
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    gt_file = pred_file.replace("_pred.json", "_gt.json")                            
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    img_width = gt_data['img_width']
    img_height = gt_data['img_height']
    
    pred_coords = np.array(pred_data['keypoint_coords'])
    pred_labels = np.array(pred_data['keypoint_label_names'])
    pred_scores = np.array(pred_data['keypoint_scores'])
    pred_relation_matrices = np.array(pred_data['keypoint_relation_scores'])
    
    gt_coords = np.array(gt_data['keypoint_coords'])
    gt_labels = np.array(gt_data['keypoint_label_names'])
    gt_relation_matrices = np.array(gt_data['relation_matrices'])
    
    # Normalize coordinates
    pred_coords_norm = pred_coords / np.array([img_width, img_height])
    gt_coords_norm = gt_coords / np.array([img_width, img_height])

    # Filter predicted keypoints by score threshold and prune relation matrices
    keep_idx = np.where(pred_scores >= score_threshold)[0]
    if keep_idx.size < pred_scores.size:
        # apply filtering
        pred_coords = pred_coords[keep_idx]
        pred_coords_norm = pred_coords_norm[keep_idx]
        pred_labels = pred_labels[keep_idx]
        pred_scores = pred_scores[keep_idx]
        # prune relation matrices if present and 3D
        if pred_relation_matrices is not None and pred_relation_matrices.ndim == 3 and pred_relation_matrices.size > 0:
            R = pred_relation_matrices.shape[2]
            pred_relation_matrices = pred_relation_matrices[np.ix_(keep_idx, keep_idx, np.arange(R))]
        else:
            # empty relations -> create empty array with consistent dims
            pred_relation_matrices = np.zeros((pred_coords.shape[0], pred_coords.shape[0], 0))
    # else keep arrays as-is

    n_gt = len(gt_labels)
    # Initialize result container with all prefix labels
    method_pcks = {label: {nl: [] for nl in noise_levels} for label in PREFIX_LABELS_ORDER}
    results = {
        'category': category,
        'base_filename': base_filename,
        'method_pcks': method_pcks
    }
    
    all_label_names = sorted(set(gt_labels) | set(pred_labels))
    label_name_to_int = {name: idx for idx, name in enumerate(all_label_names)}
    gt_label_ids = np.array([label_name_to_int[name] for name in gt_labels])
    pred_label_ids = np.array([label_name_to_int[name] for name in pred_labels])
    
    # Process each noise level and repeat
    for noise_level in noise_levels:
        for r in range(num_repeats):
            # Add noise to ground truth coordinates
            noisy_gt_coords_norm = add_noise(gt_coords_norm.copy(), noise_level, gt_relation_matrices)

            # If no predicted keypoints remain after thresholding, append zeros for all labels once per repeat
            if pred_label_ids.size == 0:
                for label in PREFIX_LABELS_ORDER:
                    results['method_pcks'][label][noise_level].append(0.0)
                continue

            # Track labels already added this (noise_level, repeat) to avoid double-counting common prefixes (e.g., "LA")
            added_labels = set()

            # Run each pipeline and collect intermediate results
            pipelines = METHOD_PIPELINES if method_pipelines is None else method_pipelines
            for pipeline in pipelines:
                # Run sequential_matching with intermediates
                try:
                    best_matching, intermediates = sequential_matching(
                        gt_label_ids,
                        gt_relation_matrices,
                        pred_label_ids,
                        pred_scores,
                        pred_relation_matrices,
                        algorithm_sequence=pipeline,
                        initial_matching=None,
                        gt_coords=noisy_gt_coords_norm,
                        pred_coords=pred_coords_norm,
                        return_intermediate=True
                    )
                except Exception as e:
                    print(f"Error in pipeline {_pipeline_to_label(pipeline)} for {base_filename}: {e}")
                    continue

                # Build cumulative prefix labels and record PCK for each step (skip 'init')
                prefix = []
                for step_name, matching, _score in intermediates:
                    if step_name == 'init':
                        continue
                    prefix.append(step_name)
                    label = _pipeline_to_label(prefix)
                    # Deduplicate label within this file/noise/repeat
                    if label in added_labels:
                        continue
                    added_labels.add(label)
                    file_pck = compute_pck(
                        gt_coords, pred_coords, matching, 
                        img_width, img_height, pck_threshold=pck_threshold
                    )
                    # Ensure label initialized (in case it's from a prefix not in PREFIX_LABELS_ORDER)
                    if label not in results['method_pcks']:
                        results['method_pcks'][label] = {nl: [] for nl in noise_levels}
                    results['method_pcks'][label][noise_level].append(file_pck)
    
    return results

class GraphMatchingAnalysis:
    def __init__(self, results_dir, output_dir, noise_levels=None, num_repeats=5, pck_threshold=0.05, 
                 num_processes=None, score_threshold=0.3, categories=None, la_only=False):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.noise_levels = noise_levels
        self.num_repeats = num_repeats
        self.pck_threshold = pck_threshold
        self.results = defaultdict(lambda: defaultdict(dict))
        self.num_processes = num_processes  # Number of processes to use (None = auto)
        self.gt_pck_per_category = {}  # Store GT matching PCK for each category
        self.completed_categories = set()
        self.score_threshold = score_threshold
        self.categories = categories  # If provided, process only these categories
        self.la_only = la_only

    def run(self):
        # Determine categories to process
        if self.categories is None:
            categories = [d for d in os.listdir(self.results_dir) if os.path.isdir(os.path.join(self.results_dir, d))]
        else:
            # preserve order of requested categories, but filter out non-existing ones
            requested = list(self.categories)
            existing = [d for d in requested if os.path.isdir(os.path.join(self.results_dir, d))]
            missing = [d for d in requested if d not in existing]
            if missing:
                print(f"Warning: These requested categories do not exist in {self.results_dir}: {missing}")
            categories = existing
        print(f"Found categories: {categories}")
        
        # Get greedy PCK values from metrics.json files
        for category in categories:
            metrics_file = os.path.join(self.results_dir, category, "metrics.json")
            if os.path.exists(metrics_file):
                greedy_pck = extract_pck_from_metrics(metrics_file, str(self.pck_threshold))
                if greedy_pck is not None:
                    self.results[category]['greedy_pck'] = greedy_pck
                gt_pck = extract_pck_from_metrics(metrics_file, str(self.pck_threshold))
                if gt_pck is not None:
                    self.results[category]['gt_pck'] = gt_pck
                    self.gt_pck_per_category[category] = gt_pck

        # Process each category sequentially, but images in parallel
        for category in categories:
            print(f"Processing category: {category}")
            category_dir = os.path.join(self.results_dir, category)
            pred_files = glob.glob(os.path.join(category_dir, "*_pred.json"))
            print(f"  Found {len(pred_files)} images.")

            worker_fn = partial(
                process_image_file,
                category=category,
                noise_levels=self.noise_levels,
                num_repeats=self.num_repeats,
                pck_threshold=self.pck_threshold,
                score_threshold=self.score_threshold,
                method_pipelines=( [['la']] if self.la_only else METHOD_PIPELINES )
            )

            # Collect results for this category
            method_pcks = defaultdict(lambda: defaultdict(list))

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                future_to_file = {
                    executor.submit(worker_fn, pred_file): pred_file
                    for pred_file in pred_files
                }
                with tqdm(total=len(pred_files), desc=f"Processing {category}", unit="file") as pbar:
                    for future in concurrent.futures.as_completed(future_to_file):
                        pred_file = future_to_file[future]
                        try:
                            result = future.result()
                            # Store PCK results, including intermediate labels
                            for method, noise_levels_data in result['method_pcks'].items():
                                for noise_level, pcks in noise_levels_data.items():
                                    method_pcks[method][noise_level].extend(pcks)
                        except Exception as e:
                            print(f"Error processing file {pred_file}: {e}")
                        pbar.update(1)

            # Aggregate mean PCKs (per noise) across files for this category
            # Keep the existing chunked averaging behavior
            for label, noise_levels_data in method_pcks.items():
                for noise_level, pcks in noise_levels_data.items():
                    if pcks:
                        chunk_size = len(pred_files)
                        mean_pcks = []
                        for i in range(0, len(pcks), chunk_size):
                            chunk = pcks[i:i+chunk_size]
                            if chunk:
                                mean_pcks.append(np.mean(chunk))
                        self.results[category][label][noise_level] = mean_pcks

            self.completed_categories.add(category)
            print(f"Category {category} completed. Generating plots...")
            plot_category_results(category, self.results[category], self.noise_levels, self.output_dir)
            plot_best_method_vs_noise(category, self.results[category], self.noise_levels, self.output_dir, self.pck_threshold, categories=categories)

        return self.results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze graph matching with varying noise levels')
    parser.add_argument('test_results_dir', help='Directory containing test_results with subfolders shared/individual/finetuned')
    parser.add_argument('--subfolder', type=str, default='shared', choices=['shared', 'individual', 'finetuned'],
                        help='Which subfolder in test_results to use')
    parser.add_argument('--noise_levels', type=float, nargs='+', 
                        default=[0.005, 0.01, 0.02, 0.03],
                        help='Noise levels to test')
    parser.add_argument('--num_repeats', type=int, default=10,
                        help='Number of repetitions per noise level')
    parser.add_argument('--pck_threshold', type=float, default=0.005,
                        help='PCK threshold')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of processes to use (default: auto)')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='Keypoint score threshold to filter predictions')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Optional list of categories to process (default: all subfolders in results_dir)')
    parser.add_argument('--la_only', action='store_true',
                        help='If set, run only Linear Assignment (LA) instead of all pipelines')

    args = parser.parse_args()

    # Compose the results_dir path
    results_dir = os.path.join(args.test_results_dir, args.subfolder)
    output_dir = os.path.join(args.test_results_dir, 'graph_matching_plots', args.subfolder)
    os.makedirs(output_dir, exist_ok=True)

    analysis = GraphMatchingAnalysis(
        results_dir=results_dir,
        output_dir=output_dir,
        noise_levels=args.noise_levels,
        num_repeats=args.num_repeats,
        pck_threshold=args.pck_threshold,
        num_processes=args.num_processes,
        score_threshold=args.score_threshold,
        categories=args.categories,
        la_only=args.la_only
    )
    analysis.run()