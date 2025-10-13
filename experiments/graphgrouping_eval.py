import os
import sys
import json
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.metrics.utils import (
    compute_pck
)
from tools.graph_grouping import (
    InstanceGroup,
    group_keypoints_into_instances,
    make_merge_fn_max_label,
    make_merge_fn_max_degree_per_label,
)
from tools.graph_matching import (
    sequential_matching,
    linear_assignment_matching,
)
# Reuse pipelines, labels, plotting and noise helper from the node-level eval
from experiments.graphmatching_eval import (
    METHOD_PIPELINES,
    PREFIX_LABELS_ORDER,
    PIPELINE_LABELS,
    _pipeline_to_label,
    plot_category_results,
    add_noise,
)

# -----------------------------------------------------------------------------------
# Grouping helpers

def _choose_merge_fn(result_dir_name: str, label_name_to_int: Dict[str, int], relation_scores: np.ndarray):
    """
    Mirror the category-specific merge function selection used in graphgrouping_vis.py.
    Returns (merge_fn, merge_fn_name).
    """
    folder_name = os.path.basename(result_dir_name).lower()
    merge_fn = make_merge_fn_max_label(max_per_label=999999, relation_scores=relation_scores)
    name = "make_merge_fn_max_label(default)"

    if "shuttering" in folder_name:
        merge_fn = make_merge_fn_max_label(max_per_label=2, relation_scores=relation_scores)
        name = "make_merge_fn_max_label(2)"
    elif "girders" in folder_name:
        merge_fn = make_merge_fn_max_degree_per_label(
            max_degree_per_label={
                label_name_to_int['lattice girder top bar welding point']: 2,
                label_name_to_int['lattice girder top bar end']: 1
            },
            relation_scores=relation_scores
        )
        name = "make_merge_fn_max_degree_per_label(girders)"
    elif "bars" in folder_name:
        merge_fn = make_merge_fn_max_degree_per_label(
            max_degree_per_label={
                label_name_to_int['lattice bars welding point']: 4,
                label_name_to_int['lattice bars end']: 1
            },
            relation_scores=relation_scores
        )
        name = "make_merge_fn_max_degree_per_label(bars)"
    elif "leitern" in folder_name:
        merge_fn = make_merge_fn_max_degree_per_label(
            max_degree_per_label={
                label_name_to_int['welding point']: 3,
            },
            relation_scores=relation_scores
        )
        name = "make_merge_fn_max_degree_per_label(leitern)"
    elif "tunnelmaster" in folder_name:
        merge_fn = make_merge_fn_max_degree_per_label(
            max_degree_per_label={
                label_name_to_int['welding point']: 4,
                label_name_to_int['end of bar']: 1,
            },
            relation_scores=relation_scores
        )
        name = "make_merge_fn_max_degree_per_label(tunnelmaster)"
    return merge_fn, name

def _group_gt_components(gt_labels: np.ndarray, gt_scores: Optional[np.ndarray], gt_relation_matrices: np.ndarray) -> List[InstanceGroup]:
    """
    Group GT nodes into connected components where any relation > 0 indicates an edge.
    """
    N = gt_relation_matrices.shape[0]
    if N == 0:
        return []
    # adjacency where any relation type > 0, excluding self
    any_rel = np.any(gt_relation_matrices > 0, axis=2)
    np.fill_diagonal(any_rel, False)
    visited = np.zeros(N, dtype=bool)
    groups: List[InstanceGroup] = []
    for i in range(N):
        if visited[i]:
            continue
        # BFS to collect connected component
        comp = []
        queue = [i]
        visited[i] = True
        while queue:
            u = queue.pop()
            comp.append(u)
            nbrs = np.where(any_rel[u])[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        comp = np.array(comp, dtype=np.int32)
        comp.sort()
        K = comp.size
        R = gt_relation_matrices.shape[2]
        adj = gt_relation_matrices[np.ix_(comp, comp, np.arange(R))]
        scores = np.ones(K, dtype=np.float32) if gt_scores is None else gt_scores[comp]
        groups.append(InstanceGroup(
            node_ids=comp,
            keypoint_labels=gt_labels[comp],
            keypoint_scores=scores,
            adjacency_matrix=adj
        ))
    return groups

def _compute_group_centroids(groups: List[InstanceGroup], coords: np.ndarray) -> np.ndarray:
    """
    Compute centroid per group using provided coordinates (e.g., normalized coords).
    """
    if not groups:
        return np.zeros((0, 2), dtype=np.float32)
    cents = []
    for g in groups:
        pts = coords[g.node_ids]
        cents.append(pts.mean(axis=0))
    return np.asarray(cents, dtype=np.float32)

def _match_groups_by_centroid(gt_centroids: np.ndarray, pred_centroids: np.ndarray) -> List[Tuple[int, int]]:
    """
    Assign GT groups to Pred groups via Hungarian on Euclidean centroid distance.
    Returns list of (gt_group_idx, pred_group_idx).
    """
    from scipy.optimize import linear_sum_assignment
    if gt_centroids.size == 0 or pred_centroids.size == 0:
        return []
    # Distance matrix (n_gt x n_pred)
    diff = gt_centroids[:, None, :] - pred_centroids[None, :, :]
    C = np.linalg.norm(diff, axis=2)
    gi, pj = linear_sum_assignment(C)
    return [(int(g), int(p)) for g, p in zip(gi, pj)]

# -----------------------------------------------------------------------------------
# Public: prepare grouping for a single file (used by visualizer)

def prepare_grouping_from_files(
    pred_json_path: str,
    score_thresh: float = 0.3,
    result_dir_name: str = "",
    min_edge_score: float = 0.3,
):
    """
    Load pred/gt JSONs, filter predictions, choose merge_fn per dataset, group predictions into instances.
    Also includes GT arrays for optional overlay.

    Returns a dict with keys:
      img_width, img_height, img_path
      pred_coords, pred_label_names, pred_scores, pred_relation_matrices
      gt_coords, gt_label_names, gt_relation_matrices, relation_names
      groups (List[InstanceGroup])
      merge_fn_name (str)
    """
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    gt_file = pred_json_path.replace("_pred.json", "_gt.json")
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    img_width = gt_data['img_width']
    img_height = gt_data['img_height']
    img_path = gt_data.get('img_path', None)
    relation_names = gt_data.get('relation_names', None)

    pred_coords = np.array(pred_data['keypoint_coords'])
    pred_label_names = np.array(pred_data['keypoint_label_names'])
    pred_scores = np.array(pred_data['keypoint_scores'])
    pred_relation_matrices = np.array(pred_data['keypoint_relation_scores'])

    # Filter by score
    keep = pred_scores >= score_thresh
    pred_coords = pred_coords[keep]
    pred_label_names = pred_label_names[keep]
    pred_scores = pred_scores[keep]
    pred_relation_matrices = pred_relation_matrices[keep][:, keep, :]

    gt_coords = np.array(gt_data['keypoint_coords'])
    gt_label_names = np.array(gt_data['keypoint_label_names'])
    gt_relation_matrices = np.array(gt_data['relation_matrices'])

    # Label mapping for merge-fn selection
    all_label_names = sorted(set(gt_label_names) | set(pred_label_names))
    label_name_to_int = {n: i for i, n in enumerate(all_label_names)}
    pred_label_ids = np.array([label_name_to_int[n] for n in pred_label_names])

    # Choose merge function (mirrors vis)
    merge_fn, merge_fn_name = _choose_merge_fn(result_dir_name, label_name_to_int, pred_relation_matrices)

    # Group predictions
    groups = group_keypoints_into_instances(
        keypoint_labels=pred_label_ids,
        keypoint_scores=pred_scores,
        relation_scores=pred_relation_matrices,
        merge_fn=merge_fn,
        min_edge_score=min_edge_score
    )

    return {
        'img_width': img_width,
        'img_height': img_height,
        'img_path': img_path if img_path and os.path.exists(img_path) else None,
        'relation_names': relation_names,
        'pred_coords': pred_coords,
        'pred_label_names': pred_label_names,
        'pred_scores': pred_scores,
        'pred_relation_matrices': pred_relation_matrices,
        'gt_coords': gt_coords,
        'gt_label_names': gt_label_names,
        'gt_relation_matrices': gt_relation_matrices,
        'groups': groups,
        'merge_fn_name': merge_fn_name,
    }

# -----------------------------------------------------------------------------------
# File-level processing for grouping eval

def _process_image_file_grouping(
    pred_file: str,
    category: str,
    noise_levels: List[float],
    num_repeats: int,
    pck_threshold: float,
    score_threshold: float,
    min_edge_score: float,
    method_pipelines: Optional[List[List[str]]] = None,
):
    """
    Process one image: group preds/GT into graphs, match groups by centroid, then per matched pair
    run sequential matching pipelines and aggregate PCK per pipeline prefix.
    """
    base_filename = os.path.basename(pred_file).replace("_pred.json", "")

    # Load pred/gt data
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    gt_file = pred_file.replace("_pred.json", "_gt.json")
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    img_width = gt_data['img_width']
    img_height = gt_data['img_height']

    pred_coords = np.array(pred_data['keypoint_coords'])
    pred_label_names = np.array(pred_data['keypoint_label_names'])
    pred_scores = np.array(pred_data['keypoint_scores'])
    pred_relation_matrices = np.array(pred_data['keypoint_relation_scores'])

    gt_coords = np.array(gt_data['keypoint_coords'])
    gt_label_names = np.array(gt_data['keypoint_label_names'])
    gt_relation_matrices = np.array(gt_data['relation_matrices'])

    # Normalize coordinates for LA steps and centroid distance
    pred_coords_norm = pred_coords / np.array([img_width, img_height])
    gt_coords_norm = gt_coords / np.array([img_width, img_height])

    # Filter predictions by score threshold and prune relations
    keep_idx = np.where(pred_scores >= score_threshold)[0]
    if keep_idx.size < pred_scores.size:
        pred_coords = pred_coords[keep_idx]
        pred_coords_norm = pred_coords_norm[keep_idx]
        pred_label_names = pred_label_names[keep_idx]
        pred_scores = pred_scores[keep_idx]
        if pred_relation_matrices is not None and pred_relation_matrices.ndim == 3 and pred_relation_matrices.size > 0:
            R = pred_relation_matrices.shape[2]
            pred_relation_matrices = pred_relation_matrices[np.ix_(keep_idx, keep_idx, np.arange(R))]
        else:
            pred_relation_matrices = np.zeros((pred_coords.shape[0], pred_coords.shape[0], 0))

    # Early out if no predictions remain
    method_pcks = {label: {nl: [] for nl in noise_levels} for label in PREFIX_LABELS_ORDER}
    if pred_coords.shape[0] == 0:
        return {
            'category': category,
            'base_filename': base_filename,
            'method_pcks': method_pcks
        }

    # Label mapping to ints
    all_label_names = sorted(set(gt_label_names) | set(pred_label_names))
    label_name_to_int = {name: idx for idx, name in enumerate(all_label_names)}
    gt_label_ids = np.array([label_name_to_int[name] for name in gt_label_names])
    pred_label_ids = np.array([label_name_to_int[name] for name in pred_label_names])

    # Group GT by connectivity
    gt_groups = _group_gt_components(gt_label_ids, None, gt_relation_matrices)

    # Choose and apply merge_fn for predicted grouping (re-using category-based rule)
    merge_fn, _merge_name = _choose_merge_fn(os.path.dirname(pred_file), label_name_to_int, pred_relation_matrices)
    pred_groups = group_keypoints_into_instances(
        keypoint_labels=pred_label_ids,
        keypoint_scores=pred_scores,
        relation_scores=pred_relation_matrices,
        merge_fn=merge_fn,
        min_edge_score=min_edge_score
    )

    # If no groups on either side, still return zeros
    if len(gt_groups) == 0 or len(pred_groups) == 0:
        return {
            'category': category,
            'base_filename': base_filename,
            'method_pcks': method_pcks
        }

    # Process each noise level/repeat
    for noise_level in noise_levels:
        for r in range(num_repeats):
            noisy_gt_coords_norm = add_noise(gt_coords_norm.copy(), noise_level, gt_relation_matrices)

            # --- Baseline: Linear Assignment BEFORE grouping (global) ---
            la_label_without = _pipeline_to_label(['la']) + " (before grouping)"
            # ensure method key exists for the new label
            if la_label_without not in method_pcks:
                method_pcks[la_label_without] = {nl: [] for nl in noise_levels}
            try:
                la_match_global = linear_assignment_matching(
                    noisy_gt_coords_norm, gt_label_ids,
                    pred_coords_norm, pred_label_ids
                )
                la_pck = compute_pck(
                    gt_coords, pred_coords, la_match_global,
                    img_width, img_height, pck_threshold=pck_threshold
                )
                method_pcks[la_label_without][noise_level].append(la_pck)
            except Exception as e:
                # If LA fails, record 0.0 to keep series aligned
                method_pcks[la_label_without][noise_level].append(0.0)

            # Compute centroids and match groups
            gt_centroids = _compute_group_centroids(gt_groups, noisy_gt_coords_norm)
            pred_centroids = _compute_group_centroids(pred_groups, pred_coords_norm)
            group_pairs = _match_groups_by_centroid(gt_centroids, pred_centroids)  # list of (gi, pj)

            # Track deduplication of labels per file/noise/repeat
            # Seed with the 'without grouping' label so the grouped LA isn't blocked
            added_labels = {la_label_without}

            # For each pipeline, aggregate per-prefix mappings across group pairs
            pipelines = METHOD_PIPELINES if method_pipelines is None else method_pipelines
            for pipeline in pipelines:
                aggregated_mappings_by_label: Dict[str, Dict[int, int]] = {}

                for (gi, pj) in group_pairs:
                    g_gt = gt_groups[gi]
                    g_pr = pred_groups[pj]

                    # Build local arrays for this pair
                    gt_ids = g_gt.node_ids
                    pr_ids = g_pr.node_ids

                    gt_lbls_local = gt_label_ids[gt_ids]
                    pr_lbls_local = pred_label_ids[pr_ids]
                    gt_rel_local = gt_relation_matrices[np.ix_(gt_ids, gt_ids, np.arange(gt_relation_matrices.shape[2]))]
                    pr_rel_local = pred_relation_matrices[np.ix_(pr_ids, pr_ids, np.arange(pred_relation_matrices.shape[2]))]
                    pr_scores_local = pred_scores[pr_ids]
                    # Coords for LA (normalized noisy GT and normalized pred)
                    gt_coords_local = noisy_gt_coords_norm[gt_ids]
                    pr_coords_local = pred_coords_norm[pr_ids]

                    try:
                        _, intermediates = sequential_matching(
                            gt_lbls_local,
                            gt_rel_local,
                            pr_lbls_local,
                            pr_scores_local,
                            pr_rel_local,
                            algorithm_sequence=pipeline,
                            initial_matching=None,
                            gt_coords=gt_coords_local,
                            pred_coords=pr_coords_local,
                            return_intermediate=True
                        )
                    except Exception as e:
                        # Skip this pair on failure
                        continue

                    # Convert local matchings to global and accumulate per-prefix label
                    prefix = []
                    for step_name, matching_local, _score in intermediates:
                        if step_name == 'init':
                            continue
                        prefix.append(step_name)
                        # Default label from pipeline prefix
                        label = _pipeline_to_label(prefix)
                        # If the prefix is exactly LA, mark that it's with grouping
                        if len(prefix) == 1 and prefix[0].lower() == 'la':
                            label = _pipeline_to_label(['la'])
                        # Convert local mapping to global mapping
                        global_map = {}
                        for li, lj in matching_local.items():
                            gi_global = int(gt_ids[int(li)])
                            pj_global = int(pr_ids[int(lj)])
                            global_map[gi_global] = pj_global
                        if label not in aggregated_mappings_by_label:
                            aggregated_mappings_by_label[label] = {}
                        aggregated_mappings_by_label[label].update(global_map)

                # For each produced label for this pipeline, compute PCK and record (avoid duplicates across pipelines)
                for label, mapping in aggregated_mappings_by_label.items():
                    if label in added_labels:
                        continue
                    added_labels.add(label)
                    file_pck = compute_pck(
                        gt_coords, pred_coords, mapping,
                        img_width, img_height, pck_threshold=pck_threshold
                    )
                    if label not in method_pcks:
                        method_pcks[label] = {nl: [] for nl in noise_levels}
                    method_pcks[label][noise_level].append(file_pck)

            # Ensure we add 0.0 for labels that got no entry this round (optional)
            # Not strictly necessary; plotting code tolerates missing series.
    return {
        'category': category,
        'base_filename': base_filename,
        'method_pcks': method_pcks
    }

# -----------------------------------------------------------------------------------
# Grouping-specific best method plot

def plot_best_method_vs_noise_grouping(category, results, noise_levels, output_dir, pck_threshold, categories=None):
    """
    Plot PCK vs noise for LA baselines (with and without grouping) and for the best pipeline (including all its prefixes).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    la_without_label = _pipeline_to_label(['la']) + " (before grouping)"
    la_with_label = _pipeline_to_label(['la'])

    # Compute means and stds for LA baselines
    la_means, la_stds = [], []
    la_g_means, la_g_stds = [], []
    for nl in noise_levels:
        pcks_wo = results.get(la_without_label, {}).get(nl, [])
        la_means.append(np.mean(pcks_wo) if pcks_wo else np.nan)
        la_stds.append(np.std(pcks_wo) if pcks_wo else 0)
        pcks_wg = results.get(la_with_label, {}).get(nl, [])
        la_g_means.append(np.mean(pcks_wg) if pcks_wg else np.nan)
        la_g_stds.append(np.std(pcks_wg) if pcks_wg else 0)

    # Compute average PCK for each final pipeline label
    avg_pck_per_label = {}
    for label in PIPELINE_LABELS:
        all_pcks = []
        for nl in noise_levels:
            all_pcks.extend(results.get(label, {}).get(nl, []))
        if all_pcks:
            avg_pck_per_label[label] = np.mean(all_pcks)

    # Determine best pipeline
    best_final_label = None
    best_pipeline = None
    best_label = None
    if avg_pck_per_label:
        best_final_label = max(avg_pck_per_label, key=avg_pck_per_label.get)
        best_idx = PIPELINE_LABELS.index(best_final_label)
        best_pipeline = METHOD_PIPELINES[best_idx]
        best_label = _pipeline_to_label(best_pipeline)

    # Plot
    plt.figure(figsize=(6, 6))
    x = noise_levels

    # Plot LA without grouping (dashed), get color
    la_color = None
    if any(np.isfinite(la_means)):
        line_la = plt.errorbar(x, la_means, yerr=la_stds, label=la_without_label, fmt='--o', capsize=5)
        la_color = line_la[0].get_color() if hasattr(line_la, '__getitem__') else line_la.get_color()
    # Plot LA with grouping (solid, same color)
    if any(np.isfinite(la_g_means)):
        plt.errorbar(x, la_g_means, yerr=la_g_stds, label=la_with_label, fmt='-o', capsize=5, color=la_color)

    # Plot best pipeline and its prefixes (if available)
    if best_pipeline is not None:
        # Prefixes (including LA with grouping)
        prefix_labels = []
        for i in range(1, len(best_pipeline)):
            if i == 1:
                prefix_labels.append(la_with_label)
            else:
                prefix_labels.append(_pipeline_to_label(best_pipeline[:i]))
        for lbl in prefix_labels[1:]:  # skip LA with grouping, already plotted
            means, stds = [], []
            for nl in noise_levels:
                pcks = results.get(lbl, {}).get(nl, [])
                means.append(np.mean(pcks) if pcks else np.nan)
                stds.append(np.std(pcks) if pcks else 0)
            if any(np.isfinite(means)):
                plt.errorbar(x, means, yerr=stds, label=lbl, fmt='-o', capsize=5)
        # Best pipeline final (thicker)
        best_means, best_stds = [], []
        for nl in noise_levels:
            pcks = results.get(best_label, {}).get(nl, [])
            best_means.append(np.mean(pcks) if pcks else np.nan)
            best_stds.append(np.std(pcks) if pcks else 0)
        if any(np.isfinite(best_means)):
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

# -----------------------------------------------------------------------------------
# Analysis runner (similar interface to graphmatching_eval.py)

class GraphGroupingAnalysis:
    def __init__(self, results_dir, output_dir, noise_levels=None, num_repeats=5, pck_threshold=0.05,
                 num_processes=None, score_threshold=0.3, min_edge_score=0.3, categories=None, la_only=False):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.noise_levels = noise_levels
        self.num_repeats = num_repeats
        self.pck_threshold = pck_threshold
        self.num_processes = num_processes
        self.score_threshold = score_threshold
        self.min_edge_score = min_edge_score
        self.categories = categories
        self.results = {}
        self.completed_categories = set()
        self.la_only = la_only

    def run(self):
        import concurrent.futures
        from functools import partial
        from tqdm import tqdm

        # Determine categories to process
        if self.categories is None:
            categories = [d for d in os.listdir(self.results_dir) if os.path.isdir(os.path.join(self.results_dir, d))]
        else:
            requested = list(self.categories)
            existing = [d for d in requested if os.path.isdir(os.path.join(self.results_dir, d))]
            missing = [d for d in requested if d not in existing]
            if missing:
                print(f"Warning: These requested categories do not exist in {self.results_dir}: {missing}")
            categories = existing
        print(f"Found categories: {categories}")

        summary_results = {}

        for category in categories:
            print(f"Processing category (grouping): {category}")
            category_dir = os.path.join(self.results_dir, category)
            pred_files = glob.glob(os.path.join(category_dir, "*_pred.json"))
            print(f"  Found {len(pred_files)} images.")

            worker_fn = partial(
                _process_image_file_grouping,
                category=category,
                noise_levels=self.noise_levels,
                num_repeats=self.num_repeats,
                pck_threshold=self.pck_threshold,
                score_threshold=self.score_threshold,
                min_edge_score=self.min_edge_score,
                method_pipelines=( [['la']] if self.la_only else METHOD_PIPELINES )
            )

            method_pcks = {}

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                future_to_file = {
                    executor.submit(worker_fn, pred_file): pred_file
                    for pred_file in pred_files
                }
                with tqdm(total=len(pred_files), desc=f"Grouping {category}", unit="file") as pbar:
                    for future in concurrent.futures.as_completed(future_to_file):
                        pred_file = future_to_file[future]
                        try:
                            result = future.result()
                            # Accumulate PCKs, including intermediate labels
                            m = result['method_pcks']
                            for label, noise_levels_data in m.items():
                                if label not in method_pcks:
                                    method_pcks[label] = {nl: [] for nl in self.noise_levels}
                                for nl, pcks in noise_levels_data.items():
                                    method_pcks[label][nl].extend(pcks)
                        except Exception as e:
                            print(f"Error processing file {pred_file}: {e}")
                        pbar.update(1)

            # Chunk-average per category (same approach as graphmatching_eval)
            aggregated = {}
            for label, nl_data in method_pcks.items():
                aggregated[label] = {}
                for nl, pcks in nl_data.items():
                    if pcks:
                        chunk_size = max(1, len(pred_files))
                        mean_pcks = []
                        for i in range(0, len(pcks), chunk_size):
                            chunk = pcks[i:i+chunk_size]
                            if chunk:
                                mean_pcks.append(np.mean(chunk))
                        aggregated[label][nl] = mean_pcks

            summary_results[category] = aggregated
            self.completed_categories.add(category)
            print(f"Category {category} completed. Generating plots...")
            os.makedirs(self.output_dir, exist_ok=True)
            plot_category_results(category, aggregated, self.noise_levels, self.output_dir)
            plot_best_method_vs_noise_grouping(category, aggregated, self.noise_levels, self.output_dir, self.pck_threshold, categories=categories)

        self.results = summary_results
        return self.results

# -----------------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze graph grouping + matching with varying noise levels')
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
    parser.add_argument('--min_edge_score', type=float, default=0.3,
                        help='Minimum edge score for grouping (default: 0.3)')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Optional list of categories to process (default: all subfolders in results_dir)')
    parser.add_argument('--la_only', action='store_true',
                        help='If set, run only Linear Assignment (LA) instead of all pipelines')

    args = parser.parse_args()

    results_dir = os.path.join(args.test_results_dir, args.subfolder)
    output_dir = os.path.join(args.test_results_dir, 'graph_grouping_plots', args.subfolder)
    os.makedirs(output_dir, exist_ok=True)

    analysis = GraphGroupingAnalysis(
        results_dir=results_dir,
        output_dir=output_dir,
        noise_levels=args.noise_levels,
        num_repeats=args.num_repeats,
        pck_threshold=args.pck_threshold,
        num_processes=args.num_processes,
        score_threshold=args.score_threshold,
        min_edge_score=args.min_edge_score,
        categories=args.categories,
        la_only=args.la_only
    )
    analysis.run()
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of processes to use (default: auto)')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='Keypoint score threshold to filter predictions')
    parser.add_argument('--min_edge_score', type=float, default=0.3,
                        help='Minimum edge score for grouping (default: 0.3)')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Optional list of categories to process (default: all subfolders in results_dir)')
    parser.add_argument('--la_only', action='store_true',
                        help='If set, run only Linear Assignment (LA) instead of all pipelines')

    args = parser.parse_args()

    results_dir = os.path.join(args.test_results_dir, args.subfolder)
    output_dir = os.path.join(args.test_results_dir, 'graph_grouping_plots', args.subfolder)
    os.makedirs(output_dir, exist_ok=True)

    analysis = GraphGroupingAnalysis(
        results_dir=results_dir,
        output_dir=output_dir,
        noise_levels=args.noise_levels,
        num_repeats=args.num_repeats,
        pck_threshold=args.pck_threshold,
        num_processes=args.num_processes,
        score_threshold=args.score_threshold,
        min_edge_score=args.min_edge_score,
        categories=args.categories,
        la_only=args.la_only
    )
    analysis.run()
