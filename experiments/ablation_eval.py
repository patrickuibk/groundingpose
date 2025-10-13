"""
Ablation evaluation and comparison script.
- Discovers ablation configs under work_dirs/ablations
- Computes metrics for each ablation and the shared baseline (using psd_train_test_eval.compute_all_metrics)
- Produces a single comparison figure (Keypoint PR, Keypoint RD, Relation PR)
  comparing all ablations against the shared model results.

Usage:
  python ablation_eval.py [<ablations_dir>] [<shared_root_dir>]

Defaults (relative to project root):
  ablations_dir   = work_dirs/ablations
  shared_root_dir = work_dirs  (expects results under work_dirs/shared[/test_results]/<category>)
"""
import os
import sys
from typing import Dict, List

# Ensure we can import sibling module
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from psd_train_test_eval import (
    compute_all_metrics,
    plot_model_comparison,
)


def _discover_subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def discover_ablation_configs(ablations_dir: str) -> List[str]:
    """List config names under ablations dir that contain predictions (test_results or category dirs)."""
    configs = []
    for name in _discover_subdirs(ablations_dir):
        cfg_dir = os.path.join(ablations_dir, name)
        if os.path.isdir(os.path.join(cfg_dir, 'test_results')):
            configs.append(name)
        else:
            # Allow raw category layout directly under the config dir
            subdirs = _discover_subdirs(cfg_dir)
            if subdirs:
                configs.append(name)
    return sorted(configs)


def results_base_for_shared(shared_root: str) -> str:
    """Return path containing category result subfolders for the shared baseline."""
    cand = os.path.join(shared_root, 'shared', 'test_results')
    if os.path.isdir(cand):
        return cand
    cand = os.path.join(shared_root, 'shared')
    return cand


def results_base_for_config(ablations_dir: str, config_name: str) -> str:
    base = os.path.join(ablations_dir, config_name)
    cand = os.path.join(base, 'test_results')
    return cand if os.path.isdir(cand) else base


def discover_categories_in_results_dir(results_base: str) -> List[str]:
    return _discover_subdirs(results_base)


# Fixed legend names for ablation configs
ABLATION_LEGEND_NAMES = {
    'shared': 'Baseline',
    'dec_off': 'No Keypoint Decoder',
    'dec2': 'Half Keypoint Decoder',
    'enc_off': 'No Feature Enhancer',
    'enc3': 'Half Feature Enhancer',
    'reldec1': 'Half Relation Decoder',
    'reldec_off': 'No Relation Decoder',
}

def main():
    # Derive sensible defaults from project structure
    project_root = os.path.dirname(os.path.dirname(_THIS_DIR))
    default_ablations_dir = os.path.join(project_root, 'grounding_dino_pose', 'work_dirs', 'ablations')
    default_shared_root = os.path.join(project_root, 'grounding_dino_pose', 'work_dirs')

    ablations_dir = sys.argv[1] if len(sys.argv) > 1 else default_ablations_dir
    shared_root = sys.argv[2] if len(sys.argv) > 2 else default_shared_root

    if not os.path.isdir(ablations_dir):
        print(f"Ablations directory not found: {ablations_dir}")
        return

    print(f"Using ablations dir: {ablations_dir}")
    print(f"Using shared root:   {shared_root}")

    # Discover models
    config_names = discover_ablation_configs(ablations_dir)
    if not config_names:
        print("No ablation configs found.")
        return
    print(f"Found ablations: {', '.join(config_names)}")

    # Discover baseline shared results
    shared_base = results_base_for_shared(shared_root)
    if not os.path.isdir(shared_base):
        print(f"Shared results not found under: {shared_root}")
        return

    # Gather categories (union across shared and all configs)
    categories_union = set(discover_categories_in_results_dir(shared_base))
    for cfg in config_names:
        categories_union.update(discover_categories_in_results_dir(results_base_for_config(ablations_dir, cfg)))
    categories = sorted(categories_union)
    print(f"Discovered categories: {', '.join(categories)}")

    # Compute metrics per model
    all_model_keypoint_rd_data: Dict[str, Dict] = {}  # rename kept consistent
    all_model_keypoint_pr_data: Dict[str, Dict] = {}
    all_model_relation_pr_data: Dict[str, Dict] = {}

    # Shared baseline
    shared_cats = discover_categories_in_results_dir(shared_base)
    (
        shared_kp_pr,
        shared_rel_pr,
        shared_kp_rd,
        kp_score_thr,
        kp_dist_thr,
    ) = compute_all_metrics(shared_base, shared_cats, model_type='')
    all_model_keypoint_pr_data['shared'] = shared_kp_pr
    all_model_relation_pr_data['shared'] = shared_rel_pr
    all_model_keypoint_rd_data['shared'] = shared_kp_rd

    # Ablations
    for cfg in config_names:
        cfg_base = results_base_for_config(ablations_dir, cfg)
        cfg_cats = discover_categories_in_results_dir(cfg_base)
        print(f"Processing {cfg} ...")
        (
            kp_pr,
            rel_pr,
            kp_rd,
            kp_score_thr,
            kp_dist_thr,
        ) = compute_all_metrics(cfg_base, cfg_cats, model_type='')
        if not cfg.startswith("rel"):
            all_model_keypoint_rd_data[cfg] = kp_rd
            all_model_keypoint_pr_data[cfg] = kp_pr
        if not cfg.startswith("enc") and not cfg.startswith("dec"):
            all_model_relation_pr_data[cfg] = rel_pr
    # Use shared plotting utility
    model_labels = ['shared'] + config_names
    plot_model_comparison(
        all_model_relation_pr_data,
        all_model_keypoint_pr_data,
        all_model_keypoint_rd_data,
        categories,
        ablations_dir,
        kp_score_thr,
        kp_dist_thr,
        model_types=model_labels,
        legend_name_map=ABLATION_LEGEND_NAMES
    )

if __name__ == '__main__':
    main()