import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import matplotlib.image as mpimg
import sys
import time
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.graph_matching import sequential_matching
from datasets.metrics.utils import compute_pck
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from experiments.graphmatching_eval import add_noise


# Rename pipelines to match algorithm names expected by sequential_matching
PIPELINES = [
    ['la'],
    ['la', 'sm', 'rrwm', 'ipfp', 'our2opt'],
    ['la', 'faq', 'rrwm', 'sm', 'ipfp', 'our2opt'],
    ['la', 'sm', 'faq', 'rrwm', 'ipfp', 'our2opt'],
    ['la', 'sm', 'ipfp', 'faq', 'rrwm', 'our2opt'],
    ['la', 'ipfp', 'sm', 'rrwm', 'faq', 'our2opt'],
]
PIPELINE_LABELS = [' + '.join(p).upper().replace('OUR2OPT', 'Our 2OPT') for p in PIPELINES]

def plot_assignment_panel(ax, gt_coords, pred_coords, gt_label_names, pred_label_names, relation_names, gt_relation_matrices, pred_scores, pred_relation_matrices, assignment, kp_color_map, rel_color_map, img=None, score_thresh=0.3, title=None):
    """
    Visualize a matching assignment between GT and predictions on the given axis.
    """
    if title:
        ax.set_title(title)
    if img is not None:
        ax.imshow(img, cmap='gray', alpha=0.8)
    ax.axis('off')
    # Draw matched relations
    for i in range(len(gt_coords)):
        for j in range(i+1, len(gt_coords)):
            for rel_type in range(gt_relation_matrices.shape[2]):
                if gt_relation_matrices[i, j, rel_type] > 0.5:
                    if i in assignment and j in assignment:
                        vi = assignment[i]
                        vj = assignment[j]
                        rel_name = relation_names[rel_type]
                        ax.plot([pred_coords[vi, 0], pred_coords[vj, 0]],
                                [pred_coords[vi, 1], pred_coords[vj, 1]],
                                color=rel_color_map[rel_name], linewidth=0.8)
    # Draw GT keypoints
    for i in range(len(gt_coords)):
        kp_name = gt_label_names[i]
        ax.plot(gt_coords[i, 0], gt_coords[i, 1], 'o', color=kp_color_map[kp_name], markersize=2, markeredgecolor='black', markeredgewidth=0.5)
    # Draw matched predicted keypoints and lines
    for i in range(len(gt_coords)):
        if i in assignment:
            v = assignment[i]
            kp_name = pred_label_names[v]
            ax.plot(pred_coords[v, 0], pred_coords[v, 1], 'x', color=kp_color_map[kp_name], markersize=4, markeredgewidth=0.5)
            ax.plot([gt_coords[i, 0], pred_coords[v, 0]],
                    [gt_coords[i, 1], pred_coords[v, 1]],
                    linestyle=':', color='purple', linewidth=0.8, alpha=0.7)
        else:
            ax.plot(gt_coords[i, 0], gt_coords[i, 1], marker='s',
                    markerfacecolor='none', markeredgecolor='red',
                    markersize=4, markeredgewidth=1.2, alpha=0.8)
    # Draw unmatched predicted keypoints
    for i in range(len(pred_coords)):
        if pred_scores[i] < score_thresh or i in assignment.values():
            continue
        kp_name = pred_label_names[i]
        ax.plot(pred_coords[i, 0], pred_coords[i, 1], 'x', color=kp_color_map[kp_name], markersize=4, markeredgewidth=0.5, alpha=0.5)

def run_method_pipeline(pipeline, gt_label_ids, gt_relation_matrices, pred_label_ids, pred_scores, pred_relation_matrices, noisy_gt_coords_norm, pred_coords_norm):
    """
    Run a sequence of algorithms using sequential_matching with coordinates support
    """
    # Use sequential_matching for all algorithms, passing coordinates for LA methods
    return sequential_matching(
        gt_label_ids,
        gt_relation_matrices,
        pred_label_ids,
        pred_scores,
        pred_relation_matrices,
        algorithm_sequence=pipeline,
        initial_matching=None,
        gt_coords=noisy_gt_coords_norm,
        pred_coords=pred_coords_norm
    )

def run_graph_match(pred_json_path, score_thresh=0.3, pck_threshold=0.005, noise_level=0.1):
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    gt_file = pred_json_path.replace("_pred.json", "_gt.json")                            
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    img_width = gt_data['img_width']
    img_height = gt_data['img_height']
    pred_coords = np.array(pred_data['keypoint_coords'])
    pred_label_names = np.array(pred_data['keypoint_label_names'])
    pred_scores = np.array(pred_data['keypoint_scores'])
    pred_relation_matrices = np.array(pred_data['keypoint_relation_scores'])
    mask = pred_scores >= score_thresh
    pred_coords = pred_coords[mask]
    pred_label_names = pred_label_names[mask]
    pred_scores = pred_scores[mask]
    pred_relation_matrices = pred_relation_matrices[mask][:, mask, :]
    gt_coords = np.array(gt_data['keypoint_coords'])
    gt_label_names = np.array(gt_data['keypoint_label_names'])
    gt_relation_matrices = np.array(gt_data['relation_matrices'])
    relation_names = gt_data['relation_names']
    pred_coords_norm = pred_coords / np.array([img_width, img_height])
    gt_coords_norm = gt_coords / np.array([img_width, img_height])
    gt_coords_norm_noisy = add_noise(
        gt_coords_norm.copy(), noise_level, 
        relation_matrices=gt_relation_matrices
    )
    gt_coords_noisy = gt_coords_norm_noisy * np.array([img_width, img_height])
    all_label_names = set(gt_label_names) | set(pred_label_names)
    label_name_to_int = {name: idx for idx, name in enumerate(all_label_names)}
    gt_label_ids = np.array([label_name_to_int[name] for name in gt_label_names])
    pred_label_ids = np.array([label_name_to_int[name] for name in pred_label_names])

    print(f"\nProcessing: {os.path.basename(pred_json_path)}")
    print(f"{'Matching Method':<40} | {'PCK@' + str(pck_threshold):<10} | {'Time'}")
    print("-" * 60)

    results = []
    for pipeline, label in zip(PIPELINES, PIPELINE_LABELS):
        start_time = time.time()
        try:
            matches = run_method_pipeline(
                pipeline,
                gt_label_ids,
                gt_relation_matrices,
                pred_label_ids,
                pred_scores,
                pred_relation_matrices,
                gt_coords_norm_noisy,
                pred_coords_norm
            )
            elapsed = time.time() - start_time
            pck = compute_pck(gt_coords, pred_coords, matches, img_width, img_height, pck_threshold)
        except Exception as e:
            matches = None
            elapsed = 0.0
            pck = 0.0
            print(f"Error in pipeline {label} for {os.path.basename(pred_json_path)}: {e}")
        results.append({
            "name": label,
            "assignment": matches,
            "pck": pck,
            "time": elapsed
        })
        print(f"{label:<40} | {pck:.4f} | {elapsed:.3f}s")

    return {
        "pred_json_path": pred_json_path,
        "pred_data": pred_data,
        "gt_data": gt_data,
        "img_width": img_width,
        "img_height": img_height,
        "pred_coords": pred_coords,
        "pred_label_names": pred_label_names,
        "pred_scores": pred_scores,
        "pred_relation_matrices": pred_relation_matrices,
        "gt_coords": gt_coords,
        "gt_label_names": gt_label_names,
        "gt_relation_matrices": gt_relation_matrices,
        "relation_names": relation_names,
        "gt_coords_noisy": gt_coords_noisy,
        "results": results
    }

def visualize_graph_match(match_result, out_dir=None, score_thresh=0.3):
    pred_json_path = match_result["pred_json_path"]
    pred_data = match_result["pred_data"]
    gt_data = match_result["gt_data"]
    img_width = match_result["img_width"]
    img_height = match_result["img_height"]
    pred_coords = match_result["pred_coords"]
    pred_label_names = match_result["pred_label_names"]
    pred_scores = match_result["pred_scores"]
    pred_relation_matrices = match_result["pred_relation_matrices"]
    gt_coords = match_result["gt_coords"]
    gt_label_names = match_result["gt_label_names"]
    gt_relation_matrices = match_result["gt_relation_matrices"]
    relation_names = match_result["relation_names"]
    gt_coords_noisy = match_result["gt_coords_noisy"]
    results = match_result["results"]

    img = None
    img_path = gt_data.get('img_path', None)
    if img_path and os.path.exists(img_path):
        img = mpimg.imread(img_path)

    keypoint_names = sorted(set(gt_label_names) | set(pred_label_names))
    cmap_kp = plt.get_cmap('jet', len(keypoint_names)+len(relation_names))
    kp_color_map = {name: cmap_kp(i) for i, name in enumerate(keypoint_names)}
    rel_color_map = {name: cmap_kp(i + len(keypoint_names)) for i, name in enumerate(relation_names)}

    fig, axes = plt.subplots(len(PIPELINES) + 2, 1, figsize=(8, 26))
    axes = axes.flatten()

    # Panel 1: GT graph
    ax = axes[0]
    ax.set_title('Noisy GT Graph')
    if img is not None:
        ax.imshow(img, cmap='gray', alpha=0.8)
    ax.axis('off')

    for i in range(len(gt_coords_noisy)):
        for j in range(i+1, len(gt_coords_noisy)):
            for rel_type in range(gt_relation_matrices.shape[2]):
                if gt_relation_matrices[i, j, rel_type] > 0.5:
                    rel_name = relation_names[rel_type]
                    ax.plot([gt_coords_noisy[i, 0], gt_coords_noisy[j, 0]],
                            [gt_coords_noisy[i, 1], gt_coords_noisy[j, 1]],
                            color=rel_color_map[rel_name], linewidth=0.8)
    for i in range(len(gt_coords_noisy)):
        kp_name = gt_label_names[i]
        ax.plot(gt_coords_noisy[i, 0], gt_coords_noisy[i, 1], 'o', color=kp_color_map[kp_name], markersize=2, markeredgecolor='black', markeredgewidth=0.5)

    # Panel 2: Predictions graph
    ax = axes[1]
    ax.set_title('Predicted Graph')
    if img is not None:
        ax.imshow(img, cmap='gray', alpha=0.8)
    ax.axis('off')
    for i in range(len(pred_coords)):
        for j in range(i+1, len(pred_coords)):
            for rel_type in range(pred_relation_matrices.shape[2]):
                prob = pred_relation_matrices[i, j, rel_type]
                if prob > 0.5:
                    rel_name = relation_names[rel_type]
                    ax.plot([pred_coords[i, 0], pred_coords[j, 0]],
                            [pred_coords[i, 1], pred_coords[j, 1]],
                            color=rel_color_map[rel_name], linewidth=0.8)
                    # Plot relation probability at midpoint with small white box
                    mid_x = (pred_coords[i, 0] + pred_coords[j, 0]) / 2
                    mid_y = (pred_coords[i, 1] + pred_coords[j, 1]) / 2
                    ax.text(
                        mid_x, mid_y, f"{prob:.2f}",
                        color=rel_color_map[rel_name], fontsize=3, ha='center', va='center',
                        bbox=dict(boxstyle='square,pad=0.05', facecolor='white', edgecolor='none', alpha=0.8)
                    )
    for i in range(len(pred_coords)):
        kp_name = pred_label_names[i]
        ax.plot(pred_coords[i, 0], pred_coords[i, 1], 'x', color=kp_color_map[kp_name], markersize=4, markeredgewidth=0.5)
        # Plot keypoint probability with small white box
        ax.text(
            pred_coords[i, 0], pred_coords[i, 1], f"{pred_scores[i]:.2f}",
            color=kp_color_map[kp_name], fontsize=3, ha='left', va='bottom',
            bbox=dict(boxstyle='square,pad=0.05', facecolor='white', edgecolor='none', alpha=0.8)
        )
    kp_legend = [Line2D([0], [0], marker='o', color='black', markerfacecolor=kp_color_map[name], markersize=8, label=name)
                 for name in keypoint_names]
    rel_legend = [Line2D([0], [0], color=rel_color_map[name], linewidth=2, label=name)
                  for name in relation_names]
    # ax.legend(handles=kp_legend + rel_legend, loc='upper right', fontsize=8, title="Keypoints & Relations")

    # Panel 3+: Matching Visualizations
    for idx, label in enumerate(PIPELINE_LABELS):
        result = next((r for r in results if r["name"] == label), None)
        if result is not None:
            plot_assignment_panel(
                axes[2+idx],
                gt_coords, pred_coords, gt_label_names, pred_label_names, relation_names,
                gt_relation_matrices, pred_scores, pred_relation_matrices,
                result["assignment"], kp_color_map, rel_color_map, img=img, score_thresh=score_thresh,
                title=f"{result['name']} (PCK: {result['pck']:.4f})"
            )

    plt.tight_layout()
    filename = os.path.splitext(os.path.basename(pred_json_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(pred_json_path), 'graph_matching_visualization')
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, f'{filename}_graph_match.png')
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved visualization to: {file_name}")

def visualize_pred_graph(pred_json_path, out_dir=None, score_thresh=0.3):
    """
    Visualize only the predicted graph (with legend) and save to a separate folder.
    Plots the image in its original resolution.
    """
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    gt_file = pred_json_path.replace("_pred.json", "_gt.json")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    img = None
    img_path = gt_data.get('img_path', None)
    img_width = gt_data.get('img_width', None)
    img_height = gt_data.get('img_height', None)
    if img_path and os.path.exists(img_path):
        img = mpimg.imread(img_path)

    pred_coords = np.array(pred_data['keypoint_coords'])
    pred_label_names = np.array(pred_data['keypoint_label_names'])
    pred_scores = np.array(pred_data['keypoint_scores'])
    pred_relation_matrices = np.array(pred_data['keypoint_relation_scores'])
    relation_names = gt_data['relation_names']

    # Threshold predictions
    mask = pred_scores >= score_thresh
    pred_coords = pred_coords[mask]
    pred_label_names = pred_label_names[mask]
    pred_scores = pred_scores[mask]
    pred_relation_matrices = pred_relation_matrices[mask][:, mask, :]

    keypoint_names = sorted(set(pred_label_names))
    cmap_kp = plt.get_cmap('jet', len(keypoint_names) + len(relation_names))
    kp_color_map = {name: cmap_kp(i) for i, name in enumerate(keypoint_names)}
    rel_color_map = {name: cmap_kp(i + len(keypoint_names)) for i, name in enumerate(relation_names)}

    # Use original image resolution if available
    if img is not None and img_width is not None and img_height is not None:
        fig_w = img_width / 80
        fig_h = img_height / 80

        print(f"Image size: {img_width}x{img_height}, Figure size: {fig_w}x{fig_h}")
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)
        ax.set_xlim([0, img_width])
        ax.set_ylim([img_height, 0])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=900)

    if img is not None:
        ax.imshow(img, cmap='gray')
    ax.axis('off')

    # Draw predicted relations and annotate probabilities
    for i in range(len(pred_coords)):
        for j in range(i+1, len(pred_coords)):
            for rel_type in range(pred_relation_matrices.shape[2]):
                prob = pred_relation_matrices[i, j, rel_type]
                if prob > score_thresh:
                    rel_name = relation_names[rel_type]
                    ax.plot([pred_coords[i, 0], pred_coords[j, 0]],
                            [pred_coords[i, 1], pred_coords[j, 1]],
                            color=rel_color_map[rel_name], linewidth=3)

    # Draw keypoints and annotate keypoint scores
    for i in range(len(pred_coords)):
        kp_name = pred_label_names[i]
        ax.plot(pred_coords[i, 0], pred_coords[i, 1], 'x', color=kp_color_map[kp_name], markersize=25, markeredgewidth=3)

    # Add legend
    kp_legend = [Line2D([0], [0], marker='x', linestyle='None', color=kp_color_map[name], markeredgewidth=8.0, markersize=40, label=name)
                 for name in keypoint_names]
    rel_legend = [Line2D([0], [0], color=rel_color_map[name], linewidth=15, label=name)
                  for name in relation_names]
    legend = ax.legend(handles=kp_legend + rel_legend, loc='upper center', fontsize=40, title="Keypoints & Relations")
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(50)

    plt.tight_layout()
    filename = os.path.splitext(os.path.basename(pred_json_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(pred_json_path), 'pred_graph_visualization')
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, f'{filename}_pred_graph.png')
    plt.savefig(file_name, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved visualization to: {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', '-r', required=True, help='Directory containing _pred.json files')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--score-thresh', type=float, default=0.3, help='Keypoint score threshold (default: 0.3)')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level for GT coordinates (default: 0.1)')
    parser.add_argument('--outdir', type=str, default='graph_matching_visualization', help='Output directory (relative to result-dir) for visualizations')
    parser.add_argument('--pred-only', action='store_true', help='Only visualize predicted graph with legend and skip graph matching')
    parser.add_argument('--pred-outdir', type=str, default='pred_graph_visualization', help='Output directory (relative to result-dir) for --pred-only visualizations')
    args = parser.parse_args()
    result_dir = args.result_dir
    out_dir = os.path.join(result_dir, args.outdir)
    for fname in os.listdir(result_dir):
        if not fname.endswith('_pred.json'):
            continue
        pred_path = os.path.join(result_dir, fname)

        # Predicted-only mode: skip matching, always visualize and save to separate folder
        if args.pred_only:
            pred_only_outdir = os.path.join(result_dir, args.pred_outdir)
            visualize_pred_graph(pred_path, out_dir=pred_only_outdir, score_thresh=args.score_thresh)
            continue

        match_result = run_graph_match(
            pred_path,
            score_thresh=args.score_thresh,
            noise_level=args.noise_level
        )
        if not args.no_vis:
            visualize_graph_match(match_result, out_dir=out_dir, score_thresh=args.score_thresh)