import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.graphgrouping_eval import prepare_grouping_from_files
from experiments.graphgrouping_eval import (
    _group_gt_components,
    _compute_group_centroids,
    _match_groups_by_centroid,
)
from experiments.graphmatching_eval import (
    METHOD_PIPELINES,
    _pipeline_to_label,
    add_noise,
)
from tools.graph_matching import sequential_matching

def visualize_groups(
    pred_coords, groups, out_path, img_width, img_height, pred_label_names, relation_names,
    adjacency_matrices=None, img_path=None, gt_coords=None, gt_label_names=None, gt_relation_matrices=None,
    no_vis_text=False, matching=None, noisy_gt_coords=None
):
    print(f"- Saving visualization to {out_path} ({len(groups)} groups)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(18, 6))
    ax = plt.gca()
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)
    plt.axis('off')

    # Plot image if available (grayscale)
    img = None
    if img_path and os.path.exists(img_path):
        import matplotlib.image as mpimg
        img = mpimg.imread(img_path)
        ax.imshow(img, extent=[0, img_width, img_height, 0], alpha=0.8, cmap='gray')

    # Use label names directly, as in graphmatching
    all_labels = sorted(set(pred_label_names))
    cmap_kp = plt.get_cmap('jet', len(all_labels)+len(relation_names))
    kp_color_map = {lbl: cmap_kp(i) for i, lbl in enumerate(all_labels)}
    rel_color_map = {name: cmap_kp(i + len(all_labels)) for i, name in enumerate(relation_names)}

    # Plot ground truth keypoints and relations if provided
    if gt_coords is not None and gt_label_names is not None and gt_relation_matrices is not None:
        # for i in range(len(gt_coords)):
        #     for j in range(i+1, len(gt_coords)):
        #         for rel_type in range(gt_relation_matrices.shape[2]):
        #             if gt_relation_matrices[i, j, rel_type] > 0.5:
        #                 rel_name = relation_names[rel_type]
        #                 ax.plot([gt_coords[i, 0], gt_coords[j, 0]],
        #                         [gt_coords[i, 1], gt_coords[j, 1]],
        #                         color=rel_color_map[rel_name], linewidth=0.8, alpha=0.5, linestyle='dashed')
        for i in range(len(gt_coords)):
            kp_name = gt_label_names[i]
            ax.plot(gt_coords[i, 0], gt_coords[i, 1], 'o', color=kp_color_map[kp_name], markersize=2, markeredgecolor='black', markeredgewidth=0.5, alpha=0.7)

    # Plot noisy GT keypoints if provided
    if noisy_gt_coords is not None and gt_label_names is not None:
        for i in range(len(noisy_gt_coords)):
            kp_name = gt_label_names[i]
            # Use gray triangle for noisy GT
            ax.plot(noisy_gt_coords[i, 0], noisy_gt_coords[i, 1], '^', color='gray', markersize=4, markeredgecolor='black', markeredgewidth=0.5, alpha=0.7, label='_noisygt')

    for group in groups:
        K = len(group.node_ids)
        # Adjacency lines first (so keypoints are on top)
        for i in range(K):
            for j in range(i+1, K):
                # Find the relation type with the maximum score for this pair
                scores = group.adjacency_matrix[i, j, :]
                rel_type = np.argmax(scores)
                score = scores[rel_type]
                if score > 0.0:
                    x1, y1 = pred_coords[group.node_ids[i]]
                    x2, y2 = pred_coords[group.node_ids[j]]
                    rel_name = relation_names[rel_type]
                    ax.plot([x1, x2], [y1, y2], color=rel_color_map[rel_name], linewidth=0.8, alpha=0.7)
                    # Plot relation probability at midpoint with small white box
                    if not no_vis_text:
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        ax.text(
                            mid_x, mid_y, f"{score:.2f}",
                            color=rel_color_map[rel_name], fontsize=3, ha='center', va='center',
                            bbox=dict(boxstyle='square,pad=0.05', facecolor='white', edgecolor='none', alpha=0.8)
                        )
        # Keypoints
        for idx, nid in enumerate(group.node_ids):
            label_name = pred_label_names[nid]
            color = kp_color_map.get(label_name, (0.5, 0.5, 0.5))
            x, y = pred_coords[nid]
            ax.plot(x, y, 'x', color=color, markersize=4, markeredgewidth=0.5, alpha=0.9)
            # Plot keypoint score
            if not no_vis_text:
                ax.text(
                    x, y, f"{group.keypoint_scores[idx]:.2f}",
                    color=color, fontsize=3, ha='left', va='bottom',
                    bbox=dict(boxstyle='square,pad=0.05', facecolor='white', edgecolor='none', alpha=0.8)
                )
    # Draw GT->Pred matched pairs if provided
    if matching is not None and gt_coords is not None:
        for gi, pj in matching.items():
            if gi < 0 or pj < 0 or gi >= len(gt_coords) or pj >= len(pred_coords):
                continue
            # If noisy_gt_coords is available, use it for the GT endpoint
            if noisy_gt_coords is not None:
                gx, gy = noisy_gt_coords[gi]
            else:
                gx, gy = gt_coords[gi]
            px, py = pred_coords[pj]
            ax.plot([gx, px], [gy, py], color='black', linewidth=0.8, alpha=0.6, linestyle='dashed')

    # Legends for keypoints and relations
    kp_legend = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=kp_color_map[lbl], markersize=8, label=str(lbl))
                 for lbl in all_labels]
    rel_legend = [plt.Line2D([0], [0], color=rel_color_map[name], linewidth=2, label=name)
                  for name in relation_names]
    # Add noisy GT legend if plotted
    legend_handles = kp_legend + rel_legend
    if noisy_gt_coords is not None:
        legend_handles.append(plt.Line2D([0], [0], marker='^', color='gray', markerfacecolor='gray', markersize=8, linestyle='None', label='Noisy GT'))

    ax.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        fontsize=8,
        title="Keypoints & Relations",
        ncol=max(1, len(legend_handles) // 4)
    )

    plt.title(f"{len(groups)} grouped instances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_graph_match(
    pred_json_path, score_thresh=0.3, result_dir_name="", out_dir=None, min_edge_score=0.3,
    no_vis_text=False, noise=0.0, match_graphs=False
):
    print(f"Processing {pred_json_path} with score_thresh={score_thresh}")
    # Use eval helper to prepare grouping and arrays
    data = prepare_grouping_from_files(
        pred_json_path,
        score_thresh=score_thresh,
        result_dir_name=result_dir_name,
        min_edge_score=min_edge_score,
    )
    img_width = data['img_width']
    img_height = data['img_height']
    pred_coords = data['pred_coords']
    pred_label_names = data['pred_label_names']
    pred_scores = data['pred_scores']
    pred_relation_matrices = data['pred_relation_matrices']
    groups = data['groups']
    gt_coords = data['gt_coords']
    gt_label_names = data['gt_label_names']
    gt_relation_matrices = data['gt_relation_matrices']
    relation_names = data['relation_names']
    img_path = data['img_path']
    print(f"- Using merge function: {data['merge_fn_name']}")
    print(f"- Grouped into {len(groups)} instances")

    mapping = None
    noisy_gt_coords = None
    if match_graphs:
        # Normalize coords and add noise to GT
        pred_coords_norm = pred_coords / np.array([img_width, img_height])
        gt_coords_norm = gt_coords / np.array([img_width, img_height])
        noisy_gt_coords_norm = add_noise(gt_coords_norm.copy(), noise, gt_relation_matrices)
        noisy_gt_coords = noisy_gt_coords_norm * np.array([img_width, img_height])

        # Labels to ints for matching
        all_label_names = sorted(set(gt_label_names) | set(pred_label_names))
        name2id = {n: i for i, n in enumerate(all_label_names)}
        gt_label_ids = np.array([name2id[n] for n in gt_label_names])
        pred_label_ids = np.array([name2id[n] for n in pred_label_names])

        # Group GT by connectivity, match groups by centroid
        gt_groups = _group_gt_components(gt_label_ids, None, gt_relation_matrices)
        gt_cents = _compute_group_centroids(gt_groups, noisy_gt_coords_norm)
        pred_cents = _compute_group_centroids(groups, pred_coords_norm)  # groups are predicted groups
        group_pairs = _match_groups_by_centroid(gt_cents, pred_cents)

        # Choose a default pipeline
        pipeline = METHOD_PIPELINES[-1]
        print(f"- Running matching pipeline: {_pipeline_to_label(pipeline)}")

        # Aggregate global mapping across pairs
        mapping = {}
        for gi, pj in group_pairs:
            g_gt = gt_groups[gi]
            g_pr = groups[pj]
            gt_ids = g_gt.node_ids
            pr_ids = g_pr.node_ids

            gt_lbls_local = gt_label_ids[gt_ids]
            pr_lbls_local = pred_label_ids[pr_ids]
            gt_rel_local = gt_relation_matrices[np.ix_(gt_ids, gt_ids, np.arange(gt_relation_matrices.shape[2]))]
            pr_rel_local = pred_relation_matrices[np.ix_(pr_ids, pr_ids, np.arange(pred_relation_matrices.shape[2]))]
            pr_scores_local = pred_scores[pr_ids]
            gt_coords_local = noisy_gt_coords_norm[gt_ids]
            pr_coords_local = pred_coords_norm[pr_ids]

            try:
                best_matching, intermediates = sequential_matching(
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
                print(f"  - Matching failed for a group pair: {e}")
                continue

            # Convert local match to global
            for li, lj in best_matching.items():
                mapping[int(gt_ids[int(li)])] = int(pr_ids[int(lj)])

    if out_dir is not None:
        img_out = os.path.join(out_dir, os.path.basename(pred_json_path).replace('_pred.json', '_groups.png'))
        visualize_groups(
            pred_coords,
            groups,
            img_out,
            img_width,
            img_height,
            pred_label_names=pred_label_names,
            relation_names=relation_names,
            img_path=img_path,
            gt_coords=np.array(gt_coords),
            gt_label_names=np.array(gt_label_names),
            gt_relation_matrices=np.array(gt_relation_matrices) if gt_relation_matrices is not None else None,
            no_vis_text=no_vis_text,
            matching=mapping,
            noisy_gt_coords=noisy_gt_coords,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', '-r', required=True, help='Directory containing _pred.json files')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--score-thresh', type=float, default=0.3, help='Keypoint score threshold (default: 0.3)')
    parser.add_argument('--outdir', type=str, default='graph_grouping_visualization', help='Output directory (relative to result-dir) for visualizations')
    parser.add_argument('--min-edge-score', type=float, default=0.3, help='Minimum edge score for grouping (default: 0.3)')
    parser.add_argument('--no-vis-text', action='store_true', help='Disable visualization of text (scores)')
    parser.add_argument('--match-graphs', action='store_true', help='Perform graph matching and plot matched pairs')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level applied to GT initial coordinates for matching')
    args = parser.parse_args()
    result_dir = args.result_dir
    out_dir = os.path.join(result_dir, args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(result_dir):
        if not fname.endswith('_pred.json'):
            continue
        result_path = os.path.join(result_dir, fname)
        run_graph_match(
            result_path,
            score_thresh=args.score_thresh,
            result_dir_name=result_dir,
            out_dir=None if args.no_vis else out_dir,
            min_edge_score=args.min_edge_score,
            no_vis_text=args.no_vis_text,
            noise=args.noise,
            match_graphs=args.match_graphs,
        )