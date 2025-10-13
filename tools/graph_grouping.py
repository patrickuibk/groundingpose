from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

@dataclass
class InstanceGroup:
    """Grouped keypoints represented explicitly by arrays.
    node_ids: (K,) global indices of the keypoints in this instance.
    keypoint_labels: (K,) labels corresponding to node_ids order.
    keypoint_scores: (K,) scores corresponding to node_ids order.
    adjacency_matrix: (K, K, R) relation scores subset for these nodes.
    """
    node_ids: np.ndarray
    keypoint_labels: np.ndarray
    keypoint_scores: np.ndarray
    adjacency_matrix: np.ndarray


# MergeFn is a callable that takes two InstanceGroup objects, two node indices (u, v),
# and a relation type integer, and returns a merged InstanceGroup or None if merging is not possible.
MergeFn = Callable[[InstanceGroup, InstanceGroup, int, int, int], Optional[InstanceGroup]]


def group_keypoints_into_instances(
    keypoint_labels: np.ndarray,
    keypoint_scores: np.ndarray,
    relation_scores: np.ndarray,
    merge_fn: MergeFn,
    min_edge_score: float = 0.0,
) -> List[InstanceGroup]:
    """Group keypoints using a single descending pass over relation scores.

    Args:
        keypoint_labels: (N,) integer labels.
        keypoint_scores: (N,) scores.
        relation_scores: (N, N, R) relation score tensor (R >= 1).
        merge_fn: (grp_u, grp_v, u, v, rel_type) -> merged InstanceGroup or None.
                  u, v are global node indices. rel_type is argmax relation type for (u,v).
        min_edge_score: Minimum edge score required to consider merging two groups.

    Returns:
        List of final InstanceGroup objects.
    """
    assert keypoint_labels.ndim == 1
    assert keypoint_scores.ndim == 1 and keypoint_scores.shape[0] == keypoint_labels.shape[0]
    assert relation_scores.ndim == 3, "relation_scores must be (N,N,R)"
    N, N2, R = relation_scores.shape
    assert N == N2, "relation_scores must be square in first two dims"
    if N == 0:
        return []

    # Initialize singleton groups
    groups: List[InstanceGroup] = []
    for i in range(N):
        groups.append(InstanceGroup(
            node_ids=np.array([i], dtype=np.int32),
            keypoint_labels=np.array([keypoint_labels[i]], dtype=keypoint_labels.dtype),
            keypoint_scores=np.array([keypoint_scores[i]], dtype=keypoint_scores.dtype),
            adjacency_matrix=np.zeros((1, 1, R), dtype=relation_scores.dtype)
        ))
    node_to_group = np.arange(N)  # global node id -> index into groups list

    # Build edge list with max score and relation type per pair (i<j)
    upper_i, upper_j = np.triu_indices(N, k=1)
    edge_relations = relation_scores[upper_i, upper_j, :]  # (E, R)
    best_relation_type = np.argmax(edge_relations, axis=1)  # (E,)
    best_relation_score = edge_relations[np.arange(edge_relations.shape[0]), best_relation_type]  # (E,)

    # Filter edges by min_edge_score
    valid_mask = best_relation_score >= min_edge_score
    sorted_indices = np.argsort(-best_relation_score[valid_mask])  # descending order

    sorted_i = upper_i[valid_mask][sorted_indices]
    sorted_j = upper_j[valid_mask][sorted_indices]
    sorted_relation_type = best_relation_type[valid_mask][sorted_indices]

    for u, v, rel_type in zip(sorted_i, sorted_j, sorted_relation_type):
        group_u_idx = node_to_group[u]
        group_v_idx = node_to_group[v]

        # if already in the same group, skip
        if group_u_idx == group_v_idx:
            continue

        group_u = groups[group_u_idx]
        group_v = groups[group_v_idx]
        merged_group = merge_fn(group_u, group_v, int(u), int(v), int(rel_type))
        if merged_group is None:
            continue

        # Place merged group in slot group_u_idx, remove group_v_idx via swap-delete
        groups[group_u_idx] = merged_group
        for nid in merged_group.node_ids:
            node_to_group[int(nid)] = group_u_idx
        last_idx = len(groups) - 1
        if group_v_idx != last_idx:
            groups[group_v_idx] = groups[last_idx]
            for nid in groups[group_v_idx].node_ids:
                node_to_group[int(nid)] = group_v_idx
        groups.pop()

    return groups


def _merge_groups(
    grp_a: InstanceGroup,
    grp_b: InstanceGroup,
    relation_scores: Optional[np.ndarray] = None
) -> InstanceGroup:
    """
    Merge two InstanceGroup objects.

    Behavior:
    - node_ids, labels, and scores are concatenated preserving order: (grp_a nodes, grp_b nodes).
    - Intra-group adjacency (the two diagonal blocks) is copied exactly from grp_a / grp_b
      so any previously computed / refined values are preserved.
    - If relation_scores (global (N,N,R)) is provided, the cross-group relations
      (off-diagonal blocks) are filled from it:
          adj[:Ka, Ka:, :] = relation_scores[grp_a.node_ids][:, grp_b.node_ids, :]
          adj[Ka:, :Ka, :] = transpose of the above
      (intra-group blocks are NOT overwritten by global scores).
    - If relation_scores is None, cross blocks remain zero.

    Args:
        grp_a: First group.
        grp_b: Second group.
        relation_scores: Optional global relation score tensor (N,N,R).

    Returns:
        New merged InstanceGroup with combined metadata and adjacency.
    """
    new_node_ids = np.concatenate([grp_a.node_ids, grp_b.node_ids])
    new_labels = np.concatenate([grp_a.keypoint_labels, grp_b.keypoint_labels])
    new_scores = np.concatenate([grp_a.keypoint_scores, grp_b.keypoint_scores])

    Ka = grp_a.node_ids.size
    Kb = grp_b.node_ids.size
    R = grp_a.adjacency_matrix.shape[2]
    assert grp_b.adjacency_matrix.shape[2] == R, "Adjacency channel mismatch"

    # Initialize adjacency with zeros then place preserved intra-group blocks.
    adj = np.zeros((Ka + Kb, Ka + Kb, R), dtype=grp_a.adjacency_matrix.dtype)
    adj[:Ka, :Ka, :] = grp_a.adjacency_matrix
    adj[Ka:, Ka:, :] = grp_b.adjacency_matrix

    if relation_scores is not None:
        # Fill only cross-group relations from global tensor.
        # Shape: (Ka, Kb, R)
        cross = relation_scores[np.ix_(grp_a.node_ids, grp_b.node_ids, np.arange(R))]
        adj[:Ka, Ka:, :] = cross
        adj[Ka:, :Ka, :] = np.transpose(cross, (1, 0, 2))

    return InstanceGroup(
        node_ids=new_node_ids,
        keypoint_labels=new_labels,
        keypoint_scores=new_scores,
        adjacency_matrix=adj
    )


def make_merge_fn_max_label(max_per_label, relation_scores: Optional[np.ndarray] = None) -> MergeFn:
    """
    Factory function to create a merge function that merges two groups only if the resulting group
    does not exceed the maximum allowed number of keypoints per label.

    Args:
        max_per_label (dict or int): If dict, maps label to max count. If int, all labels share the same max.
        relation_scores (Optional[np.ndarray]): Global relation scores array for adjacency matrix.

    Returns:
        MergeFn: A function that attempts to merge two groups (grp_u, grp_v) if the merged group
        does not exceed the max count for any label. Returns None if the limit is exceeded.
    """
    def merge_fn(grp_u, grp_v, u, v, rel_type):
        merged_labels = np.concatenate([grp_u.keypoint_labels, grp_v.keypoint_labels])
        # Count occurrences per label
        unique, counts = np.unique(merged_labels, return_counts=True)
        if isinstance(max_per_label, int):
            if np.any(counts > max_per_label):
                return None
        else:
            for lbl, cnt in zip(unique, counts):
                if cnt > max_per_label.get(int(lbl), max(counts)):
                    return None
        return _merge_groups(grp_u, grp_v, relation_scores)
    return merge_fn


def make_merge_fn_max_degree_per_label(
    max_degree_per_label: dict,
    relation_scores: np.ndarray,
) -> MergeFn:

    num_nodes = relation_scores.shape[0]
    node_degrees = np.zeros(num_nodes, dtype=np.int32)

    def _label_from_groups(grp_u: InstanceGroup, grp_v: InstanceGroup, node_id: int):
        for g in (grp_u, grp_v):
            idx = np.where(g.node_ids == node_id)[0]
            if idx.size:
                return g.keypoint_labels[idx[0]]
        return None  # should not happen

    def merge_fn(grp_u: InstanceGroup, grp_v: InstanceGroup, u: int, v: int, rel_type: int):
        label_u = _label_from_groups(grp_u, grp_v, u)
        label_v = _label_from_groups(grp_u, grp_v, v)
        max_deg_u = max_degree_per_label.get(int(label_u), np.inf)
        max_deg_v = max_degree_per_label.get(int(label_v), np.inf)

        if node_degrees[u] >= max_deg_u or node_degrees[v] >= max_deg_v:
            return None

        merged = _merge_groups(grp_u, grp_v)
        new_ids = merged.node_ids
        idx_u = int(np.where(new_ids == u)[0][0])
        idx_v = int(np.where(new_ids == v)[0][0])

        # Set adjacency between u and v from global relation_scores
        rel_vec = relation_scores[u, v, :]
        merged.adjacency_matrix[idx_u, idx_v, :] = rel_vec
        merged.adjacency_matrix[idx_v, idx_u, :] = rel_vec

        node_degrees[u] += 1
        node_degrees[v] += 1
        return merged

    return merge_fn


__all__ = [
    "InstanceGroup",
    "group_keypoints_into_instances",
    "MergeFn",
    "make_merge_fn_max_label",
    "make_merge_fn_max_degree_per_label",
]
