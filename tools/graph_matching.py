import numpy as np
from typing import Dict, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
import numba
import pygmtools as pygm
from scipy.optimize import quadratic_assignment

# --------------------------------------------------------------------------------------------
# Linear Assignment

def linear_assignment_matching(
    gt_coords: np.ndarray, 
    gt_labels: np.ndarray, 
    pred_coords: np.ndarray, 
    pred_labels: np.ndarray, 
    label_mismatch_cost: float = 1e9
    ) -> Dict[int, int]:
    """
    Match coordinates using the Hungarian algorithm (linear assignment problem).
    When labels don't match, a very high cost is assigned to discourage those assignments.
    
    Args:
    gt_coords (np.ndarray): First set of coordinates, shape (N, 2)
    gt_labels (np.ndarray): Labels for first set, shape (N,)
    pred_coords (np.ndarray): Second set of coordinates, shape (M, 2)
    pred_labels (np.ndarray): Labels for second set, shape (M,)
    label_mismatch_cost (float): High cost added when labels don't match
    
    Returns:
    dict: Mapping from indices in first set to matched indices in second set
    """

    # Vectorized cost matrix computation
    gt_coords_exp = gt_coords[:, np.newaxis, :]  # (n_gt, 1, 2)
    pred_coords_exp = pred_coords[np.newaxis, :, :]  # (1, n_pred, 2)
    dists = np.linalg.norm(gt_coords_exp - pred_coords_exp, axis=2)  # (n_gt, n_pred)

    gt_labels_exp = gt_labels[:, np.newaxis]
    pred_labels_exp = pred_labels[np.newaxis, :]
    label_mismatch = (gt_labels_exp != pred_labels_exp)
    cost_matrix = dists + label_mismatch_cost * label_mismatch.astype(np.float32)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matchings: Dict[int, int] = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    return matchings

def linear_assignment_matching_with_centroid_alignment(
    gt_coords: np.ndarray,
    gt_labels: np.ndarray,
    pred_coords: np.ndarray,
    pred_labels: np.ndarray,
    label_mismatch_cost: float = 1e9
) -> Dict[int, int]:
    """
    Perform linear assignment matching after aligning prediction coordinates to ground truth centroids.

    Args:
        gt_coords (np.ndarray): Ground truth coordinates, shape (N, 2)
        gt_labels (np.ndarray): Ground truth labels, shape (N,)
        pred_coords (np.ndarray): Predicted coordinates, shape (M, 2)
        pred_labels (np.ndarray): Predicted labels, shape (M,)
        label_mismatch_cost (float): Cost for label mismatch

    Returns:
        Dict[int, int]: Mapping from ground truth index to predicted index
    """
    # Align centroids (translation only)
    delta = gt_coords.mean(axis=0) - pred_coords.mean(axis=0)
    pred_coords_aligned = pred_coords + delta

    return linear_assignment_matching(
        gt_coords, gt_labels, pred_coords_aligned, pred_labels, label_mismatch_cost=label_mismatch_cost
    )



# --------------------------------------------------------------------------------------------
# Graph Matching with pygmtools

def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-hot encode integer labels -> (N, C). 
    """
    id2idx = {int(c): i for i, c in enumerate(classes)}
    C = len(classes)
    out = np.zeros((len(labels), C), dtype=float)
    for n, c in enumerate(labels):
        out[n, id2idx[int(c)]] = 1.0
    return out


def _relations_to_sparse(rel_mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized: Convert dense R-channel relation tensor (N,N,R) to pygmtools sparse form.
    Returns (conn: (E, 2), edge_feat: (E, R), E_count)
    Only edges with any positive relation entry are kept. Self-edges are skipped.
    """
    N, _, R = rel_mats.shape
    # Mask for off-diagonal (no self-edges)
    mask = ~np.eye(N, dtype=bool)
    # Find all (i, j) where any relation is positive and i != j
    any_rel = np.any(rel_mats > 0, axis=2)
    valid = np.where(mask & any_rel)
    if valid[0].size == 0:
        conn = np.zeros((0, 2), dtype=int)
        edge_feat = np.zeros((0, R), dtype=float)
        ne = np.array([0], dtype=np.int64)
    else:
        conn = np.stack(valid, axis=1)  # (E, 2)
        edge_feat = rel_mats[valid[0], valid[1], :].astype(float)  # (E, R)
        ne = np.array([edge_feat.shape[0]], dtype=np.int64)
    return conn, edge_feat, ne


def pygm_graph_matching(
    gt_labels: np.ndarray,               # (N1,)
    gt_relation_matrices: np.ndarray,    # (N1, N1, R)
    pred_labels: np.ndarray,             # (N2,)
    pred_scores: np.ndarray,             # (N2,)
    pred_relation_matrices: np.ndarray,  # (N2, N2, R)
    solver: dict = {'fn': 'rrwm'}
) -> Tuple[Dict[int, int], Optional[np.ndarray]]:
    """
    Graph matching with pygmtools:

    Args:
        solver: dict with keys 'fn' (solver function name as string) and 'args' (dict of additional kwargs).

    Returns:
        match_dict where match_dict maps gt index -> pred index.
    """
    pygm.set_backend('numpy')

    # Early exit: no label overlap -> no valid matches
    if len(np.intersect1d(np.unique(gt_labels), np.unique(pred_labels))) == 0:
        return {}

    N1, N2 = len(gt_labels), len(pred_labels)
    n1 = np.array([N1], dtype=np.int64)
    n2 = np.array([N2], dtype=np.int64)

    # ----- Node features (so inner-product yields: 1[label match] * pred_score_j) -----
    all_classes = np.unique(np.concatenate([gt_labels, pred_labels]))
    F1 = _one_hot(gt_labels, classes=all_classes).astype(float)      # (N1, C)
    F2 = _one_hot(pred_labels, classes=all_classes).astype(float)    # (N2, C)
    F2 = (F2 * pred_scores.reshape(-1, 1)).astype(float)  # (N2, C)

    # ----- Edge features from relation tensors -----
    conn1, edge1, ne1 = _relations_to_sparse(gt_relation_matrices)
    conn2, edge2, ne2 = _relations_to_sparse(pred_relation_matrices)

    K = pygm.utils.build_aff_mat(
        F1, edge1, conn1,
        F2, edge2, conn2,
        n1=n1, ne1=ne1, n2=n2, ne2=ne2,
        node_aff_fn=pygm.utils.inner_prod_aff_fn,
        edge_aff_fn=pygm.utils.inner_prod_aff_fn,
    )

    # --- Solver selection ---
    solver_name = solver.get('fn', 'rrwm')
    solver_args = solver.get('args', {})

    solver_map = {
        'rrwm': pygm.classic_solvers.rrwm,
        'ipfp': pygm.classic_solvers.ipfp,
        'sm': pygm.classic_solvers.sm,
        'astar': pygm.classic_solvers.astar,
        'ngm': pygm.neural_solvers.ngm,
    }
    solver_fn = solver_map[solver_name]

    # Always pass K, n1, n2, and optionally x0
    solver_kwargs = dict(K=K, n1=n1, n2=n2)
    if 'x0' in solver_args:
        init = solver_args['x0']
        X0 = np.full((N1, N2), 1.0 / (N1 * N2), dtype=float)
        for i, j in init.items():
            X0[int(i), int(j)] = 1.0
        solver_args['x0'] = X0[None, ...]  # (1, N1, N2)

    solver_kwargs.update(solver_args)
    X_soft = solver_fn(**solver_kwargs)

    X = pygm.hungarian(X_soft)

    match = {int(i): int(j) for i, j in zip(*np.where(X > 0.5))}
    return match


# --------------------------------------------------------------------------------------------
# Graph Matching with SciPy QAP

def _build_full_perm(
    n: int,
    N1: int,
    initial_matching: Optional[Dict[int,int]],
) -> np.ndarray:
    p0 = np.full(n, -1, dtype=int)
    used_cols = np.zeros(n, dtype=bool)

    for g, p in initial_matching.items():
        if p0[g] == -1 and not used_cols[p]:
            p0[g] = p
            used_cols[p] = True

    available_cols = [c for c in range(n) if not used_cols[c]]

    for i in range(N1):
        if p0[i] != -1:
            continue
        p0[i] = available_cols[0]
        available_cols.remove(p0[i])

    for i in range(N1, n):
        p0[i] = available_cols.pop(0)

    assert set(p0.tolist()) == set(range(n)), "p0 is not a full permutation"
    return p0


def scipy_qap(
    gt_relation_matrices: np.ndarray,   # (N_gt, N_gt, R)
    pred_relation_matrices: np.ndarray, # (N_pred, N_pred, R)
    pred_scores: Optional[np.ndarray] , # (N_pred,) confidence scores in [0, 1)
    initial_matching: Optional[Dict[int, int]] = None,  # gt_idx -> pred_idx (seed)
    method: str = "faq",                # 'faq' or '2opt'
) -> Tuple[Dict[int,int], dict]:
    """
    SciPy QAP using predicted node scores to scale predicted adjacency edges, with initial permutation p0.
    - pred_scores multiplies predicted edge weights: B[i,j] *= (pred_scores[i] * pred_scores[j])
    - p0 is constructed from initial_matching and passed to the solver.

    Returns:
      mapping: {gt_idx: pred_idx}
    """
    adj_gt = np.max(gt_relation_matrices, axis=2).astype(float) # (N1, N2)
    adj_pred = np.max(pred_relation_matrices, axis=2).astype(float) # (N2, N2)

    N1 = adj_gt.shape[0]
    N2 = adj_pred.shape[0]

    # Scale predicted adjacency by product of node scores
    adj_pred = adj_pred * np.outer(pred_scores, pred_scores)  # (N2, N2)

    # pad to square n x n
    n = max(N1, N2)
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)
    A[:N1, :N1] = adj_gt
    B[:N2, :N2] = adj_pred

    if initial_matching is not None:
        p0 = _build_full_perm(n, N1, initial_matching)

        options = {}
        if method == "faq":
            P0_mat = np.zeros((n, n), dtype=float)
            for i in range(n):
                P0_mat[i, int(p0[i])] = 1.0
            options['P0'] = P0_mat
        elif method == "2opt":
            options['partial_guess'] = np.vstack([np.arange(n, dtype=int), p0]).T
        else:
            raise ValueError("method must be 'faq' or '2opt'")

    # Solve QAP
    options['maximize'] = True
    res = quadratic_assignment(A, B, method=method, options=options)

    perm = np.asarray(res.col_ind, dtype=int)  # row i of A -> column perm[i] of B

    mapping: Dict[int,int] = {}
    for i in range(N1):
        j = int(perm[i])
        if j < N2:
            mapping[i] = j

    return mapping


# -----------------------------------------------------------------------------
# Custom 2Opt-Algorithm


@numba.jit(nopython=True)
def compute_node_score_jit(gt_idx, pred_idx, gt_labels, pred_labels, pred_scores):
    """Compute score for a single node match (Numba-optimized)"""
    if gt_labels[gt_idx] != pred_labels[pred_idx]:
        return 0.0
    return pred_scores[pred_idx]

@numba.jit(nopython=True)
def compute_swap_improvement_jit(
    gt_idx, current_pred_idx,
    candidate_gt_idx, candidate_pred_idx,
    gt_labels, pred_labels, pred_scores,
    pred_relation_matrices,
    gt_to_pred, pred_to_gt,
    node_scores, edge_scores, edge_counts
):
    """
    Compute the improvement in score if we swap the matches (Numba-optimized).
    """
    # Save current state
    orig_gt_to_pred = gt_to_pred.copy()
    orig_pred_to_gt = pred_to_gt.copy()
    
    # Apply the swap temporarily
    gt_to_pred[gt_idx] = candidate_pred_idx
    pred_to_gt[current_pred_idx] = -1
    if candidate_gt_idx >= 0:
        gt_to_pred[candidate_gt_idx] = current_pred_idx
        pred_to_gt[current_pred_idx] = candidate_gt_idx
    pred_to_gt[candidate_pred_idx] = gt_idx
    
    # Calculate node score changes
    new_gt_score = compute_node_score_jit(gt_idx, candidate_pred_idx, gt_labels, pred_labels, pred_scores)
    node_improvement = new_gt_score - node_scores[gt_idx]
    if candidate_gt_idx >= 0:
        new_candidate_score = compute_node_score_jit(candidate_gt_idx, current_pred_idx, gt_labels, pred_labels, pred_scores)
        node_improvement += new_candidate_score - node_scores[candidate_gt_idx]
    
    # Calculate relation score changes
    relation_improvement = 0.0
    
    # Check relations involving gt_idx
    for k in range(edge_counts[gt_idx]):
        gt_j = int(edge_scores[gt_idx, k, 0])
        rel_type = int(edge_scores[gt_idx, k, 1])
        old_score = edge_scores[gt_idx, k, 2]
        
        pred_j = gt_to_pred[gt_j]
        
        # Skip if the other node is unmatched
        if pred_j < 0:
            continue
            
        # Calculate new relation score
        new_score = pred_relation_matrices[candidate_pred_idx, pred_j, rel_type]
        relation_improvement += new_score - old_score
    
    # Check relations where gt_idx is the target
    n_gt = len(gt_labels)
    for gt_i in range(n_gt):
        for k in range(edge_counts[gt_i]):
            gt_j = int(edge_scores[gt_i, k, 0])
            if gt_j != gt_idx:
                continue
                
            rel_type = int(edge_scores[gt_i, k, 1])
            old_score = edge_scores[gt_i, k, 2]
            pred_i = gt_to_pred[gt_i]
            
            # Skip if the other node is unmatched
            if pred_i < 0:
                continue
                
            # Calculate new relation score
            new_score = pred_relation_matrices[pred_i, candidate_pred_idx, rel_type]
            relation_improvement += new_score - old_score
    
    # If candidate_gt_idx is valid, also check its relations
    if candidate_gt_idx >= 0:
        # Check relations involving candidate_gt_idx
        for k in range(edge_counts[candidate_gt_idx]):
            gt_j = int(edge_scores[candidate_gt_idx, k, 0])
            rel_type = int(edge_scores[candidate_gt_idx, k, 1])
            old_score = edge_scores[candidate_gt_idx, k, 2]
            
            pred_j = gt_to_pred[gt_j]
            
            # Skip if the other node is unmatched
            if pred_j < 0:
                continue
                
            # Calculate new relation score
            new_score = pred_relation_matrices[current_pred_idx, pred_j, rel_type]
            relation_improvement += new_score - old_score
        
        # Check relations where candidate_gt_idx is the target
        for gt_i in range(n_gt):
            for k in range(edge_counts[gt_i]):
                gt_j = int(edge_scores[gt_i, k, 0])
                if gt_j != candidate_gt_idx:
                    continue
                    
                rel_type = int(edge_scores[gt_i, k, 1])
                old_score = edge_scores[gt_i, k, 2]
                pred_i = gt_to_pred[gt_i]
                
                # Skip if the other node is unmatched
                if pred_i < 0:
                    continue
                    
                # Calculate new relation score
                new_score = pred_relation_matrices[pred_i, current_pred_idx, rel_type]
                relation_improvement += new_score - old_score
    
    # Restore original state
    for i in range(len(gt_to_pred)):
        gt_to_pred[i] = orig_gt_to_pred[i]
    for i in range(len(pred_to_gt)):
        pred_to_gt[i] = orig_pred_to_gt[i]
    
    # Combine node and relation improvements
    return node_improvement + relation_improvement

@numba.jit(nopython=True)
def apply_swap_jit(
    gt_idx, current_pred_idx, candidate_gt_idx, candidate_pred_idx, 
    gt_to_pred, pred_to_gt, node_scores, edge_scores, edge_counts,
    pred_relation_matrices,
    new_gt_score, new_candidate_score=0.0
):
    """
    Apply a swap to the current matching and update node scores and edge scores (Numba-optimized)
    """
    # Update the matchings
    gt_to_pred[gt_idx] = candidate_pred_idx
    pred_to_gt[current_pred_idx] = -1
    
    if candidate_gt_idx >= 0:
        gt_to_pred[candidate_gt_idx] = current_pred_idx
        pred_to_gt[current_pred_idx] = candidate_gt_idx
        # Update node score for candidate_gt_idx
        node_scores[candidate_gt_idx] = new_candidate_score
    
    pred_to_gt[candidate_pred_idx] = gt_idx
    
    # Update node score for gt_idx
    node_scores[gt_idx] = new_gt_score
    
    # Update edge scores for all relations involving gt_idx
    for k in range(edge_counts[gt_idx]):
        gt_j = int(edge_scores[gt_idx, k, 0])
        rel_type = int(edge_scores[gt_idx, k, 1])
        pred_j = gt_to_pred[gt_j]
        
        if pred_j >= 0:
            new_score = pred_relation_matrices[candidate_pred_idx, pred_j, rel_type]
            edge_scores[gt_idx, k, 2] = new_score
    
    # Update edge scores where gt_idx is the target
    n_gt = len(gt_to_pred)
    for gt_i in range(n_gt):
        for k in range(edge_counts[gt_i]):
            gt_j = int(edge_scores[gt_i, k, 0])
            if gt_j != gt_idx:
                continue
                
            rel_type = int(edge_scores[gt_i, k, 1])
            pred_i = gt_to_pred[gt_i]
            
            if pred_i >= 0:
                new_score = pred_relation_matrices[pred_i, candidate_pred_idx, rel_type]
                edge_scores[gt_i, k, 2] = new_score
    
    # If candidate_gt_idx is valid, also update its relations
    if candidate_gt_idx >= 0:
        # Update relations involving candidate_gt_idx
        for k in range(edge_counts[candidate_gt_idx]):
            gt_j = int(edge_scores[candidate_gt_idx, k, 0])
            rel_type = int(edge_scores[candidate_gt_idx, k, 1])
            pred_j = gt_to_pred[gt_j]
            
            if pred_j >= 0:
                new_score = pred_relation_matrices[current_pred_idx, pred_j, rel_type]
                edge_scores[candidate_gt_idx, k, 2] = new_score
        
        # Update relations where candidate_gt_idx is the target
        for gt_i in range(n_gt):
            for k in range(edge_counts[gt_i]):
                gt_j = int(edge_scores[gt_i, k, 0])
                if gt_j != candidate_gt_idx:
                    continue
                    
                rel_type = int(edge_scores[gt_i, k, 1])
                pred_i = gt_to_pred[gt_i]
                
                if pred_i >= 0:
                    new_score = pred_relation_matrices[pred_i, current_pred_idx, rel_type]
                    edge_scores[gt_i, k, 2] = new_score

@numba.jit(nopython=True)
def custom_2opt_jit(
    gt_labels, 
    gt_relation_matrices,
    pred_labels, 
    pred_scores, 
    pred_relation_matrices,
    gt_to_pred_init,  
    pred_to_gt_init,
    max_relations,
    num_iterations=30,
    strategy=0  # 0 for greedy, 1 for steepest_descent
) -> tuple:
    """
    Performs 2-opt refinement of matching between ground truth and predictions.
    Numba-optimized version using arrays instead of dictionaries.
    
    Args:
        gt_labels: Ground truth keypoint labels (numpy array)
        gt_relation_matrices: Ground truth relation matrices (numpy array)
        pred_labels: Predicted keypoint labels (numpy array)
        pred_scores: Confidence scores for predictions (numpy array)
        pred_relation_matrices: Predicted relation matrices (numpy array)
        gt_to_pred_init: Initial gt->pred mapping array (pre-filled, -1 for unmatched)
        pred_to_gt_init: Initial pred->gt mapping array (pre-filled, -1 for unmatched)
        max_relations: Maximum number of relations per node to store
        num_iterations: Maximum number of refinement iterations
        strategy: 0 for "greedy", 1 for "steepest_descent"
        
    Returns:
        tuple: (gt_to_pred mapping array, num_iterations_performed)
    """
    n_gt = len(gt_labels)
    n_pred = len(pred_labels)
    
    # Initialize from initial mapping arrays
    gt_to_pred = gt_to_pred_init.copy()
    pred_to_gt = pred_to_gt_init.copy()
    node_scores = np.zeros(n_gt, dtype=np.float32)
    
    # Calculate initial node scores
    for i in range(n_gt):
        pred_idx = gt_to_pred[i]
        if pred_idx >= 0:
            node_scores[i] = compute_node_score_jit(
                i, pred_idx, gt_labels, pred_labels, pred_scores
            )
    
    # Precompute edge scores as array: edge_scores[gt_idx, relation_idx, :] = [gt_j, rel_type, score]
    edge_scores = np.full((n_gt, max_relations, 3), -1, dtype=np.float32)
    edge_counts = np.zeros(n_gt, dtype=np.int32)
    n_gt = gt_relation_matrices.shape[0]
    n_rel = gt_relation_matrices.shape[2]
    rel_types = np.zeros((n_gt, n_gt), dtype=np.int32)
    rel_values = np.zeros((n_gt, n_gt), dtype=gt_relation_matrices.dtype)
    for i in range(n_gt):
        for j in range(n_gt):
            max_val = gt_relation_matrices[i, j, 0]
            max_idx = 0
            for r in range(1, n_rel):
                val = gt_relation_matrices[i, j, r]
                if val > max_val:
                    max_val = val
                    max_idx = r
            rel_types[i, j] = max_idx
            rel_values[i, j] = max_val
    
    for i in range(n_gt):
        count = 0
        for j in range(n_gt):
            if i == j:
                continue
                
            if rel_values[i, j] > 0:
                if count >= max_relations:
                    break
                    
                best_rel_type = int(rel_types[i, j])
                pred_i = gt_to_pred[i]
                pred_j = gt_to_pred[j]
                score = 0.0
                
                if pred_i >= 0 and pred_j >= 0:
                    score = pred_relation_matrices[pred_i, pred_j, best_rel_type]
                
                edge_scores[i, count, 0] = j              # Target GT node
                edge_scores[i, count, 1] = best_rel_type  # Relation type
                edge_scores[i, count, 2] = score          # Current score
                count += 1
        
        edge_counts[i] = count
    
    # Perform iterative refinement
    for iteration in range(num_iterations):
        # Track whether any improvement was made in this iteration
        any_improvement = False
        
        # For steepest descent strategy
        best_improvement = 0.0
        best_swap = None
        
        # Try all possible swaps
        for gt_idx in range(n_gt):
            current_pred_idx = gt_to_pred[gt_idx]
                
            # Try swapping with each compatible prediction
            for candidate_pred_idx in range(n_pred):
                if candidate_pred_idx == current_pred_idx:
                    continue
                
                # Check label compatibility
                if gt_labels[gt_idx] != pred_labels[candidate_pred_idx]:
                    continue

                # Get the GT point currently matched to candidate_pred_idx (if any)
                candidate_gt_idx = pred_to_gt[candidate_pred_idx]
                
                # Calculate improvement from this swap
                improvement = compute_swap_improvement_jit(
                    gt_idx, current_pred_idx,
                    candidate_gt_idx, candidate_pred_idx,
                    gt_labels, pred_labels, pred_scores,
                    pred_relation_matrices,
                    gt_to_pred, pred_to_gt,
                    node_scores, edge_scores, edge_counts
                )
                
                # Greedy strategy: apply swap immediately if it improves
                if strategy == 0 and improvement > 0:
                    # Calculate new node scores for the swapped nodes
                    new_gt_score = compute_node_score_jit(
                        gt_idx, candidate_pred_idx, gt_labels, pred_labels, pred_scores
                    )
                    
                    # Apply the swap and update node scores
                    new_candidate_score = 0.0
                    if candidate_gt_idx >= 0:
                        new_candidate_score = compute_node_score_jit(
                            candidate_gt_idx, current_pred_idx, gt_labels, pred_labels, pred_scores
                        )
                        
                    apply_swap_jit(
                        gt_idx, current_pred_idx, candidate_gt_idx, candidate_pred_idx,
                        gt_to_pred, pred_to_gt, node_scores, edge_scores, edge_counts,
                        pred_relation_matrices,
                        new_gt_score, new_candidate_score
                    )
                    any_improvement = True
                
                # Steepest descent strategy: remember best swap
                elif strategy == 1 and improvement > best_improvement:
                    best_improvement = improvement
                    
                    new_gt_score = compute_node_score_jit(
                        gt_idx, candidate_pred_idx, gt_labels, pred_labels, pred_scores
                    )
                    
                    new_candidate_score = 0.0
                    if candidate_gt_idx >= 0:
                        new_candidate_score = compute_node_score_jit(
                            candidate_gt_idx, current_pred_idx, gt_labels, pred_labels, pred_scores
                        )
                        
                    best_swap = (
                        gt_idx, current_pred_idx, candidate_gt_idx, candidate_pred_idx,
                        new_gt_score, new_candidate_score
                    )
        
        # For steepest descent, apply the best swap if we found one
        if strategy == 1 and best_swap is not None and best_improvement > 0:
            # Explicitly unpack values to avoid parameter ordering issues
            gt_idx, current_pred_idx, candidate_gt_idx, candidate_pred_idx, new_gt_score, new_candidate_score = best_swap
            apply_swap_jit(
                gt_idx, current_pred_idx, candidate_gt_idx, candidate_pred_idx,
                gt_to_pred, pred_to_gt, node_scores, edge_scores, edge_counts,
                pred_relation_matrices,
                new_gt_score, new_candidate_score
            )
            any_improvement = True
        
        # If no improvement was found, we're done
        if not any_improvement:
            break

    return gt_to_pred, iteration + 1

def custom_2opt(
    gt_labels, 
    gt_relation_matrices,
    pred_labels, 
    pred_scores, 
    pred_relation_matrices,
    initial_matching,
    num_iterations=30,
    strategy="steepest_descent"
) -> tuple:
    """
    Refine an initial matching between ground truth and predictions using a custom 2-opt algorithm.

    Args:
        gt_labels (np.ndarray): Ground truth keypoint labels.
        gt_relation_matrices (np.ndarray): Ground truth relation matrices.
        pred_labels (np.ndarray): Predicted keypoint labels.
        pred_scores (np.ndarray): Confidence scores for predictions.
        pred_relation_matrices (np.ndarray): Predicted relation matrices.
        initial_matching (dict): Initial mapping from gt index to pred index.
        num_iterations (int): Maximum number of refinement iterations.
        strategy (str): "greedy" or "steepest_descent".

    Returns:
        tuple: (final_matchings, iterations)
    """
    n_gt = len(gt_labels)
    n_pred = len(pred_labels)
    
    # Convert dictionary to arrays for Numba compatibility
    gt_to_pred_init = np.full(n_gt, -1, dtype=np.int32)
    pred_to_gt_init = np.full(n_pred, -1, dtype=np.int32)
    for gt_idx, pred_idx in initial_matching.items():
        gt_to_pred_init[gt_idx] = pred_idx
        pred_to_gt_init[pred_idx] = gt_idx
    
    # Infer max_relations from gt_relation_matrices
    n_gt = gt_relation_matrices.shape[0]
    # Mask out self-edges
    mask = ~np.eye(n_gt, dtype=bool)
    # Any relation > 0 (excluding self-edges)
    rel_mask = np.any(gt_relation_matrices > 0, axis=2) & mask
    # Count per node
    relations_per_node = np.sum(rel_mask, axis=1)
    max_relations = int(np.max(relations_per_node))

    gt_to_pred, iterations = custom_2opt_jit(
        gt_labels, 
        gt_relation_matrices,
        pred_labels, 
        pred_scores, 
        pred_relation_matrices,
        gt_to_pred_init,
        pred_to_gt_init,
        max_relations=max_relations,
        num_iterations=num_iterations,
        strategy= 1 if strategy == "steepest_descent" else 0
    )

    # Convert the final mapping back to a dictionary
    final_matchings = {}
    for i in range(len(gt_labels)):
        if gt_to_pred[i] >= 0:
            final_matchings[i] = int(gt_to_pred[i])
    return final_matchings, iterations

def compute_objective_value(
    gt_labels: np.ndarray,
    gt_relation_matrices: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    pred_relation_matrices: np.ndarray,
    matching: Dict[int, int]
) -> float:
    """
    Compute objective value for a matching as mean of assigned node scores + mean of assigned relation scores.
    
    Args:
        gt_labels: Ground truth labels
        gt_relation_matrices: Ground truth relation matrices (N1, N1, R)
        pred_labels: Predicted labels
        pred_scores: Predicted confidence scores
        pred_relation_matrices: Predicted relation matrices (N2, N2, R)
        matching: Dictionary mapping ground truth indices to prediction indices
    
    Returns:
        float: Objective value (higher is better)
    """
    if not matching:
        return 0.0
    
    # Compute mean node score
    node_scores = []
    for gt_idx, pred_idx in matching.items():
        if gt_labels[gt_idx] == pred_labels[pred_idx]:
            node_scores.append(pred_scores[pred_idx])
    
    node_score = np.sum(node_scores) / len(gt_labels) if node_scores else 0.0
    
    # Compute mean relation score
    relation_scores = []
    for gt_i, pred_i in matching.items():
        for gt_j, pred_j in matching.items():
            if gt_i == gt_j:
                continue
                
            # Find maximum relation score across all relation types
            for r in range(gt_relation_matrices.shape[2]):
                if gt_relation_matrices[gt_i, gt_j, r] > 0:
                    relation_scores.append(pred_relation_matrices[pred_i, pred_j, r])
    
    relation_score = np.sum(relation_scores) / np.sum(gt_relation_matrices > 0) if relation_scores else 0.0
    
    # Return combined score
    return node_score + relation_score


# -----------------------------------------------------------------------------------------------------------------------------------
# Sequential Matching with Multiple Algorithms

def sequential_matching(
    gt_labels: np.ndarray,
    gt_relation_matrices: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    pred_relation_matrices: np.ndarray,
    algorithm_sequence: list,
    initial_matching: Optional[Dict[int, int]] = None,
    gt_coords: Optional[np.ndarray] = None,
    pred_coords: Optional[np.ndarray] = None,
    return_intermediate: bool = False
) -> Dict[int, int]:
    """
    Apply a sequence of matching algorithms, keeping the best result at each step.

    Args:
        gt_labels: Ground truth labels
        gt_relation_matrices: Ground truth relation matrices (N1, N1, R)
        pred_labels: Predicted labels
        pred_scores: Predicted confidence scores
        pred_relation_matrices: Predicted relation matrices (N2, N2, R)
        algorithm_sequence: List of algorithm names to apply sequentially
        initial_matching: Optional initial matching to start with
        gt_coords: Optional ground truth coordinates for coordinate-based matching
        pred_coords: Optional prediction coordinates for coordinate-based matching
        return_intermediate: If True, return all intermediate matchings and scores

    Returns:
        Dict[int, int]: Best matching found, or (best_matching, intermediates) if return_intermediate
    """
    best_matching = initial_matching if initial_matching is not None else {}
    best_score = compute_objective_value(
        gt_labels, gt_relation_matrices, 
        pred_labels, pred_scores, pred_relation_matrices,
        best_matching
    )

    intermediates = []
    if return_intermediate:
        intermediates.append(("init", best_matching, best_score))

    for algorithm in algorithm_sequence:
        # Apply the selected algorithm
        if algorithm == "la":
            if gt_coords is None or pred_coords is None:
                print("  → Skipping 'la': coordinates not provided")
                continue
            current_matching = linear_assignment_matching(
                gt_coords, gt_labels, pred_coords, pred_labels
            )

        elif algorithm == "lac":
            if gt_coords is None or pred_coords is None:
                print("  → Skipping 'lac': coordinates not provided")
                continue
            current_matching = linear_assignment_matching_with_centroid_alignment(
                gt_coords, gt_labels, pred_coords, pred_labels
            )

        elif algorithm == "faq":
            current_matching = scipy_qap(
                gt_relation_matrices, 
                pred_relation_matrices,
                pred_scores,
                initial_matching=best_matching,
                method="faq"
            )

        elif algorithm == "2opt":
            current_matching = scipy_qap(
                gt_relation_matrices, 
                pred_relation_matrices,
                pred_scores,
                initial_matching=best_matching,
                method="2opt"
            )

        elif algorithm in ["rrwm", "ipfp", "sm"]:
            current_matching = pygm_graph_matching(
                gt_labels,
                gt_relation_matrices,
                pred_labels,
                pred_scores,
                pred_relation_matrices,
                solver={'fn': algorithm, 'args': {'x0': best_matching}}
            )

        elif algorithm == "our2opt":
            current_matching, _ = custom_2opt(
                gt_labels,
                gt_relation_matrices,
                pred_labels,
                pred_scores,
                pred_relation_matrices,
                initial_matching=best_matching
            )

        else:
            print(f"Unknown algorithm: {algorithm}, skipping")
            continue

        current_score = compute_objective_value(
            gt_labels, gt_relation_matrices, 
            pred_labels, pred_scores, pred_relation_matrices,
            current_matching
        )

        if current_score > best_score:
            best_matching = current_matching
            best_score = current_score

        if return_intermediate:
            intermediates.append((algorithm, best_matching, best_score))

    if return_intermediate:
        return best_matching, intermediates
    return best_matching