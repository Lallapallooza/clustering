"""Hungarian label alignment for visualizing predicted labels against ground truth.

When comparing predicted cluster labels to ground-truth labels, the integer IDs
are arbitrary: ``gt == [0, 0, 1, 1]`` and ``pred == [5, 5, 3, 3]`` describe the
same partition. Coloring the two panels by raw IDs would show different colors
even for perfect predictions. The fix is Hungarian matching on the contingency
matrix: pick the permutation of predicted IDs that maximizes the number of
points landing in the same (gt, pred) cell.

Noise points (labelled ``-1``) are treated as a distinct outlier marker and are
passed through unchanged on both sides. Only the intersection of non-noise
points drives the contingency matrix; the matching itself operates only on
real cluster IDs.

Rectangular contingency matrices (``k_pred != k_gt``) are handled natively by
``scipy.optimize.linear_sum_assignment``. Unmatched predicted IDs when
``k_pred > k_gt`` retain their original integer so the output is always
well-defined (the vis layer colors unmatched IDs from the palette as if they
were new clusters). The degenerate all-noise-on-both-sides case produces an
empty ``(0, 0)`` contingency matrix; SciPy returns ``([], [])`` cleanly and we
surface an identity remap.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_to_ground_truth(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap ``pred_labels`` so integer IDs match ``gt_labels`` where possible.

    Applies Hungarian matching on the negated contingency matrix over the
    intersection of non-noise points (``-1`` is preserved as ``-1`` on both
    sides). The matching maximises the sum of co-occurrence counts across the
    chosen (gt, pred) pairs, so the returned ``aligned_pred_labels`` renders
    with the same colors as ``gt_labels`` when the prediction is a pure
    permutation.

    @param gt_labels    Ground-truth labels, 1-D integer array of length ``n``.
    @param pred_labels  Predicted labels, 1-D integer array of length ``n``.
    @returns            ``(aligned_pred_labels, remap_vector)``:

        - ``aligned_pred_labels`` has the same shape and dtype as
          ``pred_labels``; values of ``-1`` are preserved. Unmatched predicted
          IDs when ``k_pred > k_gt`` retain their original integer.
        - ``remap_vector`` is an ``int32`` array of length
          ``max(max(pred_labels, default=-1) + 1, 0)``. ``remap_vector[i]`` is
          the gt-id that pred-id ``i`` was assigned to, or ``i`` itself when
          that pred-id was never matched. Pred-ids that never appear in
          ``pred_labels`` (if ``pred_labels`` has gaps) get identity too.

    The function handles:

    - Identity (``aligned == gt`` when ``pred == gt``).
    - Pure permutation (``aligned == gt``, remap_vector is the permutation).
    - Rectangular (``k_pred != k_gt``): LSA handles it natively.
    - All-noise on one side: identity remap; ``-1`` preserved on both sides.
    - All-noise on both sides: LSA on a ``(0, 0)`` cost matrix returns
      ``([], [])`` cleanly; ``pred_labels`` is returned as-is.
    - kmeans-style no-noise input: the non-noise mask is a no-op; standard
      contingency LSA applies.
    """
    gt = np.asarray(gt_labels)
    pred = np.asarray(pred_labels)
    if gt.shape != pred.shape:
        raise ValueError(
            f"gt_labels shape {gt.shape} must match pred_labels shape {pred.shape}"
        )
    if gt.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {gt.shape}")

    # The remap vector is indexed by pred-id. Size it to hold every pred-id we
    # could encounter; default is identity so any id we never see still maps to
    # itself.
    max_pred = int(pred.max()) if pred.size > 0 else -1
    remap_len = max(max_pred + 1, 0)
    remap_vector = np.arange(remap_len, dtype=np.int32)

    # Intersection of non-noise points: Hungarian matching only sees real
    # cluster IDs. -1 passes through unchanged on both sides below.
    non_noise_mask = (gt != -1) & (pred != -1)
    gt_core = gt[non_noise_mask]
    pred_core = pred[non_noise_mask]

    # Build the contingency matrix. np.unique preserves the mapping
    # pred_id -> contingency column so we can translate back after LSA.
    gt_uniq = np.unique(gt_core) if gt_core.size > 0 else np.array([], dtype=gt.dtype)
    pred_uniq = (
        np.unique(pred_core) if pred_core.size > 0 else np.array([], dtype=pred.dtype)
    )
    k_gt = int(gt_uniq.size)
    k_pred = int(pred_uniq.size)

    # The all-noise-on-both-sides branch: the contingency matrix is (0, 0).
    # SciPy's linear_sum_assignment returns (empty, empty) here without
    # raising; the assertion below pins that contract for future SciPy
    # upgrades in the test suite. Output is pred as-is, remap is identity.
    if k_gt == 0 or k_pred == 0:
        aligned = pred.astype(pred.dtype, copy=True)
        return aligned, remap_vector

    # Contingency[g, p] = count of points with gt=g and pred=p on the
    # non-noise intersection. Built via a fast flat bincount instead of a
    # Python nested loop.
    gt_to_row = {int(v): i for i, v in enumerate(gt_uniq)}
    pred_to_col = {int(v): j for j, v in enumerate(pred_uniq)}
    rows = np.fromiter(
        (gt_to_row[int(v)] for v in gt_core), dtype=np.intp, count=gt_core.size
    )
    cols = np.fromiter(
        (pred_to_col[int(v)] for v in pred_core),
        dtype=np.intp,
        count=pred_core.size,
    )
    contingency = np.zeros((k_gt, k_pred), dtype=np.int64)
    np.add.at(contingency, (rows, cols), 1)

    # linear_sum_assignment minimizes; we want maximum co-occurrence, so
    # negate. maximize=True is supported on modern SciPy but the negation
    # form works across every release shipping in this repo's dep stack.
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Translate matched rows/cols back to label space. Each matched pair
    # (pred_uniq[col] -> gt_uniq[row]) becomes an entry in remap_vector.
    # Unmatched pred-ids keep their identity mapping from the default above.
    for r, c in zip(row_ind, col_ind, strict=True):
        pred_id = int(pred_uniq[c])
        gt_id = int(gt_uniq[r])
        remap_vector[pred_id] = gt_id

    # Build aligned output. Non-noise points get remapped via the vector;
    # noise stays -1. We use the remap_vector directly as a lookup table for
    # the non-noise slice instead of building a Python dict, so large inputs
    # stay fast.
    aligned = pred.astype(pred.dtype, copy=True)
    if remap_len > 0:
        non_noise_indices = np.where(pred != -1)[0]
        if non_noise_indices.size > 0:
            aligned[non_noise_indices] = remap_vector[pred[non_noise_indices]]
    return aligned, remap_vector


__all__ = ["align_to_ground_truth"]
