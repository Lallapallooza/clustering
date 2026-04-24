from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from pybench.charts.label_align import align_to_ground_truth


def test_align_identity() -> None:
    """aligned == gt when inputs are identical."""
    gt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    pred = gt.copy()
    aligned, remap = align_to_ground_truth(gt, pred)
    assert np.array_equal(aligned, gt)
    # Remap is an identity permutation on the observed ids.
    assert int(remap[0]) == 0
    assert int(remap[1]) == 1
    assert int(remap[2]) == 2


def test_align_pure_permutation_recovers_gt() -> None:
    """A pure label permutation round-trips back to gt after alignment."""
    gt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    # Permutation: 0->2, 1->0, 2->1.
    perm = {0: 2, 1: 0, 2: 1}
    pred = np.array([perm[int(v)] for v in gt], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    assert np.array_equal(aligned, gt)
    # Remap encodes the inverse permutation: pred-id 2 -> gt-id 0, etc.
    assert int(remap[2]) == 0
    assert int(remap[0]) == 1
    assert int(remap[1]) == 2


def test_align_rectangular_k_pred_gt_k_gt() -> None:
    """k_pred > k_gt: runs, no raise, every matched pred-id maps to a gt-id."""
    # gt has 2 clusters, pred has 3 clusters.
    gt = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # Each point ends up at a valid gt-id (0 or 1) or at its original pred-id
    # when unmatched. Matched pred-ids are in remap_vector at those indices.
    assert aligned.shape == gt.shape
    # At least two pred-ids should have been matched to gt-ids.
    matched = {int(i): int(remap[i]) for i in range(remap.size) if remap[i] in (0, 1)}
    assert len(matched) >= 2


def test_align_rectangular_k_gt_gt_k_pred() -> None:
    """k_gt > k_pred: runs, no raise, matching is valid on the smaller side."""
    # gt has 3 clusters, pred has 2.
    gt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    pred = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # Two pred-ids exist; both should be remapped to a gt-id.
    assert int(remap[0]) in (0, 1, 2)
    assert int(remap[1]) in (0, 1, 2)
    # Aligned output contains only gt-space ids for non-noise.
    uniq = set(int(v) for v in aligned)
    assert uniq.issubset({0, 1, 2})


def test_align_all_noise_one_side() -> None:
    """Noise on the pred side only: aligned preserves -1 and matched gt survives."""
    gt = np.array([0, 0, 1, 1], dtype=np.int32)
    pred = np.array([-1, -1, -1, -1], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # All pred are noise -> remap is identity (empty), aligned == pred.
    assert np.array_equal(aligned, pred)
    # -1 is NEVER in the remap_vector domain (indices 0..max(pred)+1).
    # With pred = [-1, -1, -1, -1], max(pred) = -1, so remap has length 0.
    assert remap.size == 0


def test_align_all_noise_both_sides() -> None:
    """LSA on (0,0) must not raise; aligned == pred, remap is identity."""
    gt = np.array([-1, -1, -1], dtype=np.int32)
    pred = np.array([-1, -1, -1], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    assert np.array_equal(aligned, pred)
    assert remap.size == 0


def test_align_no_noise_standard_case() -> None:
    """k-means-style: no -1 on either side; standard contingency LSA."""
    gt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    # Deliberately shuffled pred ids 7, 4, 9.
    pred = np.array([7, 7, 4, 4, 9, 9], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # Aligned maps perfectly back to gt.
    assert np.array_equal(aligned, gt)
    # Each observed pred-id has a matched gt-id in remap.
    assert int(remap[7]) == 0
    assert int(remap[4]) == 1
    assert int(remap[9]) == 2


def test_align_remap_vector_shape() -> None:
    """remap_vector length is max(pred) + 1 (or 0 if pred has no non-noise)."""
    gt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    pred = np.array([2, 2, 5, 5, 8, 8], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    assert remap.shape == (9,)  # indices 0..8
    assert remap.dtype == np.int32


def test_align_preserves_minus_one_in_output() -> None:
    """-1 in pred stays -1 in aligned; -1 in gt has no influence on remap."""
    gt = np.array([0, 0, 1, 1, -1, -1], dtype=np.int32)
    pred = np.array([3, 3, 4, 4, -1, -1], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # The two noise points stay noise in aligned.
    assert aligned[4] == -1
    assert aligned[5] == -1
    # Non-noise pred-ids were mapped to gt-ids, so aligned's first four
    # entries match gt's first four.
    assert np.array_equal(aligned[:4], gt[:4])


def test_align_mismatched_shapes_raises() -> None:
    """Shape mismatch between gt and pred is a clear error."""
    with pytest.raises(ValueError, match="shape"):
        align_to_ground_truth(
            np.array([0, 0, 1], dtype=np.int32),
            np.array([0, 0, 1, 1], dtype=np.int32),
        )


def test_align_noise_mask_intersection_only() -> None:
    """Only points where BOTH sides are non-noise drive the contingency matrix."""
    # gt clusters 0 and 1; pred has id 5 at the noise-in-gt positions, which
    # must NOT influence the 5 -> gt assignment (those points are excluded).
    gt = np.array([0, 0, -1, -1, 1, 1], dtype=np.int32)
    pred = np.array([5, 5, 5, 5, 7, 7], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # The 5 <-> (0, in the intersection) count is 2. So remap[5] = 0.
    assert int(remap[5]) == 0
    # 7 <-> 1 count is 2; remap[7] = 1.
    assert int(remap[7]) == 1
    # Points where gt = -1 still have pred == 5, which is remapped to 0.
    # That is intentional per the spec: we remap pred labels uniformly;
    # the caller handles the gt=-1 regions via its own masks.
    assert int(aligned[2]) == 0
    assert int(aligned[3]) == 0


def test_align_scipy_lsa_on_zero_matrix_returns_empty() -> None:
    """Sanity check: scipy's LSA on a (0, 0) cost matrix returns empty arrays
    without raising. This backstops our all-noise-both-sides branch."""
    cost = np.zeros((0, 0), dtype=np.int64)
    rows, cols = linear_sum_assignment(cost)
    assert rows.size == 0
    assert cols.size == 0


def test_align_large_input_vectorized() -> None:
    """Smoke test at realistic size (50k points, 20 clusters)."""
    rng = np.random.default_rng(42)
    n = 50_000
    k = 20
    gt = rng.integers(0, k, size=n).astype(np.int32)
    # Pred = gt with a rotation on the label IDs.
    pred = ((gt + 7) % k).astype(np.int32)
    aligned, _ = align_to_ground_truth(gt, pred)
    # After Hungarian matching, the aligned labels must equal gt.
    assert np.array_equal(aligned, gt)


def test_align_returns_same_dtype_as_pred() -> None:
    """aligned.dtype matches pred.dtype (int32 in / int32 out)."""
    gt = np.array([0, 1, 2], dtype=np.int32)
    pred = np.array([1, 2, 0], dtype=np.int32)
    aligned, _ = align_to_ground_truth(gt, pred)
    assert aligned.dtype == np.int32


def test_align_unmatched_pred_id_keeps_original() -> None:
    """When k_pred > k_gt, the extra pred cluster keeps its original integer."""
    # gt has 2 clusters, pred has 3. Cluster 2 in pred has no gt counterpart
    # that survives matching (any row matches only once).
    gt = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int32)
    pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    aligned, remap = align_to_ground_truth(gt, pred)
    # 0 and 1 should match to gt 0 and 1 (one-to-one by count).
    matched_to_gt = {int(remap[i]) for i in (0, 1)}
    assert matched_to_gt == {0, 1}
    # The unmatched pred-id is at remap[2]; its value is either 2 (identity)
    # or some unmatched gt slot. Since k_gt=2, there is no free gt row, so
    # unmatched pred-id keeps its identity value.
    assert int(remap[2]) == 2
