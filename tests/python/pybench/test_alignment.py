"""Tests for @c pybench.alignment."""

from __future__ import annotations

import numpy as np
import pytest

from pybench.alignment import AVX2_ALIGN, as_aligned, empty_aligned


def test_as_aligned_is_32_byte_aligned() -> None:
    raw = np.random.randn(2000, 17).astype(np.float32)
    aligned = as_aligned(raw)
    assert aligned.ctypes.data % AVX2_ALIGN == 0
    assert aligned.shape == raw.shape
    assert aligned.dtype == np.float32
    assert aligned.flags.c_contiguous
    np.testing.assert_array_equal(aligned, raw)


def test_as_aligned_returns_input_when_already_aligned() -> None:
    pre = empty_aligned((1000, 128))
    pre.fill(3.0)
    out = as_aligned(pre)
    assert out is pre


def test_as_aligned_copies_when_dtype_mismatches() -> None:
    raw = np.random.randn(100, 32).astype(np.float64)
    aligned = as_aligned(raw)
    assert aligned.dtype == np.float32
    assert aligned.ctypes.data % AVX2_ALIGN == 0
    np.testing.assert_allclose(aligned, raw.astype(np.float32))


def test_empty_aligned_shape_and_alignment() -> None:
    arr = empty_aligned((500, 64), dtype=np.float32)
    assert arr.shape == (500, 64)
    assert arr.dtype == np.float32
    assert arr.ctypes.data % AVX2_ALIGN == 0
    assert arr.flags.c_contiguous


def test_alignment_must_be_multiple_of_itemsize() -> None:
    with pytest.raises(ValueError):
        as_aligned(np.ones(16, dtype=np.float32), alignment=3)
    with pytest.raises(ValueError):
        empty_aligned((4,), dtype=np.float32, alignment=3)
