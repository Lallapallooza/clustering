"""AVX2-friendly alignment helpers for clustering inputs.

The @c _clustering binding's zero-copy borrow path is gated on 32-byte alignment so the AVX2
tiers that assume aligned loads on @c X fire without a defensive memcpy per call. NumPy's
default contiguous allocator on glibc x86_64 typically lands on 16-byte boundaries, so
passing a raw @c np.ndarray forces the binding down its aligned fallback copy. The helpers
below produce 32-byte aligned buffers using the canonical over-allocate-and-slice trick.

Typical usage::

    from pybench.alignment import as_aligned

    X = as_aligned(my_array)
    labels, centroids, inertia, n_iter, converged = _clustering.kmeans(X, k=16, ...)
"""

from __future__ import annotations

import numpy as np

AVX2_ALIGN: int = 32
"""Byte alignment expected by the @c _clustering binding's zero-copy borrow path."""


def _is_aligned(arr: np.ndarray, alignment: int) -> bool:
    return arr.ctypes.data % alignment == 0


def as_aligned(data: np.ndarray, alignment: int = AVX2_ALIGN) -> np.ndarray:
    """Return a C-contiguous float32 array whose first byte is aligned to @p alignment.

    When @p data already meets the alignment + contiguity + float32 contract, it is returned
    unchanged (zero-copy). Otherwise a fresh buffer is allocated with enough headroom to slice
    down to an aligned starting offset and @p data is copied into that slice with @c np.copyto.

    @param data       Input array; any dtype castable to @c float32.
    @param alignment  Required byte alignment of the first element. Must be a multiple of the
                      output dtype's itemsize.
    """
    if (
        data.flags.c_contiguous
        and data.dtype == np.float32
        and _is_aligned(data, alignment)
    ):
        return data
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    if alignment % itemsize != 0:
        raise ValueError(
            f"alignment {alignment} is not a multiple of itemsize {itemsize}"
        )
    headroom = alignment // itemsize
    buf = np.empty(data.size + headroom, dtype=dtype)
    offset = ((-buf.ctypes.data) % alignment) // itemsize
    # The reshape's base chains back to @c buf, so the sliced view keeps the parent alive.
    aligned = buf[offset : offset + data.size].reshape(data.shape)
    np.copyto(aligned, data, casting="same_kind")
    return aligned


def empty_aligned(
    shape: tuple[int, ...],
    dtype: np.dtype | type = np.float32,
    alignment: int = AVX2_ALIGN,
) -> np.ndarray:
    """Allocate an uninitialised aligned array of the given shape and dtype.

    Companion to @ref as_aligned for callers that want to fill the buffer themselves (e.g. a
    streaming loader or a generator-driven scatter).
    """
    itemsize = np.dtype(dtype).itemsize
    if alignment % itemsize != 0:
        raise ValueError(
            f"alignment {alignment} is not a multiple of itemsize {itemsize}"
        )
    headroom = alignment // itemsize
    size = 1
    for d in shape:
        size *= d
    buf = np.empty(size + headroom, dtype=dtype)
    offset = ((-buf.ctypes.data) % alignment) // itemsize
    return buf[offset : offset + size].reshape(shape)
