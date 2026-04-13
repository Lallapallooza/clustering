"""Python round-trip tests for the nanobind <-> NDArray zero-copy adapter.

Run with:

    uv run python tests/python/test_ndarray_adapter.py

The script imports the compiled _clustering extension (built by the
scikit-build-core editable install). It exits with status 0 when every
assertion passes and raises otherwise.
"""

from __future__ import annotations

import gc
import sys

import numpy as np

import _clustering as m


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def test_contig_borrow_zero_copy_and_mutable() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    rows, cols, is_mutable = m._inplace_increment_contig(a)
    _assert(rows == 2 and cols == 3, f"shape mismatch ({rows}, {cols})")
    _assert(is_mutable is True, "writable numpy array should map to m_mutable=True")
    # Increment is visible in the original numpy buffer -> zero-copy borrow confirmed.
    expected = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    _assert(
        np.array_equal(a, expected),
        f"in-place increment not visible: {a!r} != {expected!r}",
    )


def test_readonly_borrow_flips_mutable_flag() -> None:
    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    a.setflags(write=False)
    _assert(
        a.flags.writeable is False, "precondition: numpy array must be non-writeable"
    )
    is_mutable = m._borrow_is_mutable_readonly(a)
    _assert(is_mutable is False, "read-only borrow must produce m_mutable=False")


def test_owned_to_numpy_roundtrip_preserves_values() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = m._roundtrip_zero_copy(a)
    _assert(out.shape == (3, 2), f"expected (3, 2), got {out.shape}")
    _assert(out.dtype == np.float32, f"expected float32, got {out.dtype}")
    _assert(np.array_equal(out, a), f"roundtrip value mismatch: {out!r} != {a!r}")
    # out is a fresh Owned buffer, so it must not alias the input.
    _assert(out.ctypes.data != a.ctypes.data, "roundtrip output should not alias input")


def test_owned_to_numpy_capsule_keeps_buffer_alive() -> None:
    out = m._make_owned_array(4, 5)
    _assert(out.shape == (4, 5), f"expected (4, 5), got {out.shape}")
    # Drop every reference the test has to the module-side creation; the capsule should still
    # keep the C++ storage alive.
    gc.collect()
    expected = np.arange(20, dtype=np.float32).reshape(4, 5)
    _assert(np.array_equal(out, expected), f"capsule-backed values corrupted: {out!r}")
    # Mutating the returned array must not crash and must be observable.
    out[0, 0] = 999.0
    _assert(out[0, 0] == 999.0, "capsule-backed array should be writable")


def test_strided_borrow_uses_element_strides() -> None:
    # x[::2, :] on an (n, d) contiguous source has shape (n/2, d) with byte-stride
    # 2*d*sizeof(float) along axis 0 and sizeof(float) along axis 1.
    source = np.arange(24, dtype=np.float32).reshape(6, 4)
    view = source[::2, :]
    _assert(view.shape == (3, 4), f"precondition: expected (3, 4), got {view.shape}")
    _assert(
        view.strides == (32, 4),
        f"precondition: expected (32, 4) byte-strides, got {view.strides}",
    )
    _assert(not view.flags.c_contiguous, "precondition: view must be non-contiguous")

    raw0, raw1, float_size = m._probe_stride_units(view)
    _assert(float_size == 4, f"sizeof(float) != 4, got {float_size}")
    # Nanobind stride is in elements: 8 (= 2*4 elements per row skip), 1 (= one float per column).
    _assert(raw0 == 8, f"expected nanobind stride(0)=8 elements, got {raw0}")
    _assert(raw1 == 1, f"expected nanobind stride(1)=1 element, got {raw1}")

    total, element_stride0, element_stride1 = m._sum_strided(view)
    expected_sum = float(view.sum())
    _assert(
        abs(total - expected_sum) < 1e-4,
        f"strided sum mismatch: {total} != {expected_sum}",
    )
    _assert(
        element_stride0 == 8,
        f"NDArray stride(0) must be 8 elements, got {element_stride0}",
    )
    _assert(
        element_stride1 == 1,
        f"NDArray stride(1) must be 1 element, got {element_stride1}",
    )


def test_strided_borrow_column_slice() -> None:
    # Non-contiguous along the inner axis too: [:, ::2] on (n, d). Covers the case where
    # col_stride != 1.
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = source[:, ::2]
    _assert(view.shape == (4, 3), f"precondition: expected (4, 3), got {view.shape}")
    _assert(
        view.strides == (24, 8),
        f"precondition: expected (24, 8) byte-strides, got {view.strides}",
    )
    total, element_stride0, element_stride1 = m._sum_strided(view)
    expected_sum = float(view.sum())
    _assert(
        abs(total - expected_sum) < 1e-4,
        f"column-strided sum mismatch: {total} != {expected_sum}",
    )
    _assert(
        element_stride0 == 6,
        f"NDArray stride(0) must be 6 elements, got {element_stride0}",
    )
    _assert(
        element_stride1 == 2,
        f"NDArray stride(1) must be 2 elements, got {element_stride1}",
    )


def main() -> int:
    tests = [
        test_contig_borrow_zero_copy_and_mutable,
        test_readonly_borrow_flips_mutable_flag,
        test_owned_to_numpy_roundtrip_preserves_values,
        test_owned_to_numpy_capsule_keeps_buffer_alive,
        test_strided_borrow_uses_element_strides,
        test_strided_borrow_column_slice,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {exc!r}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
