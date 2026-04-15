#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
#include <span>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/ndarray_range.h"

namespace clustering {
namespace detail {

/**
 * @brief Tag type used to disambiguate the private borrowed-view constructor on NDArray.
 *
 * Only NDArray's template friends can reach the constructor; BorrowedTag lives in @c detail
 * as a cosmetic guard so external callers cannot accidentally construct a borrowed view.
 */
struct BorrowedTag {};

/**
 * @brief Stateless allocator that returns @p Align -byte aligned blocks from std::aligned_alloc.
 *
 * @tparam T Element type.
 * @tparam Align Required alignment in bytes.
 */
template <class T, std::size_t Align> class AlignedAllocator : public std::allocator<T> {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T *;
  using const_pointer = const T *;

  // is_always_equal + POCMA/POCCA/POCS = true_type lets std::vector pointer-steal on move and
  // perform a fresh allocation on copy-assign without per-instance allocator state checks.
  using is_always_equal = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Align>;
  };

  AlignedAllocator() noexcept = default;

  template <typename U> AlignedAllocator(const AlignedAllocator<U, Align> & /*unused*/) noexcept {}

  pointer allocate(size_type n) {
    // std::aligned_alloc(Align, 0) is implementation-defined; bypass it explicitly.
    if (n == 0) {
      return nullptr;
    }
    const size_type bytes = n * sizeof(T);
    const size_type aligned_bytes = (bytes + Align - 1) / Align * Align;
    if (auto *ptr = static_cast<pointer>(std::aligned_alloc(Align, aligned_bytes))) {
      return ptr;
    }
    throw std::bad_alloc();
  }

  void deallocate(pointer ptr, size_type /*n*/) noexcept { std::free(ptr); }
};

template <class T, class U, std::size_t Align>
bool operator==(const AlignedAllocator<T, Align> &, const AlignedAllocator<U, Align> &) noexcept {
  return true;
}

template <class T, class U, std::size_t Align>
bool operator!=(const AlignedAllocator<T, Align> &, const AlignedAllocator<U, Align> &) noexcept {
  return false;
}

} // namespace detail

/**
 * @brief Tag indicating whether an NDArray owns its buffer or borrows memory from elsewhere.
 */
enum class NDArrayStorage : std::uint8_t { Owned, Borrowed };

/**
 * @brief Compile-time layout tag for NDArray.
 *
 * @c Contig guarantees row-major contiguous storage with zero offset. Instances of this layout
 * expose the chain-of-accessors @c operator[] and use the baseline flat-index formula in the
 * hot path. @c MaybeStrided makes no contiguity guarantee; only the variadic @c operator() is
 * available for element access, and it consults @c m_strides and @c m_offset.
 *
 * The split is a type-level precondition: calling @c a[i] on a @c MaybeStrided array is a
 * compile error, not a runtime check.
 */
enum class Layout : std::uint8_t { Contig, MaybeStrided };

/**
 * @brief Represents a multidimensional array (NDArray) of a fixed number of dimensions N and
 * element type T.
 *
 * NDArray is a template class that provides a high-level representation of a multi-dimensional
 * array. It offers element access, dimension information, and debug utilities. The array's
 * dimensions are defined at compile time for type safety and efficiency.
 *
 * @tparam T The type of elements stored in the NDArray.
 * @tparam N The number of dimensions of the NDArray.
 * @tparam L Layout tag; @c Layout::Contig (default) enables the @c operator[] chain.
 */
template <class T, std::size_t N, Layout L = Layout::Contig> class NDArray {
  static_assert(N >= 1, "NDArray rank must be >= 1");
  // Widened from {float, double} so integer widths (signed/unsigned) are permitted as label /
  // index / count storage. bool stays excluded: std::vector<bool> is a specialization without
  // contiguous T-addressable storage, which would silently break data(), alignedData, and the
  // AlignedAllocator<T, 32> invariant. Distance / reduction / GEMM primitives carry their own
  // float/double gates so integer NDArrays cannot reach numeric math without a compile error.
  static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                "NDArray element type must be arithmetic and not bool");

  // All NDArray template instantiations share friendship so view-producing verbs can construct
  // a result with a different rank or layout via the private BorrowedTag constructor.
  template <class U, std::size_t M, Layout LL> friend class NDArray;

public:
  /**
   * @brief Inner class providing base functionality for element access in NDArray.
   *
   * Accessors are produced only by the contiguous instantiation of NDArray; the pointer type is
   * fixed to @c Layout::Contig so constructing one from a strided array is ill-formed.
   */
  class BaseAccessor {
  public:
    /**
     * @brief Returns a pointer to the element data.
     * @return Constant pointer to the data.
     */
    const T *data() const { return this->m_ndarray->data() + this->m_index; }

  protected:
    NDArray<T, N, Layout::Contig> *m_ndarray; ///< Pointer to the NDArray.
    std::size_t m_index;                      ///< Index in the flat representation of the array.
    std::size_t m_dim;                        ///< Current dimension of the accessor.

    /**
     * @brief Constructs a BaseAccessor for a given NDArray, index, and dimension.
     *
     * @param ndarray Pointer to the NDArray.
     * @param index Index in the flat representation of the array.
     * @param dim Current dimension of the accessor.
     */
    BaseAccessor(NDArray<T, N, Layout::Contig> *ndarray, std::size_t index, std::size_t dim)
        : m_ndarray(ndarray), m_index(index), m_dim(dim) {}
  };

  /**
   * @brief Provides read-only access to NDArray elements.
   */
  class ConstAccessor : public BaseAccessor {
  public:
    /**
     * @brief Constructs a ConstAccessor for a constant NDArray.
     *
     * @param ndarray Reference to the constant NDArray.
     * @param index Index for the desired element.
     * @param dim Current dimension for the accessor.
     */
    ConstAccessor(const NDArray<T, N, Layout::Contig> &ndarray, std::size_t index, std::size_t dim)
        : BaseAccessor(const_cast<NDArray<T, N, Layout::Contig> *>(&ndarray), index, dim) {}

    ConstAccessor(const ConstAccessor &other) = default;

    /**
     * @brief Provides access to the next dimension of the NDArray.
     *
     * @param index Index in the next dimension.
     * @return A new ConstAccessor for the specified index in the next dimension.
     */
    ConstAccessor operator[](std::size_t index) const noexcept {
      assert(this->m_dim < N && index < this->m_ndarray->dim(this->m_dim + 1));
      // Contig invariant lets the chain collapse to m_index * shape[dim+1] + index at every step,
      // matching the baseline hot-loop asm clang can vectorise into an 8x-unrolled aligned load.
      const size_t new_index = (this->m_index * this->m_ndarray->dim(this->m_dim + 1)) + index;
      return ConstAccessor(*this->m_ndarray, new_index, this->m_dim + 1);
    }

    /**
     * @brief Converts the accessor to the element type T, allowing read access to the element.
     *
     * @return The element of type T at the accessor's position.
     */
    operator T() const noexcept { return this->m_ndarray->flatIndex(this->m_index); }

    /**
     * @brief Returns the flat index in the NDArray corresponding to the accessor.
     *
     * @return The flat index as a size_t.
     */
    [[nodiscard]] size_t index() const noexcept { return this->m_index; }
  };

  /**
   * @brief Provides read-write access to NDArray elements.
   */
  class Accessor : public ConstAccessor {
  public:
    /**
     * @brief Constructs an Accessor for an NDArray.
     *
     * @param ndarray Reference to the NDArray.
     * @param index Index for the desired element.
     * @param dim Current dimension for the accessor.
     */
    Accessor(NDArray<T, N, Layout::Contig> &ndarray, std::size_t index, std::size_t dim)
        : ConstAccessor(ndarray, index, dim) {}

    /**
     * @brief Provides access to the next dimension of the NDArray.
     *
     * @param index Index in the next dimension.
     * @return A new Accessor for the specified index in the next dimension.
     */
    Accessor operator[](std::size_t index) noexcept {
      assert(this->m_dim < N && index < this->m_ndarray->dim(this->m_dim + 1));
      const size_t new_index = (this->m_index * this->m_ndarray->dim(this->m_dim + 1)) + index;
      return Accessor(*this->m_ndarray, new_index, this->m_dim + 1);
    }

    /**
     * @brief Assigns a value to the element at the accessor's position.
     *
     * Asserts the underlying array is mutable: writing through an accessor produced from a
     * read-only borrow is undefined in release and trapped in debug.
     *
     * @param value The value to be assigned.
     * @return Reference to the accessor after assignment.
     */
    Accessor &operator=(T value) noexcept {
      assert(this->m_ndarray->m_mutable && "write to read-only borrow");
      this->m_ndarray->flatIndex(this->m_index) = value;
      return *this;
    }
  };

  /**
   * @brief Constructs a contiguous owned NDArray with specified dimensions.
   *
   * Only available for @c Layout::Contig; the MaybeStrided instantiation has no default
   * constructor and is produced exclusively by strided-view factory methods.
   *
   * @param dims Initializer list specifying the dimensions of the NDArray.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::Contig)
  NDArray(std::initializer_list<std::size_t> dims)
      : m_data(nullptr), m_base(nullptr), m_offset(0), m_storage(NDArrayStorage::Owned),
        m_mutable(true) {
    assert(dims.size() == N);
    std::size_t i = 0;
    std::size_t size = 1;
    for (auto d : dims) {
      m_shape[i++] = d;
      size *= d;
    }
    m_strides = computeContiguousStrides(m_shape);
    m_vec.resize(size);
    m_data = m_vec.data();
    m_base = m_data;
  }

  /**
   * @brief Constructs a contiguous owned NDArray from a runtime @c std::array of dimensions.
   *
   * Mirrors the initializer-list constructor but accepts a shape already materialized as a
   * @c std::array, the form consumed by @c reshape, @c contiguous, and @c clone when they
   * need to allocate a fresh buffer.
   *
   * @param shape Dimensions of the NDArray, one entry per axis.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::Contig)
  explicit NDArray(std::array<std::size_t, N> shape)
      : m_data(nullptr), m_base(nullptr), m_shape(shape), m_offset(0),
        m_storage(NDArrayStorage::Owned), m_mutable(true) {
    std::size_t size = 1;
    for (std::size_t k = 0; k < N; ++k) {
      size *= m_shape[k];
    }
    m_strides = computeContiguousStrides(m_shape);
    m_vec.resize(size);
    m_data = m_vec.data();
    m_base = m_data;
  }

  // Storage-aware special members: Owned arrays re-seat m_data against this->m_vec (the move
  // stole or the copy just populated it), while Borrowed arrays carry an empty m_vec and must
  // preserve the external pointer from the source.
  NDArray(const NDArray &other)
      : m_vec(other.m_vec), m_shape(other.m_shape), m_strides(other.m_strides),
        m_offset(other.m_offset), m_storage(other.m_storage), m_mutable(other.m_mutable) {
    m_data = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_data;
    m_base = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_base;
  }

  NDArray(NDArray &&other) noexcept
      : m_vec(std::move(other.m_vec)), m_shape(other.m_shape), m_strides(other.m_strides),
        m_offset(other.m_offset), m_storage(other.m_storage), m_mutable(other.m_mutable) {
    m_data = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_data;
    m_base = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_base;
    other.m_data = nullptr;
    other.m_base = nullptr;
  }

  NDArray &operator=(const NDArray &other) {
    if (this == &other) {
      return *this;
    }
    m_vec = other.m_vec;
    m_shape = other.m_shape;
    m_strides = other.m_strides;
    m_offset = other.m_offset;
    m_storage = other.m_storage;
    m_mutable = other.m_mutable;
    m_data = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_data;
    m_base = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_base;
    return *this;
  }

  NDArray &operator=(NDArray &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    m_vec = std::move(other.m_vec);
    m_shape = other.m_shape;
    m_strides = other.m_strides;
    m_offset = other.m_offset;
    m_storage = other.m_storage;
    m_mutable = other.m_mutable;
    m_data = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_data;
    m_base = (m_storage == NDArrayStorage::Owned) ? m_vec.data() : other.m_base;
    other.m_data = nullptr;
    other.m_base = nullptr;
    return *this;
  }

private:
  NDArray(clustering::detail::BorrowedTag, T *data, T *base, std::array<std::size_t, N> shape,
          std::array<std::ptrdiff_t, N> strides, std::ptrdiff_t offset, bool isMutable) noexcept
      : m_data(data), m_base(base), m_vec(), m_shape(shape), m_strides(strides), m_offset(offset),
        m_storage(NDArrayStorage::Borrowed), m_mutable(isMutable) {
    if constexpr (L == Layout::Contig) {
      assert(offset == 0 && strides == computeContiguousStrides(shape) &&
             "Contig NDArray requires contiguous strides and zero offset");
    }
  }

public:
  /**
   * @brief Provides access to the elements of the NDArray.
   *
   * Only defined for @c Layout::Contig. A @c MaybeStrided array has no @c operator[]; use the
   * variadic @c operator()(i, j, ...) to read through strides.
   *
   * @param index Index in the first dimension.
   * @return An Accessor to the specified index in the first dimension.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::Contig)
  Accessor operator[](std::size_t index) noexcept {
    assert(index < m_shape[0]);
    return Accessor(*this, index, 0);
  }

  /**
   * @brief Provides read-only access to the elements of the NDArray.
   *
   * @param index Index in the first dimension.
   * @return A ConstAccessor to the specified index in the first dimension.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::Contig)
  ConstAccessor operator[](std::size_t index) const noexcept {
    assert(index < m_shape[0]);
    return ConstAccessor(*this, index, 0);
  }

  /**
   * @brief Direct multi-index element access via strides. Available for all layouts.
   *
   * @tparam Ix Integral index pack, must have exactly N elements.
   * @param ix Indices, one per dimension.
   * @return Reference to the element at @c m_offset + sum_k ix_k * m_strides[k].
   */
  template <class... Ix> T &operator()(Ix... ix) noexcept {
    static_assert(sizeof...(Ix) == N, "operator() requires exactly N indices");
    assert(m_mutable && "write to read-only borrow");
    return m_data[computeElementOffset(std::index_sequence_for<Ix...>{}, ix...)];
  }

  template <class... Ix> const T &operator()(Ix... ix) const noexcept {
    static_assert(sizeof...(Ix) == N, "operator() requires exactly N indices");
    return m_data[computeElementOffset(std::index_sequence_for<Ix...>{}, ix...)];
  }

  /**
   * @brief Provides direct access to the flat underlying array at a specific index.
   *
   * @param index Index in the flat representation of the NDArray.
   * @return Reference to the element at the specified index.
   */
  T &flatIndex(std::size_t index) noexcept {
    assert(m_mutable && "write to read-only borrow");
    return m_data[index];
  }

  /**
   * @brief Provides read-only access to the flat underlying array at a specific index.
   *
   * @param index Index in the flat representation of the NDArray.
   * @return Constant reference to the element at the specified index.
   */
  const T &flatIndex(std::size_t index) const noexcept { return m_data[index]; }

  /**
   * @brief Returns the size of a specific dimension of the NDArray.
   *
   * @param index Index of the dimension.
   * @return Size of the specified dimension as a size_t.
   */
  size_t dim(std::size_t index) const noexcept { return m_shape[index]; }

  /**
   * @brief Returns the stride (in elements) for dimension @p index.
   */
  std::ptrdiff_t strideAt(std::size_t index) const noexcept { return m_strides[index]; }

  /**
   * @brief Reports whether the array's runtime layout is row-major contiguous with zero offset.
   *
   * For @c Layout::Contig arrays this is always true (the type encodes the guarantee). For
   * @c Layout::MaybeStrided arrays the answer is computed from @c m_strides and @c m_offset.
   */
  [[nodiscard]] bool isContiguous() const noexcept {
    if constexpr (L == Layout::Contig) {
      return true;
    } else {
      return m_offset == 0 && m_strides == computeContiguousStrides(m_shape);
    }
  }

  /**
   * @brief Reports whether writes through @c operator(), @c Accessor, or @c flatIndex are allowed.
   *
   * Owned arrays are always mutable. Borrowed arrays carry the flag supplied at borrow time:
   * @c borrow(const T*, ...) flips it off, @c borrow(T*, ...) leaves it on.
   */
  [[nodiscard]] bool isMutable() const noexcept { return m_mutable; }

  /**
   * @brief Reports whether the array owns its underlying buffer.
   *
   * Owned arrays hold their storage in @c m_vec; Borrowed arrays reference an external buffer
   * whose lifetime is the caller's responsibility.
   */
  [[nodiscard]] bool isOwned() const noexcept { return m_storage == NDArrayStorage::Owned; }

  /**
   * @brief Provides read-only access to the internal data array.
   *
   * @return Constant pointer to the data array.
   */
  const T *data() const noexcept { return m_data; }

  /**
   * @brief Provides read-write access to the internal data array.
   *
   * Asserts in debug that the array is mutable: a raw pointer grabbed from a read-only borrow
   * would otherwise corrupt the source buffer without tripping any other write-path guard.
   *
   * @return Pointer to the data array.
   */
  T *data() noexcept {
    assert(m_mutable && "write to read-only borrow");
    return m_data;
  }

  /**
   * @brief Returns the original (non-advanced) base pointer for storage-identity comparisons.
   *
   * All view verbs propagate the source's base pointer unchanged; only a fresh owned allocation
   * (or the @c clone / non-contiguous @c reshape paths) installs a new base. @c sameStorage
   * uses this to decide whether two arrays share the underlying buffer.
   */
  [[nodiscard]] T *baseData() const noexcept { return m_base; }

  /**
   * @brief Tests whether @c data() is aligned to @p A bytes.
   *
   * @tparam A Alignment to test in bytes.
   * @return True when @c data() is a null pointer or an @p A -byte boundary.
   */
  template <std::size_t A> bool isAligned() const noexcept {
    return (reinterpret_cast<std::uintptr_t>(m_data) % A) == 0;
  }

  /**
   * @brief Returns @c data() with an alignment hint of @p A bytes applied.
   *
   * Calls @c __builtin_assume_aligned so downstream SIMD intrinsics (e.g. @c _mm256_load_ps)
   * can assume an @p A -byte aligned pointer and emit aligned-load instructions under @c -O2.
   * The debug @c assert guards against feeding an unaligned borrow into this path.
   *
   * @tparam A Required alignment in bytes.
   * @return Aligned pointer into the buffer. Writes through the mutable overload on a read-only
   *         borrow are undefined; the caller is responsible for honoring the const contract.
   */
  template <std::size_t A> T *alignedData() noexcept {
    assert(isAligned<A>() && "alignedData<A>() requires A-byte aligned data");
    return static_cast<T *>(__builtin_assume_aligned(m_data, A));
  }

  template <std::size_t A> const T *alignedData() const noexcept {
    assert(isAligned<A>() && "alignedData<A>() requires A-byte aligned data");
    return static_cast<const T *>(__builtin_assume_aligned(m_data, A));
  }

  /**
   * @brief Borrows a contiguous buffer as an NDArray without taking ownership.
   *
   * Available only when @c L == Layout::Contig. The returned array shares the caller's buffer,
   * carries contiguous strides, and is writable through @c operator() and @c Accessor.
   *
   * @param ptr Non-owning pointer to the first element. Must stay alive for the view's lifetime.
   * @param shape Dimensions, one entry per axis.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::Contig)
  static NDArray borrow(T *ptr, std::array<std::size_t, N> shape) noexcept {
    return NDArray(clustering::detail::BorrowedTag{}, ptr, ptr, shape,
                   computeContiguousStrides(shape), 0, true);
  }

  /**
   * @brief Borrows a read-only contiguous buffer as an NDArray.
   *
   * Stores the caller's @c const T* as @c T* via @c const_cast and flips @c m_mutable off so any
   * write through @c operator() or @c Accessor asserts in debug.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::Contig)
  static NDArray borrow(const T *ptr, std::array<std::size_t, N> shape) noexcept {
    auto *mutPtr = const_cast<T *>(ptr);
    return NDArray(clustering::detail::BorrowedTag{}, mutPtr, mutPtr, shape,
                   computeContiguousStrides(shape), 0, false);
  }

  /**
   * @brief Borrows a strided buffer as an NDArray without taking ownership.
   *
   * Available only when @c L == Layout::MaybeStrided. Strides are in elements, not bytes; see
   * @c borrowBytes for the byte-stride entry point used at the Python binding boundary.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::MaybeStrided)
  static NDArray borrow(T *ptr, std::array<std::size_t, N> shape,
                        std::array<std::ptrdiff_t, N> strides) noexcept {
    return NDArray(clustering::detail::BorrowedTag{}, ptr, ptr, shape, strides, 0, true);
  }

  template <Layout L2 = L>
    requires(L2 == Layout::MaybeStrided)
  static NDArray borrow(const T *ptr, std::array<std::size_t, N> shape,
                        std::array<std::ptrdiff_t, N> strides) noexcept {
    auto *mutPtr = const_cast<T *>(ptr);
    return NDArray(clustering::detail::BorrowedTag{}, mutPtr, mutPtr, shape, strides, 0, false);
  }

  /**
   * @brief Rank-1 convenience borrow; avoids the @c std::array<size_t, 1>{n} boilerplate.
   */
  template <std::size_t M = N>
    requires(M == 1 && L == Layout::Contig)
  static NDArray borrow1D(T *ptr, std::size_t n) noexcept {
    return borrow(ptr, std::array<std::size_t, 1>{n});
  }

  template <std::size_t M = N>
    requires(M == 1 && L == Layout::Contig)
  static NDArray borrow1D(const T *ptr, std::size_t n) noexcept {
    return borrow(ptr, std::array<std::size_t, 1>{n});
  }

  /**
   * @brief Borrow a buffer whose strides are expressed in bytes (NumPy's convention).
   *
   * Byte strides are divided by @c sizeof(T) to recover element strides; non-divisible entries
   * are undefined and asserted in debug. The result always carries @c Layout::MaybeStrided so
   * arbitrary byte strides can be represented without a runtime contiguity check gating the
   * @c Contig type-level invariant.
   *
   * @param ptr Non-owning pointer to the first element.
   * @param shape Dimensions, one entry per axis.
   * @param stridesInBytes Byte offset between successive elements along each axis.
   * @param isMutable Whether writes through the view are permitted.
   */
  template <Layout L2 = L>
    requires(L2 == Layout::MaybeStrided)
  static NDArray borrowBytes(T *ptr, std::array<std::size_t, N> shape,
                             std::array<std::ptrdiff_t, N> stridesInBytes,
                             bool isMutable) noexcept {
    std::array<std::ptrdiff_t, N> element_strides{};
    for (std::size_t k = 0; k < N; ++k) {
      assert(stridesInBytes[k] % static_cast<std::ptrdiff_t>(sizeof(T)) == 0 &&
             "borrowBytes requires byte strides divisible by sizeof(T)");
      element_strides[k] = stridesInBytes[k] / static_cast<std::ptrdiff_t>(sizeof(T));
    }
    return NDArray(clustering::detail::BorrowedTag{}, ptr, ptr, shape, element_strides, 0,
                   isMutable);
  }

  /**
   * @brief Explicit @c std::span adapter for rank-1 borrows.
   *
   * No implicit @c std::span conversion is provided; callers spell @c fromSpan to avoid
   * overload-resolution ambiguity with the raw-pointer @c borrow overloads.
   */
  template <std::size_t M = N>
    requires(M == 1 && L == Layout::Contig)
  static NDArray fromSpan(std::span<T> s) noexcept {
    return borrow(s.data(), std::array<std::size_t, 1>{s.size()});
  }

  template <std::size_t M = N>
    requires(M == 1 && L == Layout::Contig)
  static NDArray fromSpan(std::span<const T> s) noexcept {
    return borrow(s.data(), std::array<std::size_t, 1>{s.size()});
  }

  /**
   * @brief Transposes a rank-2 NDArray into a borrowed view with swapped axes.
   *
   * The returned view reuses the source buffer; @c sameStorage returns true against the source.
   * Result is always @c Layout::MaybeStrided because transposition breaks row-major contiguity
   * for any contiguous source with more than one column.
   */
  template <std::size_t M = N>
    requires(M == 2)
  NDArray<T, 2, Layout::MaybeStrided> t() noexcept {
    return NDArray<T, 2, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{}, m_data, m_base,
        std::array<std::size_t, 2>{m_shape[1], m_shape[0]},
        std::array<std::ptrdiff_t, 2>{m_strides[1], m_strides[0]}, m_offset, m_mutable);
  }

  template <std::size_t M = N>
    requires(M == 2)
  NDArray<T, 2, Layout::MaybeStrided> t() const noexcept {
    return NDArray<T, 2, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{}, const_cast<T *>(m_data), const_cast<T *>(m_base),
        std::array<std::size_t, 2>{m_shape[1], m_shape[0]},
        std::array<std::ptrdiff_t, 2>{m_strides[1], m_strides[0]}, m_offset, false);
  }

  /**
   * @brief Returns a borrowed view of row @p i with the leading dimension dropped.
   *
   * Layout is preserved: a row of a @c Contig array is still @c Contig because the inner
   * strides remain the contiguous layout for the reduced shape.
   */
  template <std::size_t M = N>
    requires(M > 1)
  NDArray<T, N - 1, L> row(std::size_t i) noexcept {
    assert(i < m_shape[0]);
    std::array<std::size_t, N - 1> new_shape{};
    std::array<std::ptrdiff_t, N - 1> new_strides{};
    for (std::size_t k = 0; k + 1 < N; ++k) {
      new_shape[k] = m_shape[k + 1];
      new_strides[k] = m_strides[k + 1];
    }
    return NDArray<T, N - 1, L>(clustering::detail::BorrowedTag{},
                                m_data + m_offset + (static_cast<std::ptrdiff_t>(i) * m_strides[0]),
                                m_base, new_shape, new_strides, 0, m_mutable);
  }

  template <std::size_t M = N>
    requires(M > 1)
  NDArray<T, N - 1, L> row(std::size_t i) const noexcept {
    assert(i < m_shape[0]);
    std::array<std::size_t, N - 1> new_shape{};
    std::array<std::ptrdiff_t, N - 1> new_strides{};
    for (std::size_t k = 0; k + 1 < N; ++k) {
      new_shape[k] = m_shape[k + 1];
      new_strides[k] = m_strides[k + 1];
    }
    return NDArray<T, N - 1, L>(clustering::detail::BorrowedTag{},
                                const_cast<T *>(m_data) + m_offset +
                                    (static_cast<std::ptrdiff_t>(i) * m_strides[0]),
                                const_cast<T *>(m_base), new_shape, new_strides, 0, false);
  }

  /**
   * @brief Returns a borrowed rank-1 view of column @p j of a rank-2 array.
   *
   * Always @c MaybeStrided: column stride equals the row stride of the source, which is not 1
   * for any row-major source with more than one column.
   */
  template <std::size_t M = N>
    requires(M == 2)
  NDArray<T, 1, Layout::MaybeStrided> col(std::size_t j) noexcept {
    assert(j < m_shape[1]);
    return NDArray<T, 1, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{},
        m_data + m_offset + (static_cast<std::ptrdiff_t>(j) * m_strides[1]), m_base,
        std::array<std::size_t, 1>{m_shape[0]}, std::array<std::ptrdiff_t, 1>{m_strides[0]}, 0,
        m_mutable);
  }

  template <std::size_t M = N>
    requires(M == 2)
  NDArray<T, 1, Layout::MaybeStrided> col(std::size_t j) const noexcept {
    assert(j < m_shape[1]);
    return NDArray<T, 1, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{},
        const_cast<T *>(m_data) + m_offset + (static_cast<std::ptrdiff_t>(j) * m_strides[1]),
        const_cast<T *>(m_base), std::array<std::size_t, 1>{m_shape[0]},
        std::array<std::ptrdiff_t, 1>{m_strides[0]}, 0, false);
  }

  /**
   * @brief Borrowed half-open slice along a single axis.
   *
   * Returned view always carries @c MaybeStrided; callers that need the @c Contig guarantee
   * back (e.g. axis-0 slice of a contiguous source) can round-trip through @c contiguous().
   */
  NDArray<T, N, Layout::MaybeStrided> slice(std::size_t axis, std::size_t begin,
                                            std::size_t end) noexcept {
    assert(axis < N && begin <= end && end <= m_shape[axis]);
    std::array<std::size_t, N> new_shape = m_shape;
    new_shape[axis] = end - begin;
    return NDArray<T, N, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{},
        m_data + m_offset + (static_cast<std::ptrdiff_t>(begin) * m_strides[axis]), m_base,
        new_shape, m_strides, 0, m_mutable);
  }

  NDArray<T, N, Layout::MaybeStrided> slice(std::size_t axis, std::size_t begin,
                                            std::size_t end) const noexcept {
    assert(axis < N && begin <= end && end <= m_shape[axis]);
    std::array<std::size_t, N> new_shape = m_shape;
    new_shape[axis] = end - begin;
    return NDArray<T, N, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{},
        const_cast<T *>(m_data) + m_offset + (static_cast<std::ptrdiff_t>(begin) * m_strides[axis]),
        const_cast<T *>(m_base), new_shape, m_strides, 0, false);
  }

  /**
   * @brief Borrowed multi-axis slice; each @c Range applies to its corresponding axis.
   *
   * Step must be positive (no reversed views in v1). @c end defaults to the axis size via the
   * sentinel @c Range::all().
   */
  NDArray<T, N, Layout::MaybeStrided> slice(const std::array<Range, N> &ranges) noexcept {
    std::array<std::size_t, N> new_shape{};
    std::array<std::ptrdiff_t, N> new_strides{};
    std::ptrdiff_t advance = 0;
    for (std::size_t k = 0; k < N; ++k) {
      const std::size_t end = std::min(ranges[k].end, m_shape[k]);
      const std::size_t begin = ranges[k].begin;
      const std::ptrdiff_t step = ranges[k].step;
      assert(begin <= end && step > 0);
      new_shape[k] = step == 1 ? (end - begin)
                               : (end - begin + static_cast<std::size_t>(step) - 1) /
                                     static_cast<std::size_t>(step);
      new_strides[k] = m_strides[k] * step;
      advance += static_cast<std::ptrdiff_t>(begin) * m_strides[k];
    }
    return NDArray<T, N, Layout::MaybeStrided>(clustering::detail::BorrowedTag{},
                                               m_data + m_offset + advance, m_base, new_shape,
                                               new_strides, 0, m_mutable);
  }

  NDArray<T, N, Layout::MaybeStrided> slice(const std::array<Range, N> &ranges) const noexcept {
    std::array<std::size_t, N> new_shape{};
    std::array<std::ptrdiff_t, N> new_strides{};
    std::ptrdiff_t advance = 0;
    for (std::size_t k = 0; k < N; ++k) {
      const std::size_t end = std::min(ranges[k].end, m_shape[k]);
      const std::size_t begin = ranges[k].begin;
      const std::ptrdiff_t step = ranges[k].step;
      assert(begin <= end && step > 0);
      new_shape[k] = step == 1 ? (end - begin)
                               : (end - begin + static_cast<std::size_t>(step) - 1) /
                                     static_cast<std::size_t>(step);
      new_strides[k] = m_strides[k] * step;
      advance += static_cast<std::ptrdiff_t>(begin) * m_strides[k];
    }
    return NDArray<T, N, Layout::MaybeStrided>(
        clustering::detail::BorrowedTag{}, const_cast<T *>(m_data) + m_offset + advance,
        const_cast<T *>(m_base), new_shape, new_strides, 0, false);
  }

  /**
   * @brief Borrowed view with axes reordered by @p perm.
   *
   * @p perm must be a valid permutation of @c {0, ..., N-1}; validation is an assert in debug.
   */
  NDArray<T, N, Layout::MaybeStrided> permute(const std::array<std::size_t, N> &perm) noexcept {
    std::array<std::size_t, N> new_shape{};
    std::array<std::ptrdiff_t, N> new_strides{};
    for (std::size_t k = 0; k < N; ++k) {
      assert(perm[k] < N);
      new_shape[k] = m_shape[perm[k]];
      new_strides[k] = m_strides[perm[k]];
    }
    return NDArray<T, N, Layout::MaybeStrided>(clustering::detail::BorrowedTag{}, m_data, m_base,
                                               new_shape, new_strides, m_offset, m_mutable);
  }

  NDArray<T, N, Layout::MaybeStrided>
  permute(const std::array<std::size_t, N> &perm) const noexcept {
    std::array<std::size_t, N> new_shape{};
    std::array<std::ptrdiff_t, N> new_strides{};
    for (std::size_t k = 0; k < N; ++k) {
      assert(perm[k] < N);
      new_shape[k] = m_shape[perm[k]];
      new_strides[k] = m_strides[perm[k]];
    }
    return NDArray<T, N, Layout::MaybeStrided>(clustering::detail::BorrowedTag{},
                                               const_cast<T *>(m_data), const_cast<T *>(m_base),
                                               new_shape, new_strides, m_offset, false);
  }

  /**
   * @brief Returns a borrowed contiguous rank-@p M view over the same buffer with shape @p shape.
   *
   * Strict no-alloc primitive: the source must be runtime-contiguous (debug-asserted). Callers
   * that want the copy-on-nonconfig fallback should use @c reshape instead. The element count of
   * @p shape must equal the source's element count.
   *
   * @tparam M Target rank. May differ from @c N.
   * @param shape Dimensions of the returned view, one entry per axis.
   */
  template <std::size_t M>
  NDArray<T, M, Layout::Contig> view(std::array<std::size_t, M> shape) noexcept {
    assert(isContiguous() && "view<M> requires a contiguous source");
    assert(productOfShape(shape) == numel() && "view<M> must preserve element count");
    return NDArray<T, M, Layout::Contig>(
        clustering::detail::BorrowedTag{}, m_data, m_base, shape,
        NDArray<T, M, Layout::Contig>::computeContiguousStrides(shape), 0, m_mutable);
  }

  template <std::size_t M>
  NDArray<T, M, Layout::Contig> view(std::array<std::size_t, M> shape) const noexcept {
    assert(isContiguous() && "view<M> requires a contiguous source");
    assert(productOfShape(shape) == numel() && "view<M> must preserve element count");
    return NDArray<T, M, Layout::Contig>(
        clustering::detail::BorrowedTag{}, const_cast<T *>(m_data), const_cast<T *>(m_base), shape,
        NDArray<T, M, Layout::Contig>::computeContiguousStrides(shape), 0, false);
  }

  /**
   * @brief Returns a contiguous rank-@p M array with shape @p shape, copying only when needed.
   *
   * Aliases the existing buffer when the source is runtime-contiguous; allocates an owned
   * dense copy otherwise. Callers that require no-hidden-allocation guarantees should use
   * @c view instead, which asserts on non-contiguous input without the allocation fallback.
   *
   * @tparam M Target rank. May differ from @c N.
   * @param shape Dimensions of the result, one entry per axis.
   */
  template <std::size_t M> NDArray<T, M, Layout::Contig> reshape(std::array<std::size_t, M> shape) {
    assert(productOfShape(shape) == numel() && "reshape<M> must preserve element count");
    if (isContiguous()) {
      return NDArray<T, M, Layout::Contig>(
          clustering::detail::BorrowedTag{}, m_data, m_base, shape,
          NDArray<T, M, Layout::Contig>::computeContiguousStrides(shape), 0, m_mutable);
    }
    NDArray<T, M, Layout::Contig> result(shape);
    copyToContiguous(result.data());
    return result;
  }

  template <std::size_t M>
  NDArray<T, M, Layout::Contig> reshape(std::array<std::size_t, M> shape) const {
    assert(productOfShape(shape) == numel() && "reshape<M> must preserve element count");
    if (isContiguous()) {
      return NDArray<T, M, Layout::Contig>(
          clustering::detail::BorrowedTag{}, const_cast<T *>(m_data), const_cast<T *>(m_base),
          shape, NDArray<T, M, Layout::Contig>::computeContiguousStrides(shape), 0, false);
    }
    NDArray<T, M, Layout::Contig> result(shape);
    copyToContiguous(result.data());
    return result;
  }

  /**
   * @brief Returns a contiguous rank-@c N array with the same shape, copying only when needed.
   *
   * Already-contiguous sources are aliased into a borrowed view sharing storage; strided sources
   * allocate a dense owned copy. The return type drops the @c MaybeStrided tag so downstream hot
   * paths can resume the @c operator[] chain.
   */
  NDArray<T, N, Layout::Contig> contiguous() {
    if (isContiguous()) {
      return NDArray<T, N, Layout::Contig>(
          clustering::detail::BorrowedTag{}, m_data, m_base, m_shape,
          NDArray<T, N, Layout::Contig>::computeContiguousStrides(m_shape), 0, m_mutable);
    }
    NDArray<T, N, Layout::Contig> result(m_shape);
    copyToContiguous(result.data());
    return result;
  }

  NDArray<T, N, Layout::Contig> contiguous() const {
    if (isContiguous()) {
      return NDArray<T, N, Layout::Contig>(
          clustering::detail::BorrowedTag{}, const_cast<T *>(m_data), const_cast<T *>(m_base),
          m_shape, NDArray<T, N, Layout::Contig>::computeContiguousStrides(m_shape), 0, false);
    }
    NDArray<T, N, Layout::Contig> result(m_shape);
    copyToContiguous(result.data());
    return result;
  }

  /**
   * @brief Returns a freshly-allocated owned contiguous array with deep-copied contents.
   *
   * Always allocates; unlike @c contiguous, never aliases the source buffer. Equivalent to
   * NumPy's @c ndarray.copy: the caller receives an independent owner with matching values and
   * row-major layout.
   */
  NDArray<T, N, Layout::Contig> clone() const {
    NDArray<T, N, Layout::Contig> result(m_shape);
    copyToContiguous(result.data());
    return result;
  }

  /**
   * @brief Returns a formatted string representing the contents of the NDArray.
   *
   * This method is primarily used for debugging purposes, providing a visual representation
   * of the array's structure and contents. Walks the array through @c m_data, @c m_offset, and
   * @c m_strides so Borrowed views report their actual viewable contents rather than the empty
   * @c m_vec.
   *
   * @return A string containing the formatted representation of the NDArray.
   */
  std::string debugDump() const {
    std::stringstream ss;
    ss << "NDarray<" << typeid(T).name() << ", " << N << ">(";
    for (auto d : m_shape) {
      ss << d << ", ";
    }
    ss << ")\n";
    ss << "data: [";
    const std::size_t total = numel();
    if (total > 0) {
      std::array<std::size_t, N> idx{};
      for (std::size_t flat = 0; flat < total; ++flat) {
        std::ptrdiff_t off = m_offset;
        for (std::size_t k = 0; k < N; ++k) {
          off += static_cast<std::ptrdiff_t>(idx[k]) * m_strides[k];
        }
        ss << m_data[off] << ", ";
        for (std::size_t k = N; k-- > 0;) {
          if (++idx[k] < m_shape[k]) {
            break;
          }
          idx[k] = 0;
        }
      }
    }
    ss << "]\n";
    ss << "size: " << total << "\n";
    return ss.str();
  }

  // Equality between NDArrays has three plausible semantics (element-wise, storage-identity,
  // deep-value); none is obviously correct, so the operator is deleted to force callers to pick
  // an explicit intent (@c math::arrayEqual, @c sameStorage, or an explicit shape-and-element
  // comparison).
  friend bool operator==(const NDArray &, const NDArray &) = delete;
  friend bool operator!=(const NDArray &, const NDArray &) = delete;

private:
  static std::array<std::ptrdiff_t, N>
  computeContiguousStrides(const std::array<std::size_t, N> &shape) {
    std::array<std::ptrdiff_t, N> s{};
    s[N - 1] = 1;
    for (std::size_t k = N - 1; k > 0; --k) {
      s[k - 1] = s[k] * static_cast<std::ptrdiff_t>(shape[k]);
    }
    return s;
  }

  template <std::size_t M>
  static std::size_t productOfShape(const std::array<std::size_t, M> &shape) noexcept {
    std::size_t size = 1;
    for (std::size_t k = 0; k < M; ++k) {
      size *= shape[k];
    }
    return size;
  }

  std::size_t numel() const noexcept { return productOfShape(m_shape); }

  // Walk this array in row-major order and write elements densely to @p dst. Fast-path
  // @c memcpy when already contiguous, else advance a multi-index cursor and index through
  // @c m_offset + sum_k idx[k] * m_strides[k]. Shared by @c reshape, @c contiguous, @c clone.
  void copyToContiguous(T *dst) const noexcept {
    const std::size_t total = numel();
    if (total == 0) {
      return;
    }
    if (isContiguous()) {
      std::memcpy(dst, m_data + m_offset, total * sizeof(T));
      return;
    }
    std::array<std::size_t, N> idx{};
    for (std::size_t flat = 0; flat < total; ++flat) {
      std::ptrdiff_t off = m_offset;
      for (std::size_t k = 0; k < N; ++k) {
        off += static_cast<std::ptrdiff_t>(idx[k]) * m_strides[k];
      }
      dst[flat] = m_data[off];
      for (std::size_t k = N; k-- > 0;) {
        if (++idx[k] < m_shape[k]) {
          break;
        }
        idx[k] = 0;
      }
    }
  }

  template <std::size_t... Ks, class... Ix>
  std::size_t computeElementOffset(std::index_sequence<Ks...>, Ix... ix) const noexcept {
    std::ptrdiff_t off = m_offset;
    ((off += static_cast<std::ptrdiff_t>(ix) * m_strides[Ks]), ...);
    return static_cast<std::size_t>(off);
  }

  T *m_data;
  T *m_base;
  std::vector<T, clustering::detail::AlignedAllocator<T, 32>> m_vec;
  std::array<std::size_t, N> m_shape;
  std::array<std::ptrdiff_t, N> m_strides;
  std::ptrdiff_t m_offset;
  NDArrayStorage m_storage;
  bool m_mutable;
};

/**
 * @brief Returns true when @p a and @p b share the same underlying allocation.
 *
 * Every view verb propagates the parent's base pointer; fresh Owned allocations (including
 * @c clone and the non-contiguous @c reshape / @c contiguous fallbacks) install a new base.
 * Two siblings produced from the same source therefore both test true against each other and
 * against their source, regardless of which pointer-advancing verb produced them.
 */
template <class T, std::size_t NA, Layout LA, std::size_t NB, Layout LB>
bool sameStorage(const NDArray<T, NA, LA> &a, const NDArray<T, NB, LB> &b) noexcept {
  return a.baseData() == b.baseData();
}

} // namespace clustering
