#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <new>
#include <sstream>
#include <type_traits>
#include <vector>

namespace clustering {
namespace detail {

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
    size_type bytes = n * sizeof(T);
    size_type aligned_bytes = (bytes + Align - 1) / Align * Align;
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
} // namespace clustering

/**
 * @brief Tag indicating whether an NDArray owns its buffer or borrows memory from elsewhere.
 */
enum class NDArrayStorage : std::uint8_t { Owned, Borrowed };

/**
 * @brief Represents a multidimensional array (NDArray) of a fixed number of dimensions N and
 * element type T.
 *
 * NDArray is a template class that provides a high-level representation of a multi-dimensional
 * array. It offers various functionalities like element access, dimension information, and debug
 * utilities. The array's dimensions are defined at compile time for type safety and efficiency.
 *
 * @tparam T The type of elements stored in the NDArray.
 * @tparam N The number of dimensions of the NDArray.
 */
template <class T, std::size_t N> class NDArray {
  static_assert(N >= 1, "NDArray rank must be >= 1");
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "NDArray element type must be float or double");

public:
  /**
   * @brief Inner class providing base functionality for element access in NDArray.
   */
  class BaseAccessor {
  public:
    /**
     * @brief Returns a pointer to the element data.
     * @return Constant pointer to the data.
     */
    const T *data() const { return this->m_ndarray->data() + this->m_index; }

  protected:
    NDArray<T, N> *m_ndarray; ///< Pointer to the NDArray.
    std::size_t m_index;      ///< Index in the flat representation of the array.
    std::size_t m_dim;        ///< Current dimension of the accessor.

    /**
     * @brief Constructs a BaseAccessor for a given NDArray, index, and dimension.
     *
     * @param ndarray Pointer to the NDArray.
     * @param index Index in the flat representation of the array.
     * @param dim Current dimension of the accessor.
     */
    BaseAccessor(NDArray<T, N> *ndarray, std::size_t index, std::size_t dim)
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
    ConstAccessor(const NDArray<T, N> &ndarray, std::size_t index, std::size_t dim)
        : BaseAccessor(const_cast<NDArray<T, N> *>(&ndarray), index, dim) {}

    ConstAccessor(const ConstAccessor &other) = default;

    /**
     * @brief Provides access to the next dimension of the NDArray.
     *
     * @param index Index in the next dimension.
     * @return A new ConstAccessor for the specified index in the next dimension.
     */
    inline ConstAccessor operator[](std::size_t index) const noexcept {
      assert(this->m_dim < N && index < this->m_ndarray->dim(this->m_dim + 1));
      size_t new_index = this->m_index * this->m_ndarray->dim(this->m_dim + 1) + index;
      return ConstAccessor(*this->m_ndarray, new_index, this->m_dim + 1);
    }

    /**
     * @brief Converts the accessor to the element type T, allowing read access to the element.
     *
     * @return The element of type T at the accessor's position.
     */
    inline operator T() const noexcept { return this->m_ndarray->flatIndex(this->m_index); }

    /**
     * @brief Returns the flat index in the NDArray corresponding to the accessor.
     *
     * @return The flat index as a size_t.
     */
    [[nodiscard]] inline size_t index() const noexcept { return this->m_index; }
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
    Accessor(NDArray<T, N> &ndarray, std::size_t index, std::size_t dim)
        : ConstAccessor(ndarray, index, dim) {}

    /**
     * @brief Provides access to the next dimension of the NDArray.
     *
     * @param index Index in the next dimension.
     * @return A new Accessor for the specified index in the next dimension.
     */
    inline Accessor operator[](std::size_t index) noexcept {
      assert(this->m_dim < N && index < this->m_ndarray->dim(this->m_dim + 1));
      size_t new_index = this->m_index * this->m_ndarray->dim(this->m_dim + 1) + index;
      return Accessor(*this->m_ndarray, new_index, this->m_dim + 1);
    }

    /**
     * @brief Assigns a value to the element at the accessor's position.
     *
     * @param value The value to be assigned.
     * @return Reference to the accessor after assignment.
     */
    inline Accessor &operator=(T value) noexcept {
      this->m_ndarray->flatIndex(this->m_index) = value;
      return *this;
    }
  };

public:
  /**
   * @brief Constructs an NDArray with specified dimensions.
   *
   * @param dims Initializer list specifying the dimensions of the NDArray.
   */
  NDArray(std::initializer_list<std::size_t> dims)
      : m_data(nullptr), m_offset(0), m_storage(NDArrayStorage::Owned), m_mutable(true) {
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
  }

  NDArray(const NDArray &other)
      : m_vec(other.m_vec), m_shape(other.m_shape), m_strides(other.m_strides),
        m_offset(other.m_offset), m_storage(other.m_storage), m_mutable(other.m_mutable) {
    m_data = m_vec.data();
  }

  NDArray(NDArray &&other) noexcept
      : m_vec(std::move(other.m_vec)), m_shape(other.m_shape), m_strides(other.m_strides),
        m_offset(other.m_offset), m_storage(other.m_storage), m_mutable(other.m_mutable) {
    // m_data must be re-seated against this->m_vec since the move stole its buffer.
    m_data = m_vec.data();
    other.m_data = nullptr;
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
    m_data = m_vec.data();
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
    m_data = m_vec.data();
    other.m_data = nullptr;
    return *this;
  }

  /**
   * @brief Provides access to the elements of the NDArray.
   *
   * @param index Index in the first dimension.
   * @return An Accessor to the specified index in the first dimension.
   */
  inline Accessor operator[](std::size_t index) noexcept {
    assert(index < m_shape[0]);
    return Accessor(*this, index, 0);
  }

  /**
   * @brief Provides read-only access to the elements of the NDArray.
   *
   * @param index Index in the first dimension.
   * @return A ConstAccessor to the specified index in the first dimension.
   */
  inline const ConstAccessor operator[](std::size_t index) const noexcept {
    assert(index < m_shape[0]);
    return ConstAccessor(*this, index, 0);
  }

  /**
   * @brief Provides direct access to the flat underlying array at a specific index.
   *
   * @param index Index in the flat representation of the NDArray.
   * @return Reference to the element at the specified index.
   */
  inline T &flatIndex(std::size_t index) noexcept { return m_data[index]; }

  /**
   * @brief Provides read-only access to the flat underlying array at a specific index.
   *
   * @param index Index in the flat representation of the NDArray.
   * @return Constant reference to the element at the specified index.
   */
  inline const T &flatIndex(std::size_t index) const noexcept { return m_data[index]; }

  /**
   * @brief Returns the size of a specific dimension of the NDArray.
   *
   * @param index Index of the dimension.
   * @return Size of the specified dimension as a size_t.
   */
  inline const size_t dim(std::size_t index) const noexcept { return m_shape[index]; }

  /**
   * @brief Provides read-only access to the internal data array.
   *
   * @return Constant pointer to the data array.
   */
  inline const T *data() const noexcept { return m_data; }

  /**
   * @brief Provides read-write access to the internal data array.
   *
   * @return Pointer to the data array.
   */
  inline T *data() noexcept { return m_data; }

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
   * @brief Returns a formatted string representing the contents of the NDArray.
   *
   * This method is primarily used for debugging purposes, providing a visual representation
   * of the array's structure and contents.
   *
   * @return A string containing the formatted representation of the NDArray.
   */
  std::string debugDump() const noexcept {
    std::stringstream ss;
    ss << "NDarray<" << typeid(T).name() << ", " << N << ">(";
    for (auto d : m_shape) {
      ss << d << ", ";
    }
    ss << ")\n";
    ss << "data: [";
    for (auto value : m_vec) {
      ss << value << ", ";
    }
    ss << "]\n";
    ss << "size: " << m_vec.size() << "\n";
    return ss.str();
  }

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

  T *m_data;
  std::vector<T, clustering::detail::AlignedAllocator<T, 32>> m_vec;
  std::array<std::size_t, N> m_shape;
  std::array<std::ptrdiff_t, N> m_strides;
  std::ptrdiff_t m_offset;
  NDArrayStorage m_storage;
  bool m_mutable;
};
