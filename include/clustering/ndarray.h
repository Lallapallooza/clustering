#pragma once

#include <cstdint>
#include <vector>
#include <sstream>
#include <cassert>
#include <initializer_list>

/**
 * @brief Represents a multidimensional array (NDArray) of a fixed number of dimensions N and element type T.
 *
 * NDArray is a template class that provides a high-level representation of a multi-dimensional array.
 * It offers various functionalities like element access, dimension information, and debug utilities.
 * The array's dimensions are defined at compile time for type safety and efficiency.
 *
 * @tparam T The type of elements stored in the NDArray.
 * @tparam N The number of dimensions of the NDArray.
 */
template<class T, std::size_t N>
class NDArray {
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
    const T *data() const {
      return this->m_ndarray->data() + this->m_index;
    }

   protected:
    NDArray<T, N> *m_ndarray; ///< Pointer to the NDArray.
    std::size_t   m_index;    ///< Index in the flat representation of the array.
    std::size_t   m_dim;      ///< Current dimension of the accessor.

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
    inline operator T() const noexcept {
      return this->m_ndarray->flatIndex(this->m_index);
    }

    /**
     * @brief Returns the flat index in the NDArray corresponding to the accessor.
     *
     * @return The flat index as a size_t.
     */
    [[nodiscard]] inline size_t index() const noexcept {
      return this->m_index;
    }
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
    : m_dims(dims) {
    assert(dims.size() == N);
    size_t size = 1;

    for (auto dim: dims) {
      size *= dim;
    }
    m_data.resize(size);
  }

  /**
   * @brief Provides access to the elements of the NDArray.
   *
   * @param index Index in the first dimension.
   * @return An Accessor to the specified index in the first dimension.
   */
  inline Accessor operator[](std::size_t index) noexcept {
    assert(index < m_dims[0]);
    return Accessor(*this, index, 0);
  }

  /**
   * @brief Provides read-only access to the elements of the NDArray.
   *
   * @param index Index in the first dimension.
   * @return A ConstAccessor to the specified index in the first dimension.
   */
  inline const ConstAccessor operator[](std::size_t index) const noexcept {
    assert(index < m_dims[0]);
    return ConstAccessor(*this, index, 0);
  }

  /**
   * @brief Provides direct access to the flat underlying array at a specific index.
   *
   * @param index Index in the flat representation of the NDArray.
   * @return Reference to the element at the specified index.
   */
  inline T &flatIndex(std::size_t index) noexcept {
    return m_data[index];
  }

  /**
   * @brief Provides read-only access to the flat underlying array at a specific index.
   *
   * @param index Index in the flat representation of the NDArray.
   * @return Constant reference to the element at the specified index.
   */
  inline const T &flatIndex(std::size_t index) const noexcept {
    return m_data[index];
  }

  /**
   * @brief Returns the size of a specific dimension of the NDArray.
   *
   * @param index Index of the dimension.
   * @return Size of the specified dimension as a size_t.
   */
  inline const size_t dim(std::size_t index) const noexcept {
    return m_dims[index];
  }

  /**
   * @brief Provides read-only access to the internal data array.
   *
   * @return Constant pointer to the data array.
   */
  inline const T *data() const noexcept {
    return m_data.data();
  }

  /**
   * @brief Provides read-write access to the internal data array.
   *
   * @return Pointer to the data array.
   */
  inline T *data() noexcept {
    return m_data.data();
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
    for (auto dim: m_dims) {
      ss << dim << ", ";
    }
    ss << ")\n";
    ss << "data: [";
    for (auto value: m_data) {
      ss << value << ", ";
    }
    ss << "]\n";
    ss << "size: " << m_data.size() << "\n";
    return ss.str();
  }
 private:
  template <std::size_t Align = alignof(T)>
  class AlignedAllocator : public std::allocator<T> {
   public:
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;

    template <typename U>
    struct rebind {
      using other = typename NDArray<U, N>::template AlignedAllocator<Align>;
    };

    AlignedAllocator() noexcept {}

    template <typename U>
    AlignedAllocator(const typename NDArray<U, N>::template AlignedAllocator<Align>&) noexcept {}

    pointer allocate(size_type n, const_pointer hint = 0) {
      if (auto p = static_cast<pointer>(aligned_alloc(Align, n * sizeof(T))))
        return p;
      throw std::bad_alloc();
    }

    void deallocate(pointer p, size_type n) noexcept {
      free(p);
    }
  };
 private:
  std::vector<T, AlignedAllocator<32>>      m_data;  ///< Internal storage of the NDArray elements.
  std::vector<size_t> m_dims;  ///< Vector storing the dimensions of the NDArray.
};
