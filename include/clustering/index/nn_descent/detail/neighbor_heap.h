#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace clustering::index::nn_descent::detail {

/**
 * @brief Flat SoA bank of per-node bounded max-heaps collecting each node's current k-nearest
 *        neighbors.
 *
 * The bank owns a flat @c n * k storage for neighbor @c (sqDist, idx) pairs plus a parallel
 * @c n * k "new" flag array. Each @c k -slot chunk is maintained as a bounded max-heap under the
 * same total order as @ref math::detail::BoundedMaxHeap (smaller @c sqDist first; tie-break on
 * smaller @c idx). The flat layout keeps every per-node heap contiguous so the NN-Descent join
 * step walks neighbor lists through a pointer increment with no indirection.
 *
 * @par Tie-break
 * Identical to the shared @c BoundedMaxHeap: @c (a_key, a_val) < @c (b_key, b_val) when
 * @c a_key < b_key, or @c a_key == b_key and @c a_val < b_val. Load-bearing for kNN
 * reproducibility: identical distances resolve deterministically on smaller neighbor index.
 *
 * @par New-flag tracking
 * NN-Descent's local join compares each node's neighbor set against its neighbors' neighbor
 * sets. The "new" flag marks entries that were admitted since the last iteration; the join
 * visits only new pairs, avoiding redundant work on edges already processed. @ref ageEpoch
 * flips every flag to false before the next iteration retires.
 *
 * @tparam T Element type of the stored distance (must be orderable).
 */
template <class T> class NeighborHeapBank {
public:
  /// Empty-slot sentinel; unused slots hold this index so the heap's uniqueness check never
  /// matches a real neighbor. Treating the sentinel as a valid @c std::int32_t keeps the
  /// bank's storage trivially initializable and avoids a parallel "occupied" bitmap.
  static constexpr std::int32_t kEmpty = -1;

  /**
   * @brief Construct an @p n -node bank, each heap of capacity @p k.
   *
   * @param n Node count.
   * @param k Max neighbors retained per node.
   */
  NeighborHeapBank(std::size_t n, std::size_t k)
      : m_n(n), m_k(k), m_dist(n * k), m_idx(n * k, kEmpty), m_isNew(n * k, 0), m_sizes(n, 0) {}

  /**
   * @brief Admit candidate @p j as a neighbor of node @p i at squared distance @p sqDist.
   *
   * Duplicate check against @p i 's existing neighbors runs first; repeats are ignored. When the
   * heap is at capacity, the candidate is compared against the root; strictly smaller keys evict,
   * and equal-key + smaller-index evicts (the BoundedMaxHeap tie-break rule). Admitted entries
   * are marked "new" so the next join iteration visits them.
   *
   * @param i      Node index receiving the candidate.
   * @param j      Candidate neighbor; must differ from @p i (self-neighbor pushes are ignored).
   * @param sqDist Squared Euclidean distance from @p i to @p j.
   * @return @c true if the admission changed the heap contents.
   */
  bool push(std::int32_t i, std::int32_t j, T sqDist) {
    if (i == j) {
      return false;
    }
    const std::size_t base = static_cast<std::size_t>(i) * m_k;
    const std::size_t sz = m_sizes[static_cast<std::size_t>(i)];

    // Duplicate scan. Linear over current size (<= k) -- k is small enough that the scan is
    // cache-resident and cheaper than a side-band map.
    for (std::size_t s = 0; s < sz; ++s) {
      if (m_idx[base + s] == j) {
        return false;
      }
    }

    if (sz < m_k) {
      // Room in the heap: unconditional insert, sift up.
      m_dist[base + sz] = sqDist;
      m_idx[base + sz] = j;
      m_isNew[base + sz] = 1;
      siftUp(base, sz);
      m_sizes[static_cast<std::size_t>(i)] = sz + 1;
      return true;
    }
    if (m_k == 0) {
      return false;
    }

    // Full heap: admit only when the candidate is strictly smaller than the root (or equal with
    // a smaller index per the deterministic tie-break).
    const T rootDist = m_dist[base];
    const std::int32_t rootIdx = m_idx[base];
    if (sqDist < rootDist || (sqDist == rootDist && j < rootIdx)) {
      m_dist[base] = sqDist;
      m_idx[base] = j;
      m_isNew[base] = 1;
      siftDown(base, m_k);
      return true;
    }
    return false;
  }

  /// Number of neighbors currently retained for node @p i.
  [[nodiscard]] std::size_t sizeAt(std::int32_t i) const noexcept {
    return m_sizes[static_cast<std::size_t>(i)];
  }

  /// Current size of node @p i's heap, indexed directly by slot.
  [[nodiscard]] T distAt(std::int32_t i, std::size_t slot) const noexcept {
    return m_dist[(static_cast<std::size_t>(i) * m_k) + slot];
  }

  [[nodiscard]] std::int32_t idxAt(std::int32_t i, std::size_t slot) const noexcept {
    return m_idx[(static_cast<std::size_t>(i) * m_k) + slot];
  }

  [[nodiscard]] bool isNew(std::int32_t i, std::size_t slot) const noexcept {
    return m_isNew[(static_cast<std::size_t>(i) * m_k) + slot] != 0;
  }

  /**
   * @brief Clear every "new" flag without touching heap contents.
   *
   * Called at the start of each join iteration so the next pass can report the fraction of
   * neighbors admitted by this iteration alone.
   */
  void ageEpoch() noexcept { std::fill(m_isNew.begin(), m_isNew.end(), std::uint8_t{0}); }

  /**
   * @brief Re-flag every retained slot as "new" so the next join pass visits it.
   *
   * Warm-start entry point: after a prior build converged (or ran toSortedLists), the caller
   * reinvokes the join loop on the same bank. Without re-flagging, @ref JoinStep treats every
   * slot as "old" and visits zero candidate pairs, so the loop terminates immediately. This
   * method retags every occupied slot so the warm-start iteration can refine the pre-existing
   * neighbor set.
   */
  void rearmAllAsNew() noexcept {
    for (std::size_t i = 0; i < m_n; ++i) {
      const std::size_t base = i * m_k;
      const std::size_t sz = m_sizes[i];
      for (std::size_t s = 0; s < sz; ++s) {
        m_isNew[base + s] = 1;
      }
    }
  }

  /**
   * @brief Reload a node's heap from an externally-supplied neighbor list.
   *
   * Used by warm start: after @ref toSortedLists destroys heap order, the caller reloads the
   * bank from the sorted view. Each supplied @c (dist, idx) pair is placed into the storage
   * then sifted up to rebuild the heap invariant. All loaded entries are flagged "new."
   *
   * @param i       Target node.
   * @param entries Pairs of @c (sqDist, neighbor index); size must not exceed @c k.
   */
  void loadFromSorted(std::int32_t i, const std::vector<std::pair<T, std::int32_t>> &entries) {
    const std::size_t base = static_cast<std::size_t>(i) * m_k;
    const std::size_t count = entries.size();
    assert(count <= m_k);
    for (std::size_t s = 0; s < count; ++s) {
      m_dist[base + s] = entries[s].first;
      m_idx[base + s] = entries[s].second;
      m_isNew[base + s] = 1;
    }
    for (std::size_t s = count; s < m_k; ++s) {
      m_idx[base + s] = kEmpty;
      m_isNew[base + s] = 0;
    }
    m_sizes[static_cast<std::size_t>(i)] = count;
    // Rebuild the heap from the front via successive siftUp calls. After count siftUps the array
    // satisfies the max-heap invariant.
    for (std::size_t s = 1; s < count; ++s) {
      siftUp(base, s);
    }
  }

  /// Clear every heap (but keep capacity reserved).
  void clearAll() noexcept {
    std::fill(m_idx.begin(), m_idx.end(), kEmpty);
    std::fill(m_isNew.begin(), m_isNew.end(), std::uint8_t{0});
    std::fill(m_sizes.begin(), m_sizes.end(), std::size_t{0});
  }

  /**
   * @brief Extract each node's neighbors sorted ascending by squared distance.
   *
   * Performs a heap-sort in place via repeated root-pop then reverses each segment; the heap is
   * destroyed by this call. Intended for the @c neighbors() public view consumed after build.
   *
   * @return Length-@p n vector of per-node sorted neighbor lists.
   */
  template <class KnnEntry> [[nodiscard]] std::vector<std::vector<KnnEntry>> toSortedLists() {
    std::vector<std::vector<KnnEntry>> out(m_n);
    for (std::size_t i = 0; i < m_n; ++i) {
      const std::size_t base = i * m_k;
      std::size_t sz = m_sizes[i];
      out[i].reserve(sz);
      // Heap-sort: repeatedly pop the root (max) and place at the end; after sz pops the
      // storage is sorted ascending. Work in place on the heap storage.
      while (sz > 0) {
        const T rootDist = m_dist[base];
        const std::int32_t rootIdx = m_idx[base];
        --sz;
        if (sz > 0) {
          m_dist[base] = m_dist[base + sz];
          m_idx[base] = m_idx[base + sz];
          m_isNew[base] = m_isNew[base + sz];
          siftDown(base, sz);
        }
        m_dist[base + sz] = rootDist;
        m_idx[base + sz] = rootIdx;
      }
      // Storage now ascending; copy to output.
      const std::size_t finalSize = m_sizes[i];
      for (std::size_t s = 0; s < finalSize; ++s) {
        out[i].push_back(KnnEntry{m_idx[base + s], m_dist[base + s]});
      }
    }
    return out;
  }

  [[nodiscard]] std::size_t n() const noexcept { return m_n; }
  [[nodiscard]] std::size_t k() const noexcept { return m_k; }

private:
  // Total order used for sifting: larger-(dist, idx) pairs sit at the heap's root.
  [[nodiscard]] bool less(std::size_t flatA, std::size_t flatB) const noexcept {
    const T da = m_dist[flatA];
    const T db = m_dist[flatB];
    if (da < db) {
      return true;
    }
    if (db < da) {
      return false;
    }
    return m_idx[flatA] < m_idx[flatB];
  }

  void swapSlots(std::size_t flatA, std::size_t flatB) noexcept {
    using std::swap;
    swap(m_dist[flatA], m_dist[flatB]);
    swap(m_idx[flatA], m_idx[flatB]);
    swap(m_isNew[flatA], m_isNew[flatB]);
  }

  void siftUp(std::size_t base, std::size_t pos) noexcept {
    while (pos > 0) {
      const std::size_t parent = (pos - 1) / 2;
      if (less(base + parent, base + pos)) {
        swapSlots(base + parent, base + pos);
        pos = parent;
      } else {
        return;
      }
    }
  }

  void siftDown(std::size_t base, std::size_t sz) noexcept {
    std::size_t pos = 0;
    while (true) {
      const std::size_t left = (2 * pos) + 1;
      const std::size_t right = left + 1;
      std::size_t largest = pos;
      if (left < sz && less(base + largest, base + left)) {
        largest = left;
      }
      if (right < sz && less(base + largest, base + right)) {
        largest = right;
      }
      if (largest == pos) {
        return;
      }
      swapSlots(base + pos, base + largest);
      pos = largest;
    }
  }

  std::size_t m_n;
  std::size_t m_k;
  std::vector<T> m_dist;
  std::vector<std::int32_t> m_idx;
  std::vector<std::uint8_t> m_isNew; // u8 parallel to m_idx; 1 = admitted this iteration.
  std::vector<std::size_t> m_sizes;
};

} // namespace clustering::index::nn_descent::detail
