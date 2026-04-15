#pragma once

#include <cstddef>

#include "clustering/math/detail/reference_gemm.h"

namespace clustering::math::defaults {

#ifdef CLUSTERING_MATH_DEFAULT_BACKEND
/**
 * @brief Alias resolving to the default GEMM backend tag.
 *
 * Defaults to @c detail::ReferenceGemm. Override project-wide at build time with
 * @c -DCLUSTERING_MATH_DEFAULT_BACKEND=<tag-type>; override per call site by passing the
 * @c Backend template argument to @c gemm.
 */
using Backend = CLUSTERING_MATH_DEFAULT_BACKEND;
#else
/**
 * @brief Alias resolving to the default GEMM backend tag.
 *
 * Defaults to @c detail::ReferenceGemm. Override project-wide at build time with
 * @c -DCLUSTERING_MATH_DEFAULT_BACKEND=<tag-type>; override per call site by passing the
 * @c Backend template argument to @c gemm.
 */
using Backend = ::clustering::math::detail::ReferenceGemm;
#endif

#ifdef CLUSTERING_PAIRWISE_GEMM_THRESHOLD
/**
 * @brief Workload threshold at which @c pairwiseSqEuclidean switches from the per-pair SIMD
 *        kernel to the GEMM-identity kernel.
 *
 * Compared against @c n*m*d. 100000 follows FAISS's @c distance_compute_blas_threshold; the
 * exact crossover is hardware-dependent (GEMM packing/kernel cost vs. SIMD per-pair overhead)
 * and should be re-measured per target. Override project-wide at build time with
 * @c -DCLUSTERING_PAIRWISE_GEMM_THRESHOLD=<value>; must be set uniformly across the build so
 * every translation unit instantiates the pairwise template with the same dispatch body.
 */
inline constexpr std::size_t pairwiseGemmThreshold = CLUSTERING_PAIRWISE_GEMM_THRESHOLD;
#else
/**
 * @brief Workload threshold at which @c pairwiseSqEuclidean switches from the per-pair SIMD
 *        kernel to the GEMM-identity kernel.
 *
 * Compared against @c n*m*d. 100000 follows FAISS's @c distance_compute_blas_threshold; the
 * exact crossover is hardware-dependent (GEMM packing/kernel cost vs. SIMD per-pair overhead)
 * and should be re-measured per target. Override project-wide at build time with
 * @c -DCLUSTERING_PAIRWISE_GEMM_THRESHOLD=<value>; must be set uniformly across the build so
 * every translation unit instantiates the pairwise template with the same dispatch body.
 */
inline constexpr std::size_t pairwiseGemmThreshold = 100000;
#endif

#ifdef CLUSTERING_PAIRWISE_ARGMIN_MAX_D
/**
 * @brief Maximum feature dimension for which the fused @c pairwiseArgminSqEuclidean driver is
 *        used. Above this @c d, callers fall back to the materialized two-step.
 *
 * The fused outer driver keeps the full K range in-register (no K-blocking), so raising this
 * past @c 256 risks register spills that erase the measured win over the materialized path.
 * Override project-wide at build time with @c -DCLUSTERING_PAIRWISE_ARGMIN_MAX_D=<value>; must
 * be set uniformly across the build so dispatch is consistent across translation units.
 */
inline constexpr std::size_t pairwiseArgminMaxD = CLUSTERING_PAIRWISE_ARGMIN_MAX_D;
#else
/**
 * @brief Maximum feature dimension for which the fused @c pairwiseArgminSqEuclidean driver is
 *        used. Above this @c d, callers fall back to the materialized two-step.
 *
 * Matches the Faiss-measured AVX2 fused-argmin-GEMM envelope: the fused kernel keeps its full
 * K range in register, so above this @c d the accumulator path spills and the chunked
 * materialized fallback wins on cache residency.
 */
inline constexpr std::size_t pairwiseArgminMaxD = 16;
#endif

} // namespace clustering::math::defaults
