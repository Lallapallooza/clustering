#pragma once

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

} // namespace clustering::math::defaults
