#include <stdexcept>

#include "hnsw/distance.hpp"

#ifndef HNSW_HAS_AVX2
#define HNSW_HAS_AVX2 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace hnsw::detail {
#if HNSW_HAS_AVX2
float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float negative_inner_product_avx2(const float* a, const float* b, std::size_t dim) noexcept;
#endif
}  // namespace hnsw::detail

namespace {

bool cpu_has_avx2_fma() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  const bool osxsave = (ecx & (1u << 27)) != 0;
  const bool avx = (ecx & (1u << 28)) != 0;
  const bool fma = (ecx & (1u << 12)) != 0;
  if (!osxsave || !avx || !fma) return false;
  unsigned int xcr_lo = 0;
  unsigned int xcr_hi = 0;
  __asm__ volatile("xgetbv" : "=a"(xcr_lo), "=d"(xcr_hi) : "c"(0));
  // XMM and YMM state enabled by the OS.
  if ((xcr_lo & 0x6u) != 0x6u) return false;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
  return (ebx & (1u << 5)) != 0;
#else
  return false;
#endif
}

}  // namespace

namespace hnsw {

const char* metric_name(Metric metric) {
  switch (metric) {
    case Metric::L2:
      return "L2";
    case Metric::InnerProduct:
      return "InnerProduct";
    case Metric::Cosine:
      return "Cosine";
  }
  return "unknown";
}

bool avx2_supported() noexcept {
#if HNSW_HAS_AVX2
  return cpu_has_avx2_fma();
#else
  return false;
#endif
}

DistanceFunction scalar_distance_function(Metric metric) {
  switch (metric) {
    case Metric::L2:
      return &l2_squared_scalar;
    case Metric::InnerProduct:
    case Metric::Cosine:
      return &negative_inner_product_scalar;
  }
  throw std::invalid_argument("hnsw: unknown metric");
}

DistanceFunction distance_function(Metric metric) {
#if HNSW_HAS_AVX2
  static const bool use_avx = cpu_has_avx2_fma();
  if (use_avx) {
    switch (metric) {
      case Metric::L2:
        return &detail::l2_squared_avx2;
      case Metric::InnerProduct:
      case Metric::Cosine:
        return &detail::negative_inner_product_avx2;
    }
    throw std::invalid_argument("hnsw: unknown metric");
  }
#endif
  return scalar_distance_function(metric);
}

}  // namespace hnsw
