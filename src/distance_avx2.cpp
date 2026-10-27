#include <immintrin.h>

#include <cstddef>

namespace hnsw::detail {

float hsum256(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  const __m128 dup = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, dup);
  sums = _mm_add_ss(sums, _mm_movehl_ps(dup, sums));
  return _mm_cvtss_f32(sums);
}

float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  std::size_t i = 0;
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  for (; i + 32 <= dim; i += 32) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
    const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
    s2 = _mm256_fmadd_ps(d2, d2, s2);
    s3 = _mm256_fmadd_ps(d3, d3, s3);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(d, d, s0);
  }
  s0 = _mm256_add_ps(s0, s1);
  s2 = _mm256_add_ps(s2, s3);
  float sum = hsum256(_mm256_add_ps(s0, s2));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float negative_inner_product_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  std::size_t i = 0;
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  for (; i + 32 <= dim; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  for (; i + 8 <= dim; i += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  }
  s0 = _mm256_add_ps(s0, s1);
  s2 = _mm256_add_ps(s2, s3);
  float sum = hsum256(_mm256_add_ps(s0, s2));
  for (; i < dim; ++i) sum += a[i] * b[i];
  return -sum;
}

}  // namespace hnsw::detail
