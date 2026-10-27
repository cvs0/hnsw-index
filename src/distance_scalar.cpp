#include <cmath>
#include <cstddef>

#include "hnsw/distance.hpp"

namespace hnsw {

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float negative_inner_product_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.f;
  for (std::size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
  return -sum;
}

void l2_normalize(float* v, std::size_t dim) noexcept {
  double sum = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += static_cast<double>(v[i]) * static_cast<double>(v[i]);
  }
  if (!(sum > 0.0)) return;
  const float inv = static_cast<float>(1.0 / std::sqrt(sum));
  for (std::size_t i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace hnsw
