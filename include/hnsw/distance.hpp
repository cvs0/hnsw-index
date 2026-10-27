#pragma once

#include <cstddef>
#include <cstdint>

namespace hnsw {

// Quantities that the index minimizes.
// L2: squared Euclidean distance.
// InnerProduct: negative dot product.
// Cosine: negative dot product of L2-normalized inputs. This function does not normalize.
enum class Metric : std::uint32_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

const char* metric_name(Metric metric);

using DistanceFunction = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

DistanceFunction distance_function(Metric metric);
DistanceFunction scalar_distance_function(Metric metric);

bool avx2_supported() noexcept;

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float negative_inner_product_scalar(const float* a, const float* b, std::size_t dim) noexcept;

// In place. A zero vector is left unchanged.
void l2_normalize(float* v, std::size_t dim) noexcept;

}  // namespace hnsw
