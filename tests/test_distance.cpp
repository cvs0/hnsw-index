#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hnsw/distance.hpp"

namespace {

std::vector<float> rand_vec(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

void expect_close(float got, float ref) {
  const float scale = std::max(1.f, std::abs(ref));
  EXPECT_NEAR(got, ref, scale * 1e-5f);
}

}  // namespace

TEST(Distance, KnownValues) {
  const float a[] = {1.f, 2.f, 3.f};
  const float b[] = {1.f, 0.f, -1.f};
  EXPECT_FLOAT_EQ(hnsw::l2_squared_scalar(a, b, 3), 20.f);
  EXPECT_FLOAT_EQ(hnsw::negative_inner_product_scalar(a, b, 3), 2.f);
}

TEST(Distance, Normalize) {
  float v[] = {3.f, 4.f};
  hnsw::l2_normalize(v, 2);
  EXPECT_NEAR(v[0], 0.6f, 1e-6f);
  EXPECT_NEAR(v[1], 0.8f, 1e-6f);
  float z[] = {0.f, 0.f};
  hnsw::l2_normalize(z, 2);
  EXPECT_FLOAT_EQ(z[0], 0.f);
  EXPECT_FLOAT_EQ(z[1], 0.f);
}

TEST(Distance, ScalarMatchesDispatch) {
  const std::size_t dims[] = {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 128, 130};
  for (std::size_t dim : dims) {
    const std::vector<float> a = rand_vec(dim, 1);
    const std::vector<float> b = rand_vec(dim, 2);
    const float l2_s = hnsw::l2_squared_scalar(a.data(), b.data(), dim);
    const float ip_s = hnsw::negative_inner_product_scalar(a.data(), b.data(), dim);
    const float l2 = hnsw::distance_function(hnsw::Metric::L2)(a.data(), b.data(), dim);
    const float ip = hnsw::distance_function(hnsw::Metric::InnerProduct)(a.data(), b.data(), dim);
    expect_close(l2, l2_s);
    expect_close(ip, ip_s);
    expect_close(hnsw::scalar_distance_function(hnsw::Metric::L2)(a.data(), b.data(), dim), l2_s);
  }
}

TEST(Distance, IntegerVectorsMatchExactlyWhenAvx2) {
  if (!hnsw::avx2_supported()) GTEST_SKIP();
  const std::size_t dim = 128;
  std::vector<float> a(dim), b(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    a[i] = static_cast<float>(i % 5);
    b[i] = static_cast<float>((i * 3) % 5);
  }
  EXPECT_FLOAT_EQ(hnsw::distance_function(hnsw::Metric::L2)(a.data(), b.data(), dim),
                  hnsw::l2_squared_scalar(a.data(), b.data(), dim));
  EXPECT_FLOAT_EQ(hnsw::distance_function(hnsw::Metric::InnerProduct)(a.data(), b.data(), dim),
                  hnsw::negative_inner_product_scalar(a.data(), b.data(), dim));
}

TEST(Distance, MetricName) {
  EXPECT_STREQ(hnsw::metric_name(hnsw::Metric::L2), "L2");
  EXPECT_STREQ(hnsw::metric_name(hnsw::Metric::InnerProduct), "InnerProduct");
  EXPECT_STREQ(hnsw::metric_name(hnsw::Metric::Cosine), "Cosine");
}
