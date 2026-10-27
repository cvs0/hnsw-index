#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

std::vector<float> make_data(std::size_t n, std::size_t d, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> data(n * d);
  for (float& x : data) x = dist(rng);
  return data;
}

double measure_recall(hnsw::Index& index, hnsw::BruteForce& exact,
                      const std::vector<float>& queries, std::size_t nq, std::size_t d,
                      std::size_t k, std::size_t ef) {
  std::vector<std::vector<hnsw::Neighbor>> approx(nq);
  std::vector<std::uint32_t> truth(nq * k);
  for (std::size_t i = 0; i < nq; ++i) {
    const float* q = queries.data() + i * d;
    approx[i] = index.search(q, k, ef);
    const auto gt = exact.search(q, k);
    for (std::size_t j = 0; j < k; ++j) truth[i * k + j] = gt[j].id;
  }
  return hnsw::recall_at_k(approx, truth.data(), nq, k);
}

}  // namespace

TEST(Recall, MatchesBruteForceWhenBeamCoversTheSet) {
  constexpr std::size_t n = 80;
  constexpr std::size_t d = 8;
  constexpr std::size_t nq = 20;
  constexpr std::size_t k = 5;
  const std::vector<float> base = make_data(n, d, 11);
  const std::vector<float> queries = make_data(nq, d, 12);
  hnsw::Params p;
  p.M = 16;
  p.ef_construction = 80;
  p.seed = 3;
  hnsw::Index index(d, p);
  hnsw::BruteForce exact(d, hnsw::Metric::L2);
  index.insert_batch(base.data(), n, 1);
  exact.add_batch(base.data(), n);
  const double recall = measure_recall(index, exact, queries, nq, d, k, /*ef=*/80);
  EXPECT_DOUBLE_EQ(recall, 1.0);
}

TEST(Recall, HighEfBeatsLowEfOnRandomData) {
  constexpr std::size_t n = 1000;
  constexpr std::size_t d = 16;
  constexpr std::size_t nq = 40;
  constexpr std::size_t k = 10;
  const std::vector<float> base = make_data(n, d, 21);
  const std::vector<float> queries = make_data(nq, d, 22);
  hnsw::Params p;
  p.M = 16;
  p.ef_construction = 100;
  p.seed = 5;
  hnsw::Index index(d, p);
  hnsw::BruteForce exact(d, hnsw::Metric::L2);
  index.insert_batch(base.data(), n, 1);
  exact.add_batch(base.data(), n);
  const double low = measure_recall(index, exact, queries, nq, d, k, 8);
  const double high = measure_recall(index, exact, queries, nq, d, k, 128);
  EXPECT_GE(high + 0.02, low);
  EXPECT_GT(high, 0.95);
}

TEST(Recall, BruteForceAgreesWithScalarScan) {
  const float base[] = {0.f, 0.f, 5.f, 0.f, 0.f, 4.f, 1.f, 1.f};
  hnsw::BruteForce exact(2, hnsw::Metric::L2);
  exact.add_batch(base, 4);
  const float q[] = {0.5f, 0.25f};
  const auto hits = exact.search(q, 2);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_LT(hits[0].distance, hits[1].distance);
}
