#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

hnsw::Params params_small() {
  hnsw::Params p;
  p.M = 16;
  p.ef_construction = 64;
  p.seed = 7;
  p.metric = hnsw::Metric::L2;
  return p;
}

}  // namespace

TEST(Index, RejectsBadParams) {
  EXPECT_THROW(hnsw::Index(0, params_small()), std::invalid_argument);
  hnsw::Params p = params_small();
  p.M = 1;
  EXPECT_THROW(hnsw::Index(4, p), std::invalid_argument);
  p = params_small();
  p.ef_construction = 0;
  EXPECT_THROW(hnsw::Index(4, p), std::invalid_argument);
  hnsw::Index index(4, params_small());
  EXPECT_THROW((void)index.insert(nullptr), std::invalid_argument);
  EXPECT_THROW((void)index.search(nullptr, 1, 8), std::invalid_argument);
}

TEST(Index, EmptySearch) {
  hnsw::Index index(4, params_small());
  EXPECT_EQ(index.size(), 0u);
  EXPECT_EQ(index.max_level(), -1);
  EXPECT_TRUE(index.search(std::vector<float>{0, 0, 0, 0}.data(), 3, 8).empty());
}

TEST(Index, SinglePoint) {
  hnsw::Index index(2, params_small());
  const float v[] = {1.f, -2.f};
  EXPECT_EQ(index.insert(v), 0u);
  EXPECT_EQ(index.size(), 1u);
  EXPECT_EQ(index.entry_point(), 0u);
  EXPECT_EQ(index.level(0), index.max_level());
  const auto hits = index.search(v, 1, 4);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_FLOAT_EQ(hits[0].distance, 0.f);
  float out[2];
  index.copy_vector(0, out);
  EXPECT_FLOAT_EQ(out[0], 1.f);
  EXPECT_FLOAT_EQ(out[1], -2.f);
}

TEST(Index, ExactNearestOnTinySet) {
  hnsw::Index index(2, params_small());
  const float pts[] = {0.f, 0.f, 5.f, 0.f, 0.f, 5.f};
  index.insert_batch(pts, 3, 1);
  const float q[] = {1.f, 0.f};
  const auto hits = index.search(q, 1, 8);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_FLOAT_EQ(hits[0].distance, 1.f);
}

TEST(Index, TieBreakPrefersSmallerId) {
  hnsw::Params p = params_small();
  p.ef_construction = 32;
  hnsw::Index index(2, p);
  const float pts[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  index.insert_batch(pts, 5, 1);
  const float q[] = {1.f, 1.f};
  const auto hits = index.search(q, 3, 16);
  ASSERT_EQ(hits.size(), 3u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_EQ(hits[1].id, 1u);
  EXPECT_EQ(hits[2].id, 2u);
  EXPECT_FLOAT_EQ(hits[0].distance, 0.f);
}

TEST(Index, KGreaterThanSize) {
  hnsw::Index index(2, params_small());
  const float pts[] = {0.f, 0.f, 1.f, 0.f};
  index.insert_batch(pts, 2, 1);
  const auto hits = index.search(pts, 10, 1);
  EXPECT_EQ(hits.size(), 2u);
}

TEST(Index, DegreeCapsAndNoSelfLoops) {
  hnsw::Params p = params_small();
  p.M = 4;
  p.ef_construction = 32;
  hnsw::Index index(4, p);
  std::vector<float> data(200 * 4);
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = std::sin(0.17f * static_cast<float>(i));
  index.insert_batch(data.data(), 200, 1);
  EXPECT_EQ(index.M0(), 8u);
  EXPECT_EQ(index.level(index.entry_point()), index.max_level());
  for (std::uint32_t id = 0; id < index.size(); ++id) {
    const auto l0 = index.neighbors(id, 0);
    EXPECT_LE(l0.size(), index.M0());
    if (id > 0) {
      EXPECT_FALSE(l0.empty());
    }
    for (std::uint32_t nb : l0) {
      EXPECT_NE(nb, id);
      EXPECT_LT(nb, index.size());
    }
    for (int layer = 1; layer <= index.level(id); ++layer) {
      const auto upper = index.neighbors(id, layer);
      EXPECT_LE(upper.size(), index.M());
      for (std::uint32_t nb : upper) EXPECT_NE(nb, id);
    }
  }
}

TEST(Index, LevelHistogramIsGeometric) {
  hnsw::Params p = params_small();
  p.M = 4;
  p.ef_construction = 8;
  p.seed = 99;
  constexpr int n = 4000;
  hnsw::Index index(2, p);
  std::vector<float> data(static_cast<std::size_t>(n) * 2, 0.01f);
  for (int i = 0; i < n; ++i) data[static_cast<std::size_t>(i) * 2] = static_cast<float>(i);
  index.insert_batch(data.data(), n, 1);
  int ge1 = 0;
  int ge2 = 0;
  for (int i = 0; i < n; ++i) {
    const int level = index.level(static_cast<std::uint32_t>(i));
    if (level >= 1) ++ge1;
    if (level >= 2) ++ge2;
  }
  // P(L >= 1) = 1/M = 0.25, P(L >= 2) = 1/16 = 0.0625.
  EXPECT_GT(ge1, 800);
  EXPECT_LT(ge1, 1200);
  EXPECT_GT(ge2, 150);
  EXPECT_LT(ge2, 400);
}

TEST(Index, InnerProductPrefersAligned) {
  hnsw::Params p = params_small();
  p.metric = hnsw::Metric::InnerProduct;
  hnsw::Index index(2, p);
  const float pts[] = {1.f, 0.f, 0.f, 1.f, 0.2f, 0.f};
  index.insert_batch(pts, 3, 1);
  const float q[] = {1.f, 0.f};
  const auto hits = index.search(q, 1, 8);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_FLOAT_EQ(hits[0].distance, -1.f);
}

TEST(Index, CosineIgnoresMagnitude) {
  hnsw::Params p = params_small();
  p.metric = hnsw::Metric::Cosine;
  hnsw::Index index(2, p);
  // L2 nearest to (1, 0) is b. Cosine nearest is a.
  const float a[] = {10.f, 0.f};
  const float b[] = {0.f, 1.f};
  EXPECT_EQ(index.insert(a), 0u);
  EXPECT_EQ(index.insert(b), 1u);
  const float q[] = {1.f, 0.f};
  const auto hits = index.search(q, 1, 8);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, 0u);
  EXPECT_NEAR(hits[0].distance, -1.f, 1e-5f);
  float stored[2];
  index.copy_vector(0, stored);
  EXPECT_NEAR(stored[0], 1.f, 1e-5f);
  EXPECT_NEAR(stored[1], 0.f, 1e-5f);
}
