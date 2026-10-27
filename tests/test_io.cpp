#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

std::filesystem::path temp_path(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}

}  // namespace

TEST(Io, RoundTripPreservesGraphAndSearch) {
  constexpr std::size_t n = 150;
  constexpr std::size_t d = 12;
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> data(n * d);
  for (float& x : data) x = dist(rng);
  hnsw::Params p;
  p.M = 8;
  p.ef_construction = 40;
  p.seed = 1234;
  p.metric = hnsw::Metric::L2;
  hnsw::Index index(d, p);
  index.insert_batch(data.data(), n, 1);

  const auto path = temp_path("hnsw_roundtrip.bin");
  index.save(path);
  hnsw::Index loaded = hnsw::Index::load(path);
  EXPECT_EQ(loaded.size(), index.size());
  EXPECT_EQ(loaded.dim(), index.dim());
  EXPECT_EQ(loaded.M(), index.M());
  EXPECT_EQ(loaded.M0(), index.M0());
  EXPECT_EQ(loaded.max_level(), index.max_level());
  EXPECT_EQ(loaded.entry_point(), index.entry_point());
  EXPECT_EQ(loaded.params().seed, p.seed);
  for (std::uint32_t id = 0; id < n; ++id) {
    EXPECT_EQ(loaded.level(id), index.level(id));
    for (int layer = 0; layer <= index.level(id); ++layer) {
      EXPECT_EQ(loaded.neighbors(id, layer), index.neighbors(id, layer));
    }
  }
  const float* q = data.data() + 3 * d;
  const auto a = index.search(q, 8, 32);
  const auto b = loaded.search(q, 8, 32);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].id, b[i].id);
    EXPECT_FLOAT_EQ(a[i].distance, b[i].distance);
  }
  std::filesystem::remove(path);
}

TEST(Io, EmptyRoundTrip) {
  hnsw::Index index(3, {});
  const auto path = temp_path("hnsw_empty.bin");
  index.save(path);
  hnsw::Index loaded = hnsw::Index::load(path);
  EXPECT_EQ(loaded.size(), 0u);
  EXPECT_EQ(loaded.dim(), 3u);
  EXPECT_TRUE(loaded.search(std::vector<float>{1.f, 2.f, 3.f}.data(), 1, 4).empty());
  std::filesystem::remove(path);
}

TEST(Io, RejectsBadMagicAndTruncation) {
  const auto path = temp_path("hnsw_bad.bin");
  {
    std::ofstream out(path, std::ios::binary);
    const char bytes[] = {'n', 'o', 'p', 'e'};
    out.write(bytes, 4);
  }
  EXPECT_THROW((void)hnsw::Index::load(path), std::runtime_error);
  {
    std::ofstream out(path, std::ios::binary);
    const char magic[] = {'H', 'N', 'S', 'W', '0', '0', '0', '1'};
    out.write(magic, 8);
  }
  EXPECT_THROW((void)hnsw::Index::load(path), std::runtime_error);
  std::filesystem::remove(path);
  EXPECT_THROW((void)hnsw::Index::load(path), std::runtime_error);
}
