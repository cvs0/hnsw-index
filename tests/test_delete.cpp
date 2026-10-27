#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

hnsw::Params params() {
  hnsw::Params p;
  p.M = 8;
  p.ef_construction = 40;
  p.seed = 3;
  p.metric = hnsw::Metric::L2;
  return p;
}

std::vector<float> random_vectors(std::size_t n, std::size_t d, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> data(n * d);
  for (float& x : data) x = dist(rng);
  return data;
}

void expect_graph(const hnsw::Index& index) {
  ASSERT_EQ(index.level(index.entry_point()), index.max_level());
  EXPECT_TRUE(index.contains(index.entry_point()));
  const std::uint32_t hwm = static_cast<std::uint32_t>(index.slot_count());
  for (std::uint32_t id = 0; id < hwm; ++id) {
    if (!index.contains(id)) continue;
    EXPECT_LE(index.level(id), index.max_level());
    for (int layer = 0; layer <= index.level(id); ++layer) {
      const auto links = index.neighbors(id, layer);
      const std::size_t cap = layer == 0 ? index.M0() : index.M();
      EXPECT_LE(links.size(), cap);
      for (std::size_t i = 0; i < links.size(); ++i) {
        EXPECT_NE(links[i], id);
        EXPECT_TRUE(index.contains(links[i]));
        EXPECT_GE(index.level(links[i]), layer);
        EXPECT_EQ(std::count(links.begin(), links.end(), links[i]), 1);
      }
    }
  }
}

bool linked_from_somewhere(const hnsw::Index& index, std::uint32_t target) {
  const std::uint32_t hwm = static_cast<std::uint32_t>(index.slot_count());
  for (std::uint32_t id = 0; id < hwm; ++id) {
    if (!index.contains(id)) continue;
    for (int layer = 0; layer <= index.level(id); ++layer) {
      const auto links = index.neighbors(id, layer);
      if (std::find(links.begin(), links.end(), target) != links.end()) return true;
    }
  }
  return false;
}

}  // namespace

TEST(Delete, UnlinkRemovesEveryEdgeAndRepairsEntry) {
  constexpr std::size_t n = 120;
  constexpr std::size_t d = 8;
  const auto data = random_vectors(n, d, 11);
  hnsw::Index index(d, params());
  index.insert_batch(data.data(), n, 1);
  const std::uint32_t doomed = index.entry_point();
  index.erase(doomed);
  EXPECT_FALSE(index.contains(doomed));
  EXPECT_EQ(index.size(), n - 1);
  EXPECT_NE(index.entry_point(), doomed);
  EXPECT_FALSE(linked_from_somewhere(index, doomed));
  expect_graph(index);
  const auto hits = index.search(data.data() + static_cast<std::size_t>(doomed) * d, 5, 32);
  for (const auto& hit : hits) EXPECT_NE(hit.id, doomed);
}

TEST(Delete, TombstoneKeepsEdgesEraseDoesNot) {
  constexpr std::size_t n = 40;
  constexpr std::size_t d = 4;
  const auto data = random_vectors(n, d, 2);
  hnsw::Params p = params();
  hnsw::Index marked(d, p);
  hnsw::Index removed(d, p);
  marked.insert_batch(data.data(), n, 1);
  removed.insert_batch(data.data(), n, 1);
  marked.tombstone(7);
  removed.erase(7);
  EXPECT_FALSE(marked.contains(7));
  EXPECT_FALSE(removed.contains(7));
  EXPECT_EQ(marked.size(), n - 1);
  EXPECT_EQ(removed.size(), n - 1);
  EXPECT_TRUE(linked_from_somewhere(marked, 7));
  EXPECT_FALSE(linked_from_somewhere(removed, 7));
  const float* q = data.data();
  for (const auto& hit : marked.search(q, 8, 32)) EXPECT_NE(hit.id, 7u);
  for (const auto& hit : removed.search(q, 8, 32)) EXPECT_NE(hit.id, 7u);
}

TEST(Delete, ReusesSlotsAndRoundTrips) {
  constexpr std::size_t n = 90;
  constexpr std::size_t d = 6;
  constexpr std::size_t extra = (n + 2) / 3;
  auto data = random_vectors(n + extra, d, 5);
  hnsw::Index index(d, params());
  index.insert_batch(data.data(), n, 1);
  std::vector<std::uint32_t> gone;
  for (std::uint32_t id = 0; id < n; id += 3) {
    index.erase(id);
    gone.push_back(id);
  }
  EXPECT_EQ(index.slot_count(), n);
  const std::size_t live = index.size();
  index.insert_batch(data.data() + n * d, gone.size(), 1);
  EXPECT_EQ(index.size(), live + gone.size());
  EXPECT_EQ(index.slot_count(), n);
  for (std::uint32_t id : gone) EXPECT_TRUE(index.contains(id));
  expect_graph(index);

  const auto path = std::filesystem::temp_directory_path() / "hnsw_delete_roundtrip.bin";
  index.save(path);
  hnsw::Index loaded = hnsw::Index::load(path);
  EXPECT_EQ(loaded.size(), index.size());
  EXPECT_EQ(loaded.slot_count(), index.slot_count());
  EXPECT_EQ(loaded.entry_point(), index.entry_point());
  for (std::uint32_t id = 0; id < index.slot_count(); ++id) {
    EXPECT_EQ(loaded.contains(id), index.contains(id));
    if (!index.contains(id)) continue;
    EXPECT_EQ(loaded.level(id), index.level(id));
    EXPECT_EQ(loaded.neighbors(id, 0), index.neighbors(id, 0));
  }
  std::filesystem::remove(path);
  ASSERT_TRUE(index.contains(2));
  index.tombstone(2);
  EXPECT_THROW(index.tombstone(2), std::out_of_range);
  index.erase(2);
  EXPECT_THROW(index.erase(2), std::out_of_range);
  EXPECT_FALSE(index.contains(2));
}

TEST(Delete, RecallHoldsAfterRemovingThirtyPercent) {
  constexpr std::size_t n = 400;
  constexpr std::size_t d = 16;
  constexpr std::size_t nq = 30;
  constexpr std::size_t k = 10;
  const auto data = random_vectors(n + nq, d, 9);
  hnsw::Params p = params();
  p.M = 16;
  p.ef_construction = 80;
  hnsw::Index index(d, p);
  index.insert_batch(data.data(), n, 1);
  hnsw::BruteForce exact(d, hnsw::Metric::L2);
  exact.add_batch(data.data(), n);
  for (std::uint32_t id = 0; id < n; id += 3) {
    index.erase(id);
    exact.erase(id);
  }
  expect_graph(index);
  std::vector<std::vector<hnsw::Neighbor>> approx(nq);
  std::vector<std::uint32_t> truth(nq * k);
  for (std::size_t i = 0; i < nq; ++i) {
    const float* q = data.data() + (n + i) * d;
    approx[i] = index.search(q, k, 128);
    const auto gt = exact.search(q, k);
    for (std::size_t j = 0; j < k; ++j) truth[i * k + j] = gt[j].id;
  }
  EXPECT_GT(hnsw::recall_at_k(approx, truth.data(), nq, k), 0.90);
}

TEST(Delete, EmptyThenInsertAgain) {
  constexpr std::size_t d = 4;
  const auto data = random_vectors(15, d, 1);
  hnsw::Index index(d, params());
  index.insert_batch(data.data(), 15, 1);
  for (std::uint32_t id = 0; id < 15; ++id) index.erase(id);
  EXPECT_EQ(index.size(), 0u);
  EXPECT_EQ(index.max_level(), -1);
  EXPECT_TRUE(index.search(data.data(), 3, 8).empty());
  const std::uint32_t id = index.insert(data.data());
  EXPECT_TRUE(index.contains(id));
  const auto hits = index.search(data.data(), 1, 8);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].id, id);
  EXPECT_FLOAT_EQ(hits[0].distance, 0.f);
}

TEST(Delete, ConcurrentSearchEraseAndInsert) {
  constexpr std::size_t n = 300;
  constexpr std::size_t d = 8;
  constexpr std::size_t k = 10;
  const auto data = random_vectors(n + 80, d, 21);
  hnsw::Params p = params();
  p.M = 16;
  p.ef_construction = 48;
  hnsw::Index index(d, p);
  index.insert_batch(data.data(), n, 1);

  std::atomic<bool> run{true};
  std::thread searcher([&] {
    std::size_t i = 0;
    while (run.load(std::memory_order_relaxed)) {
      const float* q = data.data() + (i % n) * d;
      (void)index.search(q, 5, 16);
      ++i;
    }
  });

  std::vector<std::uint32_t> removed;
  for (std::uint32_t id = 0; id < 80; ++id) removed.push_back(id);
  std::thread eraser([&] {
    for (std::uint32_t id : removed) index.erase(id);
  });
  eraser.join();
  std::thread inserter([&] {
    for (std::size_t i = 0; i < removed.size(); ++i) {
      (void)index.insert(data.data() + (n + i) * d);
    }
  });
  inserter.join();
  run.store(false, std::memory_order_relaxed);
  searcher.join();

  EXPECT_EQ(index.size(), n);
  EXPECT_EQ(index.slot_count(), n);
  expect_graph(index);

  std::vector<float> row(d);
  constexpr std::size_t nq = 20;
  std::size_t hit = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    const float* query = data.data() + q * d;
    const auto approx = index.search(query, k, 64);
    std::vector<std::pair<float, std::uint32_t>> all;
    for (std::uint32_t id = 0; id < index.slot_count(); ++id) {
      if (!index.contains(id)) continue;
      index.copy_vector(id, row.data());
      float dist = 0.f;
      for (std::size_t j = 0; j < d; ++j) {
        const float delta = query[j] - row[j];
        dist += delta * delta;
      }
      all.push_back({dist, id});
    }
    std::sort(all.begin(), all.end());
    for (const auto& nb : approx) {
      for (std::size_t j = 0; j < k; ++j) {
        if (nb.id == all[j].second) {
          ++hit;
          break;
        }
      }
    }
  }
  EXPECT_GT(static_cast<double>(hit) / static_cast<double>(nq * k), 0.85);
}
