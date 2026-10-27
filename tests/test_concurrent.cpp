#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "hnsw/hnsw.hpp"

TEST(Concurrent, ParallelInsertPreservesIdsAndRecall) {
  constexpr std::size_t n = 800;
  constexpr std::size_t d = 8;
  constexpr std::size_t nq = 20;
  constexpr std::size_t k = 10;
  std::mt19937 rng(8);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> base(n * d);
  std::vector<float> queries(nq * d);
  for (float& x : base) x = dist(rng);
  for (float& x : queries) x = dist(rng);

  hnsw::Params p;
  p.M = 16;
  p.ef_construction = 64;
  p.seed = 15;
  hnsw::Index index(d, p);

  std::atomic<bool> run_search{true};
  std::thread searcher([&] {
    std::size_t i = 0;
    while (run_search.load(std::memory_order_relaxed)) {
      const float* q = base.data() + (i % n) * d;
      (void)index.search(q, 5, 16);
      ++i;
    }
  });

  std::vector<std::uint32_t> ids(n);
  std::vector<std::thread> writers;
  std::atomic<std::size_t> next{0};
  for (int t = 0; t < 4; ++t) {
    writers.emplace_back([&] {
      for (;;) {
        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        ids[i] = index.insert(base.data() + i * d);
      }
    });
  }
  for (std::thread& th : writers) th.join();
  run_search.store(false, std::memory_order_relaxed);
  searcher.join();

  EXPECT_EQ(index.size(), n);
  std::size_t self_hits = 0;
  std::size_t self_n = 0;
  for (std::size_t i = 0; i < n; i += 20) {
    const auto hits = index.search(base.data() + i * d, 1, 64);
    ++self_n;
    if (!hits.empty() && hits[0].id == ids[i]) ++self_hits;
  }
  // Concurrent inserts can leave a point outside the ef=64 beam of its own vector.
  EXPECT_GE(self_hits * 20, self_n * 19);

  std::vector<float> by_id(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    std::copy_n(base.data() + i * d, d, by_id.data() + static_cast<std::size_t>(ids[i]) * d);
  }
  std::vector<std::uint32_t> sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  ASSERT_EQ(sorted.size(), n);
  EXPECT_EQ(sorted.front(), 0u);
  EXPECT_EQ(sorted.back(), static_cast<std::uint32_t>(n - 1));

  hnsw::BruteForce exact(d, hnsw::Metric::L2);
  exact.add_batch(by_id.data(), n);
  std::vector<std::vector<hnsw::Neighbor>> approx(nq);
  std::vector<std::uint32_t> truth(nq * k);
  for (std::size_t i = 0; i < nq; ++i) {
    const float* q = queries.data() + i * d;
    approx[i] = index.search(q, k, 128);
    const auto gt = exact.search(q, k);
    for (std::size_t j = 0; j < k; ++j) truth[i * k + j] = gt[j].id;
  }
  EXPECT_GT(hnsw::recall_at_k(approx, truth.data(), nq, k), 0.90);
}
