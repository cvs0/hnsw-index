#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include "hnsw/distance.hpp"

namespace hnsw {

struct Params {
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  Metric metric = Metric::L2;
  std::uint64_t seed = 42;
};

struct Neighbor {
  std::uint32_t id = 0;
  // Squared L2, or the negative dot product. Cosine normalizes first, so
  // `distance` is the negative cosine similarity. Smaller is closer.
  float distance = 0.f;
};

// Fraction of approximate neighbors that land in the exact top-k.
// `truth` is row-major `nq * k`. Each result list contributes at most `k` ids.
// Returns 1 when `nq == 0` or `k == 0`.
inline double recall_at_k(const std::vector<std::vector<Neighbor>>& results,
                          const std::uint32_t* truth, std::size_t nq, std::size_t k) {
  if (nq == 0 || k == 0) return 1.0;
  if (results.size() < nq) throw std::invalid_argument("hnsw: results shorter than nq");
  std::size_t hit = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    const std::uint32_t* row = truth + q * k;
    const std::size_t n = results[q].size() < k ? results[q].size() : k;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t id = results[q][i].id;
      for (std::size_t j = 0; j < k; ++j) {
        if (id == row[j]) {
          ++hit;
          break;
        }
      }
    }
  }
  return static_cast<double>(hit) / static_cast<double>(nq * k);
}

// Multi-layer proximity graph.
// Layer 0 holds every live point and every tombstone, at most M0 = 2M neighbors.
// Layer l > 0 holds occupants with level >= l and at most M neighbors.
// Insert linking and erase take an exclusive graph mutex. Search does not.
// A freed slot is not reused until every search that started before the
// unlink has finished. save, load, and destruction require exclusive access.
class Index {
 public:
  explicit Index(std::size_t dim, Params params = {});
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  Index(Index&&) noexcept;
  Index& operator=(Index&&) noexcept;
  ~Index();

  std::size_t dim() const;
  // Live points. Tombstones and free slots are excluded.
  std::size_t size() const;
  // High-water slot count. Ids issued so far are in `[0, slot_count())`.
  std::size_t slot_count() const;
  // Bytes retained for vectors, neighbor atoms, inbound lists, and node blocks.
  std::size_t memory_bytes() const;
  std::size_t M() const;
  std::size_t M0() const;
  const Params& params() const;
  // -1 when no live point and no tombstone remains.
  int max_level() const;
  // Meaningful when max_level() >= 0. The entry may be a tombstone.
  std::uint32_t entry_point() const;
  // True only for a live point.
  bool contains(std::uint32_t id) const;
  int level(std::uint32_t id) const;
  std::vector<std::uint32_t> neighbors(std::uint32_t id, int layer) const;
  void copy_vector(std::uint32_t id, float* out) const;

  // Copies `dim` floats. Cosine storage is L2-normalized. Returns the new id.
  [[nodiscard]] std::uint32_t insert(const float* vector);

  // `threads == 1` assigns ids in row order. Concurrent calls allocate ids as
  // they take the allocator lock, so ids are not row order.
  void insert_batch(const float* vectors, std::size_t count, std::size_t threads = 1);

  // Unlink `id` at every layer, repair in-neighbors, reuse the slot.
  // Throws if `id` is not live and not tombstoned.
  void erase(std::uint32_t id);

  // Leave the vector and the edges in place. Search skips the id.
  // The slot is not reused. Throws if `id` is not live.
  void tombstone(std::uint32_t id);

  // Layer-0 beam is max(ef, k). Upper layers use beam 1.
  // Results are sorted by increasing distance, then increasing id. At most k.
  [[nodiscard]] std::vector<Neighbor> search(const float* query, std::size_t k,
                                             std::size_t ef) const;

  void save(const std::filesystem::path& path) const;
  [[nodiscard]] static Index load(const std::filesystem::path& path);

 private:
  struct Impl;
  explicit Index(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Exact scan. Not safe for concurrent add and search.
class BruteForce {
 public:
  explicit BruteForce(std::size_t dim, Metric metric);
  BruteForce(const BruteForce&) = delete;
  BruteForce& operator=(const BruteForce&) = delete;
  BruteForce(BruteForce&&) noexcept;
  BruteForce& operator=(BruteForce&&) noexcept;
  ~BruteForce();

  std::size_t dim() const;
  std::size_t size() const;
  Metric metric() const;

  [[nodiscard]] std::uint32_t add(const float* vector);
  void add_batch(const float* vectors, std::size_t count);
  // Marks the id dead and keeps its slot so later ids do not move.
  void erase(std::uint32_t id);
  // Sorted by increasing distance, then increasing id.
  [[nodiscard]] std::vector<Neighbor> search(const float* query, std::size_t k) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hnsw
