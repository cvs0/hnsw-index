#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace hnsw {

namespace {

struct BetterThanWorst {
  bool operator()(const Neighbor& a, const Neighbor& b) const {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.id < b.id;
  }
};

bool better(const Neighbor& n, const Neighbor& worst) {
  if (n.distance != worst.distance) return n.distance < worst.distance;
  return n.id < worst.id;
}

}  // namespace

struct BruteForce::Impl {
  std::size_t dim_ = 0;
  Metric metric_ = Metric::L2;
  DistanceFunction dist_ = nullptr;
  std::vector<float> data_;

  Impl(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
    if (dim_ == 0 || dim_ > 65536) throw std::invalid_argument("hnsw: dim must be in [1, 65536]");
    if (metric_ != Metric::L2 && metric_ != Metric::InnerProduct && metric_ != Metric::Cosine) {
      throw std::invalid_argument("hnsw: unknown metric");
    }
    dist_ = distance_function(metric_ == Metric::L2 ? Metric::L2 : Metric::InnerProduct);
  }

  std::uint32_t add(const float* vector) {
    if (vector == nullptr) throw std::invalid_argument("hnsw: null vector");
    const auto id = static_cast<std::uint32_t>(data_.size() / dim_);
    data_.insert(data_.end(), vector, vector + dim_);
    if (metric_ == Metric::Cosine)
      l2_normalize(data_.data() + static_cast<std::size_t>(id) * dim_, dim_);
    return id;
  }

  std::vector<Neighbor> search(const float* query, std::size_t k) const {
    if (k == 0) return {};
    if (query == nullptr) throw std::invalid_argument("hnsw: null query");
    const std::size_t n = data_.size() / dim_;
    if (n == 0) return {};
    if (k > n) k = n;

    std::vector<float> qstorage;
    const float* q = query;
    if (metric_ == Metric::Cosine) {
      qstorage.assign(query, query + dim_);
      l2_normalize(qstorage.data(), dim_);
      q = qstorage.data();
    }

    std::priority_queue<Neighbor, std::vector<Neighbor>, BetterThanWorst> heap;
    for (std::size_t id = 0; id < n; ++id) {
      Neighbor nb{static_cast<std::uint32_t>(id), dist_(q, data_.data() + id * dim_, dim_)};
      if (heap.size() < k) {
        heap.push(nb);
      } else if (better(nb, heap.top())) {
        heap.pop();
        heap.push(nb);
      }
    }
    std::vector<Neighbor> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
      out.push_back(heap.top());
      heap.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
  }
};

BruteForce::BruteForce(std::size_t dim, Metric metric)
    : impl_(std::make_unique<Impl>(dim, metric)) {}
BruteForce::BruteForce(BruteForce&&) noexcept = default;
BruteForce& BruteForce::operator=(BruteForce&&) noexcept = default;
BruteForce::~BruteForce() = default;

std::size_t BruteForce::dim() const {
  return impl_->dim_;
}
std::size_t BruteForce::size() const {
  return impl_->data_.size() / impl_->dim_;
}
Metric BruteForce::metric() const {
  return impl_->metric_;
}
std::uint32_t BruteForce::add(const float* vector) {
  return impl_->add(vector);
}
void BruteForce::add_batch(const float* vectors, std::size_t count) {
  if (count == 0) return;
  if (vectors == nullptr) throw std::invalid_argument("hnsw: null vector");
  for (std::size_t i = 0; i < count; ++i) (void)add(vectors + i * impl_->dim_);
}
std::vector<Neighbor> BruteForce::search(const float* query, std::size_t k) const {
  return impl_->search(query, k);
}

}  // namespace hnsw
