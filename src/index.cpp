#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "hnsw/hnsw.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace hnsw {

namespace {

constexpr int kMaxLevel = 16;
constexpr std::uint32_t kChunkShift = 10;
constexpr std::uint32_t kChunkSize = 1u << kChunkShift;
constexpr std::uint32_t kChunkMask = kChunkSize - 1;
constexpr char kMagic[8] = {'H', 'N', 'S', 'W', '0', '0', '0', '1'};
constexpr std::uint32_t kVersionDense = 1;
constexpr std::uint32_t kVersionSlots = 2;

enum class SlotState : std::uint8_t { Empty = 0, Live = 1, Tombstone = 2, Free = 3 };

struct Candidate {
  float dist = 0.f;
  std::uint32_t id = 0;
};

class Heap {
 public:
  enum class Order { Min, Max };

  explicit Heap(Order order) : order_(order) {}

  void clear() { a_.clear(); }
  void reserve(std::size_t n) { a_.reserve(n); }
  bool empty() const { return a_.empty(); }
  std::size_t size() const { return a_.size(); }
  const Candidate& top() const { return a_[0]; }

  void push(Candidate c) {
    a_.push_back(c);
    sift_up(a_.size() - 1);
  }

  void pop() {
    a_[0] = a_.back();
    a_.pop_back();
    if (!a_.empty()) sift_down(0);
  }

  // Ascending distance, then ascending id.
  std::vector<Candidate> take_sorted() {
    std::vector<Candidate> out;
    out.reserve(a_.size());
    while (!empty()) {
      out.push_back(top());
      pop();
    }
    if (order_ == Order::Max) std::reverse(out.begin(), out.end());
    return out;
  }

 private:
  Order order_;
  std::vector<Candidate> a_;

  bool higher(const Candidate& a, const Candidate& b) const {
    if (order_ == Order::Min) {
      if (a.dist != b.dist) return a.dist < b.dist;
      return a.id < b.id;
    }
    if (a.dist != b.dist) return a.dist > b.dist;
    return a.id > b.id;
  }

  void sift_up(std::size_t i) {
    while (i > 0) {
      const std::size_t p = (i - 1) / 2;
      if (!higher(a_[i], a_[p])) break;
      std::swap(a_[i], a_[p]);
      i = p;
    }
  }

  void sift_down(std::size_t i) {
    for (;;) {
      const std::size_t l = i * 2 + 1;
      const std::size_t r = l + 1;
      std::size_t best = i;
      if (l < a_.size() && higher(a_[l], a_[best])) best = l;
      if (r < a_.size() && higher(a_[r], a_[best])) best = r;
      if (best == i) break;
      std::swap(a_[i], a_[best]);
      i = best;
    }
  }
};

struct SearchTLS {
  std::vector<std::uint32_t> tags;
  std::uint32_t token = 1;
  std::vector<std::uint32_t> neigh;
  Heap cand{Heap::Order::Min};
  Heap result{Heap::Order::Max};
  std::vector<float> qbuf;
};

SearchTLS& search_tls() {
  static thread_local SearchTLS tls;
  return tls;
}

std::uint32_t next_token(SearchTLS& tls) {
  ++tls.token;
  if (tls.token == 0) {
    std::fill(tls.tags.begin(), tls.tags.end(), 0);
    tls.token = 1;
  }
  return tls.token;
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

void prefetch_range(const void* p, std::size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
  const auto* c = static_cast<const char*>(p);
  for (std::size_t off = 0; off < bytes; off += 64) {
    _mm_prefetch(c + off, _MM_HINT_T0);
  }
#else
  (void)p;
  (void)bytes;
#endif
}

// P(L >= l) = M^{-l} when the low 53 bits are uniform on [0, 1).
std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

bool host_little_endian() {
  const std::uint16_t x = 1;
  return *reinterpret_cast<const unsigned char*>(&x) == 1;
}

void write_u32(std::ostream& out, std::uint32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                        static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

void write_u64(std::ostream& out, std::uint64_t v) {
  write_u32(out, static_cast<std::uint32_t>(v));
  write_u32(out, static_cast<std::uint32_t>(v >> 32));
}

void write_i32(std::ostream& out, std::int32_t v) {
  write_u32(out, static_cast<std::uint32_t>(v));
}

std::uint32_t read_u32(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (in.gcount() != 4) throw std::runtime_error("hnsw: truncated file");
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t read_u64(std::istream& in) {
  const std::uint64_t lo = read_u32(in);
  const std::uint64_t hi = read_u32(in);
  return lo | (hi << 32);
}

std::int32_t read_i32(std::istream& in) {
  return static_cast<std::int32_t>(read_u32(in));
}

}  // namespace

struct Node {
  Node() = default;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  std::mutex mu;
  int level = 0;
  int slot_level = -1;
  std::atomic<SlotState> state{SlotState::Empty};
  std::atomic<std::uint32_t> gen{0};
  // seq[layer] is even when the layer's neighbor list is stable.
  std::atomic<std::uint32_t>* seq = nullptr;
  std::atomic<std::uint32_t>* degree = nullptr;
  std::atomic<std::uint32_t>* slots = nullptr;
  // inbound[layer] lists nodes whose forward list at that layer contains this id.
  std::vector<std::vector<std::uint32_t>> inbound;
};

struct NodeBlock {
  static constexpr std::uint32_t kSize = 256;
  Node nodes[kSize];
};

struct VecChunk {
  float* data = nullptr;
  // ready[i] is released after vector i in this chunk is fully written.
  std::unique_ptr<std::atomic<std::uint8_t>[]> ready;
  explicit VecChunk(std::size_t nvec, std::size_t dim) {
    const std::size_t bytes = nvec * dim * sizeof(float);
    void* mem = std::aligned_alloc(32, bytes);
    if (mem == nullptr) throw std::bad_alloc();
    data = static_cast<float*>(mem);
    std::memset(data, 0, bytes);
    ready.reset(new std::atomic<std::uint8_t>[nvec]());
  }
  ~VecChunk() { std::free(data); }
  VecChunk(const VecChunk&) = delete;
  VecChunk& operator=(const VecChunk&) = delete;
};

template <typename T>
struct AtomicTable {
  std::size_t cap = 0;
  std::unique_ptr<std::atomic<T*>[]> slots;

  static std::unique_ptr<AtomicTable> make(std::size_t cap) {
    auto t = std::make_unique<AtomicTable>();
    t->cap = cap;
    t->slots.reset(new std::atomic<T*>[cap]());
    return t;
  }
};

struct Arena {
  struct Chunk {
    std::unique_ptr<std::atomic<std::uint32_t>[]> data;
    std::size_t cap = 0;
    std::size_t used = 0;
  };
  std::vector<Chunk> chunks;

  std::atomic<std::uint32_t>* alloc(std::size_t n) {
    if (n == 0) return nullptr;
    if (chunks.empty() || chunks.back().used + n > chunks.back().cap) {
      Chunk c;
      c.cap = std::max<std::size_t>(n, 1u << 16);
      c.data.reset(new std::atomic<std::uint32_t>[c.cap]());
      chunks.push_back(std::move(c));
    }
    Chunk& c = chunks.back();
    auto* p = c.data.get() + c.used;
    c.used += n;
    return p;
  }

  std::size_t bytes() const {
    std::size_t n = 0;
    for (const Chunk& c : chunks) n += c.cap * sizeof(std::atomic<std::uint32_t>);
    return n;
  }
};

bool improves(float d, std::uint32_t id, const Candidate& worst) {
  if (d < worst.dist) return true;
  if (d > worst.dist) return false;
  return id < worst.id;
}

struct Index::Impl {
  using NodeTable = AtomicTable<Node>;
  using VecTable = AtomicTable<VecChunk>;

  std::size_t dim_ = 0;
  Params params_{};
  std::size_t M_ = 0;
  std::size_t M0_ = 0;
  std::size_t ef_construction_ = 0;
  std::uint64_t seed_ = 0;
  double mL_ = 1.0;
  DistanceFunction dist_ = nullptr;

  std::atomic<std::uint32_t> n_{0};
  std::atomic<std::uint32_t> live_{0};
  std::atomic<std::uint32_t> level_count_[kMaxLevel + 1]{};
  std::atomic<int> max_level_{-1};
  std::atomic<std::uint32_t> entry_{0};
  std::atomic<bool> ready_{false};
  std::mutex alloc_mu_;
  std::mutex graph_mu_;
  std::mutex entry_mu_;
  std::mutex reclaim_mu_;
  std::vector<std::uint32_t> freelist_;
  struct Retired {
    std::uint32_t id = 0;
    std::uint64_t epoch = 0;
  };
  std::vector<Retired> retired_;
  mutable std::atomic<std::uint64_t> epoch_clock_{1};
  static constexpr int kMaxPins = 4096;
  mutable std::atomic<std::uint64_t> pins_[kMaxPins];
  mutable std::atomic<int> pin_used_{0};

  Arena arena_;
  std::vector<std::unique_ptr<NodeBlock>> blocks_;
  std::vector<std::unique_ptr<VecChunk>> chunks_;
  std::vector<std::unique_ptr<NodeTable>> node_tables_;
  std::vector<std::unique_ptr<VecTable>> vec_tables_;
  std::atomic<NodeTable*> node_table_{nullptr};
  std::atomic<VecTable*> vec_table_{nullptr};

  Impl(std::size_t dim, Params params) : dim_(dim), params_(std::move(params)) {
    if (dim_ == 0 || dim_ > 65536) throw std::invalid_argument("hnsw: dim must be in [1, 65536]");
    if (params_.M < 2 || params_.M > 65536) {
      throw std::invalid_argument("hnsw: M must be in [2, 65536]");
    }
    if (params_.ef_construction == 0 || params_.ef_construction > 10000000) {
      throw std::invalid_argument("hnsw: ef_construction must be in [1, 10000000]");
    }
    if (params_.metric != Metric::L2 && params_.metric != Metric::InnerProduct &&
        params_.metric != Metric::Cosine) {
      throw std::invalid_argument("hnsw: unknown metric");
    }
    M_ = params_.M;
    M0_ = params_.M * 2;
    ef_construction_ = params_.ef_construction;
    seed_ = params_.seed;
    mL_ = 1.0 / std::log(static_cast<double>(M_));
    dist_ = distance_function(params_.metric == Metric::L2 ? Metric::L2 : Metric::InnerProduct);
    node_tables_.push_back(NodeTable::make(32));
    node_table_.store(node_tables_.back().get(), std::memory_order_relaxed);
    vec_tables_.push_back(VecTable::make(4));
    vec_table_.store(vec_tables_.back().get(), std::memory_order_relaxed);
    for (std::atomic<std::uint64_t>& pin : pins_) pin.store(0, std::memory_order_relaxed);
  }

  int level_for(std::uint32_t id) const {
    const std::uint64_t h =
        splitmix64(seed_ ^ (static_cast<std::uint64_t>(id) * 0x9E3779B97F4A7C15ULL));
    const double u = static_cast<double>(h >> 11) * 0x1.0p-53;
    const double x = std::max(1.0 - u, std::numeric_limits<double>::min());
    int level = static_cast<int>(std::floor(-std::log(x) * mL_));
    if (level < 0) level = 0;
    if (level > kMaxLevel) level = kMaxLevel;
    return level;
  }

  std::size_t max_m(int layer) const { return layer == 0 ? M0_ : M_; }

  std::size_t layer_offset(int layer) const {
    if (layer <= 0) return 0;
    return M0_ + static_cast<std::size_t>(layer - 1) * M_;
  }

  void ensure_storage(std::uint32_t id) {
    NodeTable* nt = node_table_.load(std::memory_order_relaxed);
    if (id >= nt->cap) {
      std::size_t cap = nt->cap;
      while (cap <= id) cap *= 2;
      auto bigger = NodeTable::make(cap);
      for (std::size_t i = 0; i < nt->cap; ++i) {
        bigger->slots[i].store(nt->slots[i].load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
      }
      NodeTable* raw = bigger.get();
      node_tables_.push_back(std::move(bigger));
      node_table_.store(raw, std::memory_order_release);
      nt = raw;
    }

    const std::uint32_t chunk = id >> kChunkShift;
    VecTable* vt = vec_table_.load(std::memory_order_relaxed);
    if (chunk >= vt->cap) {
      std::size_t cap = vt->cap;
      while (cap <= chunk) cap *= 2;
      auto bigger = VecTable::make(cap);
      for (std::size_t i = 0; i < vt->cap; ++i) {
        bigger->slots[i].store(vt->slots[i].load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
      }
      VecTable* raw = bigger.get();
      vec_tables_.push_back(std::move(bigger));
      vec_table_.store(raw, std::memory_order_release);
      vt = raw;
    }
    if (vt->slots[chunk].load(std::memory_order_relaxed) == nullptr) {
      auto owned = std::make_unique<VecChunk>(kChunkSize, dim_);
      vt->slots[chunk].store(owned.get(), std::memory_order_release);
      chunks_.push_back(std::move(owned));
    }

    const std::size_t block_index = static_cast<std::size_t>(id) / NodeBlock::kSize;
    while (blocks_.size() <= block_index) blocks_.push_back(std::make_unique<NodeBlock>());
  }

  float* vector_ptr_unlocked(std::uint32_t id) const {
    VecTable* table = vec_table_.load(std::memory_order_relaxed);
    VecChunk* chunk = table->slots[id >> kChunkShift].load(std::memory_order_relaxed);
    return chunk->data + static_cast<std::size_t>(id & kChunkMask) * dim_;
  }

  const float* vector_ptr(std::uint32_t id) const {
    const std::uint32_t chunk_index = id >> kChunkShift;
    const std::uint32_t off = id & kChunkMask;
    for (int attempt = 0; attempt < 100000; ++attempt) {
      VecTable* table = vec_table_.load(std::memory_order_acquire);
      if (table != nullptr && chunk_index < table->cap) {
        VecChunk* chunk = table->slots[chunk_index].load(std::memory_order_acquire);
        if (chunk != nullptr && chunk->ready[off].load(std::memory_order_acquire) != 0) {
          return chunk->data + static_cast<std::size_t>(off) * dim_;
        }
      }
      if ((attempt & 63) == 0)
        std::this_thread::yield();
      else
        cpu_relax();
    }
    throw std::logic_error("hnsw: missing vector");
  }

  Node* node_ptr(std::uint32_t id) const {
    for (int attempt = 0; attempt < 128; ++attempt) {
      NodeTable* table = node_table_.load(std::memory_order_acquire);
      if (table != nullptr && id < table->cap) {
        Node* node = table->slots[id].load(std::memory_order_acquire);
        if (node != nullptr) return node;
      }
      std::this_thread::yield();
    }
    throw std::logic_error("hnsw: missing node");
  }

  Node* node_at(std::uint32_t id) const {
    return &blocks_[static_cast<std::size_t>(id) / NodeBlock::kSize]->nodes[id % NodeBlock::kSize];
  }

  // Level is a function of id, so a reused slot keeps the arrays from its first adopt.
  std::uint32_t adopt_slot(std::uint32_t id, int level) {
    Node* node = node_at(id);
    if (node->slots == nullptr) {
      const std::size_t nlayers = static_cast<std::size_t>(level) + 1;
      node->seq = arena_.alloc(nlayers);
      node->degree = arena_.alloc(nlayers);
      node->slots = arena_.alloc(M0_ + static_cast<std::size_t>(level) * M_);
      node->slot_level = level;
      node->inbound.assign(nlayers, {});
      node->gen.store(1, std::memory_order_relaxed);
    } else {
      if (node->slot_level != level) throw std::logic_error("hnsw: slot level changed");
      for (auto& in : node->inbound) in.clear();
      node->gen.fetch_add(1, std::memory_order_relaxed);
    }
    node->level = level;
    for (int layer = 0; layer <= level; ++layer) {
      node->degree[layer].store(0, std::memory_order_relaxed);
      node->seq[layer].store(0, std::memory_order_relaxed);
    }
    return node->gen.load(std::memory_order_relaxed);
  }

  Node* peek_node(std::uint32_t id) const {
    NodeTable* table = node_table_.load(std::memory_order_acquire);
    if (table == nullptr || id >= table->cap) return nullptr;
    return table->slots[id].load(std::memory_order_acquire);
  }

  SlotState state_of(std::uint32_t id) const {
    Node* node = peek_node(id);
    if (node == nullptr) return SlotState::Empty;
    return node->state.load(std::memory_order_acquire);
  }

  bool is_live(std::uint32_t id) const { return state_of(id) == SlotState::Live; }

  bool occupies(std::uint32_t id) const {
    const SlotState st = state_of(id);
    return st == SlotState::Live || st == SlotState::Tombstone;
  }

  static bool contains_id(const std::vector<std::uint32_t>& v, std::uint32_t id) {
    return std::find(v.begin(), v.end(), id) != v.end();
  }

  void add_in(std::uint32_t dst, int layer, std::uint32_t src) {
    if (dst == src) return;
    Node* node = peek_node(dst);
    if (node == nullptr || layer < 0 || layer >= static_cast<int>(node->inbound.size())) return;
    auto& in = node->inbound[static_cast<std::size_t>(layer)];
    if (std::find(in.begin(), in.end(), src) == in.end()) in.push_back(src);
  }

  void remove_in(std::uint32_t dst, int layer, std::uint32_t src) {
    if (dst == src) return;
    Node* node = peek_node(dst);
    if (node == nullptr || layer < 0 || layer >= static_cast<int>(node->inbound.size())) return;
    auto& in = node->inbound[static_cast<std::size_t>(layer)];
    in.erase(std::remove(in.begin(), in.end(), src), in.end());
  }

  void write_links(std::uint32_t id, int layer, const std::vector<std::uint32_t>& neu) {
    Node* node = node_ptr(id);
    std::lock_guard<std::mutex> lock(node->mu);
    if (layer < 0 || layer > node->level) return;
    const std::vector<std::uint32_t> old = snapshot_locked(*node, layer);
    publish_locked(*node, layer, neu.data(), static_cast<std::uint32_t>(neu.size()));
    for (std::uint32_t x : old) {
      if (!contains_id(neu, x)) remove_in(x, layer, id);
    }
    for (std::uint32_t x : neu) {
      if (!contains_id(old, x)) add_in(x, layer, id);
    }
  }

  std::atomic<std::uint64_t>* pin_slot() const {
    static thread_local int index = -1;
    if (index < 0) {
      const int claimed = pin_used_.fetch_add(1, std::memory_order_relaxed);
      if (claimed >= kMaxPins) throw std::logic_error("hnsw: too many search threads");
      index = claimed;
    }
    return &pins_[index];
  }

  void try_reclaim_locked() {
    std::uint64_t min_pin = std::numeric_limits<std::uint64_t>::max();
    bool any = false;
    int used = pin_used_.load(std::memory_order_acquire);
    if (used > kMaxPins) used = kMaxPins;
    for (int i = 0; i < used; ++i) {
      const std::uint64_t p = pins_[i].load(std::memory_order_acquire);
      if (p == 0) continue;
      any = true;
      if (p < min_pin) min_pin = p;
    }
    if (!any) min_pin = std::numeric_limits<std::uint64_t>::max();
    std::size_t keep = 0;
    for (const Retired& item : retired_) {
      if (item.epoch < min_pin) {
        freelist_.push_back(item.id);
      } else {
        retired_[keep++] = item;
      }
    }
    retired_.resize(keep);
  }

  void retire(std::uint32_t id) {
    std::lock_guard<std::mutex> lock(reclaim_mu_);
    const std::uint64_t epoch = epoch_clock_.load(std::memory_order_relaxed);
    retired_.push_back(Retired{id, epoch});
    epoch_clock_.fetch_add(1, std::memory_order_release);
    try_reclaim_locked();
  }

  std::uint32_t find_occupant(int level, const std::vector<std::uint32_t>& prefer) const {
    for (std::uint32_t id : prefer) {
      Node* node = peek_node(id);
      if (node != nullptr && occupies(id) && node->level == level) return id;
    }
    const std::uint32_t hwm = n_.load(std::memory_order_relaxed);
    for (std::uint32_t id = 0; id < hwm; ++id) {
      Node* node = peek_node(id);
      if (node != nullptr && occupies(id) && node->level == level) return id;
    }
    return std::numeric_limits<std::uint32_t>::max();
  }

  // Caller holds node.mu, or is the only thread that can observe the node.
  void publish_locked(Node& node, int layer, const std::uint32_t* ids, std::uint32_t n) {
    const std::size_t cap = max_m(layer);
    if (n > cap) throw std::logic_error("hnsw: neighbor list exceeds capacity");
    auto* slots = node.slots + layer_offset(layer);
    std::atomic<std::uint32_t>& seq = node.seq[layer];
    const std::uint32_t s = seq.load(std::memory_order_relaxed);
    seq.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    node.degree[layer].store(n, std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < n; ++i) slots[i].store(ids[i], std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    seq.store(s + 2, std::memory_order_release);
  }

  std::vector<std::uint32_t> snapshot_locked(const Node& node, int layer) const {
    std::uint32_t d = node.degree[layer].load(std::memory_order_relaxed);
    const auto cap = static_cast<std::uint32_t>(max_m(layer));
    if (d > cap) d = cap;
    std::vector<std::uint32_t> out(d);
    auto* slots = node.slots + layer_offset(layer);
    for (std::uint32_t i = 0; i < d; ++i) out[i] = slots[i].load(std::memory_order_relaxed);
    return out;
  }

  void read_links(const Node& node, int layer, std::vector<std::uint32_t>& out) const {
    out.clear();
    if (layer < 0 || layer > node.level) return;
    auto* slots = node.slots + layer_offset(layer);
    const auto cap = static_cast<std::uint32_t>(max_m(layer));
    std::uint32_t spins = 0;
    for (;;) {
      const std::uint32_t s1 = node.seq[layer].load(std::memory_order_acquire);
      if ((s1 & 1u) != 0) {
        if ((++spins & 63u) == 0)
          std::this_thread::yield();
        else
          cpu_relax();
        continue;
      }
      std::uint32_t d = node.degree[layer].load(std::memory_order_relaxed);
      if (d > cap) d = cap;
      out.resize(d);
      for (std::uint32_t i = 0; i < d; ++i) out[i] = slots[i].load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint32_t s2 = node.seq[layer].load(std::memory_order_relaxed);
      if (s1 == s2) return;
    }
  }

  // Algorithm 4 with extendCandidates = false and keepPrunedConnections = false.
  // Keep e when no selected neighbor is strictly closer to e than the query is.
  std::vector<std::uint32_t> select_from(std::vector<Neighbor> cand, std::size_t max_keep,
                                         std::uint32_t self) const {
    std::sort(cand.begin(), cand.end(), [](const Neighbor& a, const Neighbor& b) {
      if (a.distance != b.distance) return a.distance < b.distance;
      return a.id < b.id;
    });
    std::vector<std::uint32_t> selected;
    selected.reserve(std::min(max_keep, cand.size()));
    for (const Neighbor& c : cand) {
      if (selected.size() >= max_keep) break;
      if (c.id == self || !is_live(c.id)) continue;
      if (std::find(selected.begin(), selected.end(), c.id) != selected.end()) continue;
      bool good = true;
      const float* cv = vector_ptr(c.id);
      for (std::uint32_t s : selected) {
        const float ds = dist_(cv, vector_ptr(s), dim_);
        if (ds < c.distance) {
          good = false;
          break;
        }
      }
      if (good) selected.push_back(c.id);
    }
    return selected;
  }

  std::vector<Neighbor> search_layer(const float* q, std::uint32_t ep, std::size_t ef,
                                     int layer) const {
    SearchTLS& tls = search_tls();
    const std::uint32_t token = next_token(tls);
    const std::uint32_t nnow = n_.load(std::memory_order_acquire);
    if (tls.tags.size() < nnow) tls.tags.resize(nnow, 0);
    if (tls.neigh.capacity() < M0_) tls.neigh.reserve(M0_);

    Heap& cand = tls.cand;
    Heap& result = tls.result;
    cand.clear();
    result.clear();
    cand.reserve(ef);
    result.reserve(ef);

    const std::size_t bytes = dim_ * sizeof(float);
    auto consider = [&](std::uint32_t id) {
      if (id >= tls.tags.size()) tls.tags.resize(static_cast<std::size_t>(id) + 1, 0);
      if (tls.tags[id] == token) return;
      const SlotState st = state_of(id);
      if (st != SlotState::Live && st != SlotState::Tombstone) return;
      tls.tags[id] = token;
      const float d = dist_(q, vector_ptr(id), dim_);
      // A tombstone stays on the candidate heap so its edges are expanded,
      // and it does not enter the result beam.
      if (st == SlotState::Tombstone) {
        cand.push(Candidate{d, id});
        return;
      }
      if (result.size() < ef || improves(d, id, result.top())) {
        cand.push(Candidate{d, id});
        result.push(Candidate{d, id});
        if (result.size() > ef) result.pop();
      }
    };

    consider(ep);
    while (!cand.empty()) {
      const Candidate c = cand.top();
      cand.pop();
      if (!result.empty() && result.size() >= ef && c.dist > result.top().dist) break;
      Node* node = node_ptr(c.id);
      if (layer > node->level) continue;
      read_links(*node, layer, tls.neigh);
      for (std::size_t i = 0; i < tls.neigh.size(); ++i) {
        if (i + 1 < tls.neigh.size() && occupies(tls.neigh[i + 1])) {
          prefetch_range(vector_ptr(tls.neigh[i + 1]), bytes);
        }
        consider(tls.neigh[i]);
      }
    }

    const std::vector<Candidate> sorted = result.take_sorted();
    std::vector<Neighbor> out;
    out.reserve(sorted.size());
    for (const Candidate& c : sorted) out.push_back(Neighbor{c.id, c.dist});
    return out;
  }

  void add_reverse(std::uint32_t eid, int layer, std::uint32_t q, std::size_t maxm) {
    if (!is_live(eid) || !is_live(q)) return;
    std::vector<std::uint32_t> cur;
    read_links(*node_ptr(eid), layer, cur);
    if (contains_id(cur, q)) return;
    cur.push_back(q);
    if (cur.size() > maxm) {
      const float* origin = vector_ptr(eid);
      std::vector<Neighbor> cand;
      cand.reserve(cur.size());
      for (std::uint32_t x : cur) {
        if (x == eid || !is_live(x)) continue;
        cand.push_back(Neighbor{x, dist_(origin, vector_ptr(x), dim_)});
      }
      cur = select_from(std::move(cand), maxm, eid);
    }
    write_links(eid, layer, cur);
  }

  void link_node(std::uint32_t id, std::uint32_t gen, const float* vec) {
    std::lock_guard<std::mutex> graph(graph_mu_);
    Node* node = peek_node(id);
    if (node == nullptr || node->gen.load(std::memory_order_relaxed) != gen) return;
    if (node->state.load(std::memory_order_relaxed) != SlotState::Live) return;
    while (!ready_.load(std::memory_order_acquire)) std::this_thread::yield();

    const int level = node->level;
    const int top = max_level_.load(std::memory_order_acquire);
    std::uint32_t ep = entry_.load(std::memory_order_acquire);
    for (int lc = top; lc > level; --lc) {
      const std::vector<Neighbor> found = search_layer(vec, ep, 1, lc);
      if (!found.empty()) ep = found.front().id;
    }
    const int limit = std::min(level, top);
    for (int lc = limit; lc >= 0; --lc) {
      const std::vector<Neighbor> found = search_layer(vec, ep, ef_construction_, lc);
      const std::vector<std::uint32_t> sel = select_from(found, max_m(lc), id);
      write_links(id, lc, sel);
      for (std::uint32_t e : sel) {
        if (e == id) continue;
        add_reverse(e, lc, id, max_m(lc));
      }
      if (!found.empty()) ep = found.front().id;
    }

    if (node->gen.load(std::memory_order_relaxed) != gen) return;
    if (level > max_level_.load(std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(entry_mu_);
      if (level > max_level_.load(std::memory_order_relaxed)) {
        entry_.store(id, std::memory_order_relaxed);
        max_level_.store(level, std::memory_order_release);
      }
    }
  }

  std::uint32_t insert(const float* vector) {
    if (vector == nullptr) throw std::invalid_argument("hnsw: null vector");
    std::unique_lock<std::mutex> alloc(alloc_mu_);
    bool reused = false;
    std::uint32_t id = 0;
    {
      std::lock_guard<std::mutex> reclaim(reclaim_mu_);
      try_reclaim_locked();
      if (!freelist_.empty()) {
        id = freelist_.back();
        freelist_.pop_back();
        reused = true;
      } else {
        id = n_.load(std::memory_order_relaxed);
      }
    }
    const bool first = max_level_.load(std::memory_order_relaxed) < 0;
    ensure_storage(id);
    VecChunk* chunk = vec_table_.load(std::memory_order_relaxed)
                          ->slots[id >> kChunkShift]
                          .load(std::memory_order_relaxed);
    if (reused) chunk->ready[id & kChunkMask].store(0, std::memory_order_release);
    float* dst = vector_ptr_unlocked(id);
    std::memcpy(dst, vector, dim_ * sizeof(float));
    if (params_.metric == Metric::Cosine) l2_normalize(dst, dim_);
    chunk->ready[id & kChunkMask].store(1, std::memory_order_release);
    const int level = level_for(id);
    const std::uint32_t gen = adopt_slot(id, level);
    Node* node = node_at(id);
    node->state.store(SlotState::Live, std::memory_order_release);
    node_table_.load(std::memory_order_relaxed)->slots[id].store(node, std::memory_order_release);
    level_count_[level].fetch_add(1, std::memory_order_relaxed);
    live_.fetch_add(1, std::memory_order_release);
    if (!reused) n_.store(id + 1, std::memory_order_release);
    if (first) {
      entry_.store(id, std::memory_order_relaxed);
      max_level_.store(level, std::memory_order_relaxed);
      ready_.store(true, std::memory_order_release);
    }
    alloc.unlock();
    if (first) return id;
    link_node(id, gen, dst);
    return id;
  }

  void insert_batch(const float* vectors, std::size_t count, std::size_t threads) {
    if (count == 0) return;
    if (vectors == nullptr) throw std::invalid_argument("hnsw: null vector");
    if (threads == 0) threads = 1;
    if (threads == 1) {
      for (std::size_t i = 0; i < count; ++i) insert(vectors + i * dim_);
      return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
      pool.emplace_back([&] {
        for (;;) {
          const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= count) break;
          insert(vectors + i * dim_);
        }
      });
    }
    for (std::thread& th : pool) th.join();
  }

  std::vector<Neighbor> search(const float* query, std::size_t k, std::size_t ef) const {
    if (k == 0) return {};
    if (query == nullptr) throw std::invalid_argument("hnsw: null query");
    if (k > 10000000 || ef > 10000000) throw std::invalid_argument("hnsw: ef/k too large");
    // Publish the sampled epoch, then confirm the clock did not move. A
    // reclaim that ran between the load and the store bumps the clock, so
    // the retry observes it. Epoch 0 means the pin is inactive.
    struct Pin {
      std::atomic<std::uint64_t>* slot = nullptr;
      Pin(std::atomic<std::uint64_t>* s, const std::atomic<std::uint64_t>& clock) : slot(s) {
        for (;;) {
          const std::uint64_t epoch = clock.load(std::memory_order_acquire);
          slot->store(epoch, std::memory_order_release);
          if (clock.load(std::memory_order_acquire) == epoch) break;
        }
      }
      ~Pin() { slot->store(0, std::memory_order_release); }
      Pin(const Pin&) = delete;
      Pin& operator=(const Pin&) = delete;
    };
    Pin pin(pin_slot(), epoch_clock_);

    const std::uint32_t live = live_.load(std::memory_order_acquire);
    if (live == 0 || max_level_.load(std::memory_order_acquire) < 0) return {};
    if (k > live) k = live;
    std::size_t beam = std::max(ef, k);
    if (beam > live) beam = live;
    if (beam == 0) return {};

    const float* q = query;
    if (params_.metric == Metric::Cosine) {
      SearchTLS& tls = search_tls();
      tls.qbuf.resize(dim_);
      std::memcpy(tls.qbuf.data(), query, dim_ * sizeof(float));
      l2_normalize(tls.qbuf.data(), dim_);
      q = tls.qbuf.data();
    }

    const int top = max_level_.load(std::memory_order_acquire);
    std::uint32_t ep = entry_.load(std::memory_order_acquire);
    for (int lc = top; lc > 0; --lc) {
      const std::vector<Neighbor> found = search_layer(q, ep, 1, lc);
      if (!found.empty()) ep = found.front().id;
    }
    std::vector<Neighbor> found = search_layer(q, ep, beam, 0);
    std::vector<Neighbor> kept;
    kept.reserve(std::min(found.size(), k));
    for (const Neighbor& hit : found) {
      if (!is_live(hit.id)) continue;
      kept.push_back(hit);
      if (kept.size() == k) break;
    }
    return kept;
  }

  void tombstone(std::uint32_t id) {
    std::lock_guard<std::mutex> graph(graph_mu_);
    if (id >= n_.load(std::memory_order_relaxed)) throw std::out_of_range("hnsw: id");
    Node* node = peek_node(id);
    if (node == nullptr || node->state.load(std::memory_order_relaxed) != SlotState::Live) {
      throw std::out_of_range("hnsw: id");
    }
    node->state.store(SlotState::Tombstone, std::memory_order_release);
    live_.fetch_sub(1, std::memory_order_release);
  }

  void erase(std::uint32_t id) {
    std::uint32_t retired_id = 0;
    bool do_retire = false;
    {
      std::lock_guard<std::mutex> graph(graph_mu_);
      if (id >= n_.load(std::memory_order_relaxed)) throw std::out_of_range("hnsw: id");
      Node* node = peek_node(id);
      if (node == nullptr) throw std::out_of_range("hnsw: id");
      const SlotState st = node->state.load(std::memory_order_relaxed);
      if (st != SlotState::Live && st != SlotState::Tombstone) throw std::out_of_range("hnsw: id");
      const int level = node->level;
      std::vector<std::vector<std::uint32_t>> outs(static_cast<std::size_t>(level) + 1);
      std::vector<std::vector<std::uint32_t>> ins(static_cast<std::size_t>(level) + 1);
      for (int layer = 0; layer <= level; ++layer) {
        read_links(*node, layer, outs[static_cast<std::size_t>(layer)]);
        ins[static_cast<std::size_t>(layer)] = node->inbound[static_cast<std::size_t>(layer)];
      }
      node->gen.fetch_add(1, std::memory_order_relaxed);
      node->state.store(SlotState::Free, std::memory_order_release);
      if (st == SlotState::Live) live_.fetch_sub(1, std::memory_order_acq_rel);
      level_count_[level].fetch_sub(1, std::memory_order_acq_rel);

      for (int layer = level; layer >= 0; --layer) {
        write_links(id, layer, {});
        for (std::uint32_t u : ins[static_cast<std::size_t>(layer)]) {
          if (u == id) continue;
          Node* un = peek_node(u);
          if (un == nullptr || layer > un->level) continue;
          std::vector<std::uint32_t> cur;
          read_links(*un, layer, cur);
          const auto it = std::find(cur.begin(), cur.end(), id);
          if (it == cur.end()) continue;
          cur.erase(it);
          if (un->state.load(std::memory_order_relaxed) != SlotState::Live) {
            write_links(u, layer, cur);
            continue;
          }
          const float* origin = vector_ptr(u);
          std::vector<Neighbor> cand;
          auto push = [&](std::uint32_t other) {
            if (other == u || other == id || !is_live(other)) return;
            cand.push_back(Neighbor{other, dist_(origin, vector_ptr(other), dim_)});
          };
          for (std::uint32_t x : cur) push(x);
          for (std::uint32_t x : outs[static_cast<std::size_t>(layer)]) push(x);
          for (std::uint32_t x : ins[static_cast<std::size_t>(layer)]) push(x);
          const std::vector<Neighbor> found = search_layer(origin, u, ef_construction_, layer);
          for (const Neighbor& hit : found) push(hit.id);
          const std::vector<std::uint32_t> sel = select_from(std::move(cand), max_m(layer), u);
          write_links(u, layer, sel);
        }
      }
      for (auto& in : node->inbound) in.clear();

      int top = kMaxLevel;
      while (top >= 0 && level_count_[top].load(std::memory_order_relaxed) == 0) --top;
      if (top < 0) {
        max_level_.store(-1, std::memory_order_relaxed);
        entry_.store(0, std::memory_order_relaxed);
        ready_.store(false, std::memory_order_release);
      } else {
        std::uint32_t ep = entry_.load(std::memory_order_relaxed);
        Node* en = peek_node(ep);
        const bool entry_ok = en != nullptr && occupies(ep) && en->level == top;
        if (!entry_ok) {
          std::vector<std::uint32_t> prefer;
          for (int layer = level; layer >= 0; --layer) {
            prefer.insert(prefer.end(), outs[static_cast<std::size_t>(layer)].begin(),
                          outs[static_cast<std::size_t>(layer)].end());
            prefer.insert(prefer.end(), ins[static_cast<std::size_t>(layer)].begin(),
                          ins[static_cast<std::size_t>(layer)].end());
          }
          ep = find_occupant(top, prefer);
          if (ep == std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("hnsw: missing entry");
          }
        }
        entry_.store(ep, std::memory_order_relaxed);
        max_level_.store(top, std::memory_order_release);
        ready_.store(true, std::memory_order_release);
      }
      retired_id = id;
      do_retire = true;
    }
    if (do_retire) retire(retired_id);
  }

  void copy_vector(std::uint32_t id, float* out) const {
    if (out == nullptr) throw std::invalid_argument("hnsw: null output");
    if (!occupies(id)) throw std::out_of_range("hnsw: id");
    std::memcpy(out, vector_ptr(id), dim_ * sizeof(float));
  }

  int level_of(std::uint32_t id) const {
    if (!occupies(id)) throw std::out_of_range("hnsw: id");
    return peek_node(id)->level;
  }

  std::vector<std::uint32_t> neighbors_of(std::uint32_t id, int layer) const {
    if (!occupies(id)) throw std::out_of_range("hnsw: id");
    Node* node = peek_node(id);
    if (layer < 0 || layer > node->level) throw std::out_of_range("hnsw: layer");
    std::vector<std::uint32_t> out;
    read_links(*node, layer, out);
    return out;
  }

  std::size_t memory_bytes() const {
    std::size_t bytes = 0;
    bytes +=
        chunks_.size() * (static_cast<std::size_t>(kChunkSize) * dim_ * sizeof(float) + kChunkSize);
    bytes += arena_.bytes();
    bytes += blocks_.size() * sizeof(NodeBlock);
    for (const auto& table : node_tables_) bytes += table->cap * sizeof(Node*);
    for (const auto& table : vec_tables_) bytes += table->cap * sizeof(VecChunk*);
    const std::uint32_t hwm = n_.load(std::memory_order_acquire);
    for (std::uint32_t id = 0; id < hwm; ++id) {
      Node* node = peek_node(id);
      if (node == nullptr) continue;
      for (const auto& in : node->inbound) bytes += in.capacity() * sizeof(std::uint32_t);
    }
    return bytes;
  }

  void write_vec(std::ostream& out, const float* v, bool bulk) const {
    if (bulk) {
      out.write(reinterpret_cast<const char*>(v),
                static_cast<std::streamsize>(dim_ * sizeof(float)));
    } else {
      for (std::size_t i = 0; i < dim_; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, v + i, 4);
        write_u32(out, bits);
      }
    }
  }

  void save(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("hnsw: cannot open " + path.string());
    const bool bulk = host_little_endian();
    const std::uint32_t hwm = n_.load(std::memory_order_acquire);
    const int max_level = max_level_.load(std::memory_order_acquire);
    const std::uint32_t entry = entry_.load(std::memory_order_acquire);
    std::vector<std::uint8_t> flags(hwm, 0);
    std::vector<std::uint32_t> ids;
    ids.reserve(hwm);
    for (std::uint32_t id = 0; id < hwm; ++id) {
      const SlotState st = state_of(id);
      if (st == SlotState::Live) {
        flags[id] = 1;
        ids.push_back(id);
      } else if (st == SlotState::Tombstone) {
        flags[id] = 2;
        ids.push_back(id);
      }
    }

    out.write(kMagic, 8);
    write_u32(out, kVersionSlots);
    write_u32(out, static_cast<std::uint32_t>(dim_));
    write_u32(out, static_cast<std::uint32_t>(params_.metric));
    write_u32(out, static_cast<std::uint32_t>(M_));
    write_u32(out, static_cast<std::uint32_t>(M0_));
    write_u32(out, static_cast<std::uint32_t>(ef_construction_));
    write_u64(out, seed_);
    write_u32(out, hwm);
    write_i32(out, max_level);
    write_u32(out, entry);
    if (!flags.empty()) {
      out.write(reinterpret_cast<const char*>(flags.data()),
                static_cast<std::streamsize>(flags.size()));
    }
    for (std::uint32_t id : ids) write_vec(out, vector_ptr(id), bulk);
    for (std::uint32_t id : ids) write_u32(out, static_cast<std::uint32_t>(level_of(id)));
    std::vector<std::uint32_t> links;
    for (std::uint32_t id : ids) {
      Node* node = peek_node(id);
      for (int layer = 0; layer <= node->level; ++layer) {
        read_links(*node, layer, links);
        write_u32(out, static_cast<std::uint32_t>(links.size()));
        for (std::uint32_t nb : links) write_u32(out, nb);
      }
    }
    if (!out) throw std::runtime_error("hnsw: write failed " + path.string());
  }

  static void read_vec(std::istream& in, float* dst, std::size_t dim, bool bulk) {
    if (bulk) {
      in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(dim * sizeof(float)));
      if (static_cast<std::size_t>(in.gcount()) != dim * sizeof(float)) {
        throw std::runtime_error("hnsw: truncated vectors");
      }
      return;
    }
    for (std::size_t i = 0; i < dim; ++i) {
      const std::uint32_t bits = read_u32(in);
      std::memcpy(dst + i, &bits, 4);
    }
  }

  static std::unique_ptr<Impl> load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("hnsw: cannot open " + path.string());
    char magic[8];
    in.read(magic, 8);
    if (in.gcount() != 8 || std::memcmp(magic, kMagic, 8) != 0) {
      throw std::runtime_error("hnsw: bad magic");
    }
    const std::uint32_t version = read_u32(in);
    if (version != kVersionDense && version != kVersionSlots) {
      throw std::runtime_error("hnsw: unsupported version");
    }
    const auto dim = static_cast<std::size_t>(read_u32(in));
    const auto metric = static_cast<Metric>(read_u32(in));
    Params params;
    params.M = read_u32(in);
    const std::size_t M0 = read_u32(in);
    params.ef_construction = read_u32(in);
    params.seed = read_u64(in);
    params.metric = metric;
    const std::uint32_t count_or_hwm = read_u32(in);
    const int max_level = read_i32(in);
    const std::uint32_t entry = read_u32(in);

    auto impl = std::make_unique<Impl>(dim, params);
    if (M0 != impl->M0_) throw std::runtime_error("hnsw: M0 does not match 2M");

    std::vector<std::uint8_t> flags;
    std::uint32_t hwm = count_or_hwm;
    if (version == kVersionDense) {
      flags.assign(hwm, 1);
    } else {
      flags.resize(hwm);
      if (hwm > 0) {
        in.read(reinterpret_cast<char*>(flags.data()), static_cast<std::streamsize>(hwm));
        if (static_cast<std::size_t>(in.gcount()) != hwm)
          throw std::runtime_error("hnsw: truncated flags");
      }
    }
    std::vector<std::uint32_t> ids;
    for (std::uint32_t id = 0; id < hwm; ++id) {
      if (flags[id] == 1 || flags[id] == 2)
        ids.push_back(id);
      else if (flags[id] != 0)
        throw std::runtime_error("hnsw: bad slot flag");
      else
        impl->freelist_.push_back(id);
    }
    if (ids.empty()) {
      if (max_level != -1 || entry != 0) throw std::runtime_error("hnsw: empty index header");
      impl->n_.store(hwm, std::memory_order_release);
      return impl;
    }
    if (entry >= hwm) throw std::runtime_error("hnsw: entry point out of range");

    const bool bulk = host_little_endian();
    for (std::uint32_t id : ids) {
      impl->ensure_storage(id);
      float* dst = impl->vector_ptr_unlocked(id);
      read_vec(in, dst, dim, bulk);
      VecChunk* chunk = impl->vec_table_.load(std::memory_order_relaxed)
                            ->slots[id >> kChunkShift]
                            .load(std::memory_order_relaxed);
      chunk->ready[id & kChunkMask].store(1, std::memory_order_release);
    }

    std::vector<int> levels(hwm, -1);
    int observed = -1;
    for (std::uint32_t id : ids) {
      const int level = static_cast<int>(read_u32(in));
      if (level < 0 || level > kMaxLevel) throw std::runtime_error("hnsw: bad level");
      levels[id] = level;
      observed = std::max(observed, level);
    }
    if (observed != max_level) throw std::runtime_error("hnsw: max_level mismatch");
    if (levels[entry] != max_level) throw std::runtime_error("hnsw: entry level mismatch");

    std::uint32_t live = 0;
    for (std::uint32_t id : ids) {
      const SlotState st = flags[id] == 1 ? SlotState::Live : SlotState::Tombstone;
      impl->adopt_slot(id, levels[id]);
      Node* node = impl->node_at(id);
      node->state.store(st, std::memory_order_release);
      impl->node_table_.load(std::memory_order_relaxed)
          ->slots[id]
          .store(node, std::memory_order_release);
      impl->level_count_[levels[id]].fetch_add(1, std::memory_order_relaxed);
      if (st == SlotState::Live) ++live;
    }
    impl->n_.store(hwm, std::memory_order_release);

    for (std::uint32_t id : ids) {
      for (int layer = 0; layer <= levels[id]; ++layer) {
        const std::uint32_t degree = read_u32(in);
        if (degree > impl->max_m(layer)) throw std::runtime_error("hnsw: degree exceeds cap");
        std::vector<std::uint32_t> links(degree);
        for (std::uint32_t i = 0; i < degree; ++i) {
          links[i] = read_u32(in);
          if (links[i] >= hwm || links[i] == id) throw std::runtime_error("hnsw: bad neighbor");
          if (levels[links[i]] < layer) throw std::runtime_error("hnsw: neighbor level");
        }
        impl->write_links(id, layer, links);
      }
    }

    impl->live_.store(live, std::memory_order_release);
    impl->entry_.store(entry, std::memory_order_relaxed);
    impl->max_level_.store(max_level, std::memory_order_relaxed);
    impl->ready_.store(true, std::memory_order_release);
    return impl;
  }
};

Index::Index(std::size_t dim, Params params)
    : impl_(std::make_unique<Impl>(dim, std::move(params))) {}

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;
Index::~Index() = default;

std::size_t Index::dim() const {
  return impl_->dim_;
}
std::size_t Index::size() const {
  return impl_->live_.load(std::memory_order_acquire);
}
std::size_t Index::slot_count() const {
  return impl_->n_.load(std::memory_order_acquire);
}
std::size_t Index::memory_bytes() const {
  return impl_->memory_bytes();
}
bool Index::contains(std::uint32_t id) const {
  return impl_->is_live(id);
}
std::size_t Index::M() const {
  return impl_->M_;
}
std::size_t Index::M0() const {
  return impl_->M0_;
}
const Params& Index::params() const {
  return impl_->params_;
}
int Index::max_level() const {
  return impl_->max_level_.load(std::memory_order_acquire);
}
std::uint32_t Index::entry_point() const {
  return impl_->entry_.load(std::memory_order_acquire);
}
int Index::level(std::uint32_t id) const {
  return impl_->level_of(id);
}
std::vector<std::uint32_t> Index::neighbors(std::uint32_t id, int layer) const {
  return impl_->neighbors_of(id, layer);
}
void Index::copy_vector(std::uint32_t id, float* out) const {
  impl_->copy_vector(id, out);
}
std::uint32_t Index::insert(const float* vector) {
  return impl_->insert(vector);
}
void Index::insert_batch(const float* vectors, std::size_t count, std::size_t threads) {
  impl_->insert_batch(vectors, count, threads);
}
void Index::erase(std::uint32_t id) {
  impl_->erase(id);
}
void Index::tombstone(std::uint32_t id) {
  impl_->tombstone(id);
}
std::vector<Neighbor> Index::search(const float* query, std::size_t k, std::size_t ef) const {
  return impl_->search(query, k, ef);
}
void Index::save(const std::filesystem::path& path) const {
  impl_->save(path);
}
Index Index::load(const std::filesystem::path& path) {
  return Index(Impl::load(path));
}

}  // namespace hnsw
