#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

struct Config {
  std::string base = "data/sift/sift_base.fvecs";
  std::string query = "data/sift/sift_query.fvecs";
  std::string csv = "bench/results/delete.csv";
  std::string meta = "bench/results/delete_meta.txt";
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t k = 10;
  std::vector<std::size_t> efs = {10, 40, 160};
  std::vector<double> fractions = {0.1, 0.3, 0.5};
  double churn = 0.1;
  std::size_t max_n = 20000;
  std::size_t max_q = 1000;
  std::uint64_t seed = 1;
};

struct Matrix {
  std::size_t n = 0;
  std::size_t dim = 0;
  std::vector<float> data;
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::size_t rss_kb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      return static_cast<std::size_t>(std::stoull(line.substr(6)));
    }
  }
  return 0;
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) {
        std::string name = line.substr(pos + 1);
        while (!name.empty() && name.front() == ' ') name.erase(name.begin());
        return name;
      }
    }
  }
  return "unknown";
}

Matrix read_fvecs(const std::string& path, std::size_t max_n) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::int32_t dim = 0;
  in.read(reinterpret_cast<char*>(&dim), 4);
  if (!in || dim <= 0) throw std::runtime_error("bad fvecs dim in " + path);
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::uint64_t>(in.tellg());
  const std::uint64_t rec = 4ull + static_cast<std::uint64_t>(dim) * 4ull;
  if (bytes % rec != 0) throw std::runtime_error("truncated fvecs " + path);
  std::size_t n = static_cast<std::size_t>(bytes / rec);
  if (max_n > 0 && max_n < n) n = max_n;
  in.seekg(0);
  Matrix m;
  m.n = n;
  m.dim = static_cast<std::size_t>(dim);
  m.data.resize(n * m.dim);
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), 4);
    if (!in || static_cast<std::size_t>(d) != m.dim) throw std::runtime_error("bad fvecs record");
    in.read(reinterpret_cast<char*>(m.data.data() + i * m.dim),
            static_cast<std::streamsize>(m.dim * sizeof(float)));
    if (!in) throw std::runtime_error("truncated fvecs record");
  }
  return m;
}

std::vector<std::size_t> parse_sizes(std::string_view text) {
  std::vector<std::size_t> out;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t comma = text.find(',', i);
    const auto n = comma == std::string_view::npos ? text.size() - i : comma - i;
    if (n > 0) out.push_back(static_cast<std::size_t>(std::stoull(std::string(text.substr(i, n)))));
    if (comma == std::string_view::npos) break;
    i = comma + 1;
  }
  return out;
}

std::vector<double> parse_fracs(std::string_view text) {
  std::vector<double> out;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t comma = text.find(',', i);
    const auto n = comma == std::string_view::npos ? text.size() - i : comma - i;
    if (n > 0) out.push_back(std::stod(std::string(text.substr(i, n))));
    if (comma == std::string_view::npos) break;
    i = comma + 1;
  }
  return out;
}

struct LiveSet {
  std::size_t dim = 0;
  std::vector<float> data;
  std::vector<char> live;
  std::size_t count = 0;

  explicit LiveSet(std::size_t d) : dim(d) {}

  void add(const float* v) {
    data.insert(data.end(), v, v + dim);
    live.push_back(1);
    ++count;
  }

  void erase(std::uint32_t id) {
    if (id >= live.size() || !live[id]) throw std::out_of_range("live id");
    live[id] = 0;
    --count;
  }

  void put(std::uint32_t id, const float* v) {
    if (static_cast<std::size_t>(id) >= live.size()) {
      data.resize((static_cast<std::size_t>(id) + 1) * dim);
      live.resize(static_cast<std::size_t>(id) + 1, 0);
    }
    if (!live[id]) ++count;
    live[id] = 1;
    std::copy_n(v, dim, data.data() + static_cast<std::size_t>(id) * dim);
  }
};

std::vector<std::uint32_t> exact_knn(const float* q, const LiveSet& set, std::size_t k) {
  struct Item {
    float dist;
    std::uint32_t id;
  };
  // Max-heap: `better` is the heap less-than, so front() is the farthest id.
  auto better = [](const Item& a, const Item& b) {
    if (a.dist != b.dist) return a.dist < b.dist;
    return a.id < b.id;
  };
  std::vector<Item> heap;
  heap.reserve(k + 1);
  const std::size_t n = set.live.size();
  for (std::size_t id = 0; id < n; ++id) {
    if (!set.live[id]) continue;
    const float* v = set.data.data() + id * set.dim;
    float dist = 0.f;
    for (std::size_t j = 0; j < set.dim; ++j) {
      const float d = q[j] - v[j];
      dist += d * d;
    }
    Item item{dist, static_cast<std::uint32_t>(id)};
    if (heap.size() < k) {
      heap.push_back(item);
      std::push_heap(heap.begin(), heap.end(), better);
    } else if (better(item, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), better);
      heap.back() = item;
      std::push_heap(heap.begin(), heap.end(), better);
    }
  }
  std::sort(heap.begin(), heap.end(), better);
  std::vector<std::uint32_t> ids;
  ids.reserve(heap.size());
  for (const Item& item : heap) ids.push_back(item.id);
  return ids;
}

void self_check_exact() {
  LiveSet set(1);
  const float vals[] = {0.f, 1.f, 3.f, 10.f};
  for (float v : vals) set.add(&v);
  const float q = 0.f;
  const auto ids = exact_knn(&q, set, 2);
  if (ids.size() != 2 || ids[0] != 0 || ids[1] != 1) {
    throw std::runtime_error("exact_knn self-check failed");
  }
}

double recall_of(const std::vector<hnsw::Neighbor>& hits, const std::uint32_t* truth,
                 std::size_t k) {
  std::size_t hit = 0;
  const std::size_t n = std::min(hits.size(), k);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      if (hits[i].id == truth[j]) {
        ++hit;
        break;
      }
    }
  }
  return static_cast<double>(hit) / static_cast<double>(k);
}

struct Row {
  std::string strategy;
  std::string phase;
  double fraction = 0;
  std::size_t ef = 0;
  double recall = 0;
  double qps = 0;
  double op_seconds = 0;
  std::size_t op_count = 0;
  std::size_t memory_bytes = 0;
  std::size_t rss_kb = 0;
  std::size_t n_live = 0;
  std::size_t n_slots = 0;
};

hnsw::Params make_params(const Config& cfg) {
  hnsw::Params p;
  p.M = cfg.M;
  p.ef_construction = cfg.ef_construction;
  p.metric = hnsw::Metric::L2;
  p.seed = cfg.seed;
  return p;
}

std::vector<std::uint32_t> permutation(std::size_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::shuffle(ids.begin(), ids.end(), rng);
  return ids;
}

void eval_queries(hnsw::Index& index, const Matrix& queries,
                  const std::vector<std::uint32_t>& truth, std::size_t k,
                  const std::vector<std::size_t>& efs, Row row, std::vector<Row>& out) {
  const std::size_t nq = queries.n;
  const std::size_t d = queries.dim;
  for (std::size_t ef : efs) {
    for (std::size_t i = 0; i < nq; ++i) {
      (void)index.search(queries.data.data() + i * d, k, ef);
    }
    const auto t0 = std::chrono::steady_clock::now();
    double rec = 0;
    for (std::size_t i = 0; i < nq; ++i) {
      const auto hits = index.search(queries.data.data() + i * d, k, ef);
      rec += recall_of(hits, truth.data() + i * k, k);
    }
    const double elapsed = seconds_since(t0);
    row.ef = ef;
    row.recall = rec / static_cast<double>(nq);
    row.qps = elapsed > 0 ? static_cast<double>(nq) / elapsed : 0;
    std::cerr << row.strategy << " " << row.phase << " f " << row.fraction << " ef " << ef
              << " recall " << row.recall << " qps " << row.qps << "\n";
    out.push_back(row);
  }
}

std::vector<std::uint32_t> ground_truth(const Matrix& queries, const LiveSet& set, std::size_t k) {
  std::vector<std::uint32_t> truth(queries.n * k);
  for (std::size_t i = 0; i < queries.n; ++i) {
    const auto ids = exact_knn(queries.data.data() + i * queries.dim, set, k);
    for (std::size_t j = 0; j < ids.size() && j < k; ++j) truth[i * k + j] = ids[j];
  }
  return truth;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return std::string(argv[++i]);
    };
    if (a == "--base")
      cfg.base = need("--base");
    else if (a == "--query")
      cfg.query = need("--query");
    else if (a == "--csv")
      cfg.csv = need("--csv");
    else if (a == "--meta")
      cfg.meta = need("--meta");
    else if (a == "--M")
      cfg.M = static_cast<std::size_t>(std::stoull(need("--M")));
    else if (a == "--ef-construction")
      cfg.ef_construction = static_cast<std::size_t>(std::stoull(need("--ef-construction")));
    else if (a == "--k")
      cfg.k = static_cast<std::size_t>(std::stoull(need("--k")));
    else if (a == "--efs")
      cfg.efs = parse_sizes(need("--efs"));
    else if (a == "--fractions")
      cfg.fractions = parse_fracs(need("--fractions"));
    else if (a == "--churn")
      cfg.churn = std::stod(need("--churn"));
    else if (a == "--max-n")
      cfg.max_n = static_cast<std::size_t>(std::stoull(need("--max-n")));
    else if (a == "--max-q")
      cfg.max_q = static_cast<std::size_t>(std::stoull(need("--max-q")));
    else if (a == "--seed")
      cfg.seed = std::stoull(need("--seed"));
    else
      throw std::runtime_error("unknown arg " + a);
  }

  self_check_exact();
  const std::size_t churn_n = static_cast<std::size_t>(cfg.churn * static_cast<double>(cfg.max_n));
  const Matrix base = read_fvecs(cfg.base, cfg.max_n + churn_n);
  const Matrix queries = read_fvecs(cfg.query, cfg.max_q);
  if (base.dim != queries.dim) throw std::runtime_error("dim mismatch");
  const std::size_t n = cfg.max_n;
  const std::size_t d = base.dim;
  if (base.n < n + churn_n) throw std::runtime_error("base shorter than n + churn");
  const auto order = permutation(n, cfg.seed);
  std::vector<Row> rows;

  auto build_index = [&](const float* data, std::size_t count) {
    hnsw::Index index(d, make_params(cfg));
    const auto t0 = std::chrono::steady_clock::now();
    index.insert_batch(data, count, 1);
    std::cerr << "build n " << count << " seconds " << seconds_since(t0) << "\n";
    return index;
  };

  for (double fraction : cfg.fractions) {
    const std::size_t cut = static_cast<std::size_t>(fraction * static_cast<double>(n));
    std::cerr << "fraction " << fraction << " cut " << cut << "\n";

    {
      hnsw::Index index = build_index(base.data.data(), n);
      LiveSet set(d);
      for (std::size_t i = 0; i < n; ++i) set.add(base.data.data() + i * d);
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < cut; ++i) {
        index.erase(order[i]);
        set.erase(order[i]);
      }
      const double op = seconds_since(t0);
      std::cerr << "erase seconds " << op << " per " << (cut ? op / static_cast<double>(cut) : 0)
                << "\n";
      const auto truth = ground_truth(queries, set, cfg.k);
      Row row;
      row.strategy = "erase";
      row.phase = "delete";
      row.fraction = fraction;
      row.op_seconds = op;
      row.op_count = cut;
      row.memory_bytes = index.memory_bytes();
      row.rss_kb = rss_kb();
      row.n_live = index.size();
      row.n_slots = index.slot_count();
      eval_queries(index, queries, truth, cfg.k, cfg.efs, row, rows);
    }

    {
      hnsw::Index index = build_index(base.data.data(), n);
      LiveSet set(d);
      for (std::size_t i = 0; i < n; ++i) set.add(base.data.data() + i * d);
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < cut; ++i) {
        index.tombstone(order[i]);
        set.erase(order[i]);
      }
      const double op = seconds_since(t0);
      std::cerr << "tombstone seconds " << op << "\n";
      const auto truth = ground_truth(queries, set, cfg.k);
      Row row;
      row.strategy = "tombstone";
      row.phase = "delete";
      row.fraction = fraction;
      row.op_seconds = op;
      row.op_count = cut;
      row.memory_bytes = index.memory_bytes();
      row.rss_kb = rss_kb();
      row.n_live = index.size();
      row.n_slots = index.slot_count();
      eval_queries(index, queries, truth, cfg.k, cfg.efs, row, rows);
    }

    {
      LiveSet set(d);
      std::vector<char> drop(n, 0);
      for (std::size_t i = 0; i < cut; ++i) drop[order[i]] = 1;
      std::vector<float> survivors;
      survivors.reserve((n - cut) * d);
      for (std::size_t id = 0; id < n; ++id) {
        if (drop[id]) continue;
        const float* v = base.data.data() + id * d;
        survivors.insert(survivors.end(), v, v + d);
        set.add(v);
      }
      const auto t0 = std::chrono::steady_clock::now();
      hnsw::Index index = build_index(survivors.data(), n - cut);
      const double op = seconds_since(t0);
      const auto truth = ground_truth(queries, set, cfg.k);
      Row row;
      row.strategy = "rebuild";
      row.phase = "delete";
      row.fraction = fraction;
      row.op_seconds = op;
      row.op_count = cut;
      row.memory_bytes = index.memory_bytes();
      row.rss_kb = rss_kb();
      row.n_live = index.size();
      row.n_slots = index.slot_count();
      eval_queries(index, queries, truth, cfg.k, cfg.efs, row, rows);
    }
  }

  if (churn_n > 0) {
    std::cerr << "churn " << churn_n << "\n";
    {
      hnsw::Index index = build_index(base.data.data(), n);
      LiveSet set(d);
      for (std::size_t i = 0; i < n; ++i) set.add(base.data.data() + i * d);
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < churn_n; ++i) {
        index.erase(order[i]);
        set.erase(order[i]);
        const float* v = base.data.data() + (n + i) * d;
        const std::uint32_t id = index.insert(v);
        set.put(id, v);
      }
      const double op = seconds_since(t0);
      std::cerr << "erase churn seconds " << op << "\n";
      const auto truth = ground_truth(queries, set, cfg.k);
      Row row;
      row.strategy = "erase";
      row.phase = "churn";
      row.fraction = cfg.churn;
      row.op_seconds = op;
      row.op_count = churn_n * 2;
      row.memory_bytes = index.memory_bytes();
      row.rss_kb = rss_kb();
      row.n_live = index.size();
      row.n_slots = index.slot_count();
      eval_queries(index, queries, truth, cfg.k, cfg.efs, row, rows);
    }
    {
      hnsw::Index index = build_index(base.data.data(), n);
      LiveSet set(d);
      for (std::size_t i = 0; i < n; ++i) set.add(base.data.data() + i * d);
      const auto t0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < churn_n; ++i) {
        index.tombstone(order[i]);
        set.erase(order[i]);
        const float* v = base.data.data() + (n + i) * d;
        const std::uint32_t id = index.insert(v);
        set.put(id, v);
      }
      const double op = seconds_since(t0);
      std::cerr << "tombstone churn seconds " << op << "\n";
      const auto truth = ground_truth(queries, set, cfg.k);
      Row row;
      row.strategy = "tombstone";
      row.phase = "churn";
      row.fraction = cfg.churn;
      row.op_seconds = op;
      row.op_count = churn_n * 2;
      row.memory_bytes = index.memory_bytes();
      row.rss_kb = rss_kb();
      row.n_live = index.size();
      row.n_slots = index.slot_count();
      eval_queries(index, queries, truth, cfg.k, cfg.efs, row, rows);
    }
    {
      std::vector<char> drop(n, 0);
      for (std::size_t i = 0; i < churn_n; ++i) drop[order[i]] = 1;
      std::vector<float> final_vecs;
      LiveSet set(d);
      for (std::size_t id = 0; id < n; ++id) {
        if (drop[id]) continue;
        const float* v = base.data.data() + id * d;
        final_vecs.insert(final_vecs.end(), v, v + d);
        set.add(v);
      }
      for (std::size_t i = 0; i < churn_n; ++i) {
        const float* v = base.data.data() + (n + i) * d;
        final_vecs.insert(final_vecs.end(), v, v + d);
        set.add(v);
      }
      const std::size_t final_n = final_vecs.size() / d;
      const auto t0 = std::chrono::steady_clock::now();
      hnsw::Index index = build_index(final_vecs.data(), final_n);
      const double op = seconds_since(t0);
      const auto truth = ground_truth(queries, set, cfg.k);
      Row row;
      row.strategy = "rebuild";
      row.phase = "churn";
      row.fraction = cfg.churn;
      row.op_seconds = op;
      row.op_count = churn_n * 2;
      row.memory_bytes = index.memory_bytes();
      row.rss_kb = rss_kb();
      row.n_live = index.size();
      row.n_slots = index.slot_count();
      eval_queries(index, queries, truth, cfg.k, cfg.efs, row, rows);
    }
  }

  std::ofstream csv(cfg.csv);
  if (!csv) throw std::runtime_error("cannot write " + cfg.csv);
  csv << "strategy,phase,fraction,ef,k,recall_at_k,qps,op_seconds,op_count,memory_bytes,rss_kb,"
         "n_live,n_slots,n_build,nq,dim,M,ef_construction\n";
  csv.setf(std::ios::fixed);
  csv.precision(6);
  for (const Row& row : rows) {
    csv << row.strategy << "," << row.phase << "," << row.fraction << "," << row.ef << "," << cfg.k
        << "," << row.recall << "," << row.qps << "," << row.op_seconds << "," << row.op_count
        << "," << row.memory_bytes << "," << row.rss_kb << "," << row.n_live << "," << row.n_slots
        << "," << n << "," << queries.n << "," << d << "," << cfg.M << "," << cfg.ef_construction
        << "\n";
  }
  std::ofstream meta(cfg.meta);
  meta << "cpu_model " << cpu_model() << "\n";
  std::ifstream mem("/proc/meminfo");
  std::string memline;
  std::getline(mem, memline);
  meta << memline << "\n";
  meta << "compiler " << __VERSION__ << "\n";
  meta << "avx2 " << (hnsw::avx2_supported() ? 1 : 0) << "\n";
  meta << "n " << n << "\n";
  meta << "nq " << queries.n << "\n";
  meta << "dim " << d << "\n";
  meta << "M " << cfg.M << "\n";
  meta << "ef_construction " << cfg.ef_construction << "\n";
  meta << "k " << cfg.k << "\n";
  meta << "churn " << cfg.churn << "\n";
  meta << "seed " << cfg.seed << "\n";
  meta << "threads 1\n";
  meta << "warmup one full untimed pass per ef, then one timed pass\n";
  std::cerr << "wrote " << cfg.csv << "\n";
  return 0;
}
