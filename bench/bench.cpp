#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace {

struct Config {
  std::string base;
  std::string query;
  std::string gt;
  std::string csv = "bench/results/hnsw.csv";
  std::string meta = "bench/results/hnsw_meta.txt";
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t k = 10;
  std::vector<std::size_t> efs = {10, 20, 40, 80, 160, 320};
  std::size_t threads = 1;
  std::size_t max_n = 0;
  std::size_t max_q = 0;
  std::uint64_t seed = 42;
  bool compute_gt = false;
  int gt_check = 16;
};

void usage() {
  std::cerr << "hnsw_bench --base FILE --query FILE [--gt FILE | --compute-gt]\n"
            << "  --M N --ef-construction N --k N --efs 10,20,40 --threads N\n"
            << "  --max-n N --max-q N --csv PATH --meta PATH --seed N --gt-check N\n";
}

std::vector<std::size_t> parse_list(std::string_view text) {
  std::vector<std::size_t> out;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t comma = text.find(',', i);
    const std::string_view tok =
        text.substr(i, comma == std::string_view::npos ? text.size() - i : comma - i);
    if (!tok.empty()) out.push_back(static_cast<std::size_t>(std::stoull(std::string(tok))));
    if (comma == std::string_view::npos) break;
    i = comma + 1;
  }
  return out;
}

struct Matrix {
  std::size_t n = 0;
  std::size_t dim = 0;
  std::vector<float> data;
};

struct IntMatrix {
  std::size_t n = 0;
  std::size_t dim = 0;
  std::vector<std::uint32_t> data;
};

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
    if (!in || d != dim) throw std::runtime_error("fvecs dim mismatch in " + path);
    in.read(reinterpret_cast<char*>(m.data.data() + i * m.dim),
            static_cast<std::streamsize>(m.dim * 4));
    if (!in) throw std::runtime_error("truncated fvecs " + path);
  }
  return m;
}

IntMatrix read_ivecs(const std::string& path, std::size_t max_n) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::int32_t dim = 0;
  in.read(reinterpret_cast<char*>(&dim), 4);
  if (!in || dim <= 0) throw std::runtime_error("bad ivecs dim in " + path);
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::uint64_t>(in.tellg());
  const std::uint64_t rec = 4ull + static_cast<std::uint64_t>(dim) * 4ull;
  if (bytes % rec != 0) throw std::runtime_error("truncated ivecs " + path);
  std::size_t n = static_cast<std::size_t>(bytes / rec);
  if (max_n > 0 && max_n < n) n = max_n;
  in.seekg(0);
  IntMatrix m;
  m.n = n;
  m.dim = static_cast<std::size_t>(dim);
  m.data.resize(n * m.dim);
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), 4);
    if (!in || d != dim) throw std::runtime_error("ivecs dim mismatch in " + path);
    std::vector<std::int32_t> row(static_cast<std::size_t>(dim));
    in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(dim * 4));
    if (!in) throw std::runtime_error("truncated ivecs " + path);
    for (std::size_t j = 0; j < m.dim; ++j)
      m.data[i * m.dim + j] = static_cast<std::uint32_t>(row[j]);
  }
  return m;
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos == std::string::npos) return line;
      std::string v = line.substr(pos + 1);
      const auto b = v.find_first_not_of(" \t");
      return b == std::string::npos ? std::string{} : v.substr(b);
    }
  }
  return "unknown";
}

std::string mem_total() {
  std::ifstream in("/proc/meminfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("MemTotal:", 0) == 0) return line;
  }
  return "MemTotal: unknown";
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<std::uint32_t> topk_from_gt(const IntMatrix& gt, std::size_t nq, std::size_t k) {
  if (gt.dim < k) throw std::runtime_error("ground truth has fewer than k neighbors");
  if (gt.n < nq) throw std::runtime_error("ground truth has fewer queries than requested");
  std::vector<std::uint32_t> out(nq * k);
  for (std::size_t q = 0; q < nq; ++q) {
    for (std::size_t j = 0; j < k; ++j) out[q * k + j] = gt.data[q * gt.dim + j];
  }
  return out;
}

std::vector<std::uint32_t> compute_gt(const Matrix& base, const Matrix& query, std::size_t k) {
  hnsw::BruteForce exact(base.dim, hnsw::Metric::L2);
  exact.add_batch(base.data.data(), base.n);
  std::vector<std::uint32_t> out(query.n * k);
  for (std::size_t q = 0; q < query.n; ++q) {
    const auto hits = exact.search(query.data.data() + q * query.dim, k);
    if (hits.size() != k) throw std::runtime_error("brute force returned fewer than k");
    for (std::size_t j = 0; j < k; ++j) out[q * k + j] = hits[j].id;
  }
  return out;
}

int check_gt(const Matrix& base, const Matrix& query, const std::vector<std::uint32_t>& truth,
             std::size_t k, int ncheck) {
  if (ncheck <= 0) return 0;
  const int n = std::min(ncheck, static_cast<int>(query.n));
  hnsw::BruteForce exact(base.dim, hnsw::Metric::L2);
  exact.add_batch(base.data.data(), base.n);
  int mismatch = 0;
  for (int q = 0; q < n; ++q) {
    const auto hits = exact.search(query.data.data() + static_cast<std::size_t>(q) * query.dim, k);
    for (std::size_t j = 0; j < k; ++j) {
      if (hits[j].id != truth[static_cast<std::size_t>(q) * k + j]) {
        if (mismatch == 0) {
          std::cerr << "gt mismatch query " << q << " rank " << j << " brute " << hits[j].id
                    << " file " << truth[static_cast<std::size_t>(q) * k + j] << "\n";
        }
        ++mismatch;
      }
    }
  }
  std::cerr << "gt_check_queries " << n << " id_mismatches " << mismatch << "\n";
  return mismatch;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return std::string(argv[++i]);
    };
    if (a == "--base")
      cfg.base = need("--base");
    else if (a == "--query")
      cfg.query = need("--query");
    else if (a == "--gt")
      cfg.gt = need("--gt");
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
      cfg.efs = parse_list(need("--efs"));
    else if (a == "--threads")
      cfg.threads = static_cast<std::size_t>(std::stoull(need("--threads")));
    else if (a == "--max-n")
      cfg.max_n = static_cast<std::size_t>(std::stoull(need("--max-n")));
    else if (a == "--max-q")
      cfg.max_q = static_cast<std::size_t>(std::stoull(need("--max-q")));
    else if (a == "--seed")
      cfg.seed = std::stoull(need("--seed"));
    else if (a == "--gt-check")
      cfg.gt_check = std::stoi(need("--gt-check"));
    else if (a == "--compute-gt")
      cfg.compute_gt = true;
    else if (a == "--help") {
      usage();
      return 0;
    } else {
      usage();
      throw std::runtime_error("unknown arg " + std::string(a));
    }
  }
  if (cfg.base.empty() || cfg.query.empty() || (cfg.gt.empty() && !cfg.compute_gt)) {
    usage();
    return 2;
  }

  const Matrix base = read_fvecs(cfg.base, cfg.max_n);
  Matrix query = read_fvecs(cfg.query, cfg.max_q);
  if (query.dim != base.dim) throw std::runtime_error("query dim != base dim");
  if (cfg.max_n > 0) cfg.compute_gt = true;

  std::vector<std::uint32_t> truth;
  if (cfg.compute_gt || cfg.gt.empty()) {
    std::cerr << "computing brute-force ground truth for " << query.n << " queries\n";
    truth = compute_gt(base, query, cfg.k);
  } else {
    const IntMatrix gt = read_ivecs(cfg.gt, query.n);
    truth = topk_from_gt(gt, query.n, cfg.k);
    if (cfg.max_n == 0 && cfg.gt_check > 0) check_gt(base, query, truth, cfg.k, cfg.gt_check);
  }

  hnsw::Params params;
  params.M = cfg.M;
  params.ef_construction = cfg.ef_construction;
  params.metric = hnsw::Metric::L2;
  params.seed = cfg.seed;
  hnsw::Index index(base.dim, params);

  std::cerr << "building n=" << base.n << " dim=" << base.dim << " M=" << cfg.M
            << " ef_construction=" << cfg.ef_construction << " threads=" << cfg.threads << "\n";
  const auto t0 = std::chrono::steady_clock::now();
  index.insert_batch(base.data.data(), base.n, cfg.threads);
  const double build_s = seconds_since(t0);
  std::cerr << "build_seconds " << build_s << " max_level " << index.max_level() << "\n";

  std::ofstream csv(cfg.csv);
  if (!csv) throw std::runtime_error("cannot write " + cfg.csv);
  csv << "system,M,M0,ef_construction,ef,k,recall_at_k,qps,build_seconds,n,nq,dim,threads\n";
  csv.setf(std::ios::fixed);
  csv.precision(6);

  for (std::size_t ef : cfg.efs) {
    std::vector<std::vector<hnsw::Neighbor>> results(query.n);
    for (std::size_t q = 0; q < query.n; ++q) {
      results[q] = index.search(query.data.data() + q * query.dim, cfg.k, ef);
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < query.n; ++q) {
      results[q] = index.search(query.data.data() + q * query.dim, cfg.k, ef);
    }
    const double elapsed = seconds_since(t1);
    const double qps = elapsed > 0.0 ? static_cast<double>(query.n) / elapsed : 0.0;
    const double recall = hnsw::recall_at_k(results, truth.data(), query.n, cfg.k);
    csv << "hnsw," << cfg.M << "," << index.M0() << "," << cfg.ef_construction << "," << ef << ","
        << cfg.k << "," << recall << "," << qps << "," << build_s << "," << base.n << "," << query.n
        << "," << base.dim << "," << cfg.threads << "\n";
    std::cerr << "ef " << ef << " recall@" << cfg.k << " " << recall << " qps " << qps << "\n";
  }
  csv.close();

  std::ofstream meta(cfg.meta);
  meta << "cpu_model " << cpu_model() << "\n";
  meta << mem_total() << "\n";
  meta << "compiler " << __VERSION__ << "\n";
  meta << "avx2 " << (hnsw::avx2_supported() ? 1 : 0) << "\n";
  meta << "build_seconds " << build_s << "\n";
  meta << "n " << base.n << "\n";
  meta << "nq " << query.n << "\n";
  meta << "dim " << base.dim << "\n";
  meta << "M " << cfg.M << "\n";
  meta << "M0 " << index.M0() << "\n";
  meta << "ef_construction " << cfg.ef_construction << "\n";
  meta << "threads " << cfg.threads << "\n";
  meta << "max_level " << index.max_level() << "\n";
  meta << "csv " << cfg.csv << "\n";
  std::cerr << "wrote " << cfg.csv << "\n";
  return 0;
}
