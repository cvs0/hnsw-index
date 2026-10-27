#include <iostream>

#include "hnsw/hnsw.hpp"

int main() {
  hnsw::Params params;
  params.M = 16;
  params.ef_construction = 32;
  params.metric = hnsw::Metric::L2;
  params.seed = 1;

  hnsw::Index index(2, params);
  const float points[] = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f};
  index.insert_batch(points, 3, 1);

  const float query[] = {0.1f, 0.9f};
  const std::vector<hnsw::Neighbor> hits = index.search(query, 1, 16);
  if (hits.size() != 1 || hits[0].id != 2) return 1;
  std::cout << hits[0].id << " " << hits[0].distance << "\n";
  return 0;
}
