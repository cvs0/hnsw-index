# hnsw-index

C++20 HNSW approximate nearest-neighbor index (Malkov and Yashunin, IEEE TPAMI 42(4), 2020). One public header, `hnsw/hnsw.hpp`. Distance kernels live in their own translation units.

## Graph

Layer 0 contains every point. Its degree cap is `M0 = 2M`.
Layer `l > 0` contains points whose level is at least `l`. Its degree cap is `M`.
Insertion writes the forward edges, then a reverse edge at each selected neighbor. Reverse insertion reruns the same prune, so adjacency is not symmetric.
The entry point is a point on the current top layer. A new point replaces it only when its level is strictly greater. Equal levels keep the earlier point.

## Layer assignment

Level is a function of `(seed, id)`. There is no RNG state.

```
m_L = 1 / ln(M)
h   = splitmix64(seed XOR (id * 0x9E3779B97F4A7C15))
u   = (h >> 11) * 2^(-53)          # [0, 1)
x   = max(1 - u, numeric_limits<double>::min())
L   = min(16, floor(-ln(x) * m_L))
```

`splitmix64` is the Steele mixer (`0x9E3779B97F4A7C15`, `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`). For uniform `u`, `P(L >= l) = M^(-l)`. The cap at 16 makes the upper tail slightly thinner. Uncapped, `E[L] = 1 / (M - 1)`.

## Neighbor selection

Paper Algorithm 4 with `extendCandidates = false` and `keepPrunedConnections = false`.

Candidates are sorted by increasing distance, then increasing id. Walk that order. Keep `e` when no already-kept neighbor `r` satisfies `dist(e, r) < dist(e, q)`. Squared L2 is monotone with L2, so the comparison is valid for the L2 metric. The forward list is that set, capped at `M0` on layer 0 and `M` above. Each selected id then receives `q` as a reverse neighbor. If the reverse list would exceed the cap, the same heuristic is run with that node as the origin.

No `prune_headroom`. FAISS 1.15.1 defaults `prune_headroom` to 0.2 and keeps `size_t(max_size - max_size * headroom)` neighbors (25 at layer 0 and 12 above, for `M = 16`), while `nb_neighbors` stays 32 and 16.

Search starts at the entry point. Layers `max_level .. 1` use beam 1. Layer 0 uses `min(n, max(ef, k))`. The candidate heap is a min-heap. The result heap is a max-heap of that beam width. A point enters when its distance is smaller than the worst result, or the distance is equal and its id is smaller. Expansion stops when the closest candidate distance is strictly greater than the worst result distance. A point is marked visited the first time it is seen. Returned ids are sorted by increasing distance, then increasing id, and truncated to `k`. `ef` or `k` above `10^7` is rejected.

| Metric | Minimized value | Storage |
| --- | --- | --- |
| L2 | squared L2 | raw floats |
| InnerProduct | negative dot product | raw floats |
| Cosine | negative dot product | L2-normalized on insert. The query is normalized into a search buffer. The distance kernel does not normalize. A zero vector is unchanged. |

## Complexity

For fixed `M` and `ef_construction`, the paper bounds expected construction by `O(N log N)` distance evaluations and expected search, at fixed `ef`, by `O(log N)`. The layer-0 beam grows with `ef`. This repository does not count distance evaluations. The measured quantities are build time, recall@10, and queries/sec.

## Memory layout

Vectors sit in chunks of 1024. `aligned_alloc(32, 1024 * dim * 4)` gives the base pointer. The byte count is a multiple of 4096. The stride is `dim` floats, with no padding. Each slot has an `atomic<uint8_t>` ready flag, release-stored after the bytes and any cosine normalization. Readers acquire-spin on that flag.

Nodes live in blocks of 256 and are never moved. Each node holds a `std::mutex`, its level, and pointers into a bump arena of `atomic<uint32_t>` (chunks of 65536). A node at level `L` owns `L + 1` seqlock counters, `L + 1` degrees, and `M0 + L * M` neighbor slots.

Id-to-node tables start at 32 pointers and id-to-chunk tables start at 4. Both double. Old tables are retained until destruction, so a published address stays valid. The `Node*` is release-stored after initialization. `n_` is release-stored after the ready flag and the node pointer.

## SIMD distance kernels

`src/distance_avx2.cpp` is compiled with `-mavx2 -mfma` and is not in the public header. `distance_function` selects it at runtime when CPUID reports OSXSAVE, AVX, and FMA, `XCR0` has bits 1 and 2 (XMM and YMM), and leaf 7 EBX bit 5 (AVX2) is set. Otherwise the scalar loops in `src/distance_scalar.cpp` are used.

The AVX2 body keeps four YMM accumulators, steps 32 floats, then 8, then a scalar tail. The horizontal sum is extract plus the SSE add chain. The build does not pass fast-math flags. `l2_normalize` accumulates the sum of squares in double. Cosine and inner product share the negative-dot kernel.

## Concurrency

`insert` holds the allocator mutex across id allocation, chunk and node growth, `memcpy`, cosine normalization, the ready release, node initialization, and the `n_` release. Graph linking runs after that unlock.

Id 0 stores the entry point, `max_level`, and the ready flag, then the `n_` release, and returns without linking. Other inserters spin until ready.

Entry promotion takes `entry_mu_` when `level > max_level`, stores the entry relaxed, then stores `max_level` with release. Search loads `max_level` with acquire, then the entry relaxed.

Each node mutex serializes writers of that node. Search does not take it. Neighbor lists are copied under a seqlock. The writer stores `seq = s + 1` relaxed, a release fence, the degree and slots relaxed, a release fence, then `seq = s + 2` with release. The odd value is the update window. The reader loads `seq` with acquire, spins while it is odd, copies degree and slots relaxed, an acquire fence, reloads `seq` relaxed, and retries unless the two sequence values match.

A thread holds at most one node mutex.

`insert_batch` with `threads == 1` assigns ids in row order. With `threads > 1`, ids follow allocator-lock order, not row order. The SIFT numbers below use `threads = 1`, so ids match the ground-truth file.

`save`, `load`, and destruction require exclusive access. `save` is not atomic with a concurrent insert. Search is safe alongside insert and other searches.

Visited generation tags and both heaps are `thread_local`. On token wrap the tag array is cleared.

## On-disk format

Little-endian. Magic `HNSW0001`, version `u32 = 1`.

| Field | Type |
| --- | --- |
| magic | 8 bytes |
| version | u32 |
| dim, metric, M, M0, ef_construction | u32 |
| seed | u64 |
| count | u32 |
| max_level | i32, `-1` when empty |
| entry_point | u32 |
| vectors | f32[count * dim], stored form (cosine already normalized) |
| levels | u32[count] |
| links | for each id, for layer `0 .. level`: u32 degree, then that many u32 neighbor ids |

Load rejects a file unless `M0 == 2M`, the entry id is in range, the entry level equals `max_level` and equals the maximum stored level, every degree is within the layer cap, and every neighbor id is `< count` and not the node itself. A little-endian host bulk-copies the floats. Any other host writes each float as its `u32` bit pattern.

## Build and run

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake options: `HNSW_BUILD_TESTS`, `HNSW_BUILD_BENCH`, `HNSW_ENABLE_AVX2`, `HNSW_ENABLE_ASAN`, `HNSW_ENABLE_UBSAN`, `HNSW_ENABLE_TSAN`, `HNSW_WERROR`. TSan cannot be combined with ASan or UBSan. Sanitizer flags are applied before the GoogleTest fetch so the test library is instrumented.

```
bash bench/fetch_sift.sh
./build/hnsw_bench \
  --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --gt data/sift/sift_groundtruth.ivecs \
  --M 16 --ef-construction 200 --k 10 \
  --efs 10,20,40,80,160,320 --threads 1 --gt-check 16 \
  --csv bench/results/hnsw.csv \
  --meta bench/results/hnsw_meta.txt
python3 bench/compare_faiss.py \
  --base data/sift/sift_base.fvecs \
  --query data/sift/sift_query.fvecs \
  --gt data/sift/sift_groundtruth.ivecs \
  --hnsw-bin ./build/hnsw_bench \
  --hnsw-csv bench/results/hnsw.csv --skip-hnsw \
  --out-csv bench/results/results.csv \
  --plot bench/results/recall_qps.png \
  --meta bench/results/meta.txt \
  --M 16 --ef-construction 200 --k 10 \
  --efs 10,20,40,80,160,320
```

`fetch_sift.sh` tries the TEXMEX tarball, then `sift-128-euclidean.hdf5` from ann-benchmarks. The measured files came from that HDF5 fallback. Python packages for the comparison: `numpy`, `faiss-cpu`, `matplotlib`. `h5py` is used only by the HDF5 conversion. Datasets are gitignored.

```cpp
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
```

`Neighbor::distance` is the minimized value from the metric table. `BruteForce` is an exact scan with the same metric and the same sort order. It is not safe for concurrent `add` and `search`.

## Benchmark

SIFT1M, 128 dimensions, L2. `n = 1000000`, `nq = 10000`, `k = 10`, `M = 16`, `M0 = 32`, `ef_construction = 200`, one thread. The timer covers index construction only. Each `ef` gets one untimed pass over the query set, then one timed pass. QPS is `nq / elapsed`. recall@10 is the fraction of returned ids that appear in the exact top 10. The ground-truth file stores 100 neighbors per query. The first 10 are used. Before the timed build, brute force on 16 base vectors matched that file with 0 id mismatches. `max_level` on this build was 5.

FAISS 1.15.1 `IndexHNSWFlat(dim, M, METRIC_L2)`, `efConstruction = 200` (the library default is 40), `faiss.omp_set_num_threads(1)`. Same warmup protocol. `nb_neighbors(0) = 32`, `nb_neighbors(1) = 16`, `prune_headroom = 0.2`.

Hardware: KVM guest, `Intel(R) Xeon(R) Processor`, family 6, model 207, stepping 2, 4 CPUs, 1 thread per core, 4 cores, 1 socket. L1d 48 KiB x4, L1i 32 KiB x4, L2 2 MiB x4, L3 320 MiB. `MemTotal: 16387280 kB`. Little endian. `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, AVX2 kernel selected. Python 3.12.3.

| ef | recall@10 | QPS | FAISS recall@10 | FAISS QPS |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.725260 | 18427.403 | 0.715710 | 32858.865 |
| 20 | 0.852450 | 11840.606 | 0.845650 | 19684.899 |
| 40 | 0.938250 | 7155.478 | 0.933700 | 11583.562 |
| 80 | 0.979680 | 3954.953 | 0.977500 | 6415.000 |
| 160 | 0.994650 | 1978.563 | 0.993620 | 3393.862 |
| 320 | 0.998440 | 1102.587 | 0.998170 | 1720.874 |

Build time: this index 468.656 s, FAISS 355.850 s. One run. Cosine and inner product are implemented and covered by the unit tests. The table is L2 only.

![recall@10 versus queries/sec](bench/results/recall_qps.png)

CSV: `bench/results/results.csv`.
