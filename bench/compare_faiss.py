#!/usr/bin/env python3
"""Run FAISS IndexHNSWFlat on the same fvecs/ivecs inputs and plot recall against QPS."""

import argparse
import csv
import platform
import subprocess
import sys
from pathlib import Path


def read_fvecs(path, max_n=0):
    import numpy as np

    raw = np.fromfile(path, dtype=np.int32, count=1)
    if raw.size != 1:
        raise RuntimeError(f"bad fvecs {path}")
    dim = int(raw[0])
    rec = 1 + dim
    data = np.fromfile(path, dtype=np.int32)
    if data.size % rec != 0:
        raise RuntimeError(f"truncated fvecs {path}")
    n = data.size // rec
    if max_n and max_n < n:
        n = max_n
        data = data[: n * rec]
    data = data.reshape(n, rec)
    if not np.all(data[:, 0] == dim):
        raise RuntimeError(f"dim mismatch in {path}")
    vecs = data[:, 1:].view(np.float32)
    return np.ascontiguousarray(vecs)


def read_ivecs(path, max_n=0):
    import numpy as np

    raw = np.fromfile(path, dtype=np.int32, count=1)
    dim = int(raw[0])
    rec = 1 + dim
    data = np.fromfile(path, dtype=np.int32)
    n = data.size // rec
    if max_n and max_n < n:
        n = max_n
        data = data[: n * rec]
    data = data.reshape(n, rec)
    return data[:, 1:]


def recall_at_k(ids, gt, k):
    hit = 0
    nq = ids.shape[0]
    for i in range(nq):
        truth = set(int(x) for x in gt[i, :k])
        for j in range(k):
            if int(ids[i, j]) in truth:
                hit += 1
    return hit / float(nq * k)


def cpu_model():
    name = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return name


def mem_total():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return line
    except OSError:
        pass
    return "MemTotal: unknown"


def run_faiss(base, query, gt, M, ef_construction, efs, k):
    import faiss
    import numpy as np
    import time

    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(base.shape[1], M, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(base)
    build = time.perf_counter() - t0
    level0 = int(index.hnsw.nb_neighbors(0))
    level1 = int(index.hnsw.nb_neighbors(1)) if index.hnsw.max_level > 0 else M
    rows = []
    for ef in efs:
        index.hnsw.efSearch = ef
        index.search(query, k)
        t1 = time.perf_counter()
        _, ids = index.search(query, k)
        elapsed = time.perf_counter() - t1
        qps = query.shape[0] / elapsed if elapsed > 0 else 0.0
        rec = recall_at_k(ids, gt, k)
        rows.append(
            {
                "system": "faiss",
                "M": M,
                "M0": level0,
                "ef_construction": ef_construction,
                "ef": ef,
                "k": k,
                "recall_at_k": rec,
                "qps": qps,
                "build_seconds": build,
                "n": base.shape[0],
                "nq": query.shape[0],
                "dim": base.shape[1],
                "threads": 1,
            }
        )
        print(f"faiss ef {ef} recall@{k} {rec:.6f} qps {qps:.3f}", file=sys.stderr)
    print(f"faiss build_seconds {build:.6f} nb_neighbors0 {level0} nb_neighbors1 {level1}", file=sys.stderr)
    print(f"faiss_version {faiss.__version__}", file=sys.stderr)
    return rows, faiss.__version__, level0, level1, build


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = [
        "system",
        "M",
        "M0",
        "ef_construction",
        "ef",
        "k",
        "recall_at_k",
        "qps",
        "build_seconds",
        "n",
        "nq",
        "dim",
        "threads",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({key: row[key] for key in fields})


def plot(rows, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    for system, marker in (("hnsw", "o"), ("faiss", "s")):
        pts = [r for r in rows if r["system"] == system]
        pts.sort(key=lambda r: int(r["ef"]))
        if not pts:
            continue
        xs = [float(r["recall_at_k"]) for r in pts]
        ys = [float(r["qps"]) for r in pts]
        ax.plot(xs, ys, marker=marker, linewidth=1.2, label=system)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(
                f"ef={r['ef']}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries/sec")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--hnsw-bin", default="")
    p.add_argument("--hnsw-csv", default="bench/results/hnsw.csv")
    p.add_argument("--out-csv", default="bench/results/results.csv")
    p.add_argument("--plot", default="bench/results/recall_qps.png")
    p.add_argument("--meta", default="bench/results/meta.txt")
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--efs", default="10,20,40,80,160,320")
    p.add_argument("--max-n", type=int, default=0)
    p.add_argument("--max-q", type=int, default=0)
    p.add_argument("--skip-hnsw", action="store_true")
    args = p.parse_args()
    efs = [int(x) for x in args.efs.split(",") if x]

    hnsw_csv = Path(args.hnsw_csv)
    if not args.skip_hnsw:
        if not args.hnsw_bin:
            raise SystemExit("--hnsw-bin is required unless --skip-hnsw")
        hnsw_csv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.hnsw_bin,
            "--base",
            args.base,
            "--query",
            args.query,
            "--gt",
            args.gt,
            "--M",
            str(args.M),
            "--ef-construction",
            str(args.ef_construction),
            "--k",
            str(args.k),
            "--efs",
            args.efs,
            "--threads",
            "1",
            "--csv",
            str(hnsw_csv),
            "--meta",
            str(Path(args.meta).with_name("hnsw_meta.txt")),
            "--gt-check",
            "16",
        ]
        if args.max_n:
            cmd.extend(["--max-n", str(args.max_n), "--compute-gt"])
        if args.max_q:
            cmd.extend(["--max-q", str(args.max_q)])
        print(" ".join(cmd), file=sys.stderr)
        subprocess.check_call(cmd)

    base = read_fvecs(args.base, args.max_n)
    query = read_fvecs(args.query, args.max_q)
    if args.max_n:
        # Subsets do not match the full-base ground-truth file. Reuse ids from the hnsw csv
        # only for the plot of that run; FAISS is evaluated with brute force below.
        gt = None
    else:
        gt = read_ivecs(args.gt, query.shape[0])

    if gt is None:
        import faiss

        faiss.omp_set_num_threads(1)
        flat = faiss.IndexFlatL2(base.shape[1])
        flat.add(base)
        _, gt = flat.search(query, args.k)

    rows, faiss_version, n0, n1, build = run_faiss(
        base, query, gt, args.M, args.ef_construction, efs, args.k
    )
    hnsw_rows = read_csv(hnsw_csv)
    combined = hnsw_rows + rows
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, combined)
    title = f"SIFT dim={base.shape[1]} n={base.shape[0]} M={args.M} efConstruction={args.ef_construction}"
    plot(combined, args.plot, title)

    hnsw_build = hnsw_rows[0]["build_seconds"] if hnsw_rows else "na"
    meta = Path(args.meta)
    meta.write_text(
        "\n".join(
            [
                f"cpu_model {cpu_model()}",
                mem_total(),
                f"platform {platform.platform()}",
                f"python {platform.python_version()}",
                f"faiss {faiss_version}",
                f"faiss_nb_neighbors_0 {n0}",
                f"faiss_nb_neighbors_1 {n1}",
                f"faiss_build_seconds {build:.6f}",
                f"hnsw_build_seconds {hnsw_build}",
                f"n {base.shape[0]}",
                f"nq {query.shape[0]}",
                f"dim {base.shape[1]}",
                f"M {args.M}",
                f"ef_construction {args.ef_construction}",
                f"k {args.k}",
                f"efs {args.efs}",
                "threads 1",
                "warmup one full untimed pass per ef, then one timed pass",
                "faiss omp threads 1",
                "",
            ]
        )
    )
    print(f"wrote {out}", file=sys.stderr)
    print(meta.read_text())


if __name__ == "__main__":
    main()
