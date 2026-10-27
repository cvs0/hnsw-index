#!/usr/bin/env python3
"""Convert an ann-benchmarks SIFT HDF5 file to TEXMEX fvecs/ivecs."""

import sys
from pathlib import Path

import h5py
import numpy as np


def write_fvecs(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, dim = arr.shape
    header = np.full((n, 1), dim, dtype=np.int32)
    packed = np.concatenate([header, arr.view(np.int32)], axis=1)
    packed.tofile(path)


def write_ivecs(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    n, dim = arr.shape
    header = np.full((n, 1), dim, dtype=np.int32)
    packed = np.concatenate([header, arr], axis=1)
    packed.tofile(path)


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as h:
        train = h["train"][:].astype(np.float32)
        test = h["test"][:].astype(np.float32)
        neighbors = h["neighbors"][:].astype(np.int32)
    write_fvecs(dst / "sift_base.fvecs", train)
    write_fvecs(dst / "sift_query.fvecs", test)
    write_ivecs(dst / "sift_groundtruth.ivecs", neighbors)
    print(f"base {train.shape} query {test.shape} gt {neighbors.shape}")


if __name__ == "__main__":
    main()
