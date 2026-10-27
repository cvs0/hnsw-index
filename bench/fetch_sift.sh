#!/usr/bin/env bash
# Download SIFT1M and write fvecs/ivecs under data/sift/.
# Prefers the TEXMEX tarball. If the FTP data connection is blocked, falls back
# to the ann-benchmarks HDF5 file and converts it.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/data/sift"
base="$root/data/sift/sift_base.fvecs"
if [[ -f "$base" ]]; then
  echo "already present: $base"
  exit 0
fi

if curl -L --retry 3 --retry-delay 2 --ftp-pasv --connect-timeout 20 \
    -o "$root/data/sift.tar.gz" \
    ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz; then
  tar -xzf "$root/data/sift.tar.gz" -C "$root/data"
fi

if [[ -f "$base" ]]; then
  ls -lh "$root/data/sift"
  exit 0
fi

hdf5="$root/data/sift-128-euclidean.hdf5"
if [[ ! -f "$hdf5" ]]; then
  curl -L --retry 4 --retry-delay 2 -o "$hdf5" \
    http://ann-benchmarks.com/sift-128-euclidean.hdf5
fi
python3 "$root/bench/hdf5_to_texmex.py" "$hdf5" "$root/data/sift"
ls -lh "$root/data/sift"
