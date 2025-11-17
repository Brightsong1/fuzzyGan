#!/bin/bash
# Convenience wrapper to launch VAE-based OpenSSL fuzzing in cyclic mode.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBRARY="${LIBRARY:-openssl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/fuzz_out}"
OSS_FUZZ_DIR="${OSS_FUZZ_DIR:-$REPO_ROOT/oss-fuzz}"
CYCLES="${CYCLES:-1000}"
FUZZ_SECONDS="${FUZZ_SECONDS:-600}"  # 10 минут на каждую итерацию по умолчанию

python3 "$REPO_ROOT/vae_fuzzing.py" \
  --library "$LIBRARY" \
  --out-dir "$OUT_DIR" \
  --oss-fuzz-dir "$OSS_FUZZ_DIR" \
  --cycles "$CYCLES" \
  --fuzz-seconds "$FUZZ_SECONDS" \
  "$@"
