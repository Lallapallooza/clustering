#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
exec taskset --cpu-list 0-7 ./build/gemm_benchmark "$@"
