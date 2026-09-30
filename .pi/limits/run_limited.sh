#!/usr/bin/env bash
# Global resource limiter for subagent analysis commands (7g523).
#
#   run_limited.sh cpu <cmd...>   # <=12 concurrent CPU jobs, 2 threads, 2 pinned cores, GPU masked
#   run_limited.sh gpu <cmd...>   # same, plus it holds one of <=4 GPU slots; CUDA_VISIBLE_DEVICES=0
#
# Slot files live in /tmp/adv_limits/{cpu,gpu}; the 13th CPU job (or 5th GPU job)
# waits, so the whole fleet can never exceed the machine budget. Locks are held
# across exec, so they are released exactly when the wrapped process exits.
set -u

KIND=${1:?usage: run_limited.sh cpu|gpu <cmd...>}
shift
[ $# -ge 1 ] || { echo "usage: run_limited.sh cpu|gpu <cmd...>" >&2; exit 2; }

CPU_DIR=/tmp/adv_limits/cpu
GPU_DIR=/tmp/adv_limits/gpu
mkdir -p "$CPU_DIR" "$GPU_DIR"

acquire() { # $1=dir $2=count $3=fd ; sets SLOT
  local dir=$1 count=$2 fd=$3 i
  while :; do
    for ((i = 0; i < count; i++)); do
      eval "exec $fd>\"$dir/$i.lock\""
      if flock -n "$fd"; then
        SLOT=$i
        return 0
      fi
      eval "exec $fd>&-"
    done
    sleep 0.3
  done
}

acquire "$CPU_DIR" 12 9
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
CORES="$((SLOT * 2)),$((SLOT * 2 + 1))"

if [ "$KIND" = "gpu" ]; then
  acquire "$GPU_DIR" 4 8
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=""
fi

exec taskset -c "$CORES" "$@"
