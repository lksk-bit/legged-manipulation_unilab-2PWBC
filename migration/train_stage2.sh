#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK="go2"
if [[ ${1:-} == --task && $# -ge 3 ]]; then
  TASK="${2}"
  shift 2
fi

if [[ $# -lt 1 || $1 == --task ]]; then
  echo "Usage: $0 [--task go2|l1w] STAGE1_OR_STAGE2_RUN [ADDITIONAL_ITERS] [NUM_ENVS] [CHECKPOINT] [OVERRIDES...]" >&2
  exit 2
fi

RUN="$1"
ITERS="${2:-3000}"
NUM_ENVS="${3:-1024}"
CHECKPOINT="${4:--1}"
shift "$(( $# < 4 ? $# : 4 ))"
exec uv run legged-train --task "${TASK}" --algo wbc2p --sim mujoco \
  stage=stage2 "algo.load_run=${RUN}" "algo.checkpoint=${CHECKPOINT}" \
  "algo.num_envs=${NUM_ENVS}" "algo.max_iterations=${ITERS}" "$@"
