#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  echo "Usage: $0 [--task go2|l1w] [NUM_ENVS] [STAGE1_ITERS] [STAGE2_ITERS] [OVERRIDES...]"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi

TASK="go2"
if [[ ${1:-} == --task ]]; then
  TASK="${2:-}"
  [[ -n "${TASK}" ]] || { usage >&2; exit 2; }
  shift 2
fi

case "${TASK}" in
  go2) TASK="go2"; TASK_NAME="Go2ArmManipLoco" ;;
  l1w|l1_w_arm_manip_loco) TASK="l1w"; TASK_NAME="L1WArmManipLoco" ;;
  *) echo "Unknown 2PWBC task: ${TASK}" >&2; usage >&2; exit 2 ;;
esac

NUM_ENVS="${1:-1024}"
STAGE1_ITERS="${2:-1000}"
STAGE2_ITERS="${3:-3000}"
shift "$(( $# < 3 ? $# : 3 ))"
LOG_ROOT="${ROOT_DIR}/logs/2pwbc-mujoco/${TASK_NAME}"
mkdir -p "${LOG_ROOT}"
STAMP="$(date +%F_%H-%M-%S)"
STAGE1_RUN="$(mktemp -d "${LOG_ROOT}/${STAMP}_stage1_XXXXXX")"

uv run legged-train --task "${TASK}" --algo wbc2p --sim mujoco \
  stage=stage1 "algo.num_envs=${NUM_ENVS}" "algo.max_iterations=${STAGE1_ITERS}" \
  "training.log_dir=${STAGE1_RUN}" "$@"

echo "Stage-1 run: ${STAGE1_RUN}"
STAGE2_RUN="$(mktemp -d "${LOG_ROOT}/${STAMP}_stage2_XXXXXX")"
bash migration/train_stage2.sh --task "${TASK}" "${STAGE1_RUN}" "${STAGE2_ITERS}" "${NUM_ENVS}" -1 \
  "training.log_dir=${STAGE2_RUN}" "$@"
echo "Stage-2 run: ${STAGE2_RUN}"
