#!/usr/bin/env bash
set -euo pipefail

# Convenience entrypoint for the complete two-stage 2PWBC run.
# Usage: ./train_2pwbc.sh [--task go2|l1w] [NUM_ENVS] [STAGE1_ITERS] [STAGE2_ITERS] [OVERRIDES...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/migration/train_2pwbc.sh" "$@"
