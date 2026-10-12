#!/usr/bin/env bash
# Local OpenDM x RoboDojo eval wrapper.
# All writes stay under /kpfs_ssd/data/ruitong_gan. Shared-disk Assets are read via symlink.
set -euo pipefail

WORKSPACE="/kpfs_ssd/data/ruitong_gan"
MINICONDA="${WORKSPACE}/miniconda"
# git clone eval tree. robodojo_runtime/RoboDojo was deleted; do not recreate a hollow dir.
EVAL_ROOT="${WORKSPACE}/RoboDojo"
# Curated GL/X11 libs only. Pointing LD_LIBRARY_PATH at the whole conda lib
# shadows system libtinfo and breaks host bash.
RUNTIME_LIBS="${WORKSPACE}/robodojo_runtime/runtime_libs"
CKPT_DIR="${WORKSPACE}/opendm/user_checkpoints/dm05_robodojo_3task_sft/checkpoint-20000"
POLICY_DIR="${EVAL_ROOT}/XPolicyLab/policy/OpenDM"
RESULT_ROOT="${WORKSPACE}/robodojo_runtime/eval_result/RoboDojo"
CKPT_NAME="${OPENDM_CKPT:-dm05_robodojo}"

export PATH="${MINICONDA}/bin:${PATH}"
# shellcheck disable=SC1091
source "${MINICONDA}/etc/profile.d/conda.sh"
conda activate base >/dev/null 2>&1 || true

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export ROBODOJO_EVAL_ROOT="${ROBODOJO_EVAL_ROOT:-${RESULT_ROOT}}"
export LD_LIBRARY_PATH="${RUNTIME_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export EVAL_ENV_TYPE="${EVAL_ENV_TYPE:-sim}"
# curobo third_party copy has no usable git tags; pretend version if scm is invoked.
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0}"

# Keep warp / ov caches and eval logs under the local workspace.
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${WORKSPACE}/robodojo_runtime/caches/warp}"
mkdir -p "${WARP_CACHE_PATH}" "${ROBODOJO_EVAL_ROOT}" \
  "${WORKSPACE}/robodojo_runtime/caches/ov" \
  "${WORKSPACE}/robodojo_runtime/caches/ov-data" \
  "${WORKSPACE}/robodojo_runtime/logs"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKSPACE}/robodojo_runtime/caches}"

if [[ ! -f "${EVAL_ROOT}/scripts/robodojo.sh" ]]; then
  echo "[run_robodojo_eval] missing ${EVAL_ROOT}/scripts/robodojo.sh" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: bash opendm/script/run_robodojo_eval.sh <command> [args]

Commands:
  doctor        Validate local assets/configs (optional --skip-isaac)
  smoke         One-episode sweep (default all runnable tasks)
  eval          Single-task eval (default task=stack_bowls, eval-num=1)
  benchmark     All 54 runnable tasks with --eval-num native
                检测到 2+ GPU 时自动 --gpu-ids（一卡一组 OpenDM+Isaac，并行不同 task）
                可用 --gpu-ids 0 或 ROBODOJO_GPU_IDS=0 强制单卡
                总日志：robodojo_runtime/logs/<run_id>.log
                分任务：robodojo_runtime/logs/<run_id>/<task>.log
                视频/_result.json：robodojo_runtime/eval_result/RoboDojo/<task>/...
  summarize     Aggregate eval_result into a markdown table
  diagnostic    A800 low-res single-step stack_bowls check (not a score)

Common flags are forwarded to scripts/robodojo.sh. Defaults:
  --policy-dir  robodojo_runtime/RoboDojo/XPolicyLab/policy/OpenDM
  --ckpt        checkpoint dir name under opendm/checkpoints (default dm05_robodojo)
                smoke 可用 --ckpt dm05_base 或 --ckpt dm05_robotwin
  --env-cfg     arx_x5
  --action-type joint
  --policy-env  opendm
  --eval-env    RoboDojo
EOF
}

resolve_ckpt_from_args() {
  local prev=""
  local arg
  for arg in "$@"; do
    if [[ "${prev}" == "--ckpt" ]]; then
      CKPT_NAME="${arg}"
    fi
    prev="${arg}"
  done
  CKPT_DIR="${WORKSPACE}/opendm/checkpoints/${CKPT_NAME}"
  if [[ ! -d "${CKPT_DIR}" && -d "${POLICY_DIR}/checkpoints/${CKPT_NAME}" ]]; then
    CKPT_DIR="${POLICY_DIR}/checkpoints/${CKPT_NAME}"
  fi
}

require_ckpt() {
  if [[ ! -f "${CKPT_DIR}/model.safetensors" ]]; then
    echo "[run_robodojo_eval] checkpoint not ready: ${CKPT_DIR}/model.safetensors" >&2
    ls -lah "${CKPT_DIR}"/model.safetensors* >&2 || true
    exit 1
  fi
  if [[ ! -f "${CKPT_DIR}/norm_stats.json" ]]; then
    echo "[run_robodojo_eval] missing norm_stats.json in ${CKPT_DIR}" >&2
    exit 1
  fi
}

common_policy_args() {
  printf '%s\n' \
    --policy-dir "${POLICY_DIR}" \
    --ckpt "${CKPT_NAME}" \
    --env-cfg arx_x5 \
    --action-type joint \
    --policy-env opendm \
    --eval-env RoboDojo
}

args_have_gpu_ids() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --gpu-ids|--policy-gpu-ids|--env-gpu-ids) return 0 ;;
      --gpu-ids=*|--policy-gpu-ids=*|--env-gpu-ids=*) return 0 ;;
    esac
  done
  return 1
}

detect_default_gpu_ids() {
  local ids n i
  if [[ -n "${ROBODOJO_GPU_IDS:-}" ]]; then
    printf '%s' "${ROBODOJO_GPU_IDS// /}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    n="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
    if [[ "${n}" -le 1 ]]; then
      printf '%s' "0"
    else
      ids=""
      for ((i = 0; i < n; i++)); do
        ids+="${ids:+,}${i}"
      done
      printf '%s' "${ids}"
    fi
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -
}

# Print --gpu-ids ... when multiple GPUs are visible and the caller did not set one.
auto_gpu_id_args() {
  if args_have_gpu_ids "$@"; then
    return 0
  fi
  local ids
  ids="$(detect_default_gpu_ids | tr -d '[:space:]')"
  if [[ -z "${ids}" ]]; then
    return 0
  fi
  if [[ "${ids}" == *","* ]]; then
    echo "[run_robodojo_eval] auto --gpu-ids ${ids} (one OpenDM+Isaac pair per GPU)" >&2
    printf '%s\n' --gpu-ids "${ids}"
  fi
}

cmd="${1:-}"
if [[ -z "${cmd}" || "${cmd}" == "-h" || "${cmd}" == "--help" ]]; then
  usage
  exit 0
fi
shift
resolve_ckpt_from_args "$@"

cd "${EVAL_ROOT}"

case "${cmd}" in
  doctor)
    bash scripts/robodojo.sh doctor \
      --policy-dir "${POLICY_DIR}" \
      --ckpt "${CKPT_NAME}" \
      --policy-env opendm \
      --sim-env RoboDojo \
      "$@"
    ;;
  diagnostic)
    require_ckpt
    export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
    export ROBODOJO_CAMERA_RESOLUTION="${ROBODOJO_CAMERA_RESOLUTION:-160x120}"
    export ROBODOJO_RENDER_PERFORMANCE="${ROBODOJO_RENDER_PERFORMANCE:-true}"
    export ROBODOJO_RESET_WARMUP_RENDERS="${ROBODOJO_RESET_WARMUP_RENDERS:-1}"
    export ROBODOJO_RESET_SETTLE_STEPS="${ROBODOJO_RESET_SETTLE_STEPS:-1}"
    export ROBODOJO_RESET_RENDER_INTERVAL="${ROBODOJO_RESET_RENDER_INTERVAL:-0}"
    export ROBODOJO_STEP_LIMIT="${ROBODOJO_STEP_LIMIT:-1}"
    bash scripts/robodojo.sh eval \
      $(common_policy_args) \
      --task stack_bowls \
      --seed 0 \
      --policy-gpu 0 \
      --env-gpu 0 \
      --eval-num 1 \
      "$@"
    ;;
  eval)
    require_ckpt
    bash scripts/robodojo.sh eval \
      $(common_policy_args) \
      --task stack_bowls \
      --seed 0 \
      --policy-gpu 0 \
      --env-gpu 0 \
      --eval-num 1 \
      "$@"
    ;;
  smoke)
    require_ckpt
    bash scripts/robodojo.sh smoke \
      $(common_policy_args) \
      --fail-fast \
      $(auto_gpu_id_args "$@") \
      "$@"
    ;;
  benchmark)
    require_ckpt
    unset ROBODOJO_CAMERA_RESOLUTION ROBODOJO_RENDER_PERFORMANCE ROBODOJO_STEP_LIMIT \
      ROBODOJO_RESET_WARMUP_RENDERS ROBODOJO_RESET_SETTLE_STEPS ROBODOJO_RESET_RENDER_INTERVAL || true
    bench_run_id="${ROBODOJO_BENCH_RUN_ID:-benchmark_${CKPT_NAME}_$(date -u +%Y%m%dT%H%M%SZ)}"
    task_log_dir="${WORKSPACE}/robodojo_runtime/logs/${bench_run_id}"
    mkdir -p "${task_log_dir}"
    echo "[run_robodojo_eval] benchmark run_id=${bench_run_id}"
    echo "[run_robodojo_eval] per-task logs=${task_log_dir}/<task>.log"
    bash scripts/robodojo.sh benchmark \
      $(common_policy_args) \
      --eval-num native \
      --seed 0 \
      --run-id "${bench_run_id}" \
      --log-dir "${task_log_dir}" \
      $(auto_gpu_id_args "$@") \
      "$@"
    ;;
  summarize)
    bash scripts/robodojo.sh summarize "$@"
    ;;
  *)
    echo "[run_robodojo_eval] unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
