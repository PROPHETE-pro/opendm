#!/usr/bin/env bash
# Tail the latest per-task eval log and keep a stable CURRENT.log symlink.
set -euo pipefail

SRC="${1:?source log dir}"
DEST="${2:?destination log dir}"
mkdir -p "${DEST}"

current=""
tail_pid=""

cleanup() {
  if [[ -n "${tail_pid}" ]]; then
    kill "${tail_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[follow_task_logs] src=${SRC}"
echo "[follow_task_logs] dest=${DEST}"
echo "[follow_task_logs] switch windows: Ctrl-b n / Ctrl-b p"

while true; do
  shopt -s nullglob
  latest=""
  newest_mtime=0
  for f in "${SRC}"/*.log; do
    mtime="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
    if (( mtime >= newest_mtime )); then
      newest_mtime="${mtime}"
      latest="${f}"
    fi
    base="$(basename "${f}")"
    ln -sfn "${f}" "${DEST}/${base}"
  done
  shopt -u nullglob

  if [[ -z "${latest}" ]]; then
    echo "[follow_task_logs] waiting for first task log..."
    sleep 2
    continue
  fi

  if [[ "${latest}" != "${current}" ]]; then
    if [[ -n "${tail_pid}" ]]; then
      kill "${tail_pid}" 2>/dev/null || true
      wait "${tail_pid}" 2>/dev/null || true
      tail_pid=""
    fi
    current="${latest}"
    ln -sfn "${current}" "${DEST}/CURRENT.log"
    echo
    echo "========== NOW: $(basename "${current}") =========="
    echo "full log: ${current}"
    echo
    tail -n 30 -F "${current}" &
    tail_pid="$!"
  fi
  sleep 2
done
