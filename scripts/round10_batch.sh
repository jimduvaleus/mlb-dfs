#!/usr/bin/env bash
# Round-10 selector batch runner (plans/variants_round10.yaml).
#
# - Resumable at slate granularity: replay_slate.py skips any slate whose
#   run dir already has a candidate_pool_debug.csv, so stopping loses at
#   most the in-flight slate.
# - Memory-contained: runs in its own systemd user scope (cgroup) with
#   MemoryHigh=10G (reclaim/throttle) and MemoryMax=12G (hard kill). An
#   OOM kill lands inside this scope only — it can never take down the
#   desktop/VS Code session. After a kill the loop relaunches python,
#   which resumes from the completed-slate state.
# - Pausable:
#     pause:  touch outputs/replay/round10.STOP && pkill -f replay_slate.py
#     resume: ./scripts/round10_batch.sh
# - Peak RSS is readable anytime while running:
#     cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/round10-batch.scope/memory.peak
cd "$(dirname "$0")/.."
STOP="$PWD/outputs/replay/round10.STOP"
rm -f "$STOP"
exec systemd-run --user --scope --unit=round10-batch \
  -p MemoryHigh=10G -p MemoryMax=12G --collect \
  nice -n 10 bash -c '
    source venv/bin/activate
    for attempt in 1 2 3; do
      python scripts/replay_slate.py --all --variants plans/variants_round10.yaml --portfolio-size 150
      rc=$?
      [ $rc -eq 0 ] && { echo "round10: batch complete"; break; }
      [ -f "'"$STOP"'" ] && { echo "round10: paused via STOP file"; break; }
      echo "round10: python exited rc=$rc (attempt $attempt/3) — resuming in 15s"
      sleep 15
    done'
