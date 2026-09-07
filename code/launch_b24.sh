#!/usr/bin/env bash
set -euo pipefail
RUN="${B24_RUN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${RUN}"
if [[ -f pipeline.pid ]] && kill -0 "$(cat pipeline.pid)" 2>/dev/null; then
  echo "B24 pipeline is already active" >&2
  exit 1
fi
# The outer watchdog leaves ample room for fixed validation and 200-shape
# inference; the inner 930-minute watchdog is the binding training limit.
nohup timeout --signal=TERM --kill-after=10m 60h bash "${RUN}/run_b24.sh" \
  > "${RUN}/pipeline.log" 2>&1 < /dev/null &
echo $! > pipeline.pid
printf 'B24_LAUNCHED pid=%s\n' "$(cat pipeline.pid)"
