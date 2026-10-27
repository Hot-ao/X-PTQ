#!/bin/bash
# 리더보드를 주기적으로 다시 만든다(기본 600초). 사용: nohup scripts/leaderboard/leaderboard_loop.sh [초] > /dev/null 2>&1 &
cd "$(dirname "$0")/../.."
while true; do
  .venv/bin/python scripts/leaderboard/make_leaderboard.py >/dev/null 2>&1 || echo "$(date '+%m-%d %H:%M') 실패" >&2
  sleep "${1:-600}"
done
