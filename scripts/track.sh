#!/bin/bash
# 대기열 밖에서 띄우는 작업(OWLv2, 점검 실행 등)을 리더보드의 진행 현황에 남긴다. worker.sh와 같은 chain.log 형식.
# 사용: scripts/track.sh <run_dir> <태그> <gpu> <명령...>    (저장소 루트 기준, 로그는 <run_dir>/<태그>.log)
cd "$(dirname "$0")/.."
D=$1; T=$2; G=$3; shift 3
mkdir -p "$D"
echo "$T start gpu$G $(date '+%m-%d %H:%M:%S')" >> "$D/chain.log"
"$@" > "$D/$T.log" 2>&1
rc=$?
echo "$T done rc=$rc gpu$G $(date '+%m-%d %H:%M:%S')" >> "$D/chain.log"
[ $rc -ne 0 ] && echo "$T rc=$rc" >> "$D/failed.log"
exit $rc
