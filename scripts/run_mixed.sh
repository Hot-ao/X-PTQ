#!/bin/bash
# GPU 하나에서 여러 대기열을 차례로 처리한다. 사용: runs/run_mixed.sh <gpu> <cpu_cores> <기다릴 패턴|-> <종류:run_dir>...
#   종류: owl(scripts/worker_owl.sh) | yolo(scripts/worker.sh). 기다릴 패턴이 '-'가 아니면 "<패턴> <gpu> " 프로세스가 끝날 때까지 기다린다.
cd "$(dirname "$0")/.."
G=$1; C=$2; W=$3; shift 3
[ "$W" != "-" ] && while pgrep -f "$W $G " >/dev/null; do sleep 120; done
for kd in "$@"; do k=${kd%%:*}; d=${kd#*:}
  if [ "$k" = owl ]; then scripts/worker_owl.sh runs/$d $G $C; else scripts/worker.sh runs/$d $G $C; fi
done
