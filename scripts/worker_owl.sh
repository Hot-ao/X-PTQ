#!/bin/bash
# OWLv2 대기열 작업자(worker.sh의 run_owl.py판). 사용법: scripts/worker_owl.sh <run_dir> <gpu> <cpu_cores>
#   <run_dir>/queue.tsv 각 줄: "<태그>\t<run_owl.py 인자>". +P 조건이 있는 줄은 보호 진단 캐시가 생길 때까지 기다린다.
cd "$(dirname "$0")/.."
D=$1; G=$2; C=$3
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PCACHE=$(.venv/bin/python -c "import sys; sys.argv=['x']; import run_owl; print(run_owl.protect_file())" 2>/dev/null)
BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g=$G '$1==g{print $2}')
others() {
  nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep -i "$BUS" | \
    awk -F', ' '{print $1}' | xargs -r -I{} ps -o user= -p {} | awk -v me="$USER" '$1 != me' | wc -l
}
while true; do
  line=$(flock $D/queue.lock bash -c "while IFS= read -r l; do t=\${l%%\$'\t'*}; grep -qx \"\$t\" $D/taken.log 2>/dev/null || { echo \"\$t\" >> $D/taken.log; echo \"\$l\"; break; }; done < $D/queue.tsv")
  [ -z "$line" ] && { echo "[gpu$G] 대기열 비었음 $(date '+%m-%d %H:%M:%S')" >> $D/chain.log; exit 0; }
  tag=${line%%$'\t'*}; args=${line#*$'\t'}
  if [[ "$args" == *"+P"* ]]; then while [ ! -s "$PCACHE" ]; do sleep 120; done; fi
  while [ "$(others)" -gt 0 ]; do sleep 120; done
  echo "$tag start gpu$G $(date '+%m-%d %H:%M:%S')" >> $D/chain.log
  nice -n 5 taskset -c $C .venv/bin/python run_owl.py $args --device $G > $D/$tag.log 2>&1
  rc=$?
  echo "$tag done rc=$rc gpu$G $(date '+%m-%d %H:%M:%S')" >> $D/chain.log
  [ $rc -ne 0 ] && echo "$tag rc=$rc" >> $D/failed.log
done
