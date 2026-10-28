#!/bin/bash
# ODinW-13 공식 배포본(GLIP, HuggingFace GLIPModel/GLIP의 odinw_35/)을 받는다. 사용법:
#   scripts/data/download_odinw13.sh [대상 폴더]      (기본 /data/taeho/odinw, 약 4GB)
# GLIP의 odinw/download.py와 같은 출처지만, 그 스크립트의 /tree/ URL은 HTML을 받아 오므로 /resolve/를 쓴다.
# 평가 split과 프롬프트를 정하는 GLIP config(configs/odinw_13/*.yaml, 13개)는 이 저장소의 configs/odinw13_glip/에
# 복사해 두었고, core/odinw.py가 읽는 위치(<대상 폴더>/glip_configs_odinw13/)로 함께 복사한다.
set -e
repo="$(cd "$(dirname "$0")/../.." && pwd)"
dst="${1:-/data/taeho/odinw}"
mkdir -p "$dst" && cd "$dst"
root=https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35
for d in AerialMaritimeDrone Aquarium CottontailRabbits EgoHands NorthAmericaMushrooms Packages PascalVOC Raccoon ShellfishOpenImages VehiclesOpenImages pistols pothole thermalDogsAndPeople; do
  echo "== $d"
  [ -d "$d" ] && { echo "already extracted"; continue; }
  wget -q -c "$root/$d.zip" -O "$d.zip"
  unzip -tq "$d.zip"
  unzip -q "$d.zip" -d .
  rm "$d.zip"
done
mkdir -p glip_configs_odinw13 && cp "$repo"/configs/odinw13_glip/*.yaml glip_configs_odinw13/
echo "DONE -> $dst  (평가: run_comparison.py ... --odinw --odinw-root $dst)"
