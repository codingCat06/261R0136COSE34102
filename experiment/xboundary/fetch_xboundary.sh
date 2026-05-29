#!/usr/bin/env bash
# fetch_xboundary.sh — X-Boundary repo 의 학습 데이터 + 핵심 src 만 sparse clone.
#
#   bash experiment/xboundary/fetch_xboundary.sh [DEST]
#
# DEST 기본 = /content/XBoundary (Colab). 우리는 *데이터·라이선스·참고* 만 받음 —
# 저자 학습 스크립트(lorra_x_boundary.py)는 우리가 3.2-3B 로 포팅했고
# train_xboundary.py 가 대체. 데이터(circuit_breakers_train_2400.json,
# circuit_breakers_val.json, ORbench_retain_set.json) 는 그대로 사용.
#
# multi_turn(SafeMT) 는 우리 over_refuse setup 이 single-turn 이라 미사용.
set -euo pipefail

DEST="${1:-/content/XBoundary}"
REPO="https://github.com/AI45Lab/X-Boundary"

if [ -d "$DEST/.git" ]; then
  echo "[fetch] 기존 clone 재사용: $DEST"
else
  echo "[fetch] sparse clone (blob:none) → $DEST"
  # --filter=blob:none + sparse cone : 데이터 + src 디렉토리만 fetch (큰 model ckpt 없음)
  git clone --no-checkout --depth 1 --filter=blob:none "$REPO" "$DEST"
  git -C "$DEST" sparse-checkout init --cone
  # data/train (학습 데이터) + src (참고용) + LICENSE 만
  git -C "$DEST" sparse-checkout set data src LICENSE README.md
  git -C "$DEST" checkout
fi

COMMIT="$(git -C "$DEST" rev-parse --short HEAD)"

# 필수 학습 파일 검증 (없으면 train_xboundary.py 가 fail)
REQUIRED=(
  "data/train/circuit_breakers_train_2400.json"   # erase set (harmful Q + harmful A)
  "data/train/circuit_breakers_val.json"          # val (eval 중 cos sim log)
  "data/train/ORbench_retain_set.json"            # boundary set (over-refusal Q + complied A)
)
MISSING=0
for f in "${REQUIRED[@]}"; do
  if [ ! -f "$DEST/$f" ]; then
    echo "ERROR: 필수 파일 없음 — $DEST/$f" >&2
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "  실제 받은 트리:" >&2
  find "$DEST/data" -maxdepth 3 -type f 2>/dev/null | sed 's/^/    /' >&2 || true
  exit 1
fi

SIZE="$(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "[fetch] OK  commit=$COMMIT  clone=$SIZE"
echo
echo "다음 단계 — 이 경로를 --xb-data-dir 로:"
echo "  --xb-data-dir '$DEST/data/train'"
