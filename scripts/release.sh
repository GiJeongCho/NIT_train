#!/usr/bin/env bash
# NIT_train 릴리스 빌드 (Linux/macOS)
#
# 버전(VERSION 파일 또는 첫 인자), 깃 커밋, 빌드 시각을 이미지에 박아 태깅한다.
#   ./scripts/release.sh                 # VERSION 파일 버전으로 빌드
#   ./scripts/release.sh 1.1.0           # 특정 버전으로 빌드
#   UP=1 ./scripts/release.sh            # 빌드 후 기동
#   CPU=1 UP=1 ./scripts/release.sh      # GPU 없는 호스트
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-$(tr -d '[:space:]' < VERSION)}"
GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo '')"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export VERSION GIT_SHA BUILD_DATE
export NIT_TRAIN_TAG="$VERSION"

FILES=(-f docker-compose.yml)
[ "${CPU:-0}" = "1" ] && FILES+=(-f docker-compose.cpu.yml)

echo "[release] nit-train $VERSION (git $GIT_SHA, $BUILD_DATE)"
docker compose "${FILES[@]}" build

if [ "${NO_LATEST:-0}" != "1" ]; then
  docker tag "nit-train:$VERSION"       "nit-train:latest"
  docker tag "nit-train-front:$VERSION" "nit-train-front:latest"
fi

[ "${UP:-0}" = "1" ] && docker compose "${FILES[@]}" up -d

echo "[release] 완료. 화면 http://localhost:8890  API http://localhost:8888/docs"
