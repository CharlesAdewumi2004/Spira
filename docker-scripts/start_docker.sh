#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="spira-env"
CONTAINER_NAME="spira-env-run"
CPU_SET="0"     # core to pin container to

echo "[*] Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "[*] Starting container '${CONTAINER_NAME}' on CPU(s): ${CPU_SET}"
echo "    Project mounted at /src, working dir = /src"
echo

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --cpuset-cpus="${CPU_SET}" \
  --cpus="1.0" \
  --privileged \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --shm-size=1g \
  -v "$PWD":/src \
  -w /src \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  "${IMAGE_NAME}" \
  bash
