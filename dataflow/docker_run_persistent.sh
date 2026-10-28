#!/bin/bash
# Persistent (detached, no --rm) FINN dev container launcher. Does NOT touch
# finn/run-docker.sh (that script is interactive/--rm and rebuilds by
# default -- not suited for a long-lived container reused across many
# `docker exec` calls over multiple days).
#
# Bakes FINN_BUILD_DIR and HOME as container-level defaults (Config.Env),
# so every future `docker exec <container> ...` inherits them automatically
# -- no more per-call `-e HOME=... -e FINN_BUILD_DIR=...` to remember.
#
# CRITICAL: FINN_BUILD_DIR points inside the always-mounted finn/ tree
# (persistent, survives Docker Desktop/WSL backend restarts), NOT /tmp
# (container default /tmp/finn_dev_<user> is wiped by a host VM restart --
# this has destroyed multi-hour build artifacts twice already).
#
# Usage: bash .docker_run [container_name]
set -e

FINN_ROOT=/home/thelegendiv/finn
FINN_BUILD_DIR="$FINN_ROOT/notebooks/enet/finn_build_tmp"
FINN_XILINX_PATH=/tools/Xilinx
FINN_XILINX_VERSION=2022.2
FINN_DOCKER_TAG="xilinx/finn:v0.10.1-10-g39f0c9a6b-dirty.xrt_202220.2.14.354_22.04-amd64-xrt"
CONTAINER_NAME="${1:-finn_persistent}"

mkdir -p "$FINN_BUILD_DIR/vivado_ip_cache"

docker run -d --init --ipc=host --name "$CONTAINER_NAME" \
  --hostname finn_dev_thelegendiv \
  -e HOME=/tmp/home_dir \
  -e SHELL=/bin/bash \
  -e FINN_ROOT="$FINN_ROOT" \
  -e FINN_BUILD_DIR="$FINN_BUILD_DIR" \
  -e VIVADO_IP_CACHE="$FINN_BUILD_DIR/vivado_ip_cache" \
  -e OHMYXILINX="$FINN_ROOT/deps/oh-my-xilinx" \
  -e XILINX_VIVADO="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION" \
  -e VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION" \
  -e HLS_PATH="$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION" \
  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 \
  -w "$FINN_ROOT" \
  -v "$FINN_ROOT:$FINN_ROOT" \
  -v "$FINN_XILINX_PATH:$FINN_XILINX_PATH" \
  -v /etc/group:/etc/group:ro \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/shadow:/etc/shadow:ro \
  --user 1000:1000 \
  "$FINN_DOCKER_TAG" sleep infinity

echo "Started persistent container: $CONTAINER_NAME"

# /home/thelegendiv is root-owned in this image even though we run as uid
# 1000 -- breaks any zsh-launched Vivado step (e.g. OOC synth) that needs to
# write ~/.Xilinx/... Fix once, right after creation (see finn-container-env
# repo memory for the full incident this papers over).
docker exec -u root "$CONTAINER_NAME" chown thelegendiv:thelegendiv /home/thelegendiv
echo "Fixed /home/thelegendiv ownership."

docker exec "$CONTAINER_NAME" bash -c 'echo HOME=$HOME; echo FINN_BUILD_DIR=$FINN_BUILD_DIR; python3 -c "import finn, qonnx; print(\"finn/qonnx importable OK\")"'
