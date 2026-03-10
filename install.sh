#!/bin/bash --login
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
PIPELINE_DIR=$(dirname "$SCRIPT_PATH")

ENV_NAME="pipe_crd"
ENV_FILE="$PIPELINE_DIR/environment.yaml"

log() {
  local ts
  ts="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[$ts] $*"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ Conda not found in PATH"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "❌ ${ENV_FILE} not found"
  exit 1
fi

# --- helper ---
conda_has_cmd() {
  conda commands 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

accept_conda_tos() {
  if conda_has_cmd tos; then
    log "conda 'tos' available → accepting ToS for required channels…"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  else
    log "conda 'tos' not available → skipping ToS acceptance."
  fi
}

# ---------------- Hash do env.yaml ----------------
ENV_HASH=$(sha256sum "$ENV_FILE" | awk '{print $1}')
HASH_FILE="$PIPELINE_DIR/.env_hash"

log "Environment hash: $ENV_HASH"

env_exists() {
  conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"
}

create_env() {
  accept_conda_tos
  log "📦 Creating environment '${ENV_NAME}'..."
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
  echo "$ENV_HASH" > "$HASH_FILE"
}

# ---------------- Lógica ----------------
if env_exists; then
  log "Environment '${ENV_NAME}' already exists."

  if [ -f "$HASH_FILE" ]; then
    EXISTING_HASH=$(cat "$HASH_FILE")

    if [ "$EXISTING_HASH" = "$ENV_HASH" ]; then
      log "✅ Environment is up-to-date."
    else
      log "⚠️ env.yaml changed. Recreating environment..."
      conda remove -n "$ENV_NAME" --all -y
      create_env
    fi
  else
    log "⚠️ No hash metadata found. Recreating environment..."
    conda remove -n "$ENV_NAME" --all -y
    create_env
  fi
else
  create_env
fi

log "✅ Installation complete."
