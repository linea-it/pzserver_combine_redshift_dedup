#!/bin/bash --login
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "$0")
PIPELINE_DIR=$(dirname "$SCRIPT_PATH")

ENV_NAME="pipe_crd"
ENV_FILE="$PIPELINE_DIR/environment.yaml"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"

log() {
  local ts
  ts="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[$ts] $*"
}

if ! command -v "$MICROMAMBA_BIN" >/dev/null 2>&1; then
  echo "❌ micromamba not found in PATH"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "❌ ${ENV_FILE} not found"
  exit 1
fi

ENV_HASH=$(sha256sum "$ENV_FILE" | awk '{print $1}')
HASH_FILE="$PIPELINE_DIR/.env_hash"

log "Environment hash: $ENV_HASH"

env_exists() {
  "$MICROMAMBA_BIN" env list --root-prefix "$MAMBA_ROOT_PREFIX" | awk '{print $1}' | grep -q "^${ENV_NAME}$"
}

create_env() {
  log "📦 Creating environment '${ENV_NAME}'..."
  "$MICROMAMBA_BIN" create --root-prefix "$MAMBA_ROOT_PREFIX" -y -n "$ENV_NAME" -f "$ENV_FILE"
  echo "$ENV_HASH" > "$HASH_FILE"
}

if env_exists; then
  log "Environment '${ENV_NAME}' already exists."

  if [ -f "$HASH_FILE" ]; then
    EXISTING_HASH=$(cat "$HASH_FILE")

    if [ "$EXISTING_HASH" = "$ENV_HASH" ]; then
      log "✅ Environment is up-to-date."
    else
      log "⚠️ env.yaml changed. Recreating environment..."
      "$MICROMAMBA_BIN" env remove --root-prefix "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" -y
      create_env
    fi
  else
    log "⚠️ No hash metadata found. Recreating environment..."
    "$MICROMAMBA_BIN" env remove --root-prefix "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" -y
    create_env
  fi
else
  create_env
fi

log "✅ Installation complete."
