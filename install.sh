#!/bin/bash --login
set -Eeuo pipefail

ENV_NAME="pipe_crd"
ENV_FILE="environment.yaml"

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

# ---------------- Hash do env.yaml ----------------
ENV_HASH=$(sha256sum "$ENV_FILE" | awk '{print $1}')
ENV_PREFIX=$(conda info --base)/envs/$ENV_NAME
HASH_FILE="$ENV_PREFIX/.env_hash"

log "Environment hash: $ENV_HASH"

env_exists() {
  conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"
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
      # ---------------- Criar environment ----------------
      log "📦 Creating environment '${ENV_NAME}'..."
      conda env create -n "$ENV_NAME" -f "$ENV_FILE"

      # ---------------- Salvar hash ----------------
      echo "$ENV_HASH" > "$HASH_FILE"
    fi
  else
    log "⚠️ No hash metadata found. Recreating environment..."
    conda remove -n "$ENV_NAME" --all -y
    # ---------------- Criar environment ----------------
    log "📦 Creating environment '${ENV_NAME}'..."
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"

    # ---------------- Salvar hash ----------------
    echo "$ENV_HASH" > "$HASH_FILE"
  fi
fi

log "✅ Installation complete."
