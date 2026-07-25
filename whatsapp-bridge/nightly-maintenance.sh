#!/bin/bash
# One-off maintenance run: transcribe the voice-note backlog, then rebuild the
# search index from scratch.
#
# Ordering matters. The backfill writes transcripts into messages.content, and
# the reindex afterwards is what folds them into the context groups — so the
# backfill runs with --skip-index (SQLite only, no bleve lock) and the bridge
# stays up for it. Only the reindex needs the service stopped, which keeps the
# outage to that stage instead of covering both.
#
# The reindex also repairs group boundaries: until today the reply join matched
# across chats and skewed every OFFSET-based context window.
set -uo pipefail

cd /home/bruno/git/whatsapp-mcp/whatsapp-bridge || exit 1

BIN=./whatsapp-bridge-bin
SERVICE=whatsapp-bridge.service

# Match the service unit: keep native math thread pools from spawning one
# thread per core on a 4-core box.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export OMP_WAIT_POLICY=passive

log() { echo "[$(date '+%F %T')] $*"; }

# Whatever happens below, the bridge must come back up.
restart_service() {
    if ! systemctl --user is-active --quiet "$SERVICE"; then
        log "Restarting $SERVICE"
        systemctl --user start "$SERVICE"
    fi
}
trap restart_service EXIT

log "=== Stage 1: voice-note transcription backfill (bridge stays up) ==="
if "$BIN" --transcribe-backfill --skip-index; then
    log "Stage 1 complete"
else
    log "Stage 1 FAILED (exit $?) — continuing to reindex anyway"
fi

log "=== Stage 2: full reindex (bridge stopped) ==="
log "Stopping $SERVICE"
systemctl --user stop "$SERVICE"

if "$BIN" --reindex all; then
    log "Stage 2 complete"
else
    log "Stage 2 FAILED (exit $?) — index may be partial, check before relying on search"
fi

log "Starting $SERVICE"
systemctl --user start "$SERVICE"
sleep 5
systemctl --user is-active --quiet "$SERVICE" && log "$SERVICE is up" || log "$SERVICE FAILED to start"

log "=== Maintenance run finished ==="
