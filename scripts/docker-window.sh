#!/usr/bin/env bash
# Run the screener container for a daily window: start now, stop at STOP_AT.
#
# Intended for cron, e.g. a weekday 05:00-15:00 local-time window:
#   0 5 * * 1-5 STOP_AT=15:00 /path/to/repo/scripts/docker-window.sh >> /path/to/screener-cron.log 2>&1
#
# The script starts the container detached (so crashes restart during the
# window), waits until STOP_AT, then stops and removes it. State stays in the
# screener-data volume, so the model cache survives between days.
set -euo pipefail

# cron usually runs with a minimal PATH; make sure docker/date/readlink resolve.
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

REPO_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")/.." && pwd)"
cd "$REPO_DIR"

STOP_AT="${STOP_AT:-15:00}"
if [[ ! "$STOP_AT" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]]; then
    echo "STOP_AT must be HH:MM in 24-hour time (got '$STOP_AT')" >&2
    exit 2
fi

DOCKER="${DOCKER:-docker}"
CONTAINER="${DOCKER_CONTAINER:-screener}"
IMAGE="${DOCKER_IMAGE:-screener:latest}"
VOLUME="${DOCKER_VOLUME:-screener-data}"
ENV_FILE="${ENV_FILE:-.env}"
CONFIG_FILE="${CONFIG_FILE:-config/config.toml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# Epoch of today's STOP_AT. Fails when the stop time has already passed.
stop_epoch() {
    local now target
    now=$(date +%s)
    target=$(date -d "today ${STOP_AT}" +%s)
    if (( target <= now )); then
        return 1
    fi
    printf '%s\n' "$target"
}

remaining_seconds() {
    local target
    if ! target=$(stop_epoch); then
        printf '0\n'
        return
    fi
    printf '%s\n' "$(( target - $(date +%s) ))"
}

cleanup() {
    "$DOCKER" stop -t 30 "$CONTAINER" >/dev/null 2>&1 || true
    "$DOCKER" rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 0' INT TERM HUP

if (( $(remaining_seconds) <= 0 )); then
    echo "STOP_AT ${STOP_AT} has already passed today; not starting" >&2
    exit 0
fi

# Build only when the image is missing; cron should not rebuild every morning.
if ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image ${IMAGE} not found; building it now"
    "$DOCKER" build -t "$IMAGE" "$REPO_DIR"
fi

# Remove a leftover container from a previous (interrupted) run.
"$DOCKER" rm -f "$CONTAINER" >/dev/null 2>&1 || true

env_args=()
if [[ -f "$ENV_FILE" ]]; then
    env_args=(--env-file "$ENV_FILE")
fi

"$DOCKER" run -d \
    --name "$CONTAINER" \
    --restart on-failure \
    --init \
    ${env_args[@]+"${env_args[@]}"} \
    -v "$(readlink -f -- "$CONFIG_FILE"):/config/config.toml:ro" \
    -e SCREENER_CONFIG=/config/config.toml \
    -v "${VOLUME}:/data" \
    "$IMAGE" >/dev/null

echo "screener started at $(date '+%F %T %Z'); stop scheduled for ${STOP_AT} local time"

# Sleep in short chunks and check the container, so a crash/suspend does not
# silently leave the cron job waiting forever.
while (( $(remaining_seconds) > 0 )); do
    status=$("$DOCKER" inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)
    case "$status" in
        running|restarting|created) ;;
        *)
            echo "container ${CONTAINER} is ${status}; ending the window early" >&2
            exit 1
            ;;
    esac

    left=$(remaining_seconds)
    if (( left > 60 )); then
        sleep 60
    else
        sleep "$left"
    fi
done

echo "screener window closed at $(date '+%F %T %Z')"
