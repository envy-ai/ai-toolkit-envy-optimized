#!/bin/sh

set -eu

TASK_LOG=/home/bart/ai-toolkit-qwen/output/scheduled-restart-20260824.log
COMFY_LOG=/home/bart/ComfyUI/scheduled-restart-20260824.log
AI_TOOLKIT_DB=/home/bart/ai-toolkit-qwen/aitk_db.db
COMFY_DIR=/home/bart/ComfyUI
COMFY_PYTHON=/home/bart/.conda/envs/comfyui/bin/python
COMFY_PATTERN='^/home/bart/.conda/envs/comfyui/bin/python main.py --listen --reserve-vram 0.2 --preview-method taesd --enable-cors-header --use-ck-attention --cache-ram 48 48$'
COMFY_HEALTH_URL=http://127.0.0.1:8188/system_stats
QUEUE_START_URL=http://127.0.0.1:8675/api/queue/0/start
QUEUE_STOP_URL=http://127.0.0.1:8675/api/queue/0/stop
QUEUE_STATUS_URL=http://127.0.0.1:8675/api/queue

exec >>"$TASK_LOG" 2>&1

date '+Scheduled operation started: %Y-%m-%d %H:%M:%S %Z'

active_training_jobs() {
    sqlite3 -batch -noheader -cmd '.timeout 5000' "$AI_TOOLKIT_DB" \
        "SELECT name FROM Job WHERE job_type = 'train' AND status IN ('running', 'stopping') ORDER BY name;"
}

active_jobs="$(active_training_jobs)"
if [ -n "$active_jobs" ]; then
    echo "Active training job(s) found; skipping ComfyUI restart and queue start:"
    printf '%s\n' "$active_jobs"
    date '+Scheduled operation skipped: %Y-%m-%d %H:%M:%S %Z'
    exit 0
fi

# If the queue is enabled but idle, pause it before the final guard check so it
# cannot launch a training job while ComfyUI is being restarted.
queue_was_running="$(
    sqlite3 -batch -noheader -cmd '.timeout 5000' "$AI_TOOLKIT_DB" \
        "SELECT COALESCE(MAX(is_running), 0) FROM Queue WHERE gpu_ids = '0';"
)"
if [ "$queue_was_running" = "1" ]; then
    echo "Pausing the idle AI Toolkit GPU-0 queue for the final training guard check"
    curl -fsS "$QUEUE_STOP_URL" >/dev/null

    active_jobs="$(active_training_jobs)"
    if [ -n "$active_jobs" ]; then
        echo "A training job started during the guard check; restoring the queue and skipping:"
        printf '%s\n' "$active_jobs"
        curl -fsS "$QUEUE_START_URL" >/dev/null
        date '+Scheduled operation skipped: %Y-%m-%d %H:%M:%S %Z'
        exit 0
    fi
fi

comfy_pids="$(pgrep -f "$COMFY_PATTERN" || true)"
if [ -n "$comfy_pids" ]; then
    echo "Stopping ComfyUI process(es): $comfy_pids"
    kill -TERM $comfy_pids

    wait_seconds=0
    while [ "$wait_seconds" -lt 30 ]; do
        remaining="$(pgrep -f "$COMFY_PATTERN" || true)"
        [ -z "$remaining" ] && break
        sleep 1
        wait_seconds=$((wait_seconds + 1))
    done

    remaining="$(pgrep -f "$COMFY_PATTERN" || true)"
    if [ -n "$remaining" ]; then
        echo "ComfyUI did not stop gracefully; terminating remaining process(es): $remaining"
        kill -KILL $remaining
    fi
else
    echo "No matching ComfyUI process was running; starting a fresh instance"
fi

cd "$COMFY_DIR"
nohup "$COMFY_PYTHON" main.py \
    --listen \
    --reserve-vram 0.2 \
    --preview-method taesd \
    --enable-cors-header \
    --use-ck-attention \
    --cache-ram 48 48 \
    >>"$COMFY_LOG" 2>&1 </dev/null &
new_comfy_pid=$!
echo "Started ComfyUI as PID $new_comfy_pid"

health_attempt=1
while [ "$health_attempt" -le 90 ]; do
    if curl -fsS "$COMFY_HEALTH_URL" >/dev/null; then
        echo "ComfyUI health check passed"
        break
    fi
    if ! kill -0 "$new_comfy_pid" 2>/dev/null; then
        echo "ComfyUI exited before becoming healthy; see $COMFY_LOG"
        exit 1
    fi
    sleep 2
    health_attempt=$((health_attempt + 1))
done

if [ "$health_attempt" -gt 90 ]; then
    echo "ComfyUI did not become healthy within 180 seconds; queue was not started"
    exit 1
fi

echo "Starting AI Toolkit GPU-0 training queue"
curl -fsS "$QUEUE_START_URL"
echo
echo "Queue status after start:"
curl -fsS "$QUEUE_STATUS_URL"
echo
date '+Scheduled operation completed: %Y-%m-%d %H:%M:%S %Z'
