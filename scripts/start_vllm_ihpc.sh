#!/bin/bash
# =============================================================================
# start_vllm_ihpc.sh — One-command vLLM deployment on EBI HPC
#
# Submits a SLURM job via SSH, waits for a compute node, starts the vLLM
# server automatically, and sets up SSH port forwarding — all from a single
# terminal.
#
# Usage:
#   bash scripts/start_vllm_ihpc.sh                  # start with defaults
#   bash scripts/start_vllm_ihpc.sh --user wuy        # specify EBI username
#   bash scripts/start_vllm_ihpc.sh --reconnect       # reconnect tunnel
#   bash scripts/start_vllm_ihpc.sh --stop             # stop server & release
#   bash scripts/start_vllm_ihpc.sh --status           # check running server
#   bash scripts/start_vllm_ihpc.sh --logs             # attach to server logs
#
# See --help for all options.
# =============================================================================
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
EBI_USER="wuy"
LOGIN_HOST="ihpc.ebi.ac.uk"
SRUN_TIME="80:00:00"
GPU_TYPE="a100"
GPU_COUNT=1
CPUS=1
MEM="32G"
LOCAL_PORT=8000
MODEL="qwen3.5-9b"
ACTION="start"
DETACH=false

REMOTE_PROJECT_ROOT="/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct"
REMOTE_NODE_SCRIPT="${REMOTE_PROJECT_ROOT}/scripts/vllm_node_setup.sh"
REMOTE_INFO_FILE="${REMOTE_PROJECT_ROOT}/.vllm_server_info"
TMUX_SESSION="vllm-srun"

# ---- Usage ------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start, manage, or reconnect to a vLLM server on EBI HPC.

Options:
  --user USER        EBI username                       (default: wuy)
  --host HOST        Login node hostname                (default: ihpc.ebi.ac.uk)
  --time HH:MM:SS   SLURM wall-time limit              (default: 4:00:00)
  --gpu-type TYPE    GPU type for SLURM --gres           (default: a100)
  --gpus N           Number of GPUs                     (default: 4)
  --cpus N           CPUs per task                      (default: 16)
  --mem SIZE         Memory (e.g. 32G)                  (default: 32G)
  --port PORT        Local port for the API             (default: 8000)
  --model MODEL      Model to serve (e.g. qwen3-8b)             (default: qwen3-8b)
  --detach           Run in background, don't block and wait for Ctrl-C

Actions (mutually exclusive, default is start):
  --stop             Stop the running server and release compute
  --status           Show server status
  --reconnect        Reconnect port forwarding to a running server
  --logs             Attach to the tmux session showing server logs

  -h, --help         Show this help message
EOF
    exit 0
}

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)      EBI_USER="$2";    shift 2 ;;
        --host)      LOGIN_HOST="$2";  shift 2 ;;
        --time)      SRUN_TIME="$2";   shift 2 ;;
        --gpu-type)  GPU_TYPE="$2";    shift 2 ;;
        --gpus)      GPU_COUNT="$2";   shift 2 ;;
        --cpus)      CPUS="$2";        shift 2 ;;
        --mem)       MEM="$2";         shift 2 ;;
        --port)      LOCAL_PORT="$2";  shift 2 ;;
        --model)     MODEL="$2";       shift 2 ;;
        --detach)    DETACH=true;      shift ;;
        --stop)      ACTION="stop";    shift ;;
        --status)    ACTION="status";  shift ;;
        --reconnect) ACTION="reconnect"; shift ;;
        --logs)      ACTION="logs";    shift ;;
        -h|--help)   usage ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"

# ---- SSH multiplexing (authenticate once, reuse connection) -----------------
SSH_CONTROL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/vllm-ssh-XXXXXX")
SSH_SOCKET="${SSH_CONTROL_DIR}/ctrl-socket"
SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o "ControlPath=${SSH_SOCKET}" -o ControlPersist=600)

close_ssh_mux() {
    ssh "${SSH_OPTS[@]}" -O exit "$LOGIN_NODE" 2>/dev/null || true
    rm -rf "$SSH_CONTROL_DIR"
}
trap close_ssh_mux EXIT

# ---- Helpers ----------------------------------------------------------------
ssh_login() {
    ssh "${SSH_OPTS[@]}" "$LOGIN_NODE" "$@"
}

read_info() {
    ssh_login "cat ${REMOTE_INFO_FILE} 2>/dev/null" || true
}

parse_info() {
    local info="$1"
    COMPUTE_HOST=$(echo "$info" | cut -d: -f1)
    REMOTE_PORT=$(echo "$info" | cut -d: -f2)
    ACTUAL_GPUS=$(echo "$info" | cut -d: -f3)
}

print_connection_info() {
    echo ""
    echo "============================================================"
    echo "  vLLM Server"
    echo "============================================================"
    echo "  Compute host:  ${COMPUTE_HOST}"
    echo "  Remote port:   ${REMOTE_PORT}"
    echo "  GPUs:          ${ACTUAL_GPUS}"
    echo "  API URL:       http://localhost:${LOCAL_PORT}/v1/"
    echo "============================================================"
}

# ---- STOP -------------------------------------------------------------------
if [[ "$ACTION" == "stop" ]]; then
    echo "Stopping vLLM server..."
    if ssh_login "tmux kill-session -t ${TMUX_SESSION} 2>/dev/null"; then
        echo "  tmux session '${TMUX_SESSION}' killed — compute node released."
    else
        echo "  No running tmux session '${TMUX_SESSION}' found."
    fi
    ssh_login "rm -f ${REMOTE_INFO_FILE} 2>/dev/null" || true
    exit 0
fi

# ---- STATUS -----------------------------------------------------------------
if [[ "$ACTION" == "status" ]]; then
    info=$(read_info)
    if [[ -z "$info" ]]; then
        echo "No vLLM server info found."
        exit 0
    fi
    parse_info "$info"
    print_connection_info
    if ssh_login "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
        echo "  Status:        RUNNING"
    else
        echo "  Status:        STALE (info file exists but tmux session gone)"
    fi
    echo ""
    exit 0
fi

# ---- LOGS -------------------------------------------------------------------
if [[ "$ACTION" == "logs" ]]; then
    info=$(read_info)
    if [[ -z "$info" ]]; then
        echo "No vLLM server info found. Start a server first."
        exit 1
    fi
    echo "Attaching to tmux session '${TMUX_SESSION}' on ${LOGIN_HOST}..."
    echo "  (Detach with Ctrl-B then D)"
    ssh -t -o ConnectTimeout=10 "$LOGIN_NODE" "tmux attach-session -t ${TMUX_SESSION}"
    exit 0
fi

# ---- RECONNECT --------------------------------------------------------------
if [[ "$ACTION" == "reconnect" ]]; then
    info=$(read_info)
    if [[ -z "$info" ]]; then
        echo "No server info found. Start a server first:"
        echo "  bash $0"
        exit 1
    fi
    parse_info "$info"
    print_connection_info
    echo ""
    echo "  Setting up port forwarding..."
    echo "  Press Ctrl-C to disconnect (server keeps running)."
    echo ""
    ssh -N -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -L "${LOCAL_PORT}:${COMPUTE_HOST}:${REMOTE_PORT}" "$LOGIN_NODE"
    exit 0
fi

# ---- START ------------------------------------------------------------------
# Check for an existing server
info=$(read_info)
if [[ -n "$info" ]]; then
    parse_info "$info"
    if ssh_login "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
        echo "A vLLM server is already running on ${COMPUTE_HOST}:${REMOTE_PORT}."
        echo ""
        echo "  Reconnect:  bash $0 --reconnect"
        echo "  Stop it:    bash $0 --stop"
        echo "  View logs:  bash $0 --logs"
        exit 1
    else
        echo "Stale info file found, cleaning up..."
        ssh_login "rm -f ${REMOTE_INFO_FILE} 2>/dev/null" || true
    fi
fi

# Verify the node-setup script exists on the remote node
if ! ssh_login "test -f ${REMOTE_NODE_SCRIPT}"; then
    echo "ERROR: Compute node script not found on HPC: ${REMOTE_NODE_SCRIPT}"
    exit 1
fi

ssh_login "rm -f ${REMOTE_INFO_FILE} 2>/dev/null" || true

echo "============================================================"
echo "  Starting vLLM on EBI HPC"
echo "============================================================"
echo "  User:    ${EBI_USER}"
echo "  Login:   ${LOGIN_HOST}"
echo "  Time:    ${SRUN_TIME}"
echo "  GPUs:    ${GPU_COUNT}x ${GPU_TYPE}"
echo "  CPUs:    ${CPUS}"
echo "  Memory:  ${MEM}"
echo "  Model:   ${MODEL}"
echo "============================================================"
echo ""

# Submit srun inside a detached tmux on the login node
echo "Submitting SLURM job..."
ssh_login "tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true; \
    tmux new-session -d -s ${TMUX_SESSION} \
    'srun -t ${SRUN_TIME} -N1 --gres=gpu:${GPU_TYPE}:${GPU_COUNT} --cpus-per-task=${CPUS} --mem=${MEM} \
     bash ${REMOTE_NODE_SCRIPT} ${REMOTE_INFO_FILE} ${MODEL}'"

echo "Waiting for compute node allocation..."

# Poll the info file (on the shared filesystem)
SECONDS=0
SPINNER=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
i=0
SSH_FAILURES=0
MAX_SSH_FAILURES=3
while true; do
    info=$(read_info)
    if [[ -n "$info" ]]; then
        break
    fi

    # Check the tmux session is still alive (job might have failed).
    # SSH checks can fail transiently (connection dropped, HPC network
    # hiccup) so we tolerate a few consecutive failures before giving up.
    if ! ssh_login "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
        SSH_FAILURES=$((SSH_FAILURES + 1))
        if [[ $SSH_FAILURES -ge $MAX_SSH_FAILURES ]]; then
            # One last chance: maybe the node started while SSH was down
            info=$(read_info)
            if [[ -n "$info" ]]; then
                break
            fi
            echo ""
            echo "ERROR: SLURM job failed or tmux session was killed."
            echo "  (SSH health-check failed ${MAX_SSH_FAILURES} times in a row)"
            echo "Check the SLURM queue with: ssh ${LOGIN_NODE} squeue -u ${EBI_USER}"
            exit 1
        fi
    else
        SSH_FAILURES=0
    fi

    elapsed_min=$((SECONDS / 60))
    elapsed_sec=$((SECONDS % 60))
    printf "\r  ${SPINNER[$((i % 10))]} Waiting... (%dm %02ds elapsed)  " "$elapsed_min" "$elapsed_sec"
    i=$((i + 1))
    sleep 3
done

printf "\r  ✔ Compute node allocated!                          \n"
echo ""

parse_info "$info"
print_connection_info
echo ""

# Start SSH tunnel in the background
echo "Setting up port forwarding (localhost:${LOCAL_PORT} -> ${COMPUTE_HOST}:${REMOTE_PORT})..."
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    "${SSH_OPTS[@]}" -L "${LOCAL_PORT}:${COMPUTE_HOST}:${REMOTE_PORT}" "$LOGIN_NODE" &
TUNNEL_PID=$!

# Clean up tunnel and SSH mux on exit
cleanup() {
    echo ""
    echo "Disconnecting port forwarding..."
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    close_ssh_mux
    echo ""
    echo "  Server keeps running on the compute node."
    echo "  Reconnect:  bash $0 --reconnect"
    echo "  View logs:  bash $0 --logs"
    echo "  Stop:       bash $0 --stop"
}
trap cleanup EXIT

# Give the tunnel a moment to establish
sleep 5

# Poll the vLLM health endpoint until the model is loaded
echo ""
echo "Waiting for vLLM to load the model (this may take a few minutes)..."
i=0
while true; do
    if curl -sf "http://localhost:${LOCAL_PORT}/health" >/dev/null 2>&1; then
        break
    fi

    # Check tunnel is alive
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo ""
        # Tunnel dropped — check if the server is still running on the compute node
        if ssh_login "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
            echo "  SSH tunnel closed, but vLLM server is still running on ${COMPUTE_HOST}."
            echo ""
            echo "  Use --reconnect to re-establish port forwarding, or --logs to view server output."
            exit 0
        else
            echo "ERROR: SSH tunnel died and server session is gone. Check your connection."
            exit 1
        fi
    fi

    elapsed_min=$((SECONDS / 60))
    elapsed_sec=$((SECONDS % 60))
    printf "\r  ${SPINNER[$((i % 10))]} Loading model... (%dm %02ds elapsed)  " "$elapsed_min" "$elapsed_sec"
    i=$((i + 1))
    sleep 5
done

printf "\r  ✔ vLLM server is READY!                             \n"
echo ""
echo "============================================================"
echo "  API base URL:  http://localhost:${LOCAL_PORT}/v1/"
echo "============================================================"
echo ""
echo "  Press Ctrl-C to disconnect (server keeps running)."
echo ""

if [[ "$DETACH" == "true" ]]; then
    trap - EXIT
    disown "$TUNNEL_PID" 2>/dev/null || true
    echo "  Running in detached mode. Tunnel PID: $TUNNEL_PID"
    exit 0
fi

# Block until user hits Ctrl-C (or tunnel dies)
wait "$TUNNEL_PID"
