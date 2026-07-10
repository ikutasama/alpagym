#!/bin/bash
# SSH tunnels: 5090 <-> A100
# Runs on the 5090. Creates two tunnels in one SSH connection:
#   1. Reverse (-R): A100 localhost:RUNTIME_PORT -> 5090 localhost:RUNTIME_PORT
#      (A100 reaches 5090's AlPaSim runtime)
#   2. Forward (-L): 5090 localhost:DRIVER_PORT -> A100 localhost:DRIVER_PORT
#      (5090 reaches A100's rollout driver in colocated mode)
#
# Usage: ./scripts/start_ssh_tunnel.sh [runtime_port] [driver_port]
#
# Requires sshpass: sudo apt install sshpass
# (Or set up key auth: ssh-copy-id -p 8040 root@10.50.121.187)

RUNTIME_PORT="${1:-5011}"
DRIVER_PORT="${2:-5012}"
A100_HOST="${A100_HOST:-10.50.121.187}"
A100_PORT="${A100_PORT:-8040}"
A100_USER="${A100_USER:-root}"
A100_PASS="${A100_PASS:-root}"

while true; do
    echo "[$(date)] SSH tunnels: runtime -R ${RUNTIME_PORT}, driver -L ${DRIVER_PORT}"
    sshpass -p "${A100_PASS}" ssh -N \
        -R "${RUNTIME_PORT}:localhost:${RUNTIME_PORT}" \
        -L "${DRIVER_PORT}:localhost:${DRIVER_PORT}" \
        -p "${A100_PORT}" \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=no \
        "${A100_USER}@${A100_HOST}"

    echo "[$(date)] SSH tunnel disconnected, reconnecting in 5s..."
    sleep 5
done
