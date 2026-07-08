#!/bin/bash
# Reverse SSH tunnel: 5090 -> A100
# Runs on the 5090. Creates a reverse tunnel so the A100 can reach the
# 5090's AlPaSim runtime gRPC port via localhost:<port>.
#
# Usage: ./scripts/start_ssh_tunnel.sh <runtime_port>
#
# The runtime port is printed by run_wizard_only.sh when the wizard starts.
#
# Requires sshpass: sudo apt install sshpass
# (Or set up key auth: ssh-copy-id -p 8040 root@10.50.121.187)

PORT="${1:-5011}"
A100_HOST="${A100_HOST:-10.50.121.187}"
A100_PORT="${A100_PORT:-8040}"
A100_USER="${A100_USER:-root}"
A100_PASS="${A100_PASS:-root}"

while true; do
    echo "[$(date)] Starting SSH tunnel: A100 localhost:${PORT} -> 5090 localhost:${PORT}"
    sshpass -p "${A100_PASS}" ssh -N \
        -R "${PORT}:localhost:${PORT}" \
        -p "${A100_PORT}" \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=no \
        "${A100_USER}@${A100_HOST}"

    echo "[$(date)] SSH tunnel disconnected, reconnecting in 5s..."
    sleep 5
done
