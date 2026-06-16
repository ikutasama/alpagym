"""Quick NCCL peer-to-peer (GPU-to-GPU) sanity check.

AlpaGym's policy and rollout replicas live on separate GPUs and exchange
weights over NCCL point-to-point send/recv. On some hosts the direct GPU-to-GPU
transport (NVLink or PCIe P2P) is advertised by NCCL but then stalls forever,
which shows up as a hang during the first weight sync with both GPUs pinned at
100%. This script reproduces that exact transport in a few seconds so you can
tell, before launching a full run, whether your topology works as-is or whether
you need ``NCCL_P2P_DISABLE=1``.

Run it on two GPUs:

    uv run --no-sync torchrun --nproc-per-node=2 install/check_nccl_p2p.py

If it hangs for ~10s and then errors, re-run with ``NCCL_P2P_DISABLE=1``:

    NCCL_P2P_DISABLE=1 uv run --no-sync torchrun --nproc-per-node=2 install/check_nccl_p2p.py

A pass with ``NCCL_P2P_DISABLE=1`` but a hang without it means your direct P2P
transport is broken; set ``NCCL_P2P_DISABLE=1`` for AlpaGym runs (weight sync
then falls back to slower shared-memory staging through host RAM).
"""

import datetime
import os

# On a timeout the NCCL watchdog otherwise waits up to 60s for a debug dump
# before tearing the process down. Skip it so the check aborts right after the
# 10s timeout below. Must be set before torch loads.
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "0")
os.environ.setdefault("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "1000")

import torch
import torch.distributed as dist


def main() -> None:
    """Send a tensor from rank 0 to rank 1 over NCCL and verify it arrives."""
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.device_count() < 2:
        if local_rank == 0:
            print(
                "This check needs 2 GPUs (P2P is GPU-to-GPU); found "
                f"{torch.cuda.device_count()}. AlpaGym's 10B model requires two "
                "GPUs, so a single-GPU host cannot run it yet."
            )
        return
    torch.cuda.set_device(local_rank)

    # A short timeout turns an indefinite P2P stall into a fast, visible failure
    # instead of a hang that looks identical to a healthy-but-slow transfer. The
    # NCCL watchdog aborts the process once a send/recv exceeds this, and the
    # few GB moved below complete well under it even over shared memory.
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=10))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        if rank == 0:
            print("This check expects exactly 2 ranks (--nproc-per-node=2).")
        dist.destroy_process_group()
        return

    # ~256 MB across several iterations so we exercise the real data path, not
    # just connection setup.
    tensor = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=local_rank)
    for step in range(20):
        if rank == 0:
            tensor.fill_(float(step))
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
            assert torch.allclose(tensor, torch.full_like(tensor, float(step)))
        torch.cuda.synchronize()

    dist.barrier()
    if rank == 1:
        p2p_disabled = os.environ.get("NCCL_P2P_DISABLE") == "1"
        suffix = " (with NCCL_P2P_DISABLE=1)" if p2p_disabled else ""
        print(f"NCCL P2P send/recv OK{suffix}: this topology works for AlpaGym.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
