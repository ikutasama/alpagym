"""Encode continuous trajectories into AutoVLA discrete action tokens.

AutoVLA uses a codebook of 2048 tokens, each encoding 6×4×2 sub-waypoints.
Decoding is a sequential rollout: each token's sub-waypoints are rotated by
the current heading and translated to the current position.

Encoding is a greedy search: at each of the 10 steps, pick the token whose
resulting position is closest to the target waypoint.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
import numpy as np


def load_codebook(path: str | Path) -> torch.Tensor:
    """Load the AutoVLA action codebook.

    Returns:
        Tensor of shape (n_bins, 6, 4, 2) — sub-waypoints per token.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "token_all" in data:
        cb = data["token_all"]["veh"]
    else:
        cb = data
    return torch.tensor(cb, dtype=torch.float32)


def decode_tokens_to_trajectory(
    codebook: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """Decode codebook indices to a trajectory via sequential rollout.

    Args:
        codebook: (n_bins, 6, 4, 2)
        token_indices: (T,) codebook indices (NOT token IDs)

    Returns:
        (T+1, 3) trajectory — (x, y, heading), first point is origin (0,0,0).
    """
    T = token_indices.shape[0]
    action_tokens = codebook[token_indices]  # (T, 6, 4, 2)

    pos_a = torch.zeros(1, 1, 2)   # (1, 1, 2)
    head_a = torch.zeros(1, 1)     # (1, 1)

    for t in range(T):
        next_token_traj = action_tokens[None, t]  # (1, 6, 4, 2)
        pos_local = next_token_traj.flatten(1, 2)  # (1, 24, 2)
        pos_now = pos_a[:, t]    # (1, 2)
        head_now = head_a[:, t]  # (1,)

        cos, sin = head_now.cos(), head_now.sin()
        rot_mat = torch.zeros(1, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        pos_global = torch.bmm(pos_local, rot_mat) + pos_now.unsqueeze(1)
        pos_global = pos_global.view(*next_token_traj.shape)  # (1, 6, 4, 2)

        pos_a_next = pos_global[:, -1].mean(dim=1)  # (1, 2)
        diff_xy = pos_global[:, -1, 0] - pos_global[:, -1, 3]  # (1, 2)
        head_a_next = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])  # (1,)

        pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
        head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

    trajectory = torch.cat([pos_a, head_a.unsqueeze(-1)], dim=-1)  # (1, T+1, 3)
    return trajectory[0]  # (T+1, 3)


def _precompute_token_effects(codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute each token's local position delta and heading delta.

    For each token, the position update is:
        pos_next = pos_now + rotate(mean(last_row), heading_now)

    The heading update is:
        heading_next = heading_now + arctan2(diff_last_row[1], diff_last_row[0])

    So we precompute:
        delta_pos_local[k] = mean(codebook[k, -1], dim=0)  -- (2,)
        delta_heading[k] = arctan2(diff[1], diff[0])       -- scalar

    Returns:
        delta_pos_local: (n_bins, 2)
        delta_heading: (n_bins,)
    """
    n_bins = codebook.shape[0]
    last_rows = codebook[:, -1]  # (n_bins, 4, 2)
    delta_pos_local = last_rows.mean(dim=1)  # (n_bins, 2)

    diff = last_rows[:, 0] - last_rows[:, 3]  # (n_bins, 2)
    delta_heading = torch.arctan2(diff[:, 1], diff[:, 0])  # (n_bins,)

    return delta_pos_local, delta_heading


def encode_trajectory_to_tokens(
    codebook: torch.Tensor,
    target_traj: torch.Tensor,
    num_poses: int = 10,
) -> torch.Tensor:
    """Encode a target trajectory into AutoVLA codebook indices via greedy search.

    Args:
        codebook: (n_bins, 6, 4, 2)
        target_traj: (num_poses, 3) — (x, y, heading) in ego frame, 0.5s intervals.
                     These are the 10 predicted poses (excluding origin).
        num_poses: number of output tokens.

    Returns:
        (num_poses,) codebook indices (int64).
    """
    delta_pos_local, delta_heading = _precompute_token_effects(codebook)
    n_bins = codebook.shape[0]

    pos_now = torch.zeros(2)
    head_now = torch.tensor(0.0)
    token_indices = []

    for t in range(num_poses):
        target = target_traj[t, :2]  # (x, y)

        # For each candidate token, compute resulting position
        cos_h, sin_h = head_now.cos(), head_now.sin()
        # delta_pos_local: (n_bins, 2)
        # Rotate by heading: [cos, -sin; sin, cos] @ delta
        rotated_x = cos_h * delta_pos_local[:, 0] - sin_h * delta_pos_local[:, 1]
        rotated_y = sin_h * delta_pos_local[:, 0] + cos_h * delta_pos_local[:, 1]
        candidate_pos = pos_now.unsqueeze(0) + torch.stack([rotated_x, rotated_y], dim=1)  # (n_bins, 2)

        # Distance to target
        dists = torch.norm(candidate_pos - target.unsqueeze(0), dim=1)  # (n_bins,)
        best_k = torch.argmin(dists).item()
        token_indices.append(best_k)

        # Update state
        pos_now = candidate_pos[best_k]
        head_now = head_now + delta_heading[best_k]

    return torch.tensor(token_indices, dtype=torch.int64)


def expert_traj_to_autovla_format(
    expert_xyz: np.ndarray,
    expert_rot: np.ndarray | None = None,
    expert_dt: float = 0.1,
    autovla_dt: float = 0.5,
    num_poses: int = 10,
) -> torch.Tensor:
    """Convert expert trajectory to AutoVLA target format.

    Expert: N waypoints @ expert_dt in ego frame (x, y, z) + rotation matrices.
    AutoVLA: num_poses waypoints @ autovla_dt in ego frame (x, y, heading).

    Args:
        expert_xyz: (N, 3) — ego-frame positions from expert model.
        expert_rot: (N, 3, 3) — ego-frame rotation matrices. If None, heading=0.
        expert_dt: time between expert waypoints (0.1s for Alpamayo 10Hz).
        autovla_dt: time between AutoVLA poses (0.5s).
        num_poses: number of output poses.

    Returns:
        (num_poses, 3) — (x, y, heading) in ego frame.
    """
    N = expert_xyz.shape[0]
    step = max(1, int(round(autovla_dt / expert_dt)))

    indices = [min(i * step, N - 1) for i in range(num_poses)]
    selected_xyz = expert_xyz[indices]  # (num_poses, 3)

    result = torch.zeros(num_poses, 3, dtype=torch.float32)
    result[:, 0] = torch.tensor(selected_xyz[:, 0], dtype=torch.float32)  # x
    result[:, 1] = torch.tensor(selected_xyz[:, 1], dtype=torch.float32)  # y

    if expert_rot is not None:
        selected_rot = expert_rot[indices]  # (num_poses, 3, 3)
        # heading = arctan2(rot[1,0], rot[0,0])
        result[:, 2] = torch.arctan2(
            torch.tensor(selected_rot[:, 1, 0], dtype=torch.float32),
            torch.tensor(selected_rot[:, 0, 0], dtype=torch.float32),
        )

    return result


def codebook_index_to_token_id(
    indices: torch.Tensor,
    action_start_id: int = 151665,
) -> torch.Tensor:
    """Convert codebook indices to AutoVLA action token IDs."""
    return indices + action_start_id


def token_id_to_codebook_index(
    token_ids: torch.Tensor,
    action_start_id: int = 151665,
) -> torch.Tensor:
    """Convert AutoVLA action token IDs to codebook indices."""
    return token_ids - action_start_id


if __name__ == "__main__":
    import sys

    codebook_path = "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA/codebook_cache/agent_vocab.pkl"
    codebook = load_codebook(codebook_path)
    print(f"Codebook shape: {codebook.shape}")
    print(f"Codebook range: [{codebook.min():.4f}, {codebook.max():.4f}]")

    # Test 1: Round-trip with random tokens
    torch.manual_seed(42)
    random_indices = torch.randint(0, codebook.shape[0], (10,))
    print(f"\nRandom indices: {random_indices.tolist()}")

    decoded = decode_tokens_to_trajectory(codebook, random_indices)
    print(f"Decoded trajectory shape: {decoded.shape}")
    print(f"Decoded trajectory:\n{decoded}")

    # Encode back
    target = decoded[1:]  # (10, 3) — skip origin
    encoded = encode_trajectory_to_tokens(codebook, target)
    print(f"\nEncoded indices: {encoded.tolist()}")
    print(f"Match: {(encoded == random_indices).all().item()}")

    # Re-decode encoded
    redecoded = decode_tokens_to_trajectory(codebook, encoded)
    redecoded_target = redecoded[1:]
    error = torch.norm(redecoded_target - target).item()
    print(f"Round-trip error: {error:.6f}")

    # Test 2: Synthetic expert trajectory (straight line + slight curve)
    t_vals = torch.linspace(0, 5, 64)  # 64 waypoints, 5s
    expert_xyz = torch.zeros(64, 3)
    expert_xyz[:, 0] = t_vals * 3.0  # x: 0 to 15m
    expert_xyz[:, 1] = t_vals * 0.5  # y: slight curve
    print(f"\n--- Expert trajectory test ---")
    print(f"Expert xyz shape: {expert_xyz.shape}")
    print(f"Expert first 3: {expert_xyz[:3]}")
    print(f"Expert last 3: {expert_xyz[-3:]}")

    target_traj = expert_traj_to_autovla_format(expert_xyz.numpy(), expert_dt=0.1)
    print(f"AutoVLA target shape: {target_traj.shape}")
    print(f"AutoVLA target:\n{target_traj}")

    encoded = encode_trajectory_to_tokens(codebook, target_traj)
    print(f"Encoded tokens: {encoded.tolist()}")

    decoded = decode_tokens_to_trajectory(codebook, encoded)
    decoded_poses = decoded[1:]  # (10, 3)
    print(f"Decoded poses:\n{decoded_poses}")

    error = torch.norm(decoded_poses[:, :2] - target_traj[:, :2]).item()
    print(f"Position encoding error: {error:.6f}")

    # Test 3: Curved trajectory
    t_vals = torch.linspace(0, 5, 64)
    expert_xyz2 = torch.zeros(64, 3)
    expert_xyz2[:, 0] = torch.cos(t_vals * 0.5) * 5 - 5
    expert_xyz2[:, 1] = torch.sin(t_vals * 0.5) * 5
    print(f"\n--- Curved trajectory test ---")
    target2 = expert_traj_to_autovla_format(expert_xyz2.numpy(), expert_dt=0.1)
    encoded2 = encode_trajectory_to_tokens(codebook, target2)
    decoded2 = decode_tokens_to_trajectory(codebook, encoded2)
    error2 = torch.norm(decoded2[1:, :2] - target2[:, :2]).item()
    print(f"Encoded: {encoded2.tolist()}")
    print(f"Curved position error: {error2:.6f}")

    print("\nAll tests passed!")
