"""Generate expert trajectories using Alpamayo-1.5 on AlPaGym rollout data.

Loads camera frames + ego history from EpisodeOutput .pt files,
feeds them to Alpamayo-1.5 expert model, and saves the expert
trajectories alongside the original observations for DAgger SFT.

Usage:
    CUDA_VISIBLE_DEVICES=2 python expert_generate.py \
        --artifacts-dir .../artifacts \
        --model-path .../Alpamayo-1.5-10B \
        --output-dir .../expert_trajectories
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from trajectory_encoder import (
    load_codebook,
    expert_traj_to_autovla_format,
    encode_trajectory_to_tokens,
    decode_tokens_to_trajectory,
)


def load_alpamayo_model(model_path: str, device: str = "cuda"):
    """Load Alpamayo-1.5 model and processor."""
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper

    print(f"Loading Alpamayo-1.5 from {model_path}...")
    model = Alpamayo1_5.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, processor


def build_expert_inputs(
    camera_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    processor,
    device: str = "cuda",
):
    """Build Alpamayo-1.5 inputs from AlPaGym observation format.

    Args:
        camera_frames: (N_cams * N_ctx, 3, H, W) uint8
        camera_indices: (N_cams,) int64
        ego_history_xyz: (1, T_hist, 3) float32
        ego_history_rot: (1, T_hist, 3, 3) float32
        processor: Alpamayo processor
        device: target device
    """
    from alpamayo1_5 import helper

    # Convert camera frames to PIL images for the processor
    n_total = camera_frames.shape[0]
    unique_cams = torch.unique(camera_indices)
    n_cams = unique_cams.shape[0]
    frames_per_cam = n_total // n_cams

    # Build frames tensor (N_total, C, H, W) as PIL images
    pil_frames = []
    for i in range(n_total):
        frame = camera_frames[i]
        if frame.ndim == 3 and frame.shape[0] in (3, 4):
            arr = frame.numpy().transpose(1, 2, 0)
        else:
            arr = frame.numpy()
        pil_frames.append(Image.fromarray(arr.astype(np.uint8)))

    # Create messages using helper
    messages = helper.create_message(
        frames=camera_frames,
        camera_indices=unique_cams,
        num_frames_per_camera=frames_per_cam,
    )

    # Process with chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Build model inputs
    # Add n_traj_group dimension: (1, T, 3) -> (1, 1, T, 3)
    if ego_history_xyz.dim() == 3:
        ego_history_xyz = ego_history_xyz.unsqueeze(1)
    if ego_history_rot.dim() == 4:
        ego_history_rot = ego_history_rot.unsqueeze(1)

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }

    return helper.to_device(model_inputs, device)


def generate_expert_trajectory(
    model,
    model_inputs: dict,
    device: str = "cuda",
    temperature: float = 0.6,
    top_p: float = 0.98,
):
    """Run Alpamayo-1.5 inference to get expert trajectory.

    Returns:
        pred_xyz: (num_traj_samples, T_future, 3) — expert positions in ego frame
        pred_rot: (num_traj_samples, T_future, 3, 3) — expert rotations
    """
    torch.cuda.manual_seed_all(42)
    with torch.autocast(device, dtype=torch.bfloat16):
        with torch.inference_mode():
            pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=1,
                max_generation_length=256,
            )

    # pred_xyz shape: (B, n_sets, n_samples, T_future, 3)
    # pred_rot shape: (B, n_sets, n_samples, T_future, 3, 3)
    # Take first batch, first set, first sample
    pred_xyz = pred_xyz[0, 0, 0]  # (T_future, 3)
    pred_rot = pred_rot[0, 0, 0]  # (T_future, 3, 3)

    return pred_xyz.cpu(), pred_rot.cpu()


def process_episode(
    pt_path: str,
    model,
    processor,
    codebook: torch.Tensor,
    action_start_id: int,
    device: str,
    max_steps: int = 22,
) -> list[dict]:
    """Process one EpisodeOutput .pt file, generating expert trajectories for each step."""
    ep = torch.load(pt_path, map_location="cpu", weights_only=False)
    samples = []

    for step_idx, po in enumerate(ep.policy_outputs):
        if step_idx >= max_steps:
            break
        if po.replay_data is None:
            continue
        payload = po.replay_data.payload
        if "model_input" not in payload:
            continue

        mi = payload["model_input"]
        if isinstance(mi, dict):
            camera_frames = mi["camera_frames"]
            camera_indices = mi["camera_indices"]
            ego_history_xyz = mi["ego_history_xyz"]
            ego_history_rot = mi["ego_history_rot"]
            route_xy = mi["route_xy"]
        else:
            camera_frames = mi.camera_frames
            camera_indices = mi.camera_indices
            ego_history_xyz = mi.ego_history_xyz
            ego_history_rot = mi.ego_history_rot
            route_xy = mi.route_xy

        if not isinstance(camera_frames, torch.Tensor):
            camera_frames = torch.as_tensor(camera_frames)
        if not isinstance(camera_indices, torch.Tensor):
            camera_indices = torch.as_tensor(camera_indices)
        if not isinstance(ego_history_xyz, torch.Tensor):
            ego_history_xyz = torch.as_tensor(ego_history_xyz)
        if not isinstance(ego_history_rot, torch.Tensor):
            ego_history_rot = torch.as_tensor(ego_history_rot)

        if camera_frames.shape[0] < 12:
            continue

        try:
            model_inputs = build_expert_inputs(
                camera_frames, camera_indices,
                ego_history_xyz, ego_history_rot,
                processor, device,
            )
            expert_xyz, expert_rot = generate_expert_trajectory(
                model, model_inputs, device,
            )
        except Exception as e:
            import traceback
            print(f"    Step {step_idx}: inference failed: {e}")
            traceback.print_exc()
            continue

        # Convert expert trajectory to AutoVLA format
        # Expert: 64 waypoints @ 0.1s (10Hz), 3D + rotation
        # AutoVLA: 10 poses @ 0.5s, (x, y, heading)
        expert_xyz_np = expert_xyz.numpy()
        expert_rot_np = expert_rot.numpy()

        target_traj = expert_traj_to_autovla_format(
            expert_xyz_np, expert_rot_np,
            expert_dt=0.1, autovla_dt=0.5, num_poses=10,
        )

        # Encode to AutoVLA action tokens
        action_indices = encode_trajectory_to_tokens(codebook, target_traj, num_poses=10)

        # Extract camera frames as PIL images for SFT
        pil_images = []
        for i in range(12):
            frame = camera_frames[i]
            if frame.ndim == 3 and frame.shape[0] in (3, 4):
                arr = frame.numpy().transpose(1, 2, 0)
            else:
                arr = frame.numpy()
            pil_images.append(Image.fromarray(arr.astype(np.uint8)))

        # Extract ego history for prompt
        ego_hist = ego_history_xyz.squeeze(0) if ego_history_xyz.dim() > 2 else ego_history_xyz
        if ego_hist.shape[0] >= 4:
            hist = ego_hist[-4:, :2]
        else:
            hist = ego_hist[:, :2]
        if hist.shape[0] < 4:
            pad = torch.zeros(4 - hist.shape[0], 2)
            hist = torch.cat([pad, hist], dim=0)

        # Velocity/acceleration
        if ego_hist.shape[0] >= 2:
            diff = ego_hist[1:] - ego_hist[:-1]
            velocity = float(torch.norm(diff[-1][:2]).item()) / 0.5
            if diff.shape[0] >= 2:
                acceleration = float(torch.norm(diff[-1][:2] - diff[-2][:2]).item()) / 0.25
            else:
                acceleration = 0.0
        else:
            velocity = 0.0
            acceleration = 0.0

        # Instruction from route
        if route_xy.shape[0] > 0:
            first_wp = route_xy[0]
            if abs(float(first_wp[0])) > abs(float(first_wp[1])):
                instruction = "turn left" if float(first_wp[0]) < 0 else "turn right"
            else:
                instruction = "move forward"
        else:
            instruction = "move forward"

        history_xy = [[f"{float(hist[i, 0]):7.2f}", f"{float(hist[i, 1]):7.2f}"]
                      for i in range(4)]

        # Decode for verification
        decoded = decode_tokens_to_trajectory(codebook, action_indices)
        decoded_xy = decoded[1:, :2].numpy()

        sample = {
            "pil_images": pil_images,
            "history_xy": history_xy,
            "velocity": velocity,
            "acceleration": acceleration,
            "instruction": instruction,
            "action_indices": action_indices.tolist(),
            "gt_xy": decoded_xy,
            "expert_xyz": expert_xyz_np,
            "step_idx": step_idx,
            "scene_id": ep.scene_id,
            "source": "alpamayo_expert",
        }
        samples.append(sample)

        del model_inputs
        torch.cuda.empty_cache()

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate expert trajectories with Alpamayo-1.5")
    parser.add_argument(
        "--artifacts-dir",
        default="/data/mnt_m62/10_personal/z59900495/workspace/latest/artifacts",
    )
    parser.add_argument(
        "--model-path",
        default="/data/mnt_m62/10_personal/z59900495/workspace/model/Alpamayo-1.5-10B",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/mnt_m62/10_personal/z59900495/workspace/expert_trajectories",
    )
    parser.add_argument(
        "--codebook-path",
        default="/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA/codebook_cache/agent_vocab.pkl",
    )
    parser.add_argument("--action-start-id", type=int, default=151665)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=22)
    args = parser.parse_args()

    codebook = load_codebook(args.codebook_path)
    print(f"Codebook loaded: {codebook.shape}")

    model, processor = load_alpamayo_model(args.model_path)

    artifacts_dir = Path(args.artifacts_dir)
    pt_files = sorted(artifacts_dir.glob("*.pt"))
    if args.max_episodes:
        pt_files = pt_files[:args.max_episodes]
    print(f"Found {len(pt_files)} episode .pt files")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    total_samples = 0
    t_start = time.time()

    for ep_idx, pt_path in enumerate(pt_files):
        try:
            samples = process_episode(
                str(pt_path), model, processor, codebook,
                args.action_start_id, "cuda",
                max_steps=args.max_steps_per_episode,
            )
        except Exception as e:
            print(f"  Episode {ep_idx} ({pt_path.name}): ERROR: {e}")
            continue

        for sample in samples:
            save_path = samples_dir / f"expert_sample_{total_samples:06d}.pt"
            torch.save(sample, save_path)
            total_samples += 1

        elapsed = time.time() - t_start
        print(f"  Episode {ep_idx+1}/{len(pt_files)}: {len(samples)} samples, "
              f"total={total_samples}, {elapsed:.0f}s")

    # Save metadata
    torch.save({
        "num_samples": total_samples,
        "action_start_id": args.action_start_id,
        "model_path": args.model_path,
        "source": "alpamayo_expert",
    }, output_dir / "metadata.pt")

    print(f"\nDone! {total_samples} expert samples saved to {samples_dir}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
