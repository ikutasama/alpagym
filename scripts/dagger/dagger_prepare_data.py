"""Prepare DAgger SFT training data from AlPaGym rollout .pt files.

Loads EpisodeOutput .pt files, extracts per-step observations and action
tokens, converts to AutoVLA SFT format (PIL images + chat messages +
action token labels), and saves as individual .pt samples.

Filtering: keeps episodes with reward above a percentile threshold and
excludes episodes with collisions.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def load_codebook(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "token_all" in data:
        return torch.tensor(data["token_all"]["veh"], dtype=torch.float32)
    return torch.tensor(data, dtype=torch.float32)


def extract_sft_samples_from_episode(
    episode_path: str,
    codebook: torch.Tensor,
    action_start_id: int = 151665,
    num_cameras: int = 3,
    frames_per_cam: int = 4,
    min_pixels: int = 28 * 28 * 128,
    max_pixels: int = 28 * 28 * 128,
) -> list[dict]:
    """Extract SFT samples from a single EpisodeOutput .pt file.

    Each step becomes one SFT sample with:
    - camera_frames: list of PIL images (3 cams × 4 frames)
    - history_xy: [4, 2] ego history in ego frame
    - velocity, acceleration: floats
    - instruction: driving instruction string
    - action_token_ids: [10] codebook indices (from replay data)
    - gt_xy: [10, 2] decoded trajectory positions (for verification)
    """
    ep = torch.load(episode_path, map_location="cpu", weights_only=False)
    samples = []

    for step_idx, po in enumerate(ep.policy_outputs):
        if po.replay_data is None:
            continue
        payload = po.replay_data.payload
        if "model_input" not in payload or "action_token_ids" not in payload:
            continue

        mi = payload["model_input"]
        action_token_ids = payload["action_token_ids"]
        if isinstance(action_token_ids, torch.Tensor):
            raw_ids = action_token_ids.cpu()
        else:
            raw_ids = torch.tensor(action_token_ids)

        # Convert token IDs (151665+) to codebook indices (0-2047)
        action_indices = raw_ids - action_start_id
        valid_mask = (action_indices >= 0) & (action_indices < codebook.shape[0])
        if not valid_mask.all():
            continue
        if action_indices.numel() < 10:
            continue

        # Extract camera frames → PIL images
        camera_frames = mi["camera_frames"] if isinstance(mi, dict) else mi.camera_frames
        if not isinstance(camera_frames, torch.Tensor):
            camera_frames = torch.as_tensor(camera_frames)
        if camera_frames.shape[0] < num_cameras * frames_per_cam:
            continue

        pil_images = []
        for i in range(num_cameras * frames_per_cam):
            frame = camera_frames[i]
            if frame.ndim == 3 and frame.shape[0] in (3, 4):
                arr = frame.numpy().transpose(1, 2, 0)
            else:
                arr = frame.numpy()
            pil_images.append(Image.fromarray(arr.astype(np.uint8)))

        # Extract ego history → last 4 xy points in ego frame
        ego_hist = mi["ego_history_xyz"] if isinstance(mi, dict) else mi.ego_history_xyz
        if ego_hist.dim() > 2:
            ego_hist = ego_hist.squeeze(0)
        if ego_hist.shape[0] >= 4:
            hist = ego_hist[-4:, :2]
        elif ego_hist.shape[0] >= 1:
            hist = ego_hist[:, :2]
        else:
            hist = torch.zeros(1, 2)
        if hist.shape[0] < 4:
            pad = torch.zeros(4 - hist.shape[0], 2)
            hist = torch.cat([pad, hist], dim=0)

        # Compute velocity/acceleration from ego history
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

        # Build instruction from route
        route_xy = mi["route_xy"] if isinstance(mi, dict) else mi.route_xy
        if route_xy.shape[0] > 0:
            first_wp = route_xy[0]
            if abs(float(first_wp[0])) > abs(float(first_wp[1])):
                instruction = "turn left" if float(first_wp[0]) < 0 else "turn right"
            else:
                instruction = "move forward"
        else:
            instruction = "move forward"

        # Decode action tokens to trajectory for verification
        try:
            from trajectory_encoder import decode_tokens_to_trajectory
            traj = decode_tokens_to_trajectory(codebook, action_indices[:10])
            gt_xy = traj[1:, :2].numpy()
        except Exception:
            gt_xy = np.zeros((10, 2), dtype=np.float32)

        # Format history_xy as fixed-width strings (same as inference)
        history_xy = [[f"{float(hist[i, 0]):7.2f}", f"{float(hist[i, 1]):7.2f}"]
                      for i in range(4)]

        sample = {
            "pil_images": pil_images,
            "history_xy": history_xy,
            "velocity": velocity,
            "acceleration": acceleration,
            "instruction": instruction,
            "action_indices": action_indices[:10].tolist(),
            "gt_xy": gt_xy if isinstance(gt_xy, np.ndarray) else gt_xy.numpy(),
            "step_idx": step_idx,
            "scene_id": ep.scene_id,
        }
        samples.append(sample)

    return samples, ep.reward.total if ep.reward else -999.0, ep.is_valid


def build_chat_messages(
    pil_images: list,
    history_xy: list,
    velocity: float,
    acceleration: float,
    instruction: str,
    action_text: str,
    min_pixels: int = 28 * 28 * 128,
    max_pixels: int = 28 * 28 * 128,
) -> list[dict]:
    """Build chat messages in the same format as PAISFTDataset._build_sample."""
    num_images = len(pil_images)
    frames_per_cam = num_images // 3
    sample_rate_hz = 2.0

    system_content = [
        {
            "type": "text",
            "text": (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle's "
                "surrounding cameras, along with the vehicle's current dynamic states. "
                "Your task is to predict the most appropriate driving action for the "
                "next five seconds."
            ),
        }
    ]

    user_content = [
        {
            "type": "text",
            "text": (
                "The autonomous vehicle is equipped with three cameras mounted at the "
                "front, left, and right, enabling a comprehensive perception of the "
                "surrounding environment."
            ),
        },
        {
            "type": "text",
            "text": f"The first video presents the front view of the vehicle, comprising {frames_per_cam} sequential frames sampled at {sample_rate_hz:g} Hz.",
        },
        {
            "type": "video",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "sample_fps": sample_rate_hz,
            "video": pil_images[:frames_per_cam],
        },
        {
            "type": "text",
            "text": f"The second video presents the front-left view of the vehicle, comprising {frames_per_cam} sequential frames sampled at {sample_rate_hz:g} Hz.",
        },
        {
            "type": "video",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "sample_fps": sample_rate_hz,
            "video": pil_images[frames_per_cam:2*frames_per_cam],
        },
        {
            "type": "text",
            "text": f"The third video presents the front-right view of the vehicle, comprising {frames_per_cam} sequential frames sampled at {sample_rate_hz:g} Hz.",
        },
        {
            "type": "video",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "sample_fps": sample_rate_hz,
            "video": pil_images[2*frames_per_cam:],
        },
        {
            "type": "text",
            "text": (
                f"The recent trajectory of the ego vehicle (x, y) in ego frame "
                f"over the past 2 seconds at 0.5s intervals is: {history_xy}. "
                f"The current velocity of the vehicle is {velocity:.3f} m/s, "
                f"and the current acceleration is {acceleration:.3f} m/s^2. "
                f"The driving instruction is: {instruction}. "
                f"Based on this information, plan the action trajectory for the "
                f"autonomous vehicle over the next five seconds."
            ),
        },
    ]

    assistant_content = [
        {
            "type": "text",
            "text": f"<answer>\nThe final output action is: {action_text}\n</answer>",
        }
    ]

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare DAgger SFT data from rollout .pt files")
    parser.add_argument(
        "--artifacts-dir",
        default="/data/mnt_m62/10_personal/z59900495/workspace/latest/artifacts",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/mnt_m62/10_personal/z59900495/workspace/dagger_sft_data",
    )
    parser.add_argument(
        "--codebook-path",
        default="/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA/codebook_cache/agent_vocab.pkl",
    )
    parser.add_argument("--action-start-id", type=int, default=151665)
    parser.add_argument("--reward-percentile", type=float, default=50.0,
                        help="Keep episodes with reward above this percentile")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Process at most this many episodes")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum number of SFT samples to produce")
    args = parser.parse_args()

    codebook = load_codebook(args.codebook_path)
    print(f"Codebook loaded: {codebook.shape}")

    artifacts_dir = Path(args.artifacts_dir)
    pt_files = sorted(artifacts_dir.glob("*.pt"))
    if args.max_episodes:
        pt_files = pt_files[:args.max_episodes]
    print(f"Found {len(pt_files)} episode .pt files")

    # First pass: load all episodes, collect rewards
    print("Loading episodes and computing rewards...")
    episode_data = []
    for i, pt_path in enumerate(pt_files):
        try:
            ep = torch.load(pt_path, map_location="cpu", weights_only=False)
            reward = ep.reward.total if ep.reward else -999.0
            episode_data.append((pt_path, reward, ep.is_valid, ep.num_steps))
            ep = None  # free memory
        except Exception as e:
            print(f"  Skipping {pt_path.name}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(pt_files)} episodes")

    if not episode_data:
        print("ERROR: No valid episodes found!")
        sys.exit(1)

    # Filter by reward percentile
    rewards = [r for _, r, valid, _ in episode_data if valid]
    if not rewards:
        print("ERROR: No valid episodes with rewards!")
        sys.exit(1)

    threshold = np.percentile(rewards, args.reward_percentile)
    print(f"\nReward stats: min={min(rewards):.4f} max={max(rewards):.4f} "
          f"median={np.median(rewards):.4f} mean={np.mean(rewards):.4f}")
    print(f"Keeping episodes with reward >= {threshold:.4f} (percentile {args.reward_percentile})")

    selected = [(p, r, v, n) for p, r, v, n in episode_data
                if v and r >= threshold]
    print(f"Selected {len(selected)}/{len(episode_data)} episodes")

    # Second pass: extract SFT samples from selected episodes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for i, (pt_path, reward, valid, num_steps) in enumerate(selected):
        try:
            samples, ep_reward, ep_valid = extract_sft_samples_from_episode(
                str(pt_path), codebook, args.action_start_id
            )
            all_samples.extend(samples)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(selected)} episodes, "
                      f"{len(all_samples)} samples so far")
        except Exception as e:
            print(f"  Error processing {pt_path.name}: {e}")

    print(f"\nTotal SFT samples: {len(all_samples)}")

    if len(all_samples) < args.min_samples:
        print(f"WARNING: Only {len(all_samples)} samples (min: {args.min_samples})")
        if len(all_samples) == 0:
            sys.exit(1)

    # Save samples
    # We save metadata + action tokens separately from images to keep file sizes manageable
    metadata_path = output_dir / "metadata.pt"
    torch.save({
        "num_samples": len(all_samples),
        "action_start_id": args.action_start_id,
        "reward_percentile": args.reward_percentile,
        "reward_threshold": threshold,
        "source_artifacts_dir": str(artifacts_dir),
    }, metadata_path)
    print(f"Saved metadata to {metadata_path}")

    # Save each sample as individual file
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(all_samples):
        save_path = samples_dir / f"sample_{i:06d}.pt"
        torch.save(sample, save_path)

    print(f"Saved {len(all_samples)} samples to {samples_dir}")

    # Print sample stats
    action_indices = [s["action_indices"] for s in all_samples]
    all_actions = torch.tensor(action_indices)
    print(f"\nAction token stats:")
    print(f"  Unique tokens used: {len(torch.unique(all_actions))}")
    print(f"  Most common: {torch.bincount(all_actions.flatten()).topk(10).indices.tolist()}")
    print(f"  Velocity range: [{min(s['velocity'] for s in all_samples):.3f}, "
          f"{max(s['velocity'] for s in all_samples):.3f}]")
    print(f"  Instructions: {set(s['instruction'] for s in all_samples)}")


if __name__ == "__main__":
    main()
