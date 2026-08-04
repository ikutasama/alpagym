#!/usr/bin/env python3
"""AutoVLA closed-loop trajectory animation.

Simulates receding-horizon rollout: every 0.5s, re-generate 5s trajectory
from the current camera frame + ego state, execute only the first 0.5s,
then re-generate. Visualizes whether the model produces laterally stable
trajectories or oscillating (left-right jitter) trajectories.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/verify_autovla_closed_loop.py \
        --sft-ckpt /path/to/step=12000-loss=1.1283.ckpt \
        --pai-data /path/to/pai_dataset \
        --clip-id <clip_id> \
        --temperature 0.01 \
        --num-steps 20 \
        --output /tmp/autovla_closed_loop.gif

If --clip-id omitted, picks first available clip.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ── Helpers ──────────────────────────────────────────────────────────────

def load_codebook(codebook_path: str) -> torch.Tensor:
    with open(codebook_path, "rb") as f:
        data = pickle.load(f)
    return torch.tensor(data["token_all"]["veh"])


def decode_tokens_to_trajectory(action_indices: list, code_book: torch.Tensor) -> np.ndarray:
    action_tokens = code_book[action_indices]
    pos_a = torch.tensor([[[0.0, 0.0]]])
    head_a = torch.tensor([[0.0]])
    for t in range(action_tokens.shape[0]):
        next_token_traj = action_tokens[None, t]
        pos_local = next_token_traj.flatten(1, 2)
        pos_now = pos_a[:, t]
        head_now = head_a[:, t]
        cos, sin = head_now.cos(), head_now.sin()
        rot_mat = torch.zeros((1, 2, 2))
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        pos_global = torch.bmm(pos_local, rot_mat) + pos_now.unsqueeze(1)
        pos_global = pos_global.view(*next_token_traj.shape)
        pos_a_next = pos_global[:, -1].mean(dim=1)
        diff_xy = pos_global[:, -1, 0] - pos_global[:, -1, 3]
        head_a_next = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
        pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
        head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
    trajectory = torch.cat([pos_a, head_a.unsqueeze(-1)], dim=-1)
    return trajectory[0].numpy()


def main():
    parser = argparse.ArgumentParser(description="AutoVLA closed-loop trajectory animation")
    parser.add_argument("--sft-ckpt", required=True)
    parser.add_argument("--pai-data", required=True)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of receding-horizon steps (each 0.5s)")
    parser.add_argument("--codebook-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="/tmp/autovla_closed_loop.gif")
    parser.add_argument("--use-history", action="store_true", default=True,
                        help="Include history waypoints in prompt (for new SFT checkpoints)")
    parser.add_argument("--no-history", action="store_true",
                        help="Use old prompt format without history waypoints")
    args = parser.parse_args()

    use_history = args.use_history and not args.no_history

    autovla_repo = os.environ.get("AUTOVLA_REPO_PATH",
                                  "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA")
    if args.codebook_path is None:
        args.codebook_path = os.path.join(autovla_repo, "codebook_cache/agent_vocab.pkl")
    if args.model_path is None:
        args.model_path = "/data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"SFT checkpoint: {args.sft_ckpt}")
    print(f"Temperature: {args.temperature}")
    print(f"History waypoints: {use_history}")
    print(f"Steps: {args.num_steps} (each 0.5s = {args.num_steps * 0.5}s total)")

    # Load codebook
    code_book = load_codebook(args.codebook_path)
    n_bins = code_book.shape[0]
    action_start_id = 151665
    action_end_id = action_start_id + n_bins

    # Load tokenizer + model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    tokenizer = processor.tokenizer
    tokenizer.add_tokens([f"<action_{i}>" for i in range(n_bins)], special_tokens=False)

    action_token_ids = []
    for i in range(n_bins):
        ids = tokenizer.encode(f"<action_{i}>", add_special_tokens=False)
        assert len(ids) == 1
        action_token_ids.append(ids[0])
    action_token_ids = torch.tensor(action_token_ids)
    tid_to_idx = {int(tid): idx for idx, tid in enumerate(action_token_ids.tolist())}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.resize_token_embeddings(len(tokenizer))

    # Load SFT checkpoint
    print(f"Loading SFT checkpoint: {args.sft_ckpt}")
    ckpt = torch.load(args.sft_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("vlm.", "", 1) if k.startswith("vlm.") else k
        cleaned[new_k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded: {len(cleaned)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.cuda().eval()

    # Load PAI data
    sys.path.insert(0, "/data/mnt_m62/10_personal/z59900495/workspace/autovla-sft-pai")
    from pai_sft_dataset import SimplePAIInterface, PAISFTDataset

    data_config = {
        "pai_data_dir": args.pai_data,
        "anchor_time_s": 2.0, "frame_interval_s": 0.5,
        "num_context_frames": 4, "validation_fraction": 0.01,
        "split_seed": 42, "max_retries": 20,
    }
    model_config = {
        "use_cot": False,
        "codebook_cache_path": args.codebook_path,
        "trajectory": {"num_poses": 10, "interval_length": 0.5, "time_horizon": 5.0},
        "video": {"min_pixels": 109760, "max_pixels": 109760},
        "codebook_vehicle_width": 2.0, "codebook_vehicle_length": 4.8,
    }

    ds_interface = SimplePAIInterface(args.pai_data)
    dataset = PAISFTDataset(ds_interface, data_config, model_config, processor,
                            split="train", max_samples=100)

    if args.clip_id:
        clip_ids = [args.clip_id]
    else:
        clip_ids = dataset.clip_ids[:1]
    print(f"Using clip: {clip_ids[0]}")

    # ── Closed-loop simulation ──────────────────────────────────────────
    # We simulate receding horizon: at each step, the model sees the SAME
    # camera frames (from the PAI clip at anchor_time=2.0s) but we vary the
    # ego history to simulate the car having moved. We track the executed
    # trajectory and show how each prediction evolves.

    # Get the base sample (cameras + ego at anchor time)
    cameras = dataset._extract_camera_frames(clip_ids[0])
    ego = dataset._extract_ego_motion(clip_ids[0])

    # GT trajectory for reference
    gt_traj = ego["gt_xy"]  # [10, 2]

    # Simulated ego state (starts at origin, heading=0)
    sim_pos = np.array([0.0, 0.0])
    sim_heading = 0.0
    sim_history = []  # past positions (relative to current)

    # Store all predictions for animation
    all_predictions = []  # list of [10, 2] arrays (each prediction in global frame)
    executed_trajectory = []  # list of [x, y] points actually driven

    from qwen_vl_utils import process_vision_info

    system_text = (
        "You are an Advanced Driver Assistance and Full Self-Driving System. "
        "You will be provided with video observations from the ego vehicle's "
        "surrounding cameras, along with the vehicle's current dynamic states. "
        "Your task is to predict the most appropriate driving action for the "
        "next five seconds."
    )

    sample_rate_hz = 1.0 / dataset.frame_interval_s
    video_desc = (
        f"comprising {dataset.num_context_frames} sequential frames sampled at "
        f"{sample_rate_hz:g} Hz."
    )
    min_px = model_config["video"]["min_pixels"]
    max_px = model_config["video"]["max_pixels"]

    velocity = ego["velocity"]
    acceleration = ego["acceleration"]

    for step in range(args.num_steps):
        # Build history waypoints (relative to current sim position)
        if use_history:
            if len(sim_history) >= 4:
                hist = sim_history[-4:]
            elif len(sim_history) >= 1:
                hist = sim_history
            else:
                hist = [[0.0, 0.0]] * 4
            # Convert to ego frame (relative to current pos+heading)
            hist_ego = []
            for h in hist:
                dx = h[0] - sim_pos[0]
                dy = h[1] - sim_pos[1]
                cos_h, sin_h = np.cos(-sim_heading), np.sin(-sim_heading)
                hx = cos_h * dx - sin_h * dy
                hy = sin_h * dx + cos_h * dy
                hist_ego.append([float(hx), float(hy)])
            velocity_text = (
                f"The recent trajectory of the ego vehicle (x, y) in ego frame "
                f"over the past 2 seconds at 0.5s intervals is: {hist_ego}. "
                f"The current velocity of the vehicle is {velocity:.3f} m/s, "
                f"and the current acceleration is {acceleration:.3f} m/s². "
                "No route or navigation command is available for this clip. Based on "
                "the observations and current dynamics, plan a safe action trajectory "
                "for the autonomous vehicle over the next five seconds."
            )
        else:
            velocity_text = (
                f"The current velocity of the vehicle is {velocity:.3f} m/s, "
                f"and the current acceleration is {acceleration:.3f} m/s². "
                "No route or navigation command is available for this clip. Based on "
                "the observations and current dynamics, plan a safe action trajectory "
                "for the autonomous vehicle over the next five seconds."
            )

        user_content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
            {"type": "text", "text": f"The first video presents the front view of the vehicle, {video_desc}"},
            {"type": "video", "min_pixels": min_px, "max_pixels": max_px, "sample_fps": sample_rate_hz, "video": cameras["front_camera"]},
            {"type": "text", "text": f"The second video presents the front-left view of the vehicle, {video_desc}"},
            {"type": "video", "min_pixels": min_px, "max_pixels": max_px, "sample_fps": sample_rate_hz, "video": cameras["front_left_camera"]},
            {"type": "text", "text": f"The third video presents the front-right view of the vehicle, {video_desc}"},
            {"type": "video", "min_pixels": min_px, "max_pixels": max_px, "sample_fps": sample_rate_hz, "video": cameras["front_right_camera"]},
            {"type": "text", "text": velocity_text},
        ]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(text=[text], images=image_inputs or None,
                                 videos=video_inputs, padding=True, return_tensors="pt")
        model_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                        for k, v in model_inputs.items()}

        torch.manual_seed(42 + step)
        with torch.no_grad():
            gen_kwargs = {
                "do_sample": True,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k if args.top_k > 0 else 0,
                "top_p": args.top_p,
            }
            prompt_completion_ids = model.generate(**model_inputs, **gen_kwargs)

        prompt_length = model_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Extract action tokens
        mask = (completion_ids[0] >= action_start_id) & (completion_ids[0] < action_end_id)
        action_token_ids_raw = completion_ids[0][mask]
        action_indices = [tid_to_idx.get(int(tid.item()), 0) for tid in action_token_ids_raw]
        while len(action_indices) < 10:
            action_indices.append(0)
        action_indices = action_indices[:10]

        # Decode trajectory (in ego frame)
        traj_ego = decode_tokens_to_trajectory(action_indices, code_book)
        traj_ego = traj_ego[1:]  # skip origin, [10, 3]

        # Transform to global frame (from current sim position+heading)
        traj_global = np.zeros((10, 2))
        cos_h, sin_h = np.cos(sim_heading), np.sin(sim_heading)
        for j in range(10):
            x_ego, y_ego = traj_ego[j, 0], traj_ego[j, 1]
            traj_global[j, 0] = cos_h * x_ego - sin_h * y_ego + sim_pos[0]
            traj_global[j, 1] = sin_h * x_ego + cos_h * y_ego + sim_pos[1]

        all_predictions.append(traj_global)

        # Execute first 0.5s (first waypoint)
        next_pos = traj_global[0]
        # Update heading based on direction to next point
        dx = next_pos[0] - sim_pos[0]
        dy = next_pos[1] - sim_pos[1]
        if np.linalg.norm([dx, dy]) > 1e-6:
            sim_heading = float(np.arctan2(dy, dx))
        sim_pos = next_pos
        sim_history.append(sim_pos.copy())
        executed_trajectory.append(sim_pos.copy())

        print(f"Step {step+1}/{args.num_steps}: pos={sim_pos.tolist()} "
              f"heading={np.degrees(sim_heading):.1f}° "
              f"actions={action_indices[:5]}...")

    # ── Animation ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        executed = np.array(executed_trajectory)
        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame):
            ax.clear()
            # GT trajectory
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", linewidth=2, alpha=0.3, label="GT")

            # Executed trajectory so far
            if len(executed) > 0:
                ax.plot(executed[:frame+1, 0], executed[:frame+1, 1],
                        "b-o", markersize=4, linewidth=2, label="executed")

            # Current prediction (faded lines for future)
            for i in range(frame + 1):
                pred = all_predictions[i]
                alpha = 0.15 if i < frame else 0.8
                color = "green" if i == frame else "gray"
                ax.plot(pred[:, 0], pred[:, 1], "-", color=color, alpha=alpha, linewidth=1.5)
                if i == frame:
                    ax.plot(pred[:, 0], pred[:, 1], "g-o", markersize=5, linewidth=2,
                            label=f"pred step {i+1}")

            # Current position
            if frame < len(executed):
                ax.plot(executed[frame, 0], executed[frame, 1], "rs", markersize=12, zorder=5)

            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"AutoVLA Closed-Loop (temp={args.temperature}, "
                        f"step {frame+1}/{args.num_steps}, "
                        f"history={'on' if use_history else 'off'})")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
            margin = 5
            all_x = np.concatenate([p[:, 0] for p in all_predictions] + [executed[:, 0]])
            all_y = np.concatenate([p[:, 1] for p in all_predictions] + [executed[:, 1]])
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(min(all_y.min(), -margin), max(all_y.max(), margin))

        anim = animation.FuncAnimation(fig, update, frames=args.num_steps,
                                       interval=500, repeat=True)
        anim.save(args.output, writer="pillow", fps=2)
        print(f"\nAnimation saved to {args.output}")
    except ImportError:
        print("\nmatplotlib not installed — printing text summary only")
        print("\nExecuted trajectory:")
        for i, p in enumerate(executed_trajectory):
            print(f"  step {i+1}: ({p[0]:.3f}, {p[1]:.3f})")
        print("\nAll predictions (first/last waypoint):")
        for i, pred in enumerate(all_predictions):
            print(f"  step {i+1}: start=({pred[0,0]:.3f},{pred[0,1]:.3f}) "
                  f"end=({pred[-1,0]:.3f},{pred[-1,1]:.3f})")


if __name__ == "__main__":
    main()
