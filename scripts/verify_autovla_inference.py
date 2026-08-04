#!/usr/bin/env python3
"""Standalone AutoVLA inference verification script.

Loads the SFT checkpoint, runs inference on a single PAI clip,
prints the full completion text (CoT + action tokens), decodes
the trajectory, and saves a visualization plot.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/verify_autovla_inference.py \
        --sft-ckpt /path/to/step=30000.ckpt \
        --pai-data /path/to/pai_dataset \
        --clip-id <clip_id> \
        --temperature 0.01 \
        --output-dir /tmp/autovla_verify

If --clip-id is omitted, picks the first available clip.
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
    return torch.tensor(data["token_all"]["veh"])  # (n_bins, 6, 4, 2)


def decode_tokens_to_trajectory(action_indices: list, code_book: torch.Tensor) -> np.ndarray:
    """Replicate AutoVLA ActionTokenizer.rollout."""
    action_tokens = code_book[action_indices]  # (T, 6, 4, 2)
    pos_a = torch.tensor([[[0.0, 0.0]]])  # [1, 1, 2]
    head_a = torch.tensor([[0.0]])  # [1, 1]

    for t in range(action_tokens.shape[0]):
        next_token_traj = action_tokens[None, t]  # [1, 6, 4, 2]
        pos_local = next_token_traj.flatten(1, 2)  # [1, 6*4, 2]
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
    return trajectory[0].numpy()  # [T+1, 3]


def plot_trajectory(traj: np.ndarray, completion_text: str, action_indices: list,
                    output_path: str, temperature: float):
    """Save trajectory visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Trajectory plot
    x, y = traj[:, 0], traj[:, 1]
    ax1.plot(x, y, "b-o", markersize=5, linewidth=2)
    ax1.plot(x[0], y[0], "go", markersize=12, label="start")
    ax1.plot(x[-1], y[-1], "rs", markersize=12, label="end")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title(f"Predicted Trajectory (temp={temperature})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    # Mark direction arrows
    for i in range(0, len(traj) - 1, 2):
        dx = traj[i + 1, 0] - traj[i, 0]
        dy = traj[i + 1, 1] - traj[i, 1]
        ax1.annotate("", xy=(traj[i, 0] + dx, traj[i, 1] + dy),
                     xytext=(traj[i, 0], traj[i, 1]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    # Completion text
    action_str = ", ".join(str(a) for a in action_indices)
    text_display = (
        f"Temperature: {temperature}\n\n"
        f"Action indices ({len(action_indices)}):\n{action_str}\n\n"
        f"Full completion text:\n{completion_text[:2000]}"
    )
    ax2.text(0.02, 0.98, text_display, transform=ax2.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             wrap=True)
    ax2.set_title("Model Output (CoT + Action Tokens)")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify AutoVLA inference")
    parser.add_argument("--sft-ckpt", required=True, help="Path to SFT checkpoint .ckpt")
    parser.add_argument("--pai-data", required=True, help="Path to PAI dataset root")
    parser.add_argument("--clip-id", default=None, help="Specific clip ID (default: first available)")
    parser.add_argument("--temperature", type=float, default=0.01, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--codebook-path", default=None,
                        help="Path to agent_vocab.pkl (default: auto-detect from AutoVLA repo)")
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen2.5-VL-3B-Instruct (default: auto-detect)")
    parser.add_argument("--output-dir", default="/tmp/autovla_verify")
    parser.add_argument("--no-history", action="store_true",
                        help="Use old prompt format without history waypoints (matches step30000 SFT checkpoint)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples at the given temperature")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect paths
    autovla_repo = os.environ.get("AUTOVLA_REPO_PATH",
                                  "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA")
    if args.codebook_path is None:
        args.codebook_path = os.path.join(autovla_repo, "codebook_cache/agent_vocab.pkl")
    if args.model_path is None:
        args.model_path = "/data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"Codebook: {args.codebook_path}")
    print(f"Model: {args.model_path}")
    print(f"SFT checkpoint: {args.sft_ckpt}")
    print(f"Temperature: {args.temperature}")

    # Load codebook
    code_book = load_codebook(args.codebook_path)
    n_bins = code_book.shape[0]
    action_start_id = 151665
    action_end_id = action_start_id + n_bins
    print(f"Codebook: {n_bins} bins, action token IDs {action_start_id}..{action_end_id - 1}")

    # Load tokenizer + model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)

    # Add action tokens
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens([f"<action_{i}>" for i in range(n_bins)], special_tokens=False)
    print(f"Added {added} action tokens to tokenizer")

    # Build action token ID → index mapping (same as inference_model.py)
    action_token_ids = []
    for i in range(n_bins):
        ids = tokenizer.encode(f"<action_{i}>", add_special_tokens=False)
        assert len(ids) == 1, f"Action token {i} mapped to {ids} (expected single token)"
        action_token_ids.append(ids[0])
    action_token_ids = torch.tensor(action_token_ids)
    tid_to_idx = {int(tid): idx for idx, tid in enumerate(action_token_ids.tolist())}
    print(f"Action token ID range: {action_token_ids.min()}..{action_token_ids.max()}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings: {len(tokenizer)} tokens")

    # Load SFT checkpoint
    print(f"Loading SFT checkpoint: {args.sft_ckpt}")
    ckpt = torch.load(args.sft_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip "vlm." prefix if present (PL checkpoint format)
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("vlm.", "", 1) if k.startswith("vlm.") else k
        cleaned[new_k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Checkpoint loaded: {len(cleaned)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    model = model.cuda().eval()

    # Load PAI data for a single clip
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages/host/src"))
    # Use the SFT dataset to get a single sample
    sys.path.insert(0, "/data/mnt_m62/10_personal/z59900495/workspace/autovla-sft-pai")
    from pai_sft_dataset import SimplePAIInterface, PAISFTDataset

    # Build a minimal config for the dataset
    data_config = {
        "pai_data_dir": args.pai_data,
        "anchor_time_s": 2.0,
        "frame_interval_s": 0.5,
        "num_context_frames": 4,
        "validation_fraction": 0.01,
        "split_seed": 42,
        "max_retries": 20,
    }
    model_config = {
        "use_cot": False,
        "codebook_cache_path": args.codebook_path,
        "trajectory": {"num_poses": 10, "interval_length": 0.5, "time_horizon": 5.0},
        "video": {"min_pixels": 109760, "max_pixels": 109760},
        "codebook_vehicle_width": 2.0,
        "codebook_vehicle_length": 4.8,
    }

    ds_interface = SimplePAIInterface(args.pai_data)
    dataset = PAISFTDataset(ds_interface, data_config, model_config, processor, split="train", max_samples=100)

    # Pick clip
    if args.clip_id:
        clip_ids = [args.clip_id]
    else:
        clip_ids = dataset.clip_ids[:1]
    print(f"Using clip: {clip_ids[0]}")

    # Build sample
    sample = dataset._build_sample(clip_ids[0])

    # If --no-history, rebuild text without history waypoints (old format).
    if args.no_history:
        # Replace the history trajectory line in text with just velocity/acceleration.
        # The old format prompt doesn't have "recent trajectory" line.
        import re
        text_old = sample["text"]
        # Remove the history trajectory sentence
        text = re.sub(
            r"The recent trajectory of the ego vehicle.*?intervals is: \[.*?\]\. ",
            "",
            text_old,
            flags=re.DOTALL,
        )
        print("Using OLD prompt format (no history waypoints)")
    else:
        text = sample["text"]

    # Print the prompt text (user content)
    print("\n" + "=" * 80)
    print("PROMPT TEXT (first 3000 chars):")
    print("=" * 80)
    print(text[:3000])
    print("=" * 80)

    # Prepare model inputs
    from qwen_vl_utils import process_vision_info
    image_inputs = sample.get("image_inputs", [])
    video_inputs = sample.get("video_inputs", [])
    model_inputs = processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in model_inputs.items()}

    prompt_length = model_inputs["input_ids"].size(1)
    print(f"Prompt length: {prompt_length} tokens")

    # Run inference N times at the given temperature
    all_trajectories = []
    all_action_indices = []
    all_completion_texts = []

    for sample_idx in range(args.num_samples):
        torch.manual_seed(42 + sample_idx)
        with torch.no_grad():
            gen_kwargs = {
                "do_sample": True,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k if args.top_k > 0 else 0,
                "top_p": args.top_p,
            }
            prompt_completion_ids = model.generate(**model_inputs, **gen_kwargs)

        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_text = processor.decode(completion_ids[0], skip_special_tokens=False)

        # Extract action tokens
        mask = (completion_ids[0] >= action_start_id) & (completion_ids[0] < action_end_id)
        action_token_ids_raw = completion_ids[0][mask]

        action_indices = []
        for tid in action_token_ids_raw:
            idx = tid_to_idx.get(int(tid.item()), 0)
            action_indices.append(idx)

        # Pad to 10 if needed
        while len(action_indices) < 10:
            action_indices.append(0)
        action_indices = action_indices[:10]

        # Decode trajectory
        traj = decode_tokens_to_trajectory(action_indices, code_book)
        traj = traj[1:]  # skip origin

        all_trajectories.append(traj)
        all_action_indices.append(action_indices)
        all_completion_texts.append(completion_text)

        print(f"\n--- Sample {sample_idx + 1}/{args.num_samples} (temp={args.temperature}) ---")
        print(f"Completion length: {completion_ids.shape[1]} tokens")
        print(f"Action tokens: {len(action_token_ids_raw)} (indices: {action_indices})")
        print(f"Trajectory first3: {traj[:3].tolist()}")
        print(f"Trajectory last3: {traj[-3:].tolist()}")
        print(f"Completion text:\n{completion_text[:500]}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Trajectory plot — overlay all samples
    colors = ["blue", "red", "green", "orange", "purple"]
    for i, traj in enumerate(all_trajectories):
        x, y = traj[:, 0], traj[:, 1]
        ax1.plot(x, y, f"{colors[i % len(colors)]}-o", markersize=5, linewidth=2,
                 label=f"sample {i+1}", alpha=0.7)
        ax1.plot(x[0], y[0], "go", markersize=10)
        ax1.plot(x[-1], y[-1], "rs", markersize=10)

    # GT trajectory
    gt_traj = sample["gt_trajectory"].numpy()
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", linewidth=2, label="GT", alpha=0.5)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title(f"Trajectories (temp={args.temperature}, {args.num_samples} samples)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Text output
    text_display = (
        f"Temperature: {args.temperature}\n"
        f"Clip: {clip_ids[0]}\n\n"
    )
    for i, (ai, ct) in enumerate(zip(all_action_indices, all_completion_texts)):
        text_display += f"--- Sample {i+1} ---\n"
        text_display += f"Action indices: {ai}\n"
        text_display += f"Completion: {ct[:300]}...\n\n"
    ax2.text(0.02, 0.98, text_display, transform=ax2.transAxes,
             fontsize=7, verticalalignment="top", fontfamily="monospace",
             wrap=True)
    ax2.set_title("Model Output")
    ax2.axis("off")

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f"verify_temp{args.temperature}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {output_path}")


if __name__ == "__main__":
    main()
