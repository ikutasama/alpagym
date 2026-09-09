"""Memory-efficient DAgger SFT training for AutoVLA.

Uses plain PyTorch (no Lightning) and loads model directly on GPU
to minimize RAM usage. Designed for systems with limited free RAM.

Usage:
    CUDA_VISIBLE_DEVICES=7 python dagger_sft_simple.py --data-dir ... --steps 500
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
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))


class DaggerDataset(Dataset):
    """Loads DAgger SFT samples and builds Qwen chat messages."""

    def __init__(self, samples_dir: str, processor, min_px: int, max_px: int, split: str = "train"):
        self.samples_dir = Path(samples_dir)
        self.processor = processor
        self.min_px = min_px
        self.max_px = max_px

        all_paths = sorted(
            list(self.samples_dir.glob("sample_*.pt")) +
            list(self.samples_dir.glob("expert_sample_*.pt"))
        )
        n_val = max(1, len(all_paths) // 10)
        self.paths = all_paths[n_val:] if split == "train" else all_paths[:n_val]
        print(f"  {split}: {len(self.paths)} samples")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info

        sample = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        pil_images = sample["pil_images"]
        action_indices = sample["action_indices"]
        action_text = "".join(f"<action_{i}>" for i in action_indices)

        fpc = len(pil_images) // 3
        sr = 2.0

        messages = [
            {"role": "system", "content": [{"type": "text", "text": (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle's "
                "surrounding cameras, along with the vehicle's current dynamic states. "
                "Your task is to predict the most appropriate driving action for the "
                "next five seconds."
            )}]},
            {"role": "user", "content": [
                {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
                {"type": "text", "text": f"The first video presents the front view of the vehicle, comprising {fpc} sequential frames sampled at {sr:g} Hz."},
                {"type": "video", "min_pixels": self.min_px, "max_pixels": self.max_px, "sample_fps": sr, "video": pil_images[:fpc]},
                {"type": "text", "text": f"The second video presents the front-left view of the vehicle, comprising {fpc} sequential frames sampled at {sr:g} Hz."},
                {"type": "video", "min_pixels": self.min_px, "max_pixels": self.max_px, "sample_fps": sr, "video": pil_images[fpc:2*fpc]},
                {"type": "text", "text": f"The third video presents the front-right view of the vehicle, comprising {fpc} sequential frames sampled at {sr:g} Hz."},
                {"type": "video", "min_pixels": self.min_px, "max_pixels": self.max_px, "sample_fps": sr, "video": pil_images[2*fpc:]},
                {"type": "text", "text": (
                    f"The recent trajectory of the ego vehicle (x, y) in ego frame "
                    f"over the past 2 seconds at 0.5s intervals is: {sample['history_xy']}. "
                    f"The current velocity of the vehicle is {sample['velocity']:.3f} m/s, "
                    f"and the current acceleration is {sample['acceleration']:.3f} m/s^2. "
                    f"The driving instruction is: {sample['instruction']}. "
                    f"Based on this information, plan the action trajectory for the "
                    f"autonomous vehicle over the next five seconds."
                )},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": f"<answer>\nThe final output action is: {action_text}\n</answer>"}]},
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_vision_id=True)

        return {"text": text, "image_inputs": image_inputs, "video_inputs": video_inputs,
                "gt_action": torch.tensor(action_indices, dtype=torch.int64)}


def collate_fn(features, processor, ignore_index, action_start_id, n_action_tokens):
    texts = [f["text"] for f in features]
    video_inputs, image_inputs = [], []
    for f in features:
        if f.get("video_inputs"):
            video_inputs.extend(f["video_inputs"])
        if f.get("image_inputs"):
            image_inputs.extend(f["image_inputs"])

    batch = processor(text=texts, images=image_inputs or None, videos=video_inputs, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = ignore_index

    assistant_marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    for row in range(labels.shape[0]):
        seq = labels[row]
        pat = torch.as_tensor(assistant_marker, dtype=seq.dtype)
        for start in range(len(seq) - len(pat) + 1):
            if torch.equal(seq[start:start + len(pat)], pat):
                labels[row, :start + len(pat)] = ignore_index
                break

    batch["labels"] = labels
    return batch


def load_sft_weights(model, ckpt_path, device="cuda"):
    """Load SFT checkpoint weights into model (memory-efficient, direct on GPU).

    The SFT checkpoint has keys like 'autovla.vlm.model.visual...' but the
    raw Qwen2_5_VLForConditionalGeneration model has keys like 'model.visual...'.
    We strip the 'autovla.' and 'vlm.' prefixes to match.
    """
    print(f"Loading SFT checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt.get("state_dict", ckpt)

    model_sd = model.state_dict()
    loaded = 0
    for src_key, value in sd.items():
        k = src_key
        for p in ("_forward_module.", "module.", "autovla.", "drivevla."):
            if k.startswith(p):
                k = k[len(p):]
        if k.startswith("vlm."):
            k = k[len("vlm."):]

        if k in model_sd and model_sd[k].shape == value.shape:
            model_sd[k].copy_(value)
            loaded += 1
            continue

        candidates = []
        if k.startswith("model.visual."):
            candidates.append("visual." + k[len("model.visual."):])
        elif k.startswith("visual."):
            candidates.append("model.visual." + k[len("visual."):])
        if k.startswith("model.language_model."):
            candidates.append("model." + k[len("model.language_model."):])
        elif k.startswith("model.") and not k.startswith("model.visual.") and not k.startswith("model.language_model."):
            candidates.append("model.language_model." + k[len("model."):])

        for cand in candidates:
            if cand in model_sd and model_sd[cand].shape == value.shape:
                model_sd[cand].copy_(value)
                loaded += 1
                break

    print(f"Loaded {loaded}/{len(model_sd)} tensors")
    del ckpt, sd
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", default="/data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sft-ckpt", default="/tmp/model/AutoVLA/autovla_sft_warmup_step5000.ckpt")
    parser.add_argument("--codebook", default="/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA/codebook_cache/agent_vocab.pkl")
    parser.add_argument("--output-dir", default="/data/mnt_m62/10_personal/z59900495/workspace/dagger_sft_output")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup processor + action tokens
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    with open(args.codebook, "rb") as f:
        cb = pickle.load(f)
    n_bins = len(cb["token_all"]["veh"])
    processor.tokenizer.add_tokens([f"<action_{i}>" for i in range(n_bins)], special_tokens=False)
    action_start = 151665
    action_end = action_start + n_bins

    min_px = max_px = 28 * 28 * 128

    # Dataset
    samples_dir = Path(args.data_dir) / "samples"
    train_ds = DaggerDataset(str(samples_dir), processor, min_px, max_px, "train")
    val_ds = DaggerDataset(str(samples_dir), processor, min_px, max_px, "val")

    collator = lambda feats: collate_fn(feats, processor, -100, action_start, n_bins)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    # Model — load directly on GPU to minimize CPU RAM
    print("Loading Qwen2.5-VL-3B on GPU...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0",
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Freeze vision backbone
    vision = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
    if vision is not None:
        vision.requires_grad_(False)
        print("Vision backbone frozen")

    # Load SFT checkpoint directly on GPU
    load_sft_weights(model, args.sft_ckpt, device="cuda:0")
    print(f"Model on GPU. Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_scale(step):
        if step < args.warmup:
            return 0.05 + 0.95 * step / args.warmup
        import math
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)

    # Training loop
    step = 0
    model.train()
    train_iter = iter(train_loader)
    t_start = time.time()

    print(f"\nStarting training: {args.steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        accepted = ("input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw",
                    "pixel_values", "image_grid_thw", "labels")
        outputs = model(**{k: batch[k] for k in accepted if k in batch})
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        # Action token accuracy
        with torch.no_grad():
            shift_labels = batch["labels"][:, 1:]
            action_mask = (shift_labels >= action_start) & (shift_labels < action_end)
            if action_mask.any():
                preds = outputs.logits[:, :-1].argmax(dim=-1)
                acc = (preds[action_mask] == shift_labels[action_mask]).float().mean().item()
            else:
                acc = 0.0

        step += 1
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"  step {step}/{args.steps}  loss={loss.item():.4f}  acc={acc:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        if step % args.save_every == 0:
            ckpt_path = output_dir / f"dagger_sft_step{step}.ckpt"
            # Save with autovla.vlm. prefix to match RL pipeline format
            sd = {f"autovla.vlm.{k}": v.cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": sd}, ckpt_path)
            del sd
            print(f"  Saved checkpoint: {ckpt_path}")

        del batch, outputs, loss
        torch.cuda.empty_cache()

    # Final checkpoint
    final_path = output_dir / "dagger_sft_final.ckpt"
    sd = {f"autovla.vlm.{k}": v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd}, final_path)
    del sd
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    # Validation
    model.eval()
    val_losses = []
    val_accs = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            accepted = ("input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw",
                        "pixel_values", "image_grid_thw", "labels")
            outputs = model(**{k: batch[k] for k in accepted if k in batch})
            val_losses.append(outputs.loss.item())
            shift_labels = batch["labels"][:, 1:]
            action_mask = (shift_labels >= action_start) & (shift_labels < action_end)
            if action_mask.any():
                preds = outputs.logits[:, :-1].argmax(dim=-1)
                acc = (preds[action_mask] == shift_labels[action_mask]).float().mean().item()
                val_accs.append(acc)

    print(f"Val loss: {np.mean(val_losses):.4f}  Val acc: {np.mean(val_accs) if val_accs else 0:.3f}")


if __name__ == "__main__":
    main()
