"""DAgger SFT training for AutoVLA using collected rollout data.

Adapts the autovla-sft-pai training pipeline to use DAgger samples
collected from AlPaGym rollout .pt files instead of the PAI dataset.

Usage:
    python dagger_sft_train.py --data-dir ../dagger_sft_data --gpu 4
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

torch.set_float32_matmul_precision("high")

DEFAULT_CONFIG = {
    "model": {
        "pretrained_model_path": "/data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct",
        "sft_model_path": "/data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Zewei-Zhou/AutoVLA/AutoVLA_PDMS_89.ckpt",
        "codebook_cache_path": "/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA/codebook_cache/agent_vocab.pkl",
        "attn_implementation": "sdpa",
        "train_vision_backbone": False,
        "train_lm_backbone": True,
        "tokens": {
            "action_start_id": 151665,
            "ignore_index": -100,
        },
        "video": {
            "min_pixels": 28 * 28 * 128,
            "max_pixels": 28 * 28 * 128,
        },
    },
    "training": {
        "batch_size": 2,
        "learning_rate": 3.0e-5,
        "epochs": 3,
        "devices": 1,
        "num_workers": 4,
        "weight_decay": 0.01,
        "lr_schedule": "cosine",
        "lr_warmup_step": 100,
        "lr_min_ratio": 0.05,
        "accumulate_grad_batches": 4,
        "max_steps": 5000,
        "checkpoint_every_n_steps": 500,
        "val_check_interval": 500,
        "gradient_clip_val": 1.0,
    },
}


class DaggerSFTDataset(Dataset):
    """Dataset of DAgger SFT samples from AlPaGym rollout data."""

    def __init__(self, samples_dir: str, processor, config: dict, split: str = "train"):
        self.samples_dir = Path(samples_dir)
        self.processor = processor
        self.config = config
        self.split = split

        all_samples = sorted(self.samples_dir.glob("sample_*.pt"))
        if not all_samples:
            raise RuntimeError(f"No samples found in {self.samples_dir}")

        # 90/10 train/val split
        n_val = max(1, len(all_samples) // 10)
        if split == "train":
            self.sample_paths = all_samples[n_val:]
        else:
            self.sample_paths = all_samples[:n_val]

        print(f"DAgger SFT {split}: {len(self.sample_paths)} samples")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        sample = torch.load(self.sample_paths[idx], map_location="cpu", weights_only=False)

        pil_images = sample["pil_images"]
        history_xy = sample["history_xy"]
        velocity = sample["velocity"]
        acceleration = sample["acceleration"]
        instruction = sample["instruction"]
        action_indices = sample["action_indices"]
        action_text = "".join(f"<action_{i}>" for i in action_indices)

        num_images = len(pil_images)
        frames_per_cam = num_images // 3
        sample_rate_hz = 2.0

        min_pixels = self.config["model"]["video"]["min_pixels"]
        max_pixels = self.config["model"]["video"]["max_pixels"]

        system_content = [
            {
                "type": "text",
                "text": (
                    "You are an Advanced Driver Assistance and Full Self-Driving System. "
                    "You will receive visual observations from the ego vehicle's cameras "
                    "and dynamic information about the vehicle's current state. "
                    "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                    "First, carefully analyze the surrounding environment by considering "
                    "traffic lights, the movements of other vehicles and pedestrians, lane "
                    "markings, and any other relevant factors.\n\n"
                    "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at "
                    "the best driving action. Otherwise, you may directly predict the final "
                    "driving action.\n\n"
                    "Structure your reasoning as follows:\n"
                    "1. **Scene Analysis**: Describe the traffic situation, including relevant "
                    "environmental cues such as traffic lights, lane markings, and the behaviors "
                    "of surrounding vehicles or pedestrians.\n"
                    "2. **Identification of Critical Objects**: Identify two to three critical "
                    "road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                    "3. **Prediction of Critical Object Behavior**: Predict the potential movements "
                    "of the identified critical objects.\n"
                    "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and "
                    "current vehicle state, reason about the desired intent of the ego vehicle.\n"
                    "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                    "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                    "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                    "Present the final action clearly after your reasoning steps."
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

        # Generate template-based CoT reasoning text for DAgger SFT target.
        # This teaches the model to produce structured reasoning before actions.
        lat_action = instruction if instruction in (
            "move forward", "turn left", "turn right",
            "change lane to left", "change lane to right"
        ) else "move forward"
        if velocity < 0.5:
            lon_action = "stop"
        elif acceleration < -1.0:
            lon_action = "quick deceleration"
        elif acceleration < -0.3:
            lon_action = "deceleration"
        elif acceleration > 1.0:
            lon_action = "quick acceleration"
        elif acceleration > 0.3:
            lon_action = "acceleration"
        else:
            lon_action = "maintain constant speed"

        cot_text = (
            f"1. **Scene Analysis**: The ego vehicle is traveling at {velocity:.2f} m/s "
            f"with acceleration {acceleration:.2f} m/s^2. The driving instruction is "
            f"to {instruction}. The recent trajectory indicates the vehicle has been "
            f"following the planned route.\n"
            f"2. **Identification of Critical Objects**: Based on the camera observations, "
            f"the key objects to monitor are surrounding vehicles and pedestrians near the "
            f"ego vehicle's intended path.\n"
            f"3. **Prediction of Critical Object Behavior**: The surrounding objects are "
            f"expected to continue their current trajectories. The ego vehicle should "
            f"maintain a safe distance.\n"
            f"4. **Ego Vehicle Intent Reasoning**: Given the instruction to {instruction} "
            f"and current speed of {velocity:.2f} m/s, the ego vehicle should execute "
            f"a {lat_action} maneuver with {lon_action}.\n"
            f"5. **Final Action Decision**: Lateral action: {lat_action}. "
            f"Longitudinal action: {lon_action}.\n"
        )

        assistant_content = [
            {
                "type": "text",
                "text": f"{cot_text}<answer>\nThe final output action is: {action_text}\n</answer>",
            }
        ]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=True,
        )

        return {
            "text": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "gt_action": torch.tensor(action_indices, dtype=torch.int64),
            "gt_trajectory": torch.tensor(sample.get("gt_xy", np.zeros((10, 2), dtype=np.float32)), dtype=torch.float32),
            "has_cot": True,
        }


class DaggerDataCollator:
    """Collate DAgger SFT samples into batches with proper label masking."""

    def __init__(self, processor, ignore_index: int, action_start_id: int, n_action_tokens: int):
        self.processor = processor
        self.ignore_index = ignore_index
        self.action_start_id = action_start_id
        self.n_action_tokens = n_action_tokens
        self.assistant_marker = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    @staticmethod
    def _find_subsequence(sequence: torch.Tensor, pattern: list) -> int | None:
        pattern_tensor = torch.as_tensor(pattern, dtype=sequence.dtype)
        for start in range(len(sequence) - len(pattern) + 1):
            if torch.equal(sequence[start:start + len(pattern)], pattern_tensor):
                return start
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        video_inputs = []
        image_inputs = []
        for f in features:
            if f.get("video_inputs"):
                video_inputs.extend(f["video_inputs"])
            if f.get("image_inputs"):
                image_inputs.extend(f["image_inputs"])

        batch = self.processor(
            text=texts,
            images=image_inputs or None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.ignore_index
        action_end_id = self.action_start_id + self.n_action_tokens

        for row, feature in enumerate(features):
            marker_start = self._find_subsequence(labels[row], self.assistant_marker)
            if marker_start is None:
                raise RuntimeError("Assistant marker not found in tokenized SFT example")
            response_start = marker_start + len(self.assistant_marker)
            labels[row, :response_start] = self.ignore_index

        batch["labels"] = labels
        batch["gt_action"] = torch.stack([f["gt_action"] for f in features])
        batch["gt_trajectory"] = torch.stack([f["gt_trajectory"] for f in features])
        return batch


class DaggerSFTModel(pl.LightningModule):
    """AutoVLA model for DAgger SFT training."""

    def __init__(self, config: dict, processor, n_action_tokens: int):
        super().__init__()
        self.config = config
        self.processor = processor
        self.n_action_tokens = n_action_tokens
        model_cfg = config["model"]

        model_path = Path(model_cfg["pretrained_model_path"])
        print(f"Loading Qwen2.5-VL from {model_path}...")
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )
        self.vlm.resize_token_embeddings(len(processor.tokenizer))
        self.vlm.config.use_cache = False
        self.vlm.gradient_checkpointing_enable()

        if not model_cfg.get("train_vision_backbone", False):
            vision_backbone = getattr(self.vlm, "visual", None)
            if vision_backbone is None:
                vision_backbone = getattr(getattr(self.vlm, "model", None), "visual", None)
            if vision_backbone is not None:
                vision_backbone.requires_grad_(False)
                print("Vision backbone frozen")

        self.lr = float(config["training"]["learning_rate"])
        self.action_start_id = int(model_cfg["tokens"]["action_start_id"])
        self.action_end_id = self.action_start_id + n_action_tokens

    def forward(self, batch):
        accepted_keys = (
            "input_ids", "attention_mask",
            "pixel_values_videos", "video_grid_thw",
            "pixel_values", "image_grid_thw",
            "labels",
        )
        return self.vlm(**{k: batch[k] for k in accepted_keys if k in batch})

    def _shared_step(self, batch, stage: str):
        outputs = self(batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {stage} loss: {loss}")

        shift_labels = batch["labels"][:, 1:]
        action_mask = (shift_labels >= self.action_start_id) & (shift_labels < self.action_end_id)
        if action_mask.any():
            predictions = outputs.logits[:, :-1].argmax(dim=-1)
            action_accuracy = (predictions[action_mask] == shift_labels[action_mask]).float().mean()
            self.log(f"{stage}_action_accuracy", action_accuracy,
                     prog_bar=stage == "val", batch_size=batch["input_ids"].shape[0])

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch["input_ids"].shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        training_cfg = self.config["training"]
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        )
        warmup = int(training_cfg.get("lr_warmup_step", 500))
        max_steps = int(training_cfg.get("max_steps", 5000))
        min_ratio = float(training_cfg.get("lr_min_ratio", 0.05))

        def lr_scale(step: int) -> float:
            if warmup > 0 and step < warmup:
                return 0.05 + 0.95 * step / warmup
            import math
            progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def configure_action_tokens(processor, model_cfg: dict) -> Tuple[int, Tuple[int, ...]]:
    codebook_path = Path(model_cfg["codebook_cache_path"])
    with codebook_path.open("rb") as f:
        codebook = pickle.load(f)
    n_bins = len(codebook["token_all"]["veh"])
    action_tokens = [f"<action_{i}>" for i in range(n_bins)]
    processor.tokenizer.add_tokens(action_tokens, special_tokens=False)
    action_ids = tuple(processor.tokenizer.convert_tokens_to_ids(action_tokens))
    expected_start = int(model_cfg["tokens"]["action_start_id"])
    expected_ids = tuple(range(expected_start, expected_start + n_bins))
    if action_ids != expected_ids:
        raise RuntimeError(
            f"Action token ID mismatch: expected [{expected_start}, {expected_start + n_bins - 1}], "
            f"got [{action_ids[0]}, {action_ids[-1]}]"
        )
    print(f"Action vocabulary: {n_bins} tokens, IDs {action_ids[0]}..{action_ids[-1]}")
    return n_bins, action_ids


def normalize_checkpoint_key(key: str) -> str:
    for prefix in ("_forward_module.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    # Strip PyTorch Lightning FSDP wrapper infix
    key = key.replace("._fsdp_wrapped_module", "")
    for prefix in ("autovla.", "drivevla."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("vlm."):
        return key
    if key.startswith("model.visual."):
        return f"vlm.visual.{key[len('model.visual.'):]}"
    if key.startswith("model.language_model."):
        return f"vlm.model.{key[len('model.language_model.'):]}"
    if key.startswith(("model.", "visual.", "lm_head.")):
        return f"vlm.{key}"
    return key


def checkpoint_key_candidates(key: str) -> Tuple[str, ...]:
    primary = normalize_checkpoint_key(key)
    candidates = [primary]
    if primary.startswith("vlm.visual."):
        candidates.append(f"vlm.model.visual.{primary[len('vlm.visual.'):]}")
    elif primary.startswith("vlm.model.visual."):
        candidates.append(f"vlm.visual.{primary[len('vlm.model.visual.'):]}")
    elif primary.startswith("vlm.model.language_model."):
        candidates.append(f"vlm.model.{primary[len('vlm.model.language_model.'):]}")
    elif primary.startswith("vlm.model."):
        candidates.append(f"vlm.model.language_model.{primary[len('vlm.model.'):]}")
    return tuple(dict.fromkeys(candidates))


def load_sft_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    print(f"Loading SFT checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    for source_key, value in state_dict.items():
        for candidate in checkpoint_key_candidates(source_key):
            if candidate in model_state and model_state[candidate].shape == value.shape:
                compatible[candidate] = value
                break

    total_numel = sum(v.numel() for v in model_state.values())
    loaded_numel = sum(model_state[k].numel() for k in compatible)
    ratio = loaded_numel / max(1, total_numel)
    print(f"Checkpoint loaded: {ratio:.2%} of parameters ({len(compatible)}/{len(model_state)} tensors)")
    model.load_state_dict(compatible, strict=False)
    del checkpoint, state_dict, compatible


def main():
    parser = argparse.ArgumentParser(description="DAgger SFT training for AutoVLA")
    parser.add_argument("--data-dir", required=True, help="Directory with DAgger SFT samples")
    parser.add_argument("--gpu", type=int, default=4, help="GPU device index")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sft-model-path", type=str, default=None,
                        help="Path to SFT checkpoint (PDMS89 .ckpt). Defaults to config.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pl.seed_everything(args.seed, workers=True)

    config = DEFAULT_CONFIG.copy()
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.lr
    config["training"]["epochs"] = args.epochs
    config["training"]["max_steps"] = args.max_steps
    config["training"]["lr_warmup_step"] = args.warmup_steps
    config["training"]["accumulate_grad_batches"] = args.accumulate_grad_batches
    config["training"]["checkpoint_every_n_steps"] = args.checkpoint_every

    model_cfg = config["model"]
    model_path = Path(model_cfg["pretrained_model_path"])
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True)
    n_action_tokens, _ = configure_action_tokens(processor, model_cfg)

    samples_dir = Path(args.data_dir) / "samples"
    train_dataset = DaggerSFTDataset(str(samples_dir), processor, config, split="train")
    val_dataset = DaggerSFTDataset(str(samples_dir), processor, config, split="val")

    collator = DaggerDataCollator(
        processor=processor,
        ignore_index=int(model_cfg["tokens"]["ignore_index"]),
        action_start_id=int(model_cfg["tokens"]["action_start_id"]),
        n_action_tokens=n_action_tokens,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )

    model = DaggerSFTModel(config, processor, n_action_tokens)
    sft_path = args.sft_model_path or model_cfg["sft_model_path"]
    if sft_path and Path(sft_path).exists():
        load_sft_checkpoint(model, sft_path)
    else:
        print(f"WARNING: SFT checkpoint not found at {sft_path}, training from base model only")

    output_dir = Path(args.output_dir or f"runs/dagger_sft/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_dir / "processor")

    ckpt_steps = config["training"]["checkpoint_every_n_steps"]
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        val_check_interval=ckpt_steps,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                filename="step={step}-loss={train_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=False,
                every_n_train_steps=ckpt_steps,
                save_top_k=-1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=CSVLogger(output_dir, name="dagger_sft"),
        enable_model_summary=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_path = output_dir / "final.ckpt"
    trainer.save_checkpoint(final_path, weights_only=True)
    print(f"\nDAgger SFT complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
