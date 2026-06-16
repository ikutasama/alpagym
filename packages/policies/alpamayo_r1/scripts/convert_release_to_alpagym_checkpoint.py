# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a released Alpamayo model into an AlpaGym RL compatible checkpoint.

Usage::

    uv run --package alpagym-alpamayo-r1 python \
      packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py \
      --input /PATH/TO/Alpamayo-1.5-10B \
      --output /PATH/TO/alpamayo-1.5-10B_alpagym_ckpt

The released checkpoint ships weights + config only. AlpaGym's Cosmos-RL
controller calls ``AutoTokenizer.from_pretrained(checkpoint_dir)``, so the
expanded tokenizer/processor (base vocab + trajectory + special tokens, vocab
155697) is generated from the released config the same way the model builds it
and written into the checkpoint. ``--vlm-name-or-path`` supplies the base
tokenizer/preprocessor; the bare HF id ``nvidia/Cosmos-Reason2-8B`` is enough
because the trajectory tokens are added programmatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from transformers import GenerationConfig

from alpamayo.utils.checkpoint_utils import (
    collect_targets,
    prepare_output_dir,
    remap_targets,
    setup_checkpoint_output,
)

DEFAULT_VLM = "nvidia/Cosmos-Reason2-8B"

TARGET_REMAP = {
    "alpamayo_r1.models.action_in_proj.": "alpagym_alpamayo_r1.submodules.action_in_proj.",
    "alpamayo_r1.diffusion.": "alpamayo1_x_rl.diffusion.",
    "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace": (
        "alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace"
    ),
    "alpamayo1_5.models.action_in_proj.": "alpagym_alpamayo_r1.submodules.action_in_proj.",
    "alpamayo1_5.models.delta_tokenizer.": "alpamayo_r1.models.delta_tokenizer.",
    "alpamayo1_5.diffusion.": "alpamayo1_x_rl.diffusion.",
    "alpamayo1_5.action_space.UnicycleAccelCurvatureActionSpace": (
        "alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace"
    ),
    "alpamayo1_5.action_space.": "alpamayo_r1.action_space.",
}

EXPERT_DEFAULTS = {
    "cotrain_vlm": False,
    "stop_grad_from_vlm": True,
    "legacy_inference_image_input_format": False,
    "include_camera_ids": True,
    "include_frame_nums": True,
    "loss_weights": {"future_traj": 1.0, "others": 1.0},
    "traj_loss_weight": 1.0,
    "expert_hist_traj_tokenizer_cfg": None,
    "hist_traj_embed_cfg": None,
    "padding_side": "left",
}


def build_expert_config(release_config: dict, vlm_name_or_path: str) -> dict:
    """Return an ExpertModel config.json derived from a released AlpamayoR1 config."""
    cfg = json.loads(json.dumps(release_config))

    cfg["model_type"] = "alpamayo_reasoning_vla_expert"
    cfg["architectures"] = ["ExpertModel"]
    remap_targets(cfg, TARGET_REMAP)

    cfg["diffusion_cfg"]["x_dims"] = None

    cfg["expert_cfg"].pop("dtype", None)
    cfg.pop("add_special_tokens", None)

    cfg["vlm_name_or_path"] = vlm_name_or_path
    for key, value in EXPERT_DEFAULTS.items():
        cfg.setdefault(key, value)

    return cfg


def save_processor(release_config: dict, vlm_name_or_path: str, output_dir: Path) -> None:
    """Generate the expanded tokenizer/processor and write it into the checkpoint.

    Builds the processor from the released config exactly as the model does:
    the base ``vlm_name_or_path`` tokenizer plus the trajectory and special tokens
    (vocab 155697). The bare HF id ``nvidia/Cosmos-Reason2-8B`` is therefore enough;
    no pre-expanded tokenizer directory is needed.
    """
    from alpamayo_r1.models.base_model import ReasoningVLAConfig

    proc_config = ReasoningVLAConfig(**{**release_config, "vlm_name_or_path": vlm_name_or_path})
    proc_config._build_processor().save_pretrained(output_dir)


def validate_checkpoint(output_dir: Path) -> None:
    """Fail fast if the converted checkpoint would not load in AlpaGym."""
    cfg = json.loads((output_dir / "config.json").read_text())

    if cfg["model_type"] != "alpamayo_reasoning_vla_expert":
        raise ValueError(f"model_type not rewritten: {cfg['model_type']!r}")

    required = (
        "expert_cfg",
        "diffusion_cfg",
        "action_space_cfg",
        "action_in_proj_cfg",
        "action_out_proj_cfg",
        "vlm_name_or_path",
        "vlm_backend",
        "vocab_size",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")

    for target in sorted(set(collect_targets(cfg).values())):
        module, _, attr = target.rpartition(".")
        if not hasattr(importlib.import_module(module), attr):
            raise ValueError(f"_target_ does not resolve: {target}")

    # Cosmos-RL's controller does AutoTokenizer.from_pretrained(checkpoint_dir), so
    # the tokenizer must be present in the checkpoint, not just referenced by id.
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (output_dir / name).is_file():
            raise FileNotFoundError(f"tokenizer file missing from checkpoint: {name}")


def main() -> None:
    """Parse CLI args, rewrite the config, symlink weights, generate the tokenizer, and validate."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Released HF checkpoint dir (config.json + safetensors).",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination AlpaGym checkpoint dir."
    )
    parser.add_argument(
        "--vlm-name-or-path",
        default=DEFAULT_VLM,
        help="VLM the released model was built on; written into config.vlm_name_or_path and used "
        "as the base tokenizer/preprocessor for the generated processor. HF Hub id or local dir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output dir.")
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    vlm_name_or_path: str = args.vlm_name_or_path

    if input_dir == output_dir:
        raise ValueError(f"--input and --output resolve to the same directory: {input_dir}")

    if not (input_dir / "config.json").is_file():
        raise FileNotFoundError(f"config.json not found in {input_dir}")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    print(f"Converting {input_dir.name} -> {output_dir}")
    print(f"  vlm_name_or_path: {vlm_name_or_path}")

    release_config = json.loads((input_dir / "config.json").read_text())
    expert_config = build_expert_config(release_config, vlm_name_or_path)
    (output_dir / "config.json").write_text(json.dumps(expert_config, indent=2) + "\n")

    weight_actions = setup_checkpoint_output(input_dir, output_dir)
    print(f"  weights: symlinked {len(weight_actions)} files (safetensors shards + index)")

    save_processor(release_config, vlm_name_or_path, output_dir)
    print("  tokenizer: generated expanded processor (vocab 155697)")

    GenerationConfig.from_pretrained(vlm_name_or_path).save_pretrained(output_dir)
    print(f"  generation_config: copied from {vlm_name_or_path}")

    validate_checkpoint(output_dir)
    print(f"\nCheckpoint ready: {output_dir}")


if __name__ == "__main__":
    main()
