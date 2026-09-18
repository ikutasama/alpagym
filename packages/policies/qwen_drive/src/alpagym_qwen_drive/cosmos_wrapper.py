# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-RL BaseModel wrapper for Qwen-Drive-1.0 Planning Expert.

Registers the ``qwen_drive`` model type with Cosmos-RL's ``ModelRegistry``
so the GRPO trainer can load and train the model.  The wrapper:

1. Registers ``QwenDriveConfig`` with ``AutoConfig`` and
   ``QwenDriveForPlanning`` with ``AutoModel``.
2. Provides ``from_pretrained`` / ``load_hf_weights`` so Cosmos-RL can
   instantiate the model on meta device and then load weights.
3. Provides ``forward`` for the GRPO training step (Phase 4: real
   flow-matching logprob; Phase 3 smoke test: stub).

The planner (Planning Expert) checkpoint lives in a separate directory
whose path is injected via :func:`set_planner_path` from the policy
bundle's ``install_runtime_bridge`` hook.
"""


import json
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils.logging import logger as cosmos_logger
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

from alpagym_qwen_drive.stochastic_sampler import (
    StochasticSamplingConfig,
    stochastic_logprob,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom weight mapper for Qwen-Drive's composite config
# ---------------------------------------------------------------------------


class QwenDriveWeightMapper(HFModelWeightMapper):
    """Weight mapper that extracts attention config from ``vlm_config.text_config``.

    ``QwenDriveConfig`` is a composite config with ``vlm_config`` (Qwen3.5 VLM)
    and ``expert_config`` (Planning Expert).  The VLM's attention parameters
    live in ``vlm_config.text_config``, which the default
    ``HFModelWeightMapper`` cannot find (it only checks top-level
    ``text_config`` / ``llm_config``).

    Additionally, Qwen3.5 has an explicit ``head_dim`` field (256) that
    differs from ``hidden_size // num_attention_heads`` (160), and uses
    ``attn_output_gate=True`` which makes vLLM store Q+gate separately
    from K+V.  This mapper corrects the head_dim and handles the
    Q-only QKV weight layout.
    """

    def __init__(self, hf_config: AutoConfig) -> None:
        # Temporarily expose text_config at the top level so the parent
        # __init__ can find the attention parameters.
        vlm_config = getattr(hf_config, "vlm_config", None)
        if vlm_config is not None and not hasattr(hf_config, "text_config"):
            text_config = getattr(vlm_config, "text_config", None)
            if text_config is not None:
                hf_config.text_config = text_config
        super().__init__(hf_config)

        # Qwen3.5 has an explicit head_dim that differs from
        # hidden_size // num_attention_heads.  Use it if available.
        if self.text_config is not None:
            explicit_head_dim = getattr(self.text_config, "head_dim", None)
            if explicit_head_dim is not None and explicit_head_dim > 0:
                self.head_dim = explicit_head_dim

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Handle Qwen3.5's Q-only QKV weight layout.

        When ``attn_output_gate=True``, vLLM may store the Q+gate weight
        separately from K+V.  In that case the ``qkv_proj`` weight has
        ``dim_0 = total_q * head_dim`` (Q+gate only) instead of the
        expected ``(total_q + 2*n_kv) * head_dim`` (Q+K+V).
        We detect this and return only the ``q_proj`` mapping, letting
        K and V be handled by their own weights.
        """
        compatible_key = self.rollout_map_local_key_to_hf_key(param_name)

        is_qkv = ("qkv_proj" in compatible_key) or (
            "qkv" in compatible_key and not self.is_vlm
        )
        if is_qkv:
            tc = self.text_config if self.text_config is not None else self.config
            dim_0 = param.shape[0]
            total_q = tc.num_attention_heads * (1 + int(self.attn_output_gate))
            expected_full = (total_q + 2 * tc.num_key_value_heads) * self.head_dim
            q_only = total_q * self.head_dim

            if dim_0 == q_only and dim_0 != expected_full:
                # Q-only weight (Q+gate, no K+V fused)
                rule = "qkv_proj" if "qkv_proj" in compatible_key else "qkv"
                q_proj_key = compatible_key.replace(rule, "q_proj")
                cosmos_logger.debug(
                    "[QwenDriveWeightMapper] Q-only QKV detected: "
                    f"{compatible_key} dim_0={dim_0} (Q+gate only)"
                )
                return [(q_proj_key, param)]

        return super().rollout_split_local_key_n_param_to_hf_key_n_param(
            param_name, param
        )

# ---------------------------------------------------------------------------
# Module-level planner path (set by bundle.install_runtime_bridge)
# ---------------------------------------------------------------------------

_PLANNER_PATH: str | None = None


def set_planner_path(path: str | None) -> None:
    """Store the Planning Expert checkpoint path for ``load_hf_weights``."""
    global _PLANNER_PATH
    _PLANNER_PATH = path
    if path:
        logger.info("QwenDriveCosmos: planner path set to %s", path)


_DEFAULT_RL_SAMPLING_CONFIG = None


def set_rl_sampling_config(config: StochasticSamplingConfig | None) -> None:
    """Store the stochastic sampling knobs shared by rollout and trainer.

    ``build_data_packer`` calls this before the trainer instantiates
    ``QwenDriveCosmos`` so the class-level default picks the configured
    sampler; the trainer's ``__init__`` copies it onto the instance.
    """
    global _DEFAULT_RL_SAMPLING_CONFIG
    _DEFAULT_RL_SAMPLING_CONFIG = config
    if config is not None:
        logger.info(
            "QwenDriveCosmos: rl sampling config set (epsilon=%.4g, m=%d)",
            config.epsilon,
            config.num_modes,
        )




# ---------------------------------------------------------------------------
# AutoConfig / AutoModel registration
# ---------------------------------------------------------------------------

def _register_autoconfig_and_model() -> None:
    """Register QwenDriveConfig and QwenDriveForPlanning with transformers."""
    try:
        from qwen_drive.configuration_qwen_drive import (
            QwenDriveConfig,
            QwenDrivePlanningExpertConfig,
        )
        from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning
    except ImportError:
        logger.debug("qwen_drive not importable yet, skipping AutoConfig registration")
        return

    for cfg_cls in (QwenDriveConfig, QwenDrivePlanningExpertConfig):
        mt = cfg_cls.model_type
        try:
            AutoConfig.register(mt, cfg_cls, exist_ok=True)
            logger.info("Registered AutoConfig for model_type=%s", mt)
        except (ValueError, TypeError):
            try:
                AutoConfig.register(mt, cfg_cls)
            except ValueError:
                pass

    try:
        AutoModel.register(QwenDriveConfig, QwenDriveForPlanning, exist_ok=True)
        logger.info("Registered AutoModel for QwenDriveConfig")
    except (ValueError, TypeError):
        try:
            AutoModel.register(QwenDriveConfig, QwenDriveForPlanning)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Cosmos-RL BaseModel wrapper
# ---------------------------------------------------------------------------


class QwenDriveCosmos(BaseModel):
    """Cosmos-RL BaseModel wrapper for Qwen-Drive-1.0.

    The wrapper holds a :class:`QwenDriveForPlanning` model.  Cosmos-RL
    instantiates it on meta device (via ``from_pretrained``), materialises
    the tensors, then calls ``load_hf_weights`` to load the actual
    safetensors checkpoints (VLM + Planning Expert).
    """

    def __init__(self, hf_config: AutoConfig) -> None:
        super().__init__(hf_config)
        self.hf_config = hf_config
        # Sampler knobs for the trainer-side likelihood re-evaluation; set by
        # ``set_rl_sampling_config`` from the policy bundle so the trainer and
        # the rollout share one source of truth.
        self.rl_sampling_config = (
            _DEFAULT_RL_SAMPLING_CONFIG or StochasticSamplingConfig()
        )

        # Build the model from config (on meta device when called inside
        # ``init_on_device("meta")`` context).
        from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning

        self.model = QwenDriveForPlanning(hf_config)

        # Freeze the VLM 鈥?only the Planning Expert is trained.
        for param in self.model.vlm.parameters():
            param.requires_grad = False
        cosmos_logger.info(
            "[QwenDriveCosmos] Froze VLM params, trainable Planning Expert params: %d",
            sum(1 for p in self.model.planning_expert.parameters() if p.requires_grad),
        )

    @staticmethod
    def supported_model_types() -> list[str]:
        """Return the HF model types this wrapper handles."""
        return ["qwen_drive"]

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str | None = None,
        max_position_embeddings: int | None = None,
        **kwargs: Any,
    ) -> "QwenDriveCosmos":
        """Construct a QwenDriveCosmos from its HF config (meta device)."""
        del model_name_or_path, max_position_embeddings  # unused
        return cls(hf_config)

    # -- weight loading -----------------------------------------------------

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: Any = None,
        device: Any = None,
        revision: str | None = None,
    ) -> None:
        """Load VLM and Planning Expert weights from safetensors checkpoints.

        ``model_name_or_path`` points at the VLM directory.  The Planning
        Expert checkpoint path is read from the module-level
        ``_PLANNER_PATH`` (set by ``bundle.install_runtime_bridge``).
        """

        model_path = model_name_or_path

        # --- Load VLM weights ---
        vlm_state = self._load_safetensors_dir(model_path, prefix_strip=None)
        if vlm_state:
            # QwenDriveForPlanning stores the VLM under self.model.vlm
            # and the top-level model under "vlm.*" in the checkpoint.
            # Map "vlm." prefix -> "" for direct loading into self.model.vlm
            vlm_prefixed = {}
            for k, v in vlm_state.items():
                if k.startswith("vlm."):
                    vlm_prefixed[k[len("vlm.") :]] = v
                elif k.startswith("model.vlm."):
                    vlm_prefixed[k[len("model.vlm.") :]] = v
                else:
                    # Keep non-VLM keys for the expert (shouldn't happen here)
                    pass
            if vlm_prefixed:
                missing, unexpected = self.model.vlm.load_state_dict(
                    vlm_prefixed, strict=False
                )
                if missing:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] VLM missing keys: %d", len(missing)
                    )
                if unexpected:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] VLM unexpected keys: %d", len(unexpected)
                    )
            cosmos_logger.info(
                "[QwenDriveCosmos] Loaded VLM weights from %s (%d tensors)",
                model_path,
                len(vlm_state),
            )

        # --- Load Planning Expert weights ---
        planner_path = _PLANNER_PATH
        if planner_path is None:
            # Fallback: look for planner-sft subdirectory
            planner_path = os.path.join(model_path, "planner-sft")
        if os.path.isdir(planner_path):
            expert_state = self._load_safetensors_dir(planner_path, prefix_strip="planning_expert.")
            if expert_state:
                target_dtype = next(self.model.planning_expert.parameters()).dtype
                expert_state = {k: v.to(target_dtype) for k, v in expert_state.items()}
                missing, unexpected = self.model.planning_expert.load_state_dict(
                    expert_state, strict=False
                )
                if missing:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] Planning Expert missing keys: %d",
                        len(missing),
                    )
                if unexpected:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] Planning Expert unexpected keys: %d",
                        len(unexpected),
                    )
                cosmos_logger.info(
                    "[QwenDriveCosmos] Loaded Planning Expert weights from %s (%d tensors)",
                    planner_path,
                    len(expert_state),
                )
            else:
                cosmos_logger.warning(
                    "[QwenDriveCosmos] No safetensors found at %s", planner_path
                )

        # --- Move to device if specified ---
        if device is not None:
            self.model = self.model.to(device)
            cosmos_logger.info("[QwenDriveCosmos] Moved model to %s", device)

    @staticmethod
    def _load_safetensors_dir(
        dir_path: str, prefix_strip: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Load all safetensors shards from a directory."""
        state: dict[str, torch.Tensor] = {}

        # Check for index file (sharded checkpoint)
        index_path = os.path.join(dir_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            for shard in shard_files:
                shard_path = os.path.join(dir_path, shard)
                if os.path.exists(shard_path):
                    state.update(load_file(shard_path))
        else:
            # Single file
            single_path = os.path.join(dir_path, "model.safetensors")
            if os.path.exists(single_path):
                state.update(load_file(single_path))

        if prefix_strip and state:
            state = {
                k.removeprefix(prefix_strip): v
                for k, v in state.items()
            }
        return state

    # -- position ids (required by BaseModel ABC) --------------------------

    def get_position_ids(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return position ids, input ids, and sequence dimension index.

        Required by Cosmos-RL's ``BaseModel`` ABC for context-parallelism
        support.  Qwen-Drive does not use context parallelism, so we
        return a simple sequential position id tensor.
        """
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            inputs = kwargs["input_ids"]
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if inputs_embeds is None:
                # No text inputs 鈥?return a dummy 1-token position
                device = next(self.parameters()).device
                inputs = torch.zeros(1, 1, dtype=torch.long, device=device)
            else:
                seq_len = inputs_embeds.size(1)
                batch = inputs_embeds.size(0)
                inputs = torch.zeros(
                    batch, seq_len, dtype=torch.long, device=inputs_embeds.device
                )

        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        return position_ids, inputs, 1  # seq_dim_idx = 1

    # -- forward (GRPO training) --------------------------------------------

    def forward(self, teacher_model: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Forward pass for GRPO training (Phase 4).

        Rebuilds the driving scene from the replayed raw inputs, re-runs the
        frozen VLM prefill (no gradients), and evaluates the differentiable
        stochastic-sampler log-probability (paper Eq. 12/14) of the recorded
        subspace actions under the current Planning Expert parameters.

        The packer stacks single-step replay rows into a leading batch dim:
        every leaf arrives as ``[B, ...]``. Conditioning tensors carry their
        original trailing shapes; the selected-sample trace leaves
        (``selected_states`` ``[B, N+1, T, D]``, ``selected_z`` ``[B, K, m]``)
        are already per-row. Rows are processed one at a time because each
        step's prompt differs (the scene cache cannot be shared across rows).
        """


        camera_frames = kwargs["camera_frames"]  # [B, C*T, 3, H, W] uint8
        batch_size = camera_frames.shape[0]
        device = next(self.model.vlm.parameters()).device

        is_padding = kwargs.get("is_padding")
        if is_padding is None:
            is_padding = torch.zeros(batch_size, dtype=torch.bool, device=device)

        log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        for row in range(batch_size):
            if bool(is_padding[row]):
                # Padding rows forward a finite zero log-prob; the trainer's
                # mask (zero advantage + KL mask) neutralizes them.
                continue
            log_probs[row] = self._replay_row_logprob(kwargs, row, device)

        kl_div = None
        if teacher_model is not None:
            kl_div = torch.zeros(batch_size, dtype=torch.float32, device=device)

        return {"log_probs": log_probs, "kl_div": kl_div}

    def _replay_row_logprob(
        self, kwargs: dict[str, Any], row: int, device: torch.device
    ) -> torch.Tensor:
        """Score one replayed step's recorded action (paper Eq. 12/14).

        Mirrors the rollout-side adapter: rebuild the ``DrivingScene``, run
        the processor and the frozen VLM prefill, then call
        :func:`stochastic_logprob` with the recorded initial noise and
        subspace coefficients. Gradients flow only through the Planning
        Expert's ``predict_endpoint``.
        """
        from qwen_drive.scene import DrivingScene
        from qwen_drive.trajectory import normalize_history

        from alpagym_qwen_drive.inference_model import (
            compute_history_velocity_acceleration,
            route_to_nav_command,
        )

        views = self._build_replay_views(kwargs, row)
        history = kwargs["ego_history_xyz"][row].float()
        history_rot = kwargs["ego_history_rot"][row].float()
        # Drop the rollout adapter's extra "set" axis if present.
        if history.dim() == 3:
            history = history[0]
        if history_rot.dim() == 4:
            history_rot = history_rot[0]
        headings = torch.atan2(history_rot[..., 1, 0], history_rot[..., 0, 0])
        history_tha = torch.stack(
            [history[:, 0], history[:, 1], headings], dim=-1
        )
        history_velocity, history_acceleration = compute_history_velocity_acceleration(
            history_tha.cpu().numpy(), dt=0.1
        )
        nav_command = route_to_nav_command(kwargs["route_xy"][row].float())
        driving_command = torch.zeros(4, dtype=torch.float32)
        driving_command[nav_command] = 1.0

        scene = DrivingScene(
            views=views,
            history=history_tha.cpu().numpy(),
            history_velocity=history_velocity,
            history_acceleration=history_acceleration,
            ego_velocity=history_velocity[-1],
            ego_acceleration=history_acceleration[-1],
            driving_command=driving_command.numpy(),
            nav_command=nav_command,
            token=f"replay_row_{row}",
        )
        inputs = self.model.processor(scene, with_reasoning=False, device=device)
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        scene_cache, anchor = self.model._prefill(inputs)
        scale = self.model.trajectory_scale(device)
        normalized_history = normalize_history(inputs["history"].float(), scale)

        config = self.rl_sampling_config
        return stochastic_logprob(
            self.model.planning_expert,
            scene_cache=scene_cache,
            position_anchor=anchor,
            history=normalized_history,
            history_velocity=inputs["history_velocity"].float(),
            history_acceleration=inputs["history_acceleration"].float(),
            nav_command=inputs["nav_command"],
            ego_status=inputs["ego_status"].float(),
            states=kwargs["selected_states"][row].float(),
            z=kwargs["selected_z"][row].float(),
            config=config,
        )

    def _build_replay_views(
        self, kwargs: dict[str, Any], row: int
    ) -> dict:
        """Group one replay row's frames into Qwen-Drive view lists.

        Mirrors the rollout-side ``_build_camera_views``: frames ordered
        camera-major, view names assigned in fixed order.
        """
        from qwen_drive.scene import CameraFrame

        from alpagym_qwen_drive.inference_model import QWEN_DRIVE_VIEWS

        camera_frames = kwargs["camera_frames"][row]  # [C*T, 3, H, W]
        camera_indices = kwargs["camera_indices"][row]  # [C*T]
        unique_cams = torch.unique(camera_indices)
        views = {}
        for cam_idx_pos, cam_id_val in enumerate(unique_cams.tolist()):
            cam_mask = camera_indices == cam_id_val
            cam_frames = camera_frames[cam_mask]
            view_name = (
                QWEN_DRIVE_VIEWS[cam_idx_pos]
                if cam_idx_pos < len(QWEN_DRIVE_VIEWS)
                else QWEN_DRIVE_VIEWS[0]
            )
            frame_list = []
            for t in range(cam_frames.shape[0]):
                frame_hwc = cam_frames[t].permute(1, 2, 0).cpu().numpy()
                if frame_hwc.shape[2] == 1:
                    frame_hwc = np.repeat(frame_hwc, 3, axis=2)
                frame_list.append(
                    CameraFrame(image=Image.fromarray(frame_hwc.astype(np.uint8)))
                )
            views[view_name] = frame_list
        for view_name in QWEN_DRIVE_VIEWS:
            if view_name not in views:
                views[view_name] = views.get(QWEN_DRIVE_VIEWS[0], [])
        return views

    # -- parallelism / FSDP -------------------------------------------------

    @property
    def parallelize_fn(self) -> tuple[Callable, "QwenDriveCosmos"]:
        """Return ``(parallelize_fn, self)`` for single-GPU / DDP."""

        def parallelize(
            model: torch.nn.Module,
            parallel_dims: Any,
            config: Any,
            pp_loss_fn: Callable | None = None,
        ):
            # For single-GPU or DDP replicate
            if (
                hasattr(parallel_dims, "dp_replicate_enabled")
                and parallel_dims.dp_replicate_enabled
            ):
                from torch.distributed._composable.replicate import replicate

                replicate(model, device_mesh=parallel_dims.mesh, bucket_cap_mb=100)
                cosmos_logger.info("[QwenDriveCosmos] Applied DDP (replicate)")
            return None, None

        return parallelize, self

    def _apply_fsdp2(self, dp_mesh: Any, fsdp_config: dict, reshard_fn: Any) -> None:
        """Apply FSDP2 sharding (no-op for single GPU)."""
        from torch.distributed.fsdp import fully_shard

        # Shard the planning expert layers
        expert = self.model.planning_expert
        if hasattr(expert, "layers"):
            items = list(enumerate(expert.layers))
            for idx, blk in items:
                fully_shard(
                    blk,
                    **fsdp_config,
                    reshard_after_forward=reshard_fn(idx, len(items)),
                )
            fully_shard(expert, **fsdp_config, reshard_after_forward=True)
            cosmos_logger.info(
                "[QwenDriveCosmos] Sharded %d planning expert layers", len(items)
            )

        fully_shard(self, **fsdp_config, reshard_after_forward=True)
        cosmos_logger.info("[QwenDriveCosmos] Applied FSDP2 to full model")

    def post_to_empty_hook(self, cosmos_config: Any) -> None:
        """Warm up after meta->device materialisation.

        No-op for Qwen-Drive; the VLM and expert are already materialised
        by ``load_hf_weights``.
        """

    def separate_model_parts(self) -> list[torch.nn.Module]:
        """Return model sub-modules for per-part parallelization."""
        return [self]

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """Return estimated parameter count and FLOPs."""
        return 0, 0

    def apply_pipeline_split(self, pp_rank: int, pp_size: int) -> None:
        """Pipeline parallel is not supported."""
        raise NotImplementedError("Pipeline parallel is not supported for Qwen-Drive.")

    # -- parameter delegation -----------------------------------------------

    def named_parameters(self, *args: Any, **kwargs: Any):
        return self.model.named_parameters(*args, **kwargs)

    def named_modules(self, *args: Any, **kwargs: Any):
        return self.model.named_modules(*args, **kwargs)

    def parameters(self, *args: Any, **kwargs: Any):
        return self.model.parameters(*args, **kwargs)

    def modules(self, *args: Any, **kwargs: Any):
        return self.model.modules(*args, **kwargs)


# ---------------------------------------------------------------------------
# ModelRegistry registration
# ---------------------------------------------------------------------------

_register_autoconfig_and_model()

if "qwen_drive" not in ModelRegistry._MODEL_REGISTRY:
    ModelRegistry.register(QwenDriveWeightMapper)(QwenDriveCosmos)
    cosmos_logger.info("[QwenDriveCosmos] Registered with ModelRegistry")
