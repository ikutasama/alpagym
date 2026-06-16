# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import cosmos_rl.utils.util as cosmos_util
import torch
from alpamayo_r1.models.base_model import TRAJ_TOKEN
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import AutoConfig, AutoProcessor

from ....common.vla_constant import IGNORE_LABEL_ID, SPECIAL_TOKENS
from ....utils.vla_helpers.get_label_mask import get_role_mask
from ...collate_fn import qwen_collate_processed_samples


class AlpaQwenPacker(DataPacker):
    """Alpamayo data packer wrapper over Qwen VLM data packer."""

    def __init__(self, *args, pad_to_fixed_length: bool = False, **kwargs) -> None:
        """Initialize the packer.

        Args:
            pad_to_fixed_length: When True, every collated batch is
                padded/truncated to exactly ``policy.model_max_length`` —
                ``sft_compute_max_len`` returns that value so the trainer's
                ``min(model_max_length, sft_compute_max_len(...))`` collapses to
                it. When False (default), returns the longest sample in the
                batch (legacy variable-length behavior). Opt in per experiment
                by setting ``data.train.data_packer.pad_to_fixed_length: true``.
        """
        super().__init__(*args, **kwargs)
        self._pad_to_fixed_length = pad_to_fixed_length

    def setup(self, config: CosmosConfig, *args, **kwargs) -> None:
        """Initialize the AlpaQwenPacker.

        Args:
            config: CosmosConfig object containing the model configuration.
                - policy.model_name_or_path: Alpamayo1 checkpoint path. The checkpoint
                  must contain processor/tokenizer files.
                - policy.model_max_length: target length used when fixed-length
                  padding is enabled (see ``pad_to_fixed_length`` in
                  :meth:`__init__`).
        """
        model_config = AutoConfig.from_pretrained(config.policy.model_name_or_path)

        self._padding_side = model_config.padding_side
        self._model_max_length = config.policy.model_max_length
        self._initialize_from_params(
            model_name_or_path=config.policy.model_name_or_path,
            traj_vocab_size=model_config.traj_vocab_size,
            min_pixels=model_config.min_pixels,
            max_pixels=model_config.max_pixels,
        )

    def _initialize_from_params(
        self,
        model_name_or_path: str | None = None,
        traj_vocab_size: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        *,
        vlm_name_or_path: str | None = None,
    ) -> None:
        """Initialize the AlpaQwenPacker from parameters.

        Args:
            model_name_or_path: Alpamayo1 checkpoint path (must contain processor/tokenizer
                files) or a base VLM name/path (e.g. ``nvidia/Cosmos-Reason2-2B``).
                Preferred over the deprecated ``vlm_name_or_path`` alias.
            traj_vocab_size: Trajectory vocabulary size.
            min_pixels: Minimum pixels for image processing.
            max_pixels: Maximum pixels for image processing.
            vlm_name_or_path: Deprecated alias for ``model_name_or_path``.
        """
        if model_name_or_path is None and vlm_name_or_path is not None:
            model_name_or_path = vlm_name_or_path
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.hf_processor = cosmos_util.retry(AutoProcessor.from_pretrained)(
            model_name_or_path, trust_remote_code=True, **processor_kwargs
        )
        self.tokenizer = self.hf_processor.tokenizer

        # Add traj tokens to the tokenizer
        if traj_vocab_size is not None:
            discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
            num_new_tokens = self.tokenizer.add_tokens(discrete_tokens)
            assert num_new_tokens in (0, len(discrete_tokens))
            self.tokenizer.traj_token_start_idx = self.tokenizer.convert_tokens_to_ids("<i0>")
            self.tokenizer.traj_token_end_idx = self.tokenizer.convert_tokens_to_ids(
                f"<i{traj_vocab_size - 1}>"
            )

        # Add all special tokens to the tokenizer
        special_tokens = list(SPECIAL_TOKENS.values())
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        # Add mapping from traj token names to ids
        self.tokenizer.traj_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

    def sft_process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process the single sample from alpamayo-cosmos dataset for SFT.

        Args:
            sample (dict[str, Any]): The sample with the following required keys:
                - messages: list[dict[str, Any]]
                - image_frames: torch.Tensor
                - generation_mode: bool

        Returns:
            dict[str, Any]: The processed sample with the following keys:
                - image_frames: torch.Tensor
                - labels_mask: torch.Tensor
                - tokenized_data: dict[str, Any]
                    - input_ids: torch.Tensor
                    - attention_mask: torch.Tensor
                    - pixel_values: torch.Tensor
                    - image_grid_thw: torch.Tensor
        """
        generation_mode = sample.pop("generation_mode")
        messages = sample.pop("messages")
        has_assistant_content = messages[-1]["role"] == "assistant"
        text = self.hf_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation_mode and not has_assistant_content,
            add_vision_id=False,
            # only when we prefill the assistant message, we need to continue the final message
            continue_final_message=generation_mode and has_assistant_content,
        )

        # NOTE: manually convert to float and normalize to [0, 1] to match original behavior
        # This ensures consistent floating-point precision with the original processor
        images = sample["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tokenized_data = dict(
            self.hf_processor(
                text=text,
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
            )
        )

        # remove the batch dimension of the tokenized data
        tokenized_data["input_ids"] = tokenized_data["input_ids"][0]
        tokenized_data["attention_mask"] = tokenized_data["attention_mask"][0]
        sample["tokenized_data"] = tokenized_data

        # generate label masks when not in generation mode
        if not generation_mode:
            sample["labels_mask"] = get_role_mask(
                tokenizer=self.tokenizer,
                tokens=sample["tokenized_data"]["input_ids"],
                bos_token="<|im_start|>",
                eos_token="<|im_end|>",
                role="assistant",
            )
        else:
            sample["labels_mask"] = torch.zeros_like(
                sample["tokenized_data"]["input_ids"],
                dtype=torch.bool,
                device=sample["tokenized_data"]["input_ids"].device,
            )

        # add traj_data field
        if "ego_history_xyz" in sample:
            sample["traj_data"] = {
                "ego_history_xyz": sample["ego_history_xyz"],
                "ego_future_xyz": sample["ego_future_xyz"],
                "ego_history_rot": sample["ego_history_rot"],
                "ego_future_rot": sample["ego_future_rot"],
            }
        else:
            sample["traj_data"] = None

        return sample

    def sft_collate_fn(
        self,
        processed_samples: list[Any],
        computed_max_len: int | None = None,
        pad_token_id: int | None = None,
        ignore_label_id: int = IGNORE_LABEL_ID,
    ) -> dict[str, Any]:
        """Collate the processed samples into a batch.

        Args:
            processed_samples (List[Any]): The processed sample.
            computed_max_len (int): If provided, force all sequence-length
                tensors (``input_ids``, ``attention_mask``, ``labels``,
                ``labels_mask``) to exactly this length, producing a fixed-shape
                batch. Shorter sequences are padded on ``self._padding_side``;
                longer sequences are always truncated from the right (regardless
                of padding side) so the prefix containing image-placeholder
                tokens is preserved. If None, fall back to padding to the
                longest sequence in the batch (variable-shape batch); disable
                fixed-length padding upstream by setting
                ``policy.model_max_length=null``.
            pad_token_id (int): The pad token id used to pad when seq length is different.
            ignore_label_id (int): The ignore label id for padded tokens.

        Returns:
            Dict[str, Any]: The collated batch.
        """
        # consistency check
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            assert pad_token_id == self.tokenizer.pad_token_id
        assert ignore_label_id == IGNORE_LABEL_ID

        batch = qwen_collate_processed_samples(
            processed_samples,
            pad_token_id=pad_token_id,
            ignore_label_id=ignore_label_id,
            padding_side=self._padding_side,
            max_len=computed_max_len,
        )
        return batch

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        """Compute the maximum sequence length of the processed samples."""
        raise NotImplementedError("Not implemented")

    def sft_compute_max_len(self, processed_samples: list[Any]) -> int:
        """Compute the target sequence length for the next collated batch.

        When ``pad_to_fixed_length`` is enabled (see :meth:`setup`), always
        return ``policy.model_max_length`` so the trainer's
        ``min(model_max_length, sft_compute_max_len(...))`` collapses to
        ``model_max_length`` — every step pads/truncates to exactly that.
        Otherwise return the longest sample length in the batch.
        """
        if getattr(self, "_pad_to_fixed_length", False):
            return self._model_max_length
        return max([len(x["tokenized_data"]["input_ids"]) for x in processed_samples])

    def get_rollout_input(self, item: Any) -> Any:
        """Get the rollout input for the qwen2.5 model."""
        raise NotImplementedError("Not implemented")

    def rollout_collate_fn(self, items: list[Any]) -> list[Any]:
        """Collate the rollout items into a batch."""
        raise NotImplementedError("Not implemented")

    def get_policy_input(
        self,
        sample: Any,
        rollout_output: str | None = None,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """Get the policy input for the qwen2.5 model."""
        raise NotImplementedError("Not implemented")

    def policy_collate_fn(
        self, processed_samples: list[Any], computed_max_len: int
    ) -> dict[str, Any]:
        """Collate the policy items into a batch."""
        raise NotImplementedError("Not implemented")


def _preprocess_with_packer(
    data: dict[str, Any],
    packer: "AlpaQwenPacker",
) -> dict[str, Any]:
    """Preprocess data by delegating to :meth:`AlpaQwenPacker.sft_process_sample`.

    Returns:
        The processed sample dict (tokenized_data, labels_mask, traj_data, …).
    """
    return packer.sft_process_sample(data)
