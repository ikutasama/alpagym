# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the conversation template for VLM models."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch

from ...common.vla_constant import SPECIAL_TOKENS, dt_marker
from .. import constants

TokenLayout = Literal["camera_ts", "timetick_camera", "timetick_ts"]
FrameLabel = Literal["none", "frame_num", "dt_token"]


USER_PROMPT_TEMPLATE = {
    "cot": "output the chain-of-thought reasoning of the driving process",
    "cot_auto_labeling": "output a comprehensive analysis of the driving scene in JSON format",
    "meta_action": "output meta actions",
    "traj_future": "output the future trajectory",
}


def get_component_str(
    start_str: str,
    end_str: str,
    content_str: str | None = None,
    padding_str: str | None = None,
    ask_for_component: bool = False,
) -> str:
    """Get the component string for the VLA model.

    Args:
        start_str (str): The start string of the component.
        end_str (str): The end string of the component.
        content_str (str | None): The content string of the component.
        padding_str (str | None): The padding string of the component.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        component_str (str): The component string for the VLA model.
    """
    # always add the start string
    component_str = [start_str]

    # if ask this component, we only add start string
    if not ask_for_component:
        assert (content_str is None) != (padding_str is None), (
            "Exactly one of content_str or padding_str must be provided"
        )
        if content_str is not None:
            # add content string directly
            component_str.append(content_str)
        elif padding_str is not None:
            # use padding string as placeholder
            component_str.append(padding_str)
        component_str.append(end_str)
    return "".join(component_str)


def construct_system_prompt() -> list[dict[str, str]]:
    """Construct the system message for the VLA model.

    Args:
        config (DataPreprocessConfig): The configuration for the data processing.

    Returns:
        system_prompt (list): The list of system message prompts for the VLA model.
    """
    system_prompt = "You are a driving assistant that generates safe and accurate actions."
    return [{"type": "text", "text": system_prompt}]


def construct_user_prompt(components: list[str]) -> list[dict[str, str]]:
    """Construct the input prompt for the VLA  model."""
    prompt_components: list[str] = []
    # remember to preserve the order
    for component in components:
        if component in USER_PROMPT_TEMPLATE:
            prompt_components.append(USER_PROMPT_TEMPLATE[component])

    prompt = ", then ".join(prompt_components) + "."
    return [{"type": "text", "text": prompt}]


def construct_image(
    data: dict[str, Any],
    include_camera_ids: bool,
    camera_ids: torch.Tensor,
    token_layout: TokenLayout = "camera_ts",
    frame_label: FrameLabel = "frame_num",
) -> list[dict[str, str]]:
    """Construct the image description prompt for the VLA model.

    Image data is flat: ``data["image_frames"]`` has shape
    ``(N_total, 1, 3, H, W)`` and ``camera_ids`` has length ``N_total``. The
    upstream sort dispatch (see ``alpamayo.data.data_utils.sort_images_*``)
    decides whether ``N_total`` groups are camera-major
    (``[cam0_f0, cam0_f1, ..., cam1_f0, ...]``) or timestep-major
    (``[cam0_t0, cam1_t0, ..., camN_t0, cam0_t1, ...]``). ``construct_image``
    itself only iterates the flat list and emits labels.

    Args:
        data: Must contain ``image_frames`` (4-D per image as noted above) and,
            if ``frame_label="dt_token"``, ``absolute_timestamps`` of length
            ``N_total`` (int64 µs).
        include_camera_ids: Whether to prepend each camera's display name.
        camera_ids: Per-image camera id of length ``N_total`` (already sorted
            to match the chosen layout).
        token_layout: Must match the upstream sort. ``camera_ts`` is the
            legacy layout, ``timetick_camera`` is the streaming layout.
        frame_label: What fills the per-frame label slot. ``"frame_num"`` (default)
            emits absolute frame indices — legacy behavior. ``"dt_token"`` emits
            plain-text ``<dt NNN ms>`` markers so the model sees a streaming-
            friendly frame-rate signal; the dt marker reuses Qwen3-VL's digit
            priors and tokenizes to a fixed 8-token span (no vocab surgery).
            ``"none"`` suppresses the label slot entirely.

    Returns:
        List of message dicts interleaving ``{"type": "text"}`` labels and
        ``{"type": "image"}`` entries; each image entry carries a 4-D tensor
        (leading singleton T=1) so the Qwen processor treats it as one image.
    """
    images = data["image_frames"]
    n_total = images.shape[0]
    if camera_ids.shape[0] != n_total:
        raise ValueError(
            f"camera_ids has length {camera_ids.shape[0]} but image_frames has {n_total} entries"
        )

    abs_ts = None
    if frame_label == "dt_token":
        abs_ts = data.get("absolute_timestamps")
        if abs_ts is None:
            raise ValueError("frame_label='dt_token' but data has no 'absolute_timestamps' field")

    if token_layout == "camera_ts":
        return _construct_image_camera(images, include_camera_ids, camera_ids, frame_label, abs_ts)
    elif token_layout in ("timetick_camera", "timetick_ts"):
        # Both layouts produce the same flat shape (N_frames blocks of N_cams);
        # only the within-tick image order differs (cam-id vs abs-ts), and the
        # render helper just iterates the flat list.
        return _construct_image_timetick(
            images, include_camera_ids, camera_ids, frame_label, abs_ts
        )
    else:
        raise ValueError(
            "token_layout must be 'camera_ts', 'timetick_camera', or 'timetick_ts', "
            f"got {token_layout!r}"
        )


def _construct_image_camera(
    images: torch.Tensor,
    include_camera_ids: bool,
    camera_ids: torch.Tensor,
    frame_label: FrameLabel,
    absolute_timestamps: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Iterate the camera-sorted flat list, emitting ``cam: frame X [img] ...``.

    ``frame_idx`` counts within each camera and resets at camera boundaries.
    With ``frame_label="dt_token"`` (caller passes ``absolute_timestamps``), the
    ``frame_idx`` label is replaced by a per-image ``<dt NNN ms>`` marker
    computed *within each camera segment* (cross-camera gaps would be
    meaningless under camera-major sort, so segments restart).
    """
    n_total = images.shape[0]
    dt_tokens = (
        _dt_tokens_per_camera_segment(absolute_timestamps, camera_ids, n_total)
        if frame_label == "dt_token"
        else None
    )
    messages: list[dict[str, Any]] = []
    prev_cam_id = None
    frame_idx = 0
    for i in range(n_total):
        cam_id = int(camera_ids[i].item())
        if prev_cam_id is not None and cam_id != prev_cam_id:
            frame_idx = 0
        if include_camera_ids and frame_idx == 0:
            messages.append(
                {"type": "text", "text": f"{constants.CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]}: "}
            )
        if dt_tokens is not None:
            messages.append({"type": "text", "text": dt_tokens[i]})
        elif frame_label == "frame_num":
            messages.append({"type": "text", "text": f"frame {frame_idx} "})
        messages.append({"type": "image", "image": images[i]})
        prev_cam_id = cam_id
        frame_idx += 1
    return messages


def _dt_tokens_per_camera_segment(
    absolute_timestamps: torch.Tensor | None,
    camera_ids: torch.Tensor,
    n_total: int,
) -> list[str] | None:
    """Per-image ``<dt NNN ms>`` markers, computed within each camera segment.

    Camera-major layout groups consecutive images by camera; cross-segment
    timestamp gaps would be meaningless (cam1 starts again at the clip start),
    so each segment is treated independently. The first image of each segment
    reuses the next within-segment gap (matches the first-tick rule used by
    ``timetick_camera``); singleton segments fall back to 100 ms.
    """
    if absolute_timestamps is None:
        return None
    ts = absolute_timestamps
    if ts.dim() == 2 and ts.shape[-1] == 1:
        ts = ts.squeeze(-1)
    if ts.numel() != n_total:
        raise ValueError(
            f"absolute_timestamps has {ts.numel()} values but expected {n_total} (N_total)"
        )
    if ts.dtype != torch.int64:
        raise TypeError(
            f"absolute_timestamps must be int64 microseconds, got {ts.dtype}; "
            "upstream is expected to provide the canonical int64 µs schema"
        )
    cam_list = camera_ids.tolist()
    tokens: list[str | None] = [None] * n_total
    i = 0
    while i < n_total:
        j = i
        while j < n_total and cam_list[j] == cam_list[i]:
            j += 1
        seg = ts[i:j]
        seg_len = j - i
        if seg_len == 1:
            tokens[i] = dt_marker(100_000)
        else:
            deltas = (seg[1:] - seg[:-1]).tolist()
            tokens[i] = dt_marker(deltas[0])
            for k, d in enumerate(deltas, start=1):
                tokens[i + k] = dt_marker(d)
        i = j
    return tokens  # type: ignore[return-value]


def _construct_image_timetick(
    images: torch.Tensor,
    include_camera_ids: bool,
    camera_ids: torch.Tensor,
    frame_label: FrameLabel,
    absolute_timestamps: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Iterate the timestep-sorted flat list, emitting per-timestep groups.

    Input assumptions:
      * The sequence is laid out as ``N_frames`` blocks of ``N_cams`` images
        each. The within-block camera order is whatever the upstream sort chose:
        ``sort_images_by_timetick_camera`` produces a fixed cam-id order;
        ``sort_images_by_timestep`` (used by the ``timetick_ts`` layout)
        lets cameras land in actual capture-time order, which can vary tick to
        tick under sub-tick trigger jitter.
      * If ``absolute_timestamps`` is given, it has length ``N_total`` (possibly
        with a trailing singleton) and is in microseconds (int64), epoch-
        anchored. We use absolute (not relative) timestamps because the
        relative origin re-anchors at each sample's ``camera_tmin`` and is
        slated for deprecation; absolute is stable across rolling-window
        samples and remains the canonical time field.

    Output layout (without dt tokens)::

        frame 0: camF [img] camL [img] ... camB [img]
        frame 1: camF [img] ...
        ...

    With dt tokens, the ``frame t`` label is replaced by a plain-text
    ``<dt NNN ms>`` marker before every tick (including the first; see
    ``_dt_tokens_for_timesteps`` for the first-tick rule). Streaming needs
    position-invariant markers since absolute frame indices die with eviction.
    The marker reuses the tokenizer's existing digit+"ms" priors rather than
    allocating new special tokens.
    """
    n_total = images.shape[0]
    n_cams = int(torch.unique(camera_ids).numel())
    if n_total % n_cams != 0:
        raise ValueError(
            f"timestep-major: N_total={n_total} not divisible by N_cams={n_cams}; "
            "upstream ``sort_images_by_timetick_camera`` should have grouped "
            "images into whole-timestep blocks"
        )
    n_frames = n_total // n_cams

    dt_tokens = (
        _dt_tokens_for_timesteps(absolute_timestamps, n_frames, n_cams)
        if frame_label == "dt_token"
        else None
    )

    messages: list[dict[str, Any]] = []
    for i in range(n_total):
        t, c_in_block = divmod(i, n_cams)
        if c_in_block == 0:
            if dt_tokens is not None:
                # Δt tokens replace the absolute frame index — streaming-friendly.
                messages.append({"type": "text", "text": dt_tokens[t]})
            elif frame_label == "frame_num":
                messages.append({"type": "text", "text": f"frame {t} "})
        cam_id = int(camera_ids[i].item())
        if include_camera_ids:
            messages.append(
                {
                    "type": "text",
                    "text": f"{constants.CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]}: ",
                }
            )
        messages.append({"type": "image", "image": images[i]})
    return messages


def _dt_tokens_for_timesteps(
    absolute_timestamps: torch.Tensor | None,
    n_frames: int,
    n_cams: int,
) -> list[str] | None:
    """Build per-timestep ``<dt NNN ms>`` marker strings from per-image int64
    µs timestamps. Diffs in int64 (epoch values ~10^15 µs need exact subtraction),
    averages across cams in float seconds. The first tick reuses the next gap so
    every tick looks the same at streaming steady state; falls back to 100 ms
    (10 Hz) when only one tick is present.
    """
    if absolute_timestamps is None:
        return None
    ts = absolute_timestamps
    if ts.dim() == 2 and ts.shape[-1] == 1:
        ts = ts.squeeze(-1)
    if ts.numel() != n_frames * n_cams:
        raise ValueError(
            f"absolute_timestamps has {ts.numel()} values but expected "
            f"{n_frames}*{n_cams}={n_frames * n_cams} (N_frames * N_cams)"
        )
    # Require int64 µs from upstream — silently casting float→int would
    # truncate sub-µs noise into bias, and casting float32 epoch values
    # would already have lost precision before we got here. Fail fast.
    if ts.dtype != torch.int64:
        raise TypeError(
            f"absolute_timestamps must be int64 microseconds, got {ts.dtype}; "
            "upstream is expected to provide the canonical int64 µs schema"
        )
    ts_per_block = ts.view(n_frames, n_cams)
    # Per-camera Δ in int64 µs (exact); mean across cams stays in int64 via
    # ``sum // n_cams``. torch.mean rejects int tensors but sum+div is fine
    # and avoids a float round-trip. Values are ~10^5 µs × ≤10 cams, nowhere
    # near int64 overflow. Sub-µs truncation of the mean is a non-issue once
    # we quantize to ms in ``dt_marker``.
    dt_per_cam_us = ts_per_block[1:] - ts_per_block[:-1]  # int64 µs
    dt_us_mean = dt_per_cam_us.sum(dim=-1) // n_cams

    dt_list = dt_us_mean.tolist()
    # Use the first observed Δ as the first tick's marker so train and steady-
    # state inference share a single alphabet (no <dt begin> sentinel).
    first_dt_us = dt_list[0] if dt_list else 100_000  # 10 Hz fallback
    tokens: list[str] = [dt_marker(first_dt_us)]
    for dt_us in dt_list:
        tokens.append(dt_marker(dt_us))
    return tokens


def construct_camera_calibration(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the camera calibration prompt for the VLA model."""
    assert "camera_model_dict" in data, "camera_model_dict not found in data"
    camera_model_dict = data["camera_model_dict"]
    camera_calibration = ""
    for camera_name, camera_model in camera_model_dict.items():
        camera_calibration += (
            f"{constants.CAMERA_NAMES_TO_DISPLAY_NAMES[camera_name]} parameters: \n"
        )
        intrinsics = camera_model["intrinsics"]
        camera_calibration += f"width: {intrinsics._width}, height: {intrinsics._height}, "
        camera_calibration += f"horizontal FOV: {np.degrees(intrinsics._horizontal_fov)[0]:.2f}°, "
        camera_calibration += f"vertical FOV: {np.degrees(intrinsics._vertical_fov)[0]:.2f}°"
        camera_calibration += "\n"

        pinhole_intrinsics = intrinsics.to_pinhole()
        pinhole_intrinsics_str = np.array2string(
            pinhole_intrinsics, precision=2, suppress_small=True
        )
        extrinsics_str = np.array2string(
            np.asarray(camera_model["extrinsics"]), precision=2, suppress_small=True
        )
        intrinsics_str = f"Intrinsics matrix: {pinhole_intrinsics_str}\n"
        extrinsics_str = f"Extrinsics matrix: {extrinsics_str}\n"
        camera_calibration += intrinsics_str
        camera_calibration += extrinsics_str
    return [{"type": "text", "text": camera_calibration}]


def construct_traj_history(num_tokens_per_history_traj: int) -> list[dict[str, str]]:
    """Construct the trajectory history prompt for the VLA model.

    Args:
        num_tokens_per_history_traj (int): The number of tokens per history trajectory.

    Returns:
        traj_history_component (list): The list of trajectory history prompts for the VLA model.
    """
    traj_history_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["traj_history_start"],
                end_str=SPECIAL_TOKENS["traj_history_end"],
                padding_str=SPECIAL_TOKENS["traj_history"] * num_tokens_per_history_traj,
            ),
        }
    ]
    return traj_history_component


def construct_route_xy(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the route XY prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.

    Returns:
        route_xy_component (list): The list of route XY prompts for the VLA model.
    """
    # remove batch dimension and only keep XY
    route_xy = data["route_xy"][0, ..., 0:2]
    # filter out nan values
    valid_mask = ~torch.isnan(route_xy).any(dim=-1)
    route_xy = route_xy[valid_mask].numpy().tolist()
    route_xy_str = [f"[{', '.join([f'{v:.2f}' for v in row])}]" for row in route_xy]
    route_xy_str = ", ".join(route_xy_str)

    route_xy_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str="The route waypoints are: ",
                end_str=". ",
                content_str=route_xy_str,
            ),
        }
    ]
    return route_xy_component


def construct_question(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the question prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.

    Returns:
        question_component (list): The list of question prompts for the VLA model.
    """
    question_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["question_start"],
                end_str=SPECIAL_TOKENS["question_end"],
                content_str=data["question"],
            ),
        }
    ]
    return question_component


def construct_traj_future(
    num_tokens_per_future_traj: int, ask_for_component: bool = False
) -> list[dict[str, str]]:
    """Construct the trajectory future prompt for the VLA model.

    Args:
        num_tokens_per_future_traj (int): The number of tokens per future trajectory.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        traj_future_component (list): The list of trajectory future prompts for the VLA model.
    """
    traj_future_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["traj_future_start"],
                end_str=SPECIAL_TOKENS["traj_future_end"],
                padding_str=SPECIAL_TOKENS["traj_future"] * num_tokens_per_future_traj,
                ask_for_component=ask_for_component,
            ),
        }
    ]
    return traj_future_component


def construct_answer(data: dict[str, Any], ask_for_component: bool = False) -> list[dict[str, str]]:
    """Construct the answer prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        answer_component (list): The list of answer prompts for the VLA model.
    """
    # if not asking for answer, we must have the answer in data
    answer = None
    if not ask_for_component:
        assert "answer" in data, "answer not found in data but `answer` in `components_order`"
        answer = data["answer"]

    answer_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["answer_start"],
                end_str=SPECIAL_TOKENS["answer_end"],
                content_str=answer,
                ask_for_component=ask_for_component,
            ),
        }
    ]
    return answer_component


def construct_box(data: dict[str, Any], ask_for_component: bool = False) -> list[dict[str, str]]:
    """Construct the bounding box prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        box_component (list): The list of bounding box prompts for the VLA model.
    """
    # if not asking for box, we must have the box in data
    box = None
    if not ask_for_component:
        # box is a special case of answer, we use answer to store the box
        assert "answer" in data, "answer not found in data but `answer` in `components_order`"
        box = data["answer"]

        # for empty box, we set it to "[]".
        # This should be fixed when generating the dataset.
        if box[0] != "[":
            box = "[]"

        # assume the format of box is [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        assert box[0] == "[" and box[-1] == "]", f"box format is incorrect: {box}"

    box_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["box_start"],
                end_str=SPECIAL_TOKENS["box_end"],
                content_str=box,
                ask_for_component=ask_for_component,
            ),
        }
    ]
    return box_component


def construct_cot(data: dict[str, Any], ask_for_component: bool = False) -> list[dict[str, str]]:
    """Construct the chain-of-thought prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        cot_component (list): The list of chain-of-thought prompts for the VLA model.
    """
    # if not asking for cot, we must have the cot in data
    cot = None
    if not ask_for_component:
        assert "cot" in data, "cot not found in data but `cot` in `components_order`"
        cot = data["cot"]

    cot_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["cot_start"],
                end_str=SPECIAL_TOKENS["cot_end"],
                content_str=cot,
                ask_for_component=ask_for_component,
            ),
        }
    ]
    return cot_component


def construct_meta_action(
    data: dict[str, Any], ask_for_component: bool = False
) -> list[dict[str, str]]:
    """Construct the meta action prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        meta_action_component (list): The list of meta action prompts for the VLA model.
    """
    # if not asking for meta_action, we must have the meta_action in data
    meta_action = None
    if not ask_for_component:
        assert "meta_action_strings" in data, (
            "meta_action not found in data but `meta_action` in `components_order`"
        )
        meta_action = data["meta_action_strings"]

    meta_action_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["meta_action_start"],
                end_str=SPECIAL_TOKENS["meta_action_end"],
                content_str=meta_action,
                ask_for_component=ask_for_component,
            ),
        }
    ]
    return meta_action_component


def construct_nav_instruction(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the navigation instruction prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.

    Returns:
        nav_instruction_component (list): The list of navigation instruction prompts.
    """
    assert "nav_text" in data, (
        "nav_text not found in data but `nav_instruction` in `components_order`"
    )
    nav_text = data["nav_text"][0]
    assert len(nav_text) > 0, "nav_text is empty"

    nav_instruction_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["route_start"],
                end_str=SPECIAL_TOKENS["route_end"],
                content_str=nav_text,
            ),
        }
    ]
    return nav_instruction_component


def split_user_and_assistant_components(components_order: list[str]) -> tuple[list[str], list[str]]:
    """Split the components_order into user and assistant components.

    ``prompt`` / ``question`` indicates the last user component, and the assistant components are
    everything after it.
    """
    if "prompt" in components_order:
        keyword = "prompt"
    elif "question" in components_order:
        keyword = "question"
    else:
        raise ValueError(f"Invalid components_order: {components_order}")
    keyword_index = components_order.index(keyword)
    user_components = components_order[: keyword_index + 1]
    assistant_components = components_order[keyword_index + 1 :]
    return user_components, assistant_components


def build_conversation(
    data: dict[str, Any],
    num_tokens_per_history_traj: int,
    num_tokens_per_future_traj: int,
    components_order: list[str],
    components_prompt: list[str] | None,
    generation_mode: bool,
    include_camera_ids: bool = False,
    token_layout: TokenLayout = "camera_ts",
    frame_label: FrameLabel = "frame_num",
) -> list[dict[str, Any]]:
    """Compose the conversation messages for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        num_tokens_per_history_traj (int): The number of tokens per history trajectory.
        num_tokens_per_future_traj (int): The number of tokens per future trajectory.
        components_order (list[str]): The order of the components.
        components_prompt (list[str] | None): The prompt of the components.
        generation_mode (bool): Whether to use the generation mode.
        include_camera_ids (bool): Whether to include camera IDs as text before images.

    Returns:
        messages (list[dict[str, Any]]): The list of message dictionaries for the VLA model.
    """
    user_components, assistant_components = split_user_and_assistant_components(components_order)

    def build_message(
        role: str,
        components: list[str],
    ) -> dict[str, Any]:
        """Build a message for the VLA model."""
        content: list[dict[str, Any]] = []
        for component in components:
            ask_for_component = generation_mode and component == components_order[-1]
            match component:
                case "prompt":
                    if components_prompt is None:
                        raise ValueError("components_prompt is required for prompt component")
                    content.extend(
                        construct_user_prompt(
                            components_prompt if generation_mode else assistant_components
                        )
                    )
                case "camera_calibration":
                    content.extend(construct_camera_calibration(data=data))
                case "image":
                    content.extend(
                        construct_image(
                            data=data,
                            include_camera_ids=include_camera_ids,
                            camera_ids=camera_ids,
                            token_layout=token_layout,
                            frame_label=frame_label,
                        )
                    )
                case "route_xy":
                    content.extend(construct_route_xy(data=data))
                case "nav_instruction":
                    content.extend(construct_nav_instruction(data=data))
                case "traj_history":
                    content.extend(
                        construct_traj_history(
                            num_tokens_per_history_traj=num_tokens_per_history_traj
                        )
                    )
                case "question":
                    content.extend(construct_question(data=data))
                case "answer":
                    content.extend(construct_answer(data=data, ask_for_component=ask_for_component))
                case "box":
                    content.extend(construct_box(data=data, ask_for_component=ask_for_component))
                case "cot":
                    content.extend(construct_cot(data=data, ask_for_component=ask_for_component))
                case "meta_action":
                    content.extend(
                        construct_meta_action(data=data, ask_for_component=ask_for_component)
                    )
                case "traj_future":
                    content.extend(
                        construct_traj_future(
                            num_tokens_per_future_traj=num_tokens_per_future_traj,
                            ask_for_component=ask_for_component,
                        )
                    )
        return {
            "role": role,
            "content": content,
        }

    camera_ids = data.get("camera_indices", None)
    if include_camera_ids and camera_ids is None:
        raise ValueError("camera_indices is required in data when include_camera_ids=True")

    system_messages = {"role": "system", "content": construct_system_prompt()}
    user_messages = build_message(role="user", components=user_components)
    assistant_messages = build_message(role="assistant", components=assistant_components)

    messages = [system_messages, user_messages, assistant_messages]
    return messages


def _resolve_frame_label(cfg_get: Callable[..., Any]) -> FrameLabel:
    """Resolve ``frame_label`` from a config, applying legacy back-compat.

    Legacy fields ``include_frame_nums`` (bool) and ``use_dt_tokens`` (bool) are
    mapped to the new ``frame_label`` enum:
      * ``use_dt_tokens=True`` → ``"dt_token"`` (dt always wins, mirroring old
        runtime behavior where dt overrode frame numbers).
      * ``include_frame_nums=False`` → ``"none"``.
      * ``include_frame_nums=True`` (default) → ``"frame_num"``.
    Configs with the new ``frame_label`` field bypass the shim entirely.
    """
    explicit = cfg_get("frame_label", None)
    if explicit is not None:
        return explicit
    if cfg_get("use_dt_tokens", False):
        return "dt_token"
    if not cfg_get("include_frame_nums", True):
        return "none"
    return "frame_num"
