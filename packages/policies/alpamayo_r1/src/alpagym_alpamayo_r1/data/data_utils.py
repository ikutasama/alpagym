# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Util functions for data processing
import torch


def _reorder_image_data(
    image_data: dict[str, torch.Tensor], permutation: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Apply ``permutation`` to every tensor whose first dim matches it.

    Scalar-like tensors (e.g. ``camera_tmin``) and any tensor with a different
    leading dimension pass through unchanged — they are not per-image.
    """
    n = permutation.numel()
    result: dict[str, torch.Tensor] = {}
    for key, value in image_data.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == n:
            result[key] = value[permutation]
        else:
            result[key] = value
    return result


def sort_images_by_camera_ids(image_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sort image data tensors by their camera ids (stable).

    Produces the camera-major ordering ``[cam0_f0, cam0_f1, ..., cam1_f0, ...]``
    (assuming the within-camera input order is already chronological).

    Args:
        image_data: Dictionary containing at least ``"camera_indices"`` and other
            tensors from image processor output (e.g. ``image_frames``,
            ``absolute_timestamps``, ``camera_tmin``, ``relative_timestamps``).

    Returns:
        New dictionary with the same keys, where applicable tensors are sorted
        by ``camera_indices``.
    """
    permutation = torch.argsort(image_data["camera_indices"], stable=True)
    return _reorder_image_data(image_data, permutation)


def sort_images_by_timestep(
    image_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Sort image data tensors by ``absolute_timestamps`` ascending.

    Pure timestamp sort. Cameras with sub-millisecond trigger jitter end up in
    capture order (which can flip across ticks); use
    :func:`sort_images_by_timetick_camera` when a deterministic per-tick camera
    order is required (e.g. streaming KV eviction).
    """
    permutation = torch.argsort(image_data["absolute_timestamps"], stable=True)
    return _reorder_image_data(image_data, permutation)


def sort_images_by_timetick_camera(
    image_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Lex-sort image data by ``(tick_idx, camera_indices)`` ascending.

    Produces the streaming-friendly tick-major ordering
    ``[cam0_t0, cam1_t0, ..., camN_t0, cam0_t1, ...]``: each tick is a
    contiguous block of ``N_cams`` tokens that can be evicted as a unit.

    Tick membership is recovered from ``absolute_timestamps`` by chunking the
    time-sorted images into groups of ``N_cams`` (the i-th N_cams images in
    sorted-by-time order all belong to tick ``i // N_cams``). Sub-millisecond
    camera trigger jitter within a tick is folded into the tick bucket, so the
    secondary cam-id sort is jitter-invariant — every tick exposes cameras in
    the same order.

    **Assumption on input:** the per-image ``absolute_timestamps`` are sampled
    *around* tick anchors (one image per camera per tick, with sub-tick
    capture jitter much smaller than the inter-tick gap). The chunk-by-N_cams
    bucketing only recovers tick membership when this holds — e.g. clipgt-style
    multi-cam captures at a fixed tick rate. Datasets with arbitrary
    timestamps (skipped frames, frames per camera at independent rates) will
    *not* group correctly with this routine.

    Requires:
      - per-image ``absolute_timestamps`` (tensor, length matching
        ``camera_indices``);
      - the number of images divisible by the number of unique camera ids in
        the sample (every tick exposes the same cam set).
    """
    ts = image_data["absolute_timestamps"]
    cam = image_data["camera_indices"]
    if ts.shape[0] != cam.shape[0]:
        raise ValueError(
            f"absolute_timestamps ({ts.shape[0]}) and camera_indices ({cam.shape[0]}) "
            "must have the same length for tick-major sort"
        )
    n = ts.shape[0]
    n_cams = int(cam.unique().numel())
    if n % n_cams != 0:
        raise ValueError(
            f"image count ({n}) is not divisible by unique camera count ({n_cams}); "
            "tick-major sort assumes every tick exposes the same N_cams images"
        )
    # Recover per-image tick_idx: in time-sorted order, the i-th image belongs
    # to tick ``i // n_cams``. This collapses sub-tick jitter into one bucket
    # per tick, so the secondary cam sort below is jitter-invariant.
    perm_by_ts = torch.argsort(ts, stable=True)
    tick_idx = torch.empty(n, dtype=torch.long, device=ts.device)
    tick_idx[perm_by_ts] = torch.arange(n, device=ts.device) // n_cams

    # Lex-sort: primary by tick_idx, secondary by camera id (stable).
    perm_secondary = torch.argsort(cam, stable=True)
    perm_primary = torch.argsort(tick_idx[perm_secondary], stable=True)
    permutation = perm_secondary[perm_primary]
    return _reorder_image_data(image_data, permutation)


def num_batches(total_num: int, batch_size: int, drop_last: bool) -> int:
    """Get the number of batches for a given total number of items."""
    if drop_last:
        return total_num // batch_size
    else:
        return (total_num + batch_size - 1) // batch_size
