"""Alpamayo-1.5 expert inference service for online DAgger.

Persistent ZMQ ROUTER socket on a dedicated GPU.  Receives observation
dicts (camera_frames, ego_history_xyz, ego_history_rot, camera_indices),
runs Alpamayo-1.5 inference, converts the expert trajectory to AutoVLA
codebook indices, and returns them.

Usage (GPU 2)::

    CUDA_VISIBLE_DEVICES=2 HF_HOME=... HF_HUB_OFFLINE=1 \
    PYTHONPATH=... python expert_service.py --port 5557 \
        --model-path /tmp/alpamayo_model \
        --codebook-path .../agent_vocab.pkl

The client side is ``expert_client.py`` (async, fire-and-forget with
future collection).
"""

from __future__ import annotations

import faulthandler
import io
import pickle
import signal
import sys
import time
from pathlib import Path

faulthandler.enable()
signal.signal(signal.SIGPIPE, signal.SIG_IGN)

import numpy as np
import torch
import zmq

sys.path.insert(0, str(Path(__file__).parent))
from trajectory_encoder import (
    load_codebook,
    expert_traj_to_autovla_format,
    encode_trajectory_to_tokens,
    codebook_index_to_token_id,
)


def _build_expert_inputs(
    camera_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    processor,
    device: str,
):
    """Build Alpamayo-1.5 model inputs from raw observation tensors."""
    from alpamayo1_5 import helper
    from PIL import Image

    n_total = camera_frames.shape[0]
    unique_cams = torch.unique(camera_indices)
    n_cams = unique_cams.shape[0]
    frames_per_cam = n_total // n_cams

    pil_frames = []
    for i in range(n_total):
        frame = camera_frames[i]
        if frame.ndim == 3 and frame.shape[0] in (3, 4):
            arr = frame.numpy().transpose(1, 2, 0)
        else:
            arr = frame.numpy()
        pil_frames.append(Image.fromarray(arr.astype(np.uint8)))

    messages = helper.create_message(
        frames=camera_frames,
        camera_indices=unique_cams,
        num_frames_per_camera=frames_per_cam,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    if ego_history_xyz.dim() == 3:
        ego_history_xyz = ego_history_xyz.unsqueeze(1)
    if ego_history_rot.dim() == 4:
        ego_history_rot = ego_history_rot.unsqueeze(1)

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    return helper.to_device(model_inputs, device)


def _run_expert_inference(model, model_inputs, device: str):
    """Run Alpamayo-1.5 inference and return expert xyz/rot on CPU."""
    torch.cuda.manual_seed_all(42)
    with torch.autocast(device, dtype=torch.bfloat16):
        with torch.inference_mode():
            pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
            )
    pred_xyz = pred_xyz[0, 0, 0]
    pred_rot = pred_rot[0, 0, 0]
    return pred_xyz.cpu(), pred_rot.cpu()


def serve(port: int, model_path: str, codebook_path: str, device: str = "cuda") -> None:
    """Run the expert service loop."""
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper

    print(f"Loading codebook from {codebook_path}...")
    codebook = load_codebook(codebook_path)
    print(f"Codebook: {codebook.shape}")

    print(f"Loading Alpamayo-1.5 from {model_path}...")
    model = Alpamayo1_5.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(f"tcp://*:{port}")
    print(f"Expert service listening on tcp://*:{port}")

    while True:
        try:
            parts = sock.recv_multipart()
        except KeyboardInterrupt:
            print("Shutting down expert service.")
            break

        identity = parts[0]
        msg = parts[-1]

        try:
            buf = io.BytesIO(msg)
            payload = pickle.load(buf)

            camera_frames = payload["camera_frames"]
            camera_indices = payload["camera_indices"]
            ego_history_xyz = payload["ego_history_xyz"]
            ego_history_rot = payload["ego_history_rot"]

            if not isinstance(camera_frames, torch.Tensor):
                camera_frames = torch.as_tensor(camera_frames)
            if not isinstance(camera_indices, torch.Tensor):
                camera_indices = torch.as_tensor(camera_indices)
            if not isinstance(ego_history_xyz, torch.Tensor):
                ego_history_xyz = torch.as_tensor(ego_history_xyz)
            if not isinstance(ego_history_rot, torch.Tensor):
                ego_history_rot = torch.as_tensor(ego_history_rot)

            t0 = time.time()
            model_inputs = _build_expert_inputs(
                camera_frames, camera_indices,
                ego_history_xyz, ego_history_rot,
                processor, device,
            )
            expert_xyz, expert_rot = _run_expert_inference(model, model_inputs, device)

            expert_xyz_np = expert_xyz.numpy()
            expert_rot_np = expert_rot.numpy()
            target_traj = expert_traj_to_autovla_format(
                expert_xyz_np, expert_rot_np,
                expert_dt=0.1, autovla_dt=0.5, num_poses=10,
            )
            action_indices = encode_trajectory_to_tokens(codebook, target_traj, num_poses=10)
            action_token_ids = codebook_index_to_token_id(action_indices)
            elapsed = time.time() - t0

            response = pickle.dumps({
                "action_indices": action_token_ids.tolist(),
                "expert_xyz": expert_xyz_np,
                "expert_rot": expert_rot_np,
                "elapsed": elapsed,
            })

            del model_inputs
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            response = pickle.dumps({"error": str(e)})

        try:
            sock.send_multipart([identity] + parts[1:-1] + [response])
        except Exception:
            pass

    sock.close()
    ctx.term()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Alpamayo-1.5 expert inference service")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--model-path", default="/tmp/alpamayo_model")
    parser.add_argument(
        "--codebook-path",
        default="/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA/codebook_cache/agent_vocab.pkl",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    serve(args.port, args.model_path, args.codebook_path, args.device)


if __name__ == "__main__":
    main()
