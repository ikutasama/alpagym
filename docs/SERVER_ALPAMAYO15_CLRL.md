# Alpamayo 1.5 双卡闭环强化学习

这套入口面向一台 Linux x86_64 GPU 服务器：GPU 0 运行 Cosmos-RL 的策略与
rollout worker，GPU 1 运行 AlpaSim。建议两张卡各有至少 40 GiB 显存；官方
smoke 配置已在 2 x RTX 6000 Ada（每张 50 GiB）上验证。

## 1. 拉取与准备

```bash
git clone <你的-fork-url>
cd alpagym
hf auth login
scripts/server_clrl.sh prepare
```

`prepare` 会检查系统依赖、执行锁定版本的 `uv sync`、下载
`nvidia/Alpamayo-1.5-10B`，并转换为 Cosmos-RL 能训练的 checkpoint。过程可
重复执行；已经完成的下载和转换会跳过。

模型约 21 GB，Python/CUDA 环境、容器镜像与缓存还需要约 100–150 GB；每个
NuRec scene 约 1.5 GB。NuRec 是 gated dataset，运行前需要在 Hugging Face
页面获批并完成 `hf auth login`。

## 2. 先跑一次 smoke

```bash
scripts/server_clrl.sh smoke
```

它执行一个 episode 和一个训练 step。产物在 `tmp/alpagym-runs/`，Hydra 日志
在 `outputs/`。只有这一步完整结束后再开始多步训练：

```bash
MAX_NUM_STEPS=50 scripts/server_clrl.sh train
```

训练 preset 每 5 step 保存 Cosmos resume checkpoint 和 safetensors，默认保留
最近 3 份。可用 `SCENE_ID=clipgt-...` 替换默认场景；要启用 W&B：

```bash
export WANDB_API_KEY=...
ENABLE_WANDB=1 scripts/server_clrl.sh train
```

## 3. NCCL 卡住时

脚本会先做 45 秒双卡 P2P 探测。如果直接 P2P 不通，改用共享内存传输：

```bash
NCCL_P2P_DISABLE=1 scripts/server_clrl.sh smoke
NCCL_P2P_DISABLE=1 MAX_NUM_STEPS=50 scripts/server_clrl.sh train
```

若进程崩溃后 GPU 仍被占用，按官方 onboarding 的清理命令回收残留进程和
Wizard 容器，再重新执行 smoke。

## 4. 单机 8 x H200

H200 preset 将 GPU 0 分配给训练策略，GPU 1–3 分配给三个 rollout replica，
GPU 4–7 分配给四路 AlpaSim。先执行一个训练 step 的全链路验证：

```bash
scripts/server_clrl.sh smoke-h200
```

通过后运行默认 50 step 训练，或显式设置步数：

```bash
MAX_NUM_STEPS=50 scripts/server_clrl.sh train-h200
```

默认每组生成 6 条闭环轨迹，由三个 rollout GPU 各处理两条，trainer 每次消费
完整的 6 条 GRPO group。不要设置 `CUDA_VISIBLE_DEVICES` 重排物理卡号，因为
AlpaSim topology 明确占用物理 GPU 4–7。

默认单 scene 只适合验证。正式训练可切换到已获权限的 NuRec suite：

```bash
TEST_SUITE_ID=public_2507 MAX_NUM_STEPS=50 scripts/server_clrl.sh train-h200
```
