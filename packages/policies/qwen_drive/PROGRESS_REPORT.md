# Qwen-Drive-1.0 AlpaGym 适配□□进度报告

## 日期: 2025-09-17

---

## 丢□、□□体目标

□Qwen-Drive-1.0 Planning Expert（SFT 训练完成）接□AlpaGym 闭环仿真框架，实现：
1. **Phase 3（当前）**: 纯推理基□□加载 SFT 模型，在 AlpaSim 仿真环境中运行闭环评□2. **Phase 4（未来）**: GRPO 强化学习训练 □加入 flow-matching logprob 计算□Cosmos-RL 训练桥接

---

## 二□□已完成工作

### 2.1 Qwen-Drive SFT 训练（已完成□- **模型**: Qwen-Drive-1.0 Planning Expert（flow-matching 轨迹生成□- **保守 10K checkpoint** 为最□SFT 模型
- 路径: `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/`
- 备份: NFS `qwen-drive-sft/checkpoints/conservative_10k_final/`
- 完整模型（VLM+planner□ `/tmp/qd_model/` (11G)

### 2.2 AlpaGym 适配器开发（已完成代码编写）
创建□`alpagym-qwen-drive` 策略插件包，位于 `packages/policies/qwen_drive/`□
#### 文件清单
| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 包定义，entry-points: `qwen_drive = alpagym_qwen_drive.bundle:get_bundle` |
| `src/alpagym_qwen_drive/bundle.py` | PolicyBundle 5 个钩子（setup_tokenizer, build_data_packer, install_runtime_bridge, load_inference_model, build_model_inputs□|
| `src/alpagym_qwen_drive/inference_model.py` | QwenDriveInferenceModel □实现 InferenceModel 协议（sample_trajectories_from_data, build_policy_replay_data, get_model, set_model□|
| `src/alpagym_qwen_drive/configs/policy/qwen_drive.yaml` | 策略配置□ cameras, 4 context frames, 50 future waypoints, step_dt_us=100000□|
| `src/alpagym_qwen_drive/configs/experiment/qwen_drive_a100_1gpu_inference.yaml` | 实验配置□ GPU inference, rollout_replicas=1, policy_replicas=0□|

#### 适配器架□```
AlpaSim □PolicyInput □AlpamayoPolicy._preprocess() □BatchedModelInput
                                                              □                                    QwenDriveInferenceModel.sample_trajectories_from_data()
                                                              □                                    _infer_single() per batch row:
                                      1. _build_camera_views() □DrivingScene views
                                         (uint8 CHW □PIL □CameraFrame, 3 cameras × 4 frames)
                                      2. _extract_ego_history() □history/velocity/acceleration
                                         (ego_history_xyz + ego_history_rot □[H,3] + heading)
                                      3. route_to_nav_command() □nav_command (0/1/2)
                                      4. Build DrivingScene □model.generate_trajectory()
                                      5. trajectories [N,50,3] □pred_xyz [1,K,T,3] + pred_rot [1,K,T,3,3]
                                                              □                                    BatchedModelOutput □AlpamayoPolicy._postprocess() □PolicyOutput □AlpaSim
```

#### 关键设计决策
- **logprob = None**: Phase 3 纯推理，不计□flow-matching 密度
- **policy_replicas = 0**: 无训练进□- **install_runtime_bridge() = no-op**: 不安□Cosmos-RL 桥接
- **heading_to_rotation_matrix()**: Qwen-Drive 输出 (x, y, heading)，转换为 AlpaGym 霢□要的 SO(3) 旋转矩阵
- **route_to_nav_command()**: 启发式将连续 route waypoints 转为离散导航命令（GO STRAIGHT / TURN LEFT / TURN RIGHT□- **bundle.py 直接导入**: `from alpagym_runtime.policies.registry import PolicyBundle`（venv 修复□cosmos_rl 可用，无霢□懒加载）

### 2.3 pyproject.toml 修改
- 添加 `"packages/policies/qwen_drive"` □`[tool.uv.workspace] members`
- flash-attn/torch 源保持原□URL（本□wheel 路径仅在本地使用，不提交□
### 2.4 AlpaGym venv 恢复（已完成□- **问题**: `uv sync` 之前因网络中断破坏了 venv
- **解决**: 
  - 设置代理 `http://z59900495:753951tc-@proxysg.huawei.com:8080`
  - flash-attn/torch 改用本地 wheel
  - venv 创建□`/tmp/alpagym_venv`（本地文件系统，避免 NFS I/O 瓶颈□  - 使用 `UV_PROJECT_ENVIRONMENT=/tmp/alpagym_venv` 指定 venv 路径
  - `UV_LINK_MODE=copy` 避免 hardlink 跨文件系统问□- **验证通过**: torch 2.8.0+cu128, flash_attn 2.8.3, hydra 1.3.2, redis, cosmos_rl, alpagym_runtime 全部导入成功
- **qwen_drive bundle 加载成功**: `get_policy_bundle('qwen_drive')` 返回 `PolicyBundle` 实例

### 2.5 transformers 版本升级（已完成□- **问题**: Qwen-Drive-1.0 模型 config 使用 `model_type: qwen3_5`，需□`transformers>=5.14.0`，但 AlpaGym 原始环境使用 `transformers==4.57.1`
- **解决**:
  - 升级□`transformers==5.14.1`（从 PyPI 直接下载 wheel 安装，因 uv cache 有损坏的 wheel□  - 升级 `safetensors==0.8.0`（同样从 PyPI 直接下载 wheel□  - 降级 `huggingface_hub==1.5.0`（transformers 5.14.1 霢□□>=1.5.0, □1.31.0 删除□`is_offline_mode`□  - 保持 `diffusers==0.37.1`□.40.0 霢□□`get_cached_repo_tree` 不兼□hub 1.5.0□  - 保持 `click==8.3.3`□.5.0 删除□`click.command` 装饰器）
  - 保持 `datasets==5.0.1`
- **验证通过**: cosmos_rl 导入成功，qwen_drive bundle 加载成功

### 2.6 Qwen-Drive 模型加载验证（已完成□- **成功加载** `QwenDriveForPlanning.from_pretrained()`:
  - VLM: Qwen3.5 (5579.1M total params)
  - Planning Expert: PlanningExpert (separate SFT checkpoint)
  - 723 weight files loaded in <1s
  - Model moved to GPU (cuda:1), eval mode set
- **模型结构确认**:
  - `model.vlm`: VLM 主干 (AutoModelForImageTextToText)
  - `model.planning_expert`: Planning Expert (flow-matching trajectory generator)
  - `model.processor`: 懒加载的 QwenDriveProcessor (property)
  - `model.generate_trajectory(scene, mode, num_samples, num_steps, seed)` □`QwenDriveOutput(trajectories=[N,50,3])`

---

## 三□□当前进行中

### 3.1 霢□要外□AI 辅助的问□
#### 问题 1: pyproject.toml 本地 wheel 路径（已解决□**原始问题**: 代理环境□GitHub releases □PyTorch 下载站有 SSL 证书问题□invalid peer certificate: UnknownIssuer"）□□**当前方案**: pyproject.toml 保持原始 URL，不提交本地 wheel 路径。本地□□过临时修改 pyproject.toml + 本地 wheel 文件来完□`uv sync`，sync 完成后恢复原□URL□**注意事项**: 如果霢□要重□`uv sync`，需要再次临时改为本□wheel 路径。本□wheel 文件位于 alpagym 根目录□□
#### 问题 2: venv 位置□/tmp 是否合□□？
当前 venv □`/tmp/alpagym_venv`（本□overlay 文件系统），因为 NFS 上的 venv 创建/同步极慢（每□uv sync 超时 5 分钟以上）□□**风险**: `/tmp` 是容器内临时文件系统，容器重启后 venv 会丢失□□**缓解**: 已创□`/tmp/activate_alpagym.sh` 濢□活脚本，重启后只霢□重新运行 `uv sync` 即可恢复□**问题**: 是否有更好的位置？或□NFS □I/O 性能问题是暂时□□的□
#### 问题 3: bundle.py 导入方式（已解决□**原始问题**: `bundle.py` 在模块顶部直□`from alpagym_runtime.policies.registry import PolicyBundle`，会触发 `registry.py` □`cosmos.packer` □`cosmos_rl` 整个依赖链加载□□**当前方案**: 保持直接导入（原始方式）。venv 修复□cosmos_rl 可用，`get_policy_bundle('qwen_drive')` 已验证成功□□
#### 问题 4: autovla 训练任务共用 venv
当前有另丢□□autovla 训练任务（PID 2472853）正在使用同丢□□`.venv/bin/python3`，□□过 `uv run --no-sync` 启动□- 该进程在 venv 被破坏前已启动，模块已加载到内存□- venv 恢复后（□/tmp），旧的 `.venv` 路径不再指向有效 venv
- **问题**: autovla 进程是否会因 venv 路径变化而崩溃？如果它需要重□import 模块（例□fork 子进程）会□□样□
#### 问题 5: transformers 版本冲突（已解决□**原始问题**: Qwen-Drive-1.0 模型霢□□`transformers>=5.14.0`（config 中使□`model_type: qwen3_5`），□AlpaGym 原始环境使用 `transformers==4.57.1`□**当前方案**: 升级□`transformers==5.14.1`。uv cache 中有损坏□wheel（标□5.14.1 但实际包□4.57.1 代码），通过直接□PyPI 下载 wheel 文件安装解决。同时需要升□`huggingface_hub` □1.x、`datasets` □5.x、`diffusers` 到最新版本以保持兼容□**风险**: transformers 5.x 是大版本升级，可能与 cosmos_rl 或其□AlpaGym 依赖不兼容□□需要验证完整导入链□
#### 问题 6: AlpaSim 连接共享
- AlpaSim 在远程服务器 `mti@10.174.175.151` 上运□- 当前通过 SSH 隧道暴露端口 5011（runtime server）和 5013（反向□□道□- autovla 任务正在使用这个隧道进行训练
- AlpaSim capacity=4，理论上支持 4 个并□session
- **问题**: 是否可以共用同一个隧道？还是霢□要建立第二个 SSH 隧道到不同端口？
- **阻碍**: 直接 SSH □10.174.175.151 返回 "Permission denied"，密码未□
---

## 四□□下丢□步计□
1. ~~**Git push 当前修改**~~ □已完□(commit 6e3c6e0)
2. ~~**验证 Qwen-Drive 模型加载**~~ □已完□(5.6B params loaded on GPU)
3. **建立 AlpaSim 连接** □端口 5011 已确认开放，霢□测试闭环推理能否连接
4. **运行闭环推理** □□qwen_drive_a100_1gpu_inference 配置启动闭环评估
5. **调试 inference_model.py** □验证 camera view 构建、ego history 提取、trajectory 后处理是否正□6. **端到端测□* □□AlpaSim 获取丢□帧数□□构建 DrivingScene □generate_trajectory □返回 PolicyOutput

---

## 五□□环境信□
| 项目 | □|
|------|-----|
| AlpaGym 路径 | `/data/mnt_m62/10_personal/z59900495/workspace/alpagym` |
| Qwen-Drive 上游源码 | `/data/mnt_m62/10_personal/z59900495/workspace/a_0914_qwendrive/Qwen-Drive-1.0/src/` |
| Qwen-Drive 完整模型 | `/tmp/qd_model/` (11G) |
| SFT checkpoint | `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/` |
| venv 路径 | `/tmp/alpagym_venv` (UV_PROJECT_ENVIRONMENT) |
| 濢□活脚□| `/tmp/activate_alpagym.sh` |
| GPU 使用 | GPU 1 (空闲), GPU 5-7 (autovla 训练□ |
| AlpaSim | `mti@10.174.175.151`, 端口 5011/5013 via SSH tunnel |
| 代理 | `http://z59900495:753951tc-@proxysg.huawei.com:8080` |
| Git remote | `https://ikutasama:***@github.com/ikutasama/alpagym.git` |
| Git branch | `main` |

---

## 六□□文件变更清□
### 新增文件
- `packages/policies/qwen_drive/pyproject.toml`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/__init__.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/inference_model.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/__init__.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/policy/qwen_drive.yaml`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/experiment/qwen_drive_a100_1gpu_inference.yaml`

### 修改文件
- `pyproject.toml` □仅添□qwen_drive workspace member（flash-attn/torch 源保持原□URL□- `uv.lock` □uv sync 自动更新

### 未修改（保持原状□- `packages/runtime/src/alpagym_runtime/policies/registry.py` □无修□- `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py` □直接导入 PolicyBundle（无懒加载）

---

## 七□□需要外□AI 棢□视的重点

1. **适配器代码正确□□*: `inference_model.py` 中的 camera view 构建、ego history 提取、trajectory 后处理□□辑是否正确□2. **配置文件完整□*: `qwen_drive.yaml` □`qwen_drive_a100_1gpu_inference.yaml` 是否遗漏必要字段□3. **pyproject.toml 变更**: 仅添□qwen_drive workspace member，flash-attn/torch 源保持原□URL（已解决□4. **venv 策略**: /tmp venv + UV_PROJECT_ENVIRONMENT 方案是否可行？有无更好的持久化方案？
5. **AlpaSim 共用**: □autovla 训练任务共用 AlpaSim 隧道的可行□□和风险□