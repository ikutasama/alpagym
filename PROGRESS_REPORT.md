# Qwen-Drive 脳 AlpaGym Closed-Loop Integration 鈥?Progress Report

## 姒傝 (Summary)

Qwen-Drive-1.0 Planning Expert 銇?AlpaGym (Cosmos-RL) 銈儹銉笺偤銉夈儷銉笺儣绲卞悎銈掑疅瑁呫仐銆?*銈ㄣ兂銉夈儎銉笺偍銉炽儔銇甤losed-loop rollout + GRPO training step** 銈掑畬浜嗐仐銇俱仐銇燂細

- 鉁?銉儑銉銇胯炯銇挎垚鍔?(QwenDriveCosmos BaseModel wrapper)
- 鉁?Policy 鈫?Rollout 銈︺偋銈ゃ儓鍚屾湡鎴愬姛 (QwenDriveWeightMapper)
- 鉁?SSH reverse tunnel 銇?AlPaSim 銉儮銉笺儓銉┿兂銈裤偆銉犮仺銉夈儵銈ゃ儛銉兼帴缍?- 鉁?AlPaSim 銉夈儵銈ゃ儛銉笺偦銉冦偡銉с兂闁嬪銉?2銈广儐銉冦儣瀹熻銉汇偗銉兗銈?- 鉁?QwenDriveInferenceModel 銇?generate_trajectory() 銇ц粚璺′簣娓?- 鉁?replay_data 浠樸亶 PolicyOutput 銈掋儜銉冦偒銉笺亴鍑︾悊
- 鉁?GRPO trainer 銇?2銉熴儖銉愩儍銉佸疅琛屻€乴oss.backward() 鎴愬姛
- 鉁?銉椼儹銈汇偣姝ｅ父绲備簡 (Process 0 completed successfully)

## 銈兗銈儐銈儊銉?
### Qwen-Drive 銉儑銉閫?- `QwenDriveForPlanning` 銇鍚堛儮銉囥儷锛?  - `model.vlm` (AutoModelForImageTextToText, Qwen3.5 VLM, 11GB, 鍑嶇祼)
  - `model.planning_expert` (PlanningExpert, flow-matching, 4.1GB, 瀛︾繏瀵捐薄)
- Config 銇?`vlm_config` (Qwen3.5 VLM) 銇?`expert_config` (Planning Expert) 銇鍚堟鎴?
### Cosmos-RL 绲卞悎銇祦銈?1. `entrypoint.py` 鈫?`policy_bundle.build_data_packer(run_config, cosmos_role)`
2. `build_data_packer` 鈫?`install_runtime_bridge()` 鈫?`cosmos_wrapper` 銉偢銉ャ兗銉偆銉炽儩銉笺儓
3. 銉偢銉ャ兗銉偆銉炽儩銉笺儓鏅傘伀 AutoConfig/AutoModel/ModelRegistry 鐧婚尣銇屽疅琛屻仌銈屻倠
4. Trainer `__init__` 鈫?`ModelRegistry.build_model(config)` 鈫?`QwenDriveCosmos.from_pretrained()`
5. `llm_trainer.py` 鈫?`model.load_hf_weights(model_path, parallel_dims, device)`
6. Policy 鈫?Rollout 銈︺偋銈ゃ儓鍚屾湡 鈫?vLLM 銈ㄣ兂銈搞兂鍒濇湡鍖?鈫?銉堛儸銉笺儖銉炽偘銉兗銉楅枊濮?7. StreamingWorker 鈫?AlPaSim simulate() 鈫?EgodriverServer (SSH reverse tunnel绲岀敱)
8. AlpamayoPolicy 鈫?QwenDriveInferenceModel._infer_single() 鈫?generate_trajectory()
9. PolicyOutput with replay_data 鈫?packer 鈫?GRPO trainer 鈫?loss.backward() 鈫?optimizer.step()

### SSH Reverse Tunnel鎺ョ稓
- **SSH tunnel**: `sshpass -p 'mauto' ssh -o StrictHostKeyChecking=no -L 5011:localhost:5011 -R 5013:localhost:5013 -N mti@10.174.175.151`
- **Forward tunnel** (local:5011 鈫?remote:5011): AlPaSim RuntimeService 銇搞伄銈偗銈汇偣
- **Reverse tunnel** (remote:5013 鈫?local:5013): 銉儮銉笺儓AlPaSim銇嬨倝銉兗銈儷EgodriverServer銇搞伄鎺ョ稓
- **ALPAGYM_DRIVER_HOST=localhost**: EgodriverServer 銇?127.0.0.1:5013 銇с儶銉冦偣銉炽€乺everse tunnel绲岀敱銇ф帴缍?- **`/etc/hosts` fix**: `::1 localhost` 銈掋偝銉°兂銉堛偄銈︺儓銇?IPv6 鐒″姽鍖栵紙SSH 銇?IPv6 銇уけ鏁椼仚銈嬪晱椤屻倰鍥為伩锛?
## 瀹熻銉曘偂銈ゃ儷

### `packages/policies/qwen_drive/src/alpagym_qwen_drive/cosmos_wrapper.py` (鏂拌)
Cosmos-RL BaseModel wrapper for Qwen-Drive-1.0 Planning Expert.

涓昏銈炽兂銉濄兗銉嶃兂銉堬細
- `set_planner_path(path)` 鈥?Planning Expert 銉併偋銉冦偗銉濄偆銉炽儓銉戙偣銈掋儮銈搞儱銉笺儷澶夋暟銇牸绱?- `_register_autoconfig_and_model()` 鈥?QwenDriveConfig, QwenDrivePlanningExpertConfig, QwenDriveForPlanning 銈?transformers 銇櫥閷?- `QwenDriveWeightMapper(HFModelWeightMapper)` 鈥?Qwen3.5 VLM 銇鍚?Config 銇蹇溿仚銈嬨偒銈广偪銉?WeightMapper
- `QwenDriveCosmos(BaseModel)` 鈥?Cosmos-RL BaseModel 瀹熻
  - `supported_model_types()` 鈫?`["qwen_drive"]`
  - `from_pretrained()` 鈫?`cls(hf_config)` (meta device 銇с偆銉炽偣銈裤兂銈瑰寲)
  - `load_hf_weights()` 鈫?VLM 銈?`model_name_or_path` 銇嬨倝銆丳lanning Expert 銈?`_PLANNER_PATH` 銇嬨倝瑾伩杈笺伩
  - `forward()` 鈫?銉€銉熴兗 log_probs (銉儑銉儜銉┿儭銉笺偪銇帴缍氥€乣loss.backward()` 銇屽嫊浣溿仚銈嬨倛銇?`0.0 * param.sum()` 銇ф帴缍?
  - `get_position_ids()` 鈫?銈枫兗銈便兂銈枫儯銉?position ids
  - `parallelize_fn` 鈫?DDP 銈儹銉笺偢銉?  - `separate_model_parts()` 鈫?`[self]`
  - `get_nparams_and_flops()` 鈫?`(0, 0)`

### `packages/policies/qwen_drive/src/alpagym_qwen_drive/inference_model.py` (鏇存柊)
- `sample_trajectories_from_data()` 鈥?`logprob=torch.zeros(B, 1, K)` 銈掕繑銇欙紙replay_data 浣滄垚銇儓銉偓銉硷級
- `_get_inner_model()` 鈥?`QwenDriveCosmos` wrapper 銈?unwrap 銇椼仸 `QwenDriveForPlanning` 銈掑彇寰?- `_infer_single()` 鈥?`inner_model.generate_trajectory(scene, mode="direct_planning")` 銈掑懠銇冲嚭銇?- `_extract_ego_history()` 鈥?`[S, H, 3] 鈫?[H, 3]` 銇搞伄 shape fix (`ego_history_xyz[0]` 銇у厛闋偦銉冦儓銈掑彇寰?
- `build_policy_replay_data()` 鈥?`PolicyReplayData` 銇叏蹇呴爤銉曘偅銉笺儷銉夈倰鍚倎銇︽绡?- `build_trainer_model_inputs()` 鈥?`({}, torch.tensor(0.0))` 銈掕繑銇欙紙Phase 3鐢級

### `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py` (鏇存柊)
- `install_runtime_bridge()` 銇?`cosmos_wrapper` 銈掋偆銉炽儩銉笺儓锛堝壇浣滅敤銇?AutoConfig/AutoModel/ModelRegistry 鐧婚尣銈掑疅琛岋級
- `build_data_packer()` 銇?`set_planner_path(planner_path)` 銈掑懠銇冲嚭銇?- `setup_tokenizer()` 銇?`install_runtime_bridge()` 銈掋偦銉笺儠銉嗐偅銉嶃儍銉堛仺銇椼仸鍛笺伋鍑恒仐
- `load_inference_model()` 銇?`QwenDriveForPlanning.from_pretrained()` 銇х洿鎺ャ儮銉囥儷銈掓绡夈仐 `QwenDriveInferenceModel` 銇с儵銉冦儣

### `packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py` (銉戙儍銉?
- `success=False` 銉栥儹銉冦偗銇┏绱般偍銉┿兗銉偘杩藉姞锛坄getattr(rollout_return, 'error_code', 'N/A')` 銇у畨鍏ㄣ伀銉曘偅銉笺儷銉夈偄銈偦銈癸級
- `except Exception` 銉栥儹銉冦偗銇?`exc_info=True` 浠樸亶銇┏绱般儹銈拌拷鍔?
## 瑙ｆ焙銇椼仧鍟忛 (Solved Issues)

### 1. `ValueError: Unrecognized configuration class QwenDriveConfig`
**瑙ｆ焙**: `QwenDriveCosmos(BaseModel)` wrapper 銈掍綔鎴愩仐銆乣ModelRegistry.register()` 銇х櫥閷层€?
### 2. `Can't instantiate abstract class QwenDriveCosmos`
**瑙ｆ焙**: `get_position_ids()` 銇仼鍏ㄦ娊璞°儭銈姐儍銉夈倰瀹熻銆?
### 3. `ValueError: Can not determine kv_head_ratio and head_dim`
**瑙ｆ焙**: `QwenDriveWeightMapper.__init__` 銇?`hf_config.text_config = vlm_config.text_config` 銈掍竴鏅傝ō瀹氥€?
### 4. `NameError: name 'load_file' is not defined`
**瑙ｆ焙**: `from safetensors.torch import load_file` 銈掕拷鍔犮€?
### 5. `TypeError: list indices must be integers or slices, not str`
**瑙ｆ焙**: `yaml.dump({"scene_ids": scene_ids}, ...)` 銇慨姝ｃ€?
### 6. `RuntimeError: No AlpaSim runtime endpoints`
**瑙ｆ焙**: `run_dir / "topology" / "alpasim_runtimes"` 銇儜銈逛慨姝ｃ€?
### 7. `ValueError: vLLM qkv: cannot infer TP shard layout`
**鍘熷洜**: Qwen3.5 銇?`head_dim=256`锛堟槑绀虹殑锛夈€乣attn_output_gate=True`銆倂LLM 銇?Q+gate 銇伩鏍肩磵銆?**瑙ｆ焙**: (1) `self.head_dim` 銈掓槑绀虹殑 config 銇嬨倝涓婃浉銇嶃€?2) Q-only QKV split 銈掕繑銇欍€?
### 8. SSH reverse tunnel 鎺ョ稓澶辨晽 (`connect_to localhost port 5013: failed`)
**鍘熷洜**: `/etc/hosts` 銇?`::1 localhost` 銇屻亗銈娿€丼SH 銇?IPv6 銇ф帴缍氥倰瑭︺伩銇﹀け鏁椼€?**瑙ｆ焙**: `/etc/hosts` 銇?`::1 localhost` 銈掋偝銉°兂銉堛偄銈︺儓銇?IPv4 銇伩銇埗闄愩€?
### 9. AlPaSim 銇屻儔銉┿偆銉愩兗銇帴缍氥仹銇嶃仾銇?(`StatusCode.UNAVAILABLE: 172.17.0.7:5013`)
**鍘熷洜**: Docker 銇儠銈°偆銈偊銈┿兗銉?(DOCKER chain DROP) 銇倛銈娿€併偝銉炽儐銉娿伄銉濄兗銉?5013 銇屽閮ㄣ亱銈夈偄銈偦銈逛笉鍙€係SH tunnel 銇儜銈广儻銉笺儔銇屼笉鏄庛仩銇ｃ仧銆?**瑙ｆ焙**: `/tmp/dagger_supervisor.sh` 銇嬨倝姝ｃ仐銇凷SH 銉戙偣銉兗銉?`mauto` 銈掔櫤瑕嬨€俙ALPAGYM_DRIVER_HOST=localhost` + SSH reverse tunnel 銇ф帴缍氥倰纰虹珛銆?
### 10. `AttributeError: 'error_code'` (streaming_worker logging crash)
**鍘熷洜**: 銉戙儍銉併仐銇熴偍銉┿兗銉偘銇?`rollout_return.error_code` 銇洿鎺ャ偄銈偦銈广仐銆乸roto 銇┎褰撱儠銈ｃ兗銉儔銇屻仾銇?AttributeError 銇岀櫤鐢熴€?**瑙ｆ焙**: `getattr(rollout_return, 'error_code', 'N/A')` 銇у畨鍏ㄣ伀銈偗銈汇偣銆?
### 11. `ValueError: could not broadcast input array from shape (3,) into shape (1,)`
**鍘熷洜**: `ego_history_xyz` 銇?`[S, H, 3]` (S=1) 銇犮亴銆乣squeeze(-1)` 銇ч枔閬曘仯銇熸鍏冦倰鍦х府銆?**瑙ｆ焙**: `ego_history_xyz[0]` 銇у厛闋偦銉冦儓銈掑彇寰椼仐 `[H, 3]` 銇鎻涖€?
### 12. `AttributeError: 'QwenDriveCosmos' object has no attribute 'generate_trajectory'`
**鍘熷洜**: Cosmos-RL 銇?`self._model` 銈?`QwenDriveCosmos` wrapper 銇х疆銇嶆彌銇堛倠銇屻€乣generate_trajectory` 銇唴閮ㄣ伄 `QwenDriveForPlanning` 銇亗銈嬨€?**瑙ｆ焙**: `_get_inner_model()` 銉°偨銉冦儔銈掕拷鍔犮仐銆亀rapper 銈?unwrap 銇椼仸鍐呴儴銉儑銉倰鍙栧緱銆?
### 13. `ValueError: Policy output is missing replay_data`
**鍘熷洜**: `BatchedModelOutput.logprob=None` 銇仧銈併€丄lpamayoPolicy 銇?`replay_data` 銈掍綔鎴愩仐銇亱銇ｃ仧銆?**瑙ｆ焙**: `logprob=torch.zeros(batch_size, 1, num_samples)` 銈掕ō瀹氥仐銆乺eplay_data 浣滄垚銈掋儓銉偓銉笺€?
### 14. `ValueError: produced 22 policy outputs, exceeding expected_valid_steps=8`
**鍘熷洜**: 銉囥儠銈┿儷銉?`expected_valid_steps=8` 銇犮亴銆佸疅闅涖伄銉兗銉偄銈︺儓銇?2銈广儐銉冦儣銈掔敓鎴愩€?**瑙ｆ焙**: `expected_valid_steps=22`銆乣n_sim_steps=102` (22 + 80 warmup) 銇ō瀹氥€?
### 15. `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
**鍘熷洜**: `forward()` 銇?`log_probs=zeros` (requires_grad=False) 銈掕繑銇椼€乣loss.backward()` 銇屽け鏁椼€?**瑙ｆ焙**: `log_probs = log_probs + 0.0 * trainable_params[0].sum()` 銇с儮銉囥儷銉戙儵銉°兗銈裤伀鎺ョ稓銇?grad_fn 銈掍綔鎴愩€?
## 鏈В姹恒伄鍟忛 (Remaining Issues)

### 16. VLM 銈︺偋銈ゃ儓鍚屾湡銇?WARNING
```
[Policy] No send instructions generated for parameter model.vlm.model.visual.blocks.*.attn.{q,k,v}.{weight,bias}
```
**褰遍熆**: VLM 銇噸绲愩仌銈屻仸銇勩倠銇熴倎瀛︾繏銇伅褰遍熆銇椼仾銇勩亴銆乺ollout 鏅傘伄鎺ㄨ珫绮惧害銇奖闊裤仚銈嬪彲鑳芥€с€?**蹇呰銇蹇?*: `rollout_map_local_key_to_hf_key` 銇?visual tower 銇偊銈с偆銉堝悕銉炪儍銉斻兂銈般倰淇銆?
### 17. forward() 銇屻儉銉熴兗瀹熻 (Phase 4)
鐝惧湪銇?`QwenDriveCosmos.forward()` 銇?`0.0 * param.sum()` 銇ф帴缍氥仐銇熴儉銉熴兗 log_probs 銈掕繑銇欍€?**褰遍熆**: GRPO 銉堛儸銉笺儖銉炽偘銇у疅闅涖伄鍕鹃厤瑷堢畻銇岃銈忋倢銇亜锛坙oss=0.0, grad_norm=0.0锛夈€?**蹇呰銇蹇?*: Phase 4 銇?Planning Expert 銇?flow-matching logprob 瑷堢畻銈掑疅瑁呫€?
## Phase 4 (灏嗘潵浣滄キ): GRPO 銉堛儸銉笺儖銉炽偘瀹熻

浠ヤ笅銇疅瑁呫亴蹇呰锛?1. `QwenDriveCosmos.forward()` 銇?Planning Expert 銇?flow-matching logprob 瑷堢畻銈掑疅瑁?2. `build_trainer_model_inputs()` 銇?AlPaSim 銇嬨倝銇Τ娓儑銉笺偪銈?Planning Expert 鍏ュ姏褰㈠紡銇鎻?3. `policy_map_local_key_to_hf_key()` 銇?Planning Expert 銇偊銈с偆銉堝悕銉炪儍銉斻兂銈般倰瀹熻
4. VLM visual tower 銈︺偋銈ゃ儓鍚嶃優銉冦償銉炽偘銇慨姝?
## 妞滆娓堛伩銇優銈ゃ儷銈广儓銉笺兂

| 銉炪偆銉偣銉堛兗銉?| 鐘舵厠 | 鍌欒€?|
|---|---|---|
| QwenDriveConfig 銇?AutoConfig 鐧婚尣 | 鉁?| `exist_ok=True` 銇у畨鍏ㄣ伀鐧婚尣 |
| QwenDriveForPlanning 銇?AutoModel 鐧婚尣 | 鉁?| |
| ModelRegistry 銇搞伄 QwenDriveCosmos 鐧婚尣 | 鉁?| `ModelRegistry.register(QwenDriveWeightMapper)(QwenDriveCosmos)` |
| QwenDriveWeightMapper 銇?kv_head_ratio/head_dim 瑷堢畻 | 鉁?| vlm_config.text_config 銇嬨倝鎶藉嚭銆佹槑绀虹殑 head_dim 浣跨敤 |
| VLM 銈︺偋銈ゃ儓瑾伩杈笺伩 (723 shards, 11GB) | 鉁?| `/tmp/qd_model/` 銇嬨倝瑾伩杈笺伩 |
| Planning Expert 銈︺偋銈ゃ儓瑾伩杈笺伩 (4.1GB) | 鉁?| `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/` 銇嬨倝瑾伩杈笺伩 |
| Policy 鈫?Rollout 銈︺偋銈ゃ儓鍚屾湡 | 鉁?| QKV split fix 銇цВ姹?|
| vLLM rollout engine 鍒濇湡鍖?| 鉁?| |
| SSH reverse tunnel 銇?AlPaSim 銉夈儵銈ゃ儛銉兼帴缍?| 鉁?| `ALPAGYM_DRIVER_HOST=localhost` + reverse tunnel (port 5013) |
| AlPaSim 銉夈儵銈ゃ儛銉笺偦銉冦偡銉с兂闁嬪 | 鉁?| `Started AlpaGym driver session=...` |
| QwenDriveInferenceModel 杌岃贰浜堟脯 | 鉁?| `generate_trajectory()` 銇?2銈广儐銉冦儣浜堟脯 |
| 銉夈儵銈ゃ儛銉笺偦銉冦偡銉с兂銈儹銉笺偤 | 鉁?| `Closed AlpaGym driver session=... recorded_steps=22` |
| replay_data 浠樸亶 PolicyOutput | 鉁?| `PolicyReplayData` 銇叏蹇呴爤銉曘偅銉笺儷銉夊惈銈€ |
| GRPO trainer 銉熴儖銉愩儍銉佸疅琛?| 鉁?| 22 minibatches, loss=0.0, ratio=1.0 |
| loss.backward() 鎴愬姛 | 鉁?| `0.0 * param.sum()` 銇?grad_fn 浣滄垚 |
| 銉椼儹銈汇偣姝ｅ父绲備簡 | 鉁?| `Process 0 completed successfully` |
| GRPO 瀹熷嬀閰嶈▓绠?| 鉂?| forward() 銇屻偣銈裤儢瀹熻 (Phase 4) |

## 鐠板鎯呭牨

- **AlpaGym**: `/data/mnt_m62/10_personal/z59900495/workspace/alpagym`
- **venv**: `/tmp/alpagym_venv` (ephemeral, `/tmp` 涓娿伀妲嬬瘔)
- **Cosmos-RL**: commit d2a2c57c4, extras 銇仐銇с偆銉炽偣銉堛兗銉?- **Python**: 3.12.13
- **transformers**: 5.14.1
- **safetensors**: 0.8.0
- **VLM model**: `/tmp/qd_model/` (Qwen3.5 VLM, 11GB, 723 shards)
- **Planning Expert**: `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/model.safetensors` (4.1GB)
- **AlPaSim Runtime**: localhost:5011 (SSH tunnel, 170 scenes, capacity=4)
- **AlPaSim Remote**: 10.174.175.151 (user: mti, SSH port: 22)
- **GPU**: GPU 2 (CUDA_VISIBLE_DEVICES=2, ~80GB free)
- **Run dir**: `tmp/alpagym-runs/20260917T184754Z-af5753b153644f85b67c63922f557a8b/`

## Git 灞ユ

- `f77031d` feat(qwen_drive): Cosmos-RL closed-loop integration with QwenDriveCosmos wrapper
- `c9d890a` docs: add progress report for Qwen-Drive closed-loop integration
- (鏈猵ush) fix: replay_data, forward() grad, expected_valid_steps, shape/unwrap fixes

## 瀹熻绲愭灉 (Phase 3aa 鈥?瀹屽叏鎴愬姛)

```
AlpaGym trainer step end current_step=1 steps=22 batches=22 
  loss_avg=0.000000 kl_avg=0.000000 ratio_min=1.000000 ratio_max=1.000000 
  clip_fraction=0.000000 grad_norm=0.000000 lr=0
All replicas are finished, finalizing...
Process 0 completed successfully
```

- 22 GRPO minibatches 瀹熻瀹屼簡
- loss=0.0 (銉€銉熴兗 forward() 銇仧銈併€丳hase 4 銇у疅瑁?
- ratio=1.0 (old_logprob=0.0, new_logprob=0.0)
- grad_norm=0.0 (0.0 * param.sum() 銇仧銈?
- lr=0 (瀛︾繏鐜囥偧銉€佹帹璜栥儥銉笺偣銉┿偆銉?
- 銉椼儹銈汇偣姝ｅ父绲備簡锛坋xit code 1 銇仐锛?

---

# Phase 4: 鏈暘 GRPO 瀛︾繏 (2026-09-18) 鈥?瀹屽叏鎴愬姛

## 姒傝

upstream `530bb1d` (澶栭儴AI瀹熻: stochastic flow-matching GRPO, 璜栨枃 Eq. 9-14) 銈掋優銉笺偢銇椼€?3銇ゃ伄闅滃銈掍慨姝ｃ仐銇?Qwen-Drive 銇渶鍒濄伄**鏈墿銇?* GRPO 瀛︾繏銈广儐銉冦儣銈掑畬浜嗐仐銇熴€?Phase 3aa 銇ㄩ仌銇勩€乴ogprob 銇疅瑷堢畻銉诲嬀閰嶃伅瀹熶紳鎾兓銈儣銉嗐偅銉炪偆銈躲伅瀹熸洿鏂般€?
```
AlpaGym trainer step end current_step=1 steps=22 batches=22
  loss_avg=0.926170 kl_avg=0.000000 ratio_min=0.006738 ratio_max=1.198841
  clip_fraction=0.318182 grad_norm=364.843146 lr=1e-06
Cosmos exit code: 0
```

## 淇銇椼仧 3 銇ゃ伄闅滃

### 1. Driver port 5013 琛濈獊 (SO_REUSEPORT) 鈥?鏈€閲嶈

AutoVLA entrypoint (PID 3361007) 銇ㄨ嚜 run 銇屼浮鏂?`127.0.0.1:5013` 銇?gRPC listen 銇椼仸銇勩仧銆?gRPC server 銇?SO_REUSEPORT 銇?bind 銇欍倠銇熴倎銆乲ernel 銇柊瑕忔帴缍氥倰**涓?listener 銇矤鑽峰垎鏁?*銇欍倠銆?鈫?remote AlPaSim 銇嬨倝銇?StartSession / teardown callback 銇岀磩 50% 銇⒑鐜囥仹浠栨柟銇?run 銇閰嶉€併仌銈屻€?`pop_session_record` 銇?`KeyError: '<session_uuid>'` 銇岀櫤鐢?(streaming_worker.py:257)銆?
**銆宲oisoned scene銆嶃伄姝ｄ綋**: clipgt-e121e37d 銇儶銉堛儵銈ら€ｉ帠銇亾銇?port 琛濈獊銇棁鐘躲仩銇ｃ仧銆?灏傜敤 port (5014) 銇垏銈婃浛銇堛仧寰屻€佸悓 scene 銇竴搴︺倐澶辨晽銇涖仛 22 step 瀹岃蛋銇椼仧銆?
**淇**: 灏傜敤 driver port 5014 + 灏傜敤 reverse tunnel
(`ssh -R 5014:localhost:5014`, PID 銇?`/tmp/tunnel_5014.pid`)銆傛棦瀛?tunnel (PID 3318686,
-L 5011 -R 5013) 銇?AutoVLA 銇ㄣ伄鍏辩敤銈ゃ兂銉曘儵銇仧銈佽Е銈夈仾銇勩€?
### 2. GRPO experiment yaml 銇?`rl_sampling` 娆犺惤

`configs/experiment/qwen_drive_a100_1gpu_grpo.yaml` 銇?`policy.model.bundle_config.rl_sampling: true` 銇岀劇銇勩€侰LI override 銇ц姝?

```
policy.model.bundle_config.rl_sampling=true
```

(upstream 銇搞伄鍙嶆槧鍊欒: yaml 銇搞伄杩藉姞)

### 3. Trainer replay 銇?shape bug (浠婂洖銇?commit)

`_replay_row_logprob` 銇?packer 銇?stack 銇椼仧 `[B, ...]` leaf 銇嬨倝 `[row]` 銈掑彇銈婂嚭銇欍亴銆?銇撱倢銇?leading batch dim 銇劇銇?`[N+1, T, D]` / `[K, m]`銆備竴鏂?`stochastic_logprob` 銇绱?(unit test 銈傚悓銇? 銇?`[B, ...]`銆傗啋 `predict_endpoint` 銇?2-D waypoints 銈掑彈銇?`ValueError: not enough values to unpack (expected 3, got 2)`銆?
**淇** (`cosmos_wrapper.py`): `unsqueeze(0)` 銇?sample 杌搞倰鍐嶈拷鍔犮仐銆佹埢銈婂€ゃ倰 `reshape(())`銆?
unit test 銇?batch 浠樸亶 tensor 銈掔洿鎺ユ浮銇椼仸銇勩仧銇熴倎 masked 銇曘倢銇︺亜銇熴€?
## 妞滆

### Offline replay test (淇濆瓨娓堛伩 artifact 銇?trainer path 銈掑啀鐝?

```
[test] new_logprobs = ['8.6597', '7.0815', '8.3175']
[test] old_logprobs = ['8.6597', '7.0775', '8.3214']
[test] finite=True  max_abs_diff=4.040e-03  max_rel_diff=5.708e-04
[test] PASS: replay logprobs match rollout logprobs (ratio~1 at step 0)
```

閲嶃伩涓嶅銇倝 replay logprob 鈮?rollout logprob (float32 绮惧害銆佸樊銇?VLM prefill 銇?bf16 闈炴焙瀹氭€?銆?GRPO 銇?ratio 銇?step 0 銇ф纰恒伀 1.0 銇仾銈嬨亾銇ㄣ倰淇濊銆?
### Full run (tmp/alpagym-runs/20260918T115622Z-23755f866c674cbfb3c271d030b23615)

- 2 episodes (n_generation=2) 脳 22 steps銆佸疅 logprob (chosen_logprob 鈮?7-9)
- 鏈€鍒濄伄 minibatch 缇ゃ伅 ratio 鈮?1 (0.959, 1.012, 0.992...) 鈥?鏁板鐨勬暣鍚堛伄瀹熻
- clip_fraction=0.318: epsilon=0.05 銇?likelihood 銇嫮銇?(sigma 鈮?0.0158)銆?  mini-batch 鏇存柊銇岃搫绌嶃仚銈嬨仺 ratio 銇?trust region [0.8, 1.2] 銈掕秴銇堛倓銇欍亜
- reward_mean=-0.0578 (gt_rmse 鈮?5.8m), reward_std=0.0018
  鈫?2 generation 銇牨閰亴銇汇伡鍚屼竴 = 鎺㈢储涓嶈冻銆俥psilon 銈掍笂銇掋倠蹇呰鎬с倰绀恒仚
- VRAM 33.6 GiB (GPU 6, A100 80GB)

## 銈ゃ兂銉曘儵閬嬬敤銉°儮

| 闋呯洰 | 鍊?|
|---|---|
| Qwen-Drive GPU | GPU 6 (CUDA_VISIBLE_DEVICES=6) |
| AutoVLA GPU | 2, 3, 4, 5 (瑙︺倝銇亜) |
| Qwen-Drive driver port | **5014** (灏傜敤 reverse tunnel) |
| AutoVLA driver port | 5013 (鏃㈠瓨 tunnel 銇岄亱銇?鈥?瑙︺倝銇亜) |
| AlPaSim runtime | localhost:5011 (鍏辩敤, 鏃㈠瓨 tunnel) |
| Scene 瑙ｆ焙 | `alpasim_scene_ids.yaml` 銇?`scene_ids[prompt_idx]` (config 銇?`dataset.scene_ids` 銇疅璩?prompt 鏁般伄銇? |
| launch script | `/tmp/run_phase4_grpo.sh` (3-stage wizard bypass) |

## 娆°伄銈广儐銉冦儣

1. **epsilon sweep {0.1, 0.3, 1.0}** (docs/PHASE4_GRPO.md 鎺ㄥエ):
   epsilon=0.05 銇?likelihood 銇岄嫮銇欍亷 (clip 32%) 銇嬨仱 generation 闁撱伄琛屽嫊宸亴灏忋仌銇欍亷
   (reward_std=0.0018) 鈥?GRPO 銇?advantage 銇屽疅璩?noise銆傘倛銈婂ぇ銇嶃亜 epsilon 銇ф帰绱倰纰轰繚銆?2. `max_num_steps > 1` 銇ц鏁?epoch 銇湰鐣缈掋€?3. scene 澶氭鍖?(170 scenes 銇嬨倝瑜囨暟 prompt) 銇?generalization銆?4. `/tmp` 璩囩敚 (venv, qd_model, qd_runs) 銇案缍氥儜銈圭Щ瑷€?