# Qwen-Drive 脳 AlpaGym Closed-Loop Integration 鈥?Progress Report

## 姒傝 (Summary)

Qwen-Drive-1.0 Planning Expert 銇?AlpaGym (Cosmos-RL) 銈儹銉笺偤銉夈儷銉笺儣绲卞悎銈掑疅瑁呫仐銆併偣銉兗銈儐銈广儓銇т互涓嬨倰纰鸿獚銇椼伨銇椼仧锛?
- 鉁?銉儑銉銇胯炯銇挎垚鍔?(QwenDriveCosmos BaseModel wrapper)
- 鉁?Policy 鈫?Rollout 銈︺偋銈ゃ儓鍚屾湡鎴愬姛 (QwenDriveWeightMapper)
- 鉁?銉堛儸銉笺儖銉炽偘銉兗銉楄捣鍕曟垚鍔?(vLLM rollout engine 鍒濇湡鍖栧畬浜?
- 鈿狅笍 AlPaSim 銉夈儵銈ゃ儛銉笺儩銉笺儓鎺ョ稓銈ㄣ儵銉?(銉堛兂銉嶃儷瑷畾銇晱椤屻€併偝銉笺儔涓嶅叿鍚堛仹銇仾銇?

## 銈兗銈儐銈儊銉?
### Qwen-Drive 銉儑銉閫?- `QwenDriveForPlanning` 銇鍚堛儮銉囥儷锛?  - `model.vlm` (AutoModelForImageTextToText, Qwen3.5 VLM, 11GB, 鍑嶇祼)
  - `model.planning_expert` (PlanningExpert, flow-matching, 4.1GB, 瀛︾繏瀵捐薄)
- Config 銇?`vlm_config` (Qwen3.5 VLM) 銇?`expert_config` (Planning Expert) 銇鍚堟鎴?
### Cosmos-RL 绲卞悎銇祦銈?1. `entrypoint.py` 鈫?`policy_bundle.build_data_packer(run_config, cosmos_role)`
2. `build_data_packer` 鈫?`install_runtime_bridge()` 鈫?`cosmos_wrapper` 銉偢銉ャ兗銉偆銉炽儩銉笺儓
3. 銉偢銉ャ兗銉偆銉炽儩銉笺儓鏅傘伀 AutoConfig/AutoModel/ModelRegistry 鐧婚尣銇屽疅琛屻仌銈屻倠
4. Trainer `__init__` 鈫?`ModelRegistry.build_model(config)` 鈫?`QwenDriveCosmos.from_pretrained()`
5. `llm_trainer.py` 鈫?`model.load_hf_weights(model_path, parallel_dims, device)`
6. Policy 鈫?Rollout 銈︺偋銈ゃ儓鍚屾湡 鈫?vLLM 銈ㄣ兂銈搞兂鍒濇湡鍖?鈫?銉堛儸銉笺儖銉炽偘銉兗銉楅枊濮?
## 瀹熻銉曘偂銈ゃ儷

### `packages/policies/qwen_drive/src/alpagym_qwen_drive/cosmos_wrapper.py` (鏂拌)
Cosmos-RL BaseModel wrapper for Qwen-Drive-1.0 Planning Expert.

涓昏銈炽兂銉濄兗銉嶃兂銉堬細
- `set_planner_path(path)` 鈥?Planning Expert 銉併偋銉冦偗銉濄偆銉炽儓銉戙偣銈掋儮銈搞儱銉笺儷澶夋暟銇牸绱?- `_register_autoconfig_and_model()` 鈥?QwenDriveConfig, QwenDrivePlanningExpertConfig, QwenDriveForPlanning 銈?transformers 銇櫥閷?- `QwenDriveWeightMapper(HFModelWeightMapper)` 鈥?Qwen3.5 VLM 銇鍚?Config 銇蹇溿仚銈嬨偒銈广偪銉?WeightMapper
- `QwenDriveCosmos(BaseModel)` 鈥?Cosmos-RL BaseModel 瀹熻
  - `supported_model_types()` 鈫?`["qwen_drive"]`
  - `from_pretrained()` 鈫?`cls(hf_config)` (meta device 銇с偆銉炽偣銈裤兂銈瑰寲)
  - `load_hf_weights()` 鈫?VLM 銈?`model_name_or_path` 銇嬨倝銆丳lanning Expert 銈?`_PLANNER_PATH` 銇嬨倝瑾伩杈笺伩
  - `forward()` 鈫?銉€銉熴兗 `{"log_probs": zeros, "kl_div": None}` (Phase 4 銇у疅瑁呬簣瀹?
  - `get_position_ids()` 鈫?銈枫兗銈便兂銈枫儯銉?position ids
  - `parallelize_fn` 鈫?DDP 銈儹銉笺偢銉?  - `separate_model_parts()` 鈫?`[self]`
  - `get_nparams_and_flops()` 鈫?`(0, 0)`

### `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py` (鏇存柊)
- `install_runtime_bridge()` 銇?`cosmos_wrapper` 銈掋偆銉炽儩銉笺儓锛堝壇浣滅敤銇?AutoConfig/AutoModel/ModelRegistry 鐧婚尣銈掑疅琛岋級
- `build_data_packer()` 銇?`set_planner_path(planner_path)` 銈掑懠銇冲嚭銇?- `setup_tokenizer()` 銇?`install_runtime_bridge()` 銈掋偦銉笺儠銉嗐偅銉嶃儍銉堛仺銇椼仸鍛笺伋鍑恒仐

### `scripts/run_qwen_drive_smoke.sh` (鏇存柊)
3銉曘偋銉笺偤銈广儮銉笺偗銉嗐偣銉堬細
1. Phase 1: CLI 銈掍竴鏅傚疅琛屻仐銇?run directory 銇?config 銉曘偂銈ゃ儷銈掍綔鎴?2. Phase 2: 鏃㈠瓨銇?AlPaSim 銉┿兂銈裤偆銉?(localhost:5011) 銇嬨倝 topology registry 銈掓绡?3. Phase 3: `ALPAGYM_COSMOS_ONLY=1` 銇?Cosmos-RL 銈掕捣鍕?
## 瑙ｆ焙銇椼仧鍟忛 (Solved Issues)

### 1. `ValueError: Unrecognized configuration class QwenDriveConfig for AutoModelForCausalLM`
**鍘熷洜**: Cosmos-RL 銇?`ModelRegistry._MODEL_REGISTRY` 銇仾銇?model_type 銈?`AutoModelForCausalLM` 銇с儠銈┿兗銉儛銉冦偗銇椼倛銇嗐仺銇椼仧銆?**瑙ｆ焙**: `QwenDriveCosmos(BaseModel)` wrapper 銈掍綔鎴愩仐銆乣ModelRegistry.register(WeightMapper)(QwenDriveCosmos)` 銇х櫥閷层€?
### 2. `Can't instantiate abstract class QwenDriveCosmos without an implementation for abstract method 'get_position_ids'`
**鍘熷洜**: BaseModel ABC 銇娊璞°儭銈姐儍銉夈亴鏈疅瑁呫€?**瑙ｆ焙**: `get_position_ids()` 銇仼鍏ㄦ娊璞°儭銈姐儍銉夈倰瀹熻銆?
### 3. `ValueError: Can not determine kv_head_ratio and head_dim from config: QwenDriveConfig`
**鍘熷洜**: `HFModelWeightMapper.__init__` 銇?`config.num_key_value_heads`, `config.text_config`, `config.llm_config` 銇爢銇儊銈с儍銈仚銈嬨亴銆乣QwenDriveConfig` 銇?`vlm_config` 銈掓寔銇ゃ仧銈佸叏銇﹀け鏁椼€?**瑙ｆ焙**: `QwenDriveWeightMapper.__init__` 銇?`hf_config.text_config = vlm_config.text_config` 銈掍竴鏅傜殑銇ō瀹氥仐銇︺亱銈?`super().__init__()` 銈掑懠銇冲嚭銇椼€?
### 4. `NameError: name 'load_file' is not defined`
**鍘熷洜**: `cosmos_wrapper.py` 銇?`_load_safetensors_dir` 銉°偨銉冦儔銇?`safetensors.torch.load_file` 銈掋偆銉炽儩銉笺儓銇椼仸銇勩仾銇嬨仯銇熴€?**瑙ｆ焙**: `from safetensors.torch import load_file` 銈掕拷鍔犮€?
### 5. `TypeError: list indices must be integers or slices, not str`
**鍘熷洜**: `alpasim_scene_ids.yaml` 銇屻儣銉兗銉炽儶銈广儓褰㈠紡銇ф浉銇嶈炯銇俱倢銇︺亜銇熴亴銆乣entrypoint.py` 銇?`{"scene_ids": [...]}` 銇?dict 褰㈠紡銈掓湡寰呫仐銇︺亜銇熴€?**瑙ｆ焙**: Phase 2 銈广偗銉儣銉堛仹 `yaml.dump({"scene_ids": scene_ids}, ...)` 銇慨姝ｃ€?
### 6. `RuntimeError: No AlpaSim runtime endpoints in tmp/.../topology`
**鍘熷洜**: topology registry 銉曘偂銈ゃ儷銈?`topology_registry/alpasim_runtimes/` 銇浉銇嶈炯銈撱仹銇勩仧銇屻€佸疅闅涖伄銉戙偣銇?`topology/alpasim_runtimes/` 銇犮仯銇熴€?**瑙ｆ焙**: `run_dir / "topology" / "alpasim_runtimes"` 銇慨姝ｃ€?
### 7. `CUDA out of memory` (GPU 3)
**鍘熷洜**: GPU 3 銇屼粬銉椼儹銈汇偣銇?74GB 浣跨敤涓仩銇ｃ仧銆?**瑙ｆ焙**: `CUDA_VISIBLE_DEVICES=0` 銇鏇达紙GPU 0 銇?9.5GB 浣跨敤銆?1GB 绌恒亶锛夈€?
### 8. `ValueError: vLLM qkv: cannot infer TP shard layout from dim_0=8192, head_dim=256, total_q_heads=32, n_kv=4`
**鍘熷洜**: Qwen3.5 VLM 銇?text_config 銇?`head_dim=256`锛堟槑绀虹殑瑷畾銆乣hidden_size/num_attention_heads=160` 銇ㄣ伅鐣般仾銈嬶級銆乣attn_output_gate=True`銆倂LLM 銇?Q+gate 銈?`qkv_proj` 銇?K+V 銇ㄣ伅鍒ャ伀鏍肩磵銇欍倠銇熴倎銆乣dim_0=8192` 銇?Q+gate 銇伩锛圞+V 銇仐锛夈€俙HFModelWeightMapper._rollout_split_qkv_weight` 銇?Q+K+V 鍏ㄣ仸鍚個銇ㄤ划瀹氥仐銇﹀垎鍓层倰瑭︺伩銈嬨亴銆乣units=32` 銇?`total_q_heads=32` 銇瓑銇椼亸 K+V 鍒嗐亴瀛樺湪銇椼仾銇勩仧銈佸け鏁椼€?**瑙ｆ焙**:
1. `QwenDriveWeightMapper.__init__` 銇?`self.head_dim` 銈?config 銇槑绀虹殑 `head_dim` 銉曘偅銉笺儷銉?(256) 銇т笂鏇搞亶
2. `rollout_split_local_key_n_param_to_hf_key_n_param` 銈掋偑銉笺儛銉笺儵銈ゃ儔銇椼€乣dim_0 == total_q * head_dim` 銇牬鍚堛伅 Q-only 銇ㄥ垽瀹氥仐銇?`q_proj` 銇伩杩斻仚

## 鏈В姹恒伄鍟忛 (Remaining Issues)

### 9. AlPaSim 銉夈儵銈ゃ儛銉笺儩銉笺儓鎺ョ稓銈ㄣ儵銉?(銉堛兂銉嶃儷鍟忛)
```
StatusCode.UNAVAILABLE: failed to connect to all addresses;
last error: UNKNOWN: ipv4:127.0.0.1:33969: Failed to connect to remote host: Connection refused
```
**鍘熷洜**: AlPaSim RuntimeService 銇?localhost:5011锛圫SH 銉堛兂銉嶃儷绲岀敱锛夈仹銈偗銈汇偣鍙兘銇犮亴銆併偡銉笺兂鍓层倞褰撱仸鏅傘伀鍕曠殑銇壊銈婂綋銇︺倝銈屻倠銉夈儵銈ゃ儛銉笺偍銉炽儔銉濄偆銉炽儓锛堛儩銉笺儓 33969 銇仼锛夈伅銉堛兂銉嶃儷銇曘倢銇︺亜銇亜銆?**褰遍熆**: 銉堛儸銉笺儖銉炽偘銉兗銉椼伅璧峰嫊銇欍倠銇屻€併偍銉斻偨銉笺儔瀹熻鏅傘伀銉夈儵銈ゃ儛銉笺伀鎺ョ稓銇с亶銇氥儶銉堛儵銈ゃ倰绻般倞杩斻仚銆?**蹇呰銇蹇?*: 
- AlPaSim 銉┿兂銈裤偆銉犮倰銉兗銈儷銇ц捣鍕曘仚銈嬶紙銉堛兂銉嶃儷涓嶈锛?- 銇俱仧銇嫊鐨勩儩銉笺儓銈傘儓銉炽儘銉仚銈嬭ō瀹氥倰杩藉姞
- 銇俱仧銇?AlPaSim 銉┿兂銈裤偆銉犮伄銉夈儵銈ゃ儛銉笺儩銉笺儓銈掑浐瀹氥仚銈?
### 10. VLM 銈︺偋銈ゃ儓鍚屾湡銇?WARNING
```
[Policy] No send instructions generated for parameter model.vlm.model.visual.blocks.*.attn.{q,k,v}.{weight,bias}
```
**鍘熷洜**: VLM 銇?visual tower 銇偊銈с偆銉堛亴 vLLM 銇?HF 銉儑銉仹鍚嶅墠銉炪儍銉斻兂銈般亴涓€鑷淬仐銇亜銆?**褰遍熆**: VLM 銇噸绲愩仌銈屻仸銇勩倠銇熴倎瀛︾繏銇伅褰遍熆銇椼仾銇勩亴銆乺ollout 鏅傘伄鎺ㄨ珫绮惧害銇奖闊裤仚銈嬪彲鑳芥€с亴銇傘倠銆?**蹇呰銇蹇?*: `rollout_map_local_key_to_hf_key` 銇?visual tower 銇偊銈с偆銉堝悕銉炪儍銉斻兂銈般倰淇銆?
### 11. forward() 銇屻儉銉熴兗瀹熻
**鍘熷洜**: 鐝惧湪銇?`QwenDriveCosmos.forward()` 銇?`{"log_probs": zeros, "kl_div": None}` 銈掕繑銇欍偣銈裤儢瀹熻銆?**褰遍熆**: GRPO 銉堛儸銉笺儖銉炽偘銇у疅闅涖伄鍕鹃厤瑷堢畻銇岃銈忋倢銇亜銆?**蹇呰銇蹇?*: Phase 4 銇?Planning Expert 銇?flow-matching logprob 瑷堢畻銈掑疅瑁呫€?
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
| 銉堛儸銉笺儖銉炽偘銉兗銉楄捣鍕?| 鉁?| 銈ㄣ償銈姐兗銉夊疅琛屻倰闁嬪 |
| AlPaSim 銉夈儵銈ゃ儛銉兼帴缍?| 鈿狅笍 | 銉濄兗銉?33969 銇屻儓銉炽儘銉仌銈屻仸銇勩仾銇?|
| GRPO 鍕鹃厤瑷堢畻 | 鉂?| forward() 銇屻偣銈裤儢瀹熻 (Phase 4) |

## 鐠板鎯呭牨

- **AlpaGym**: `/data/mnt_m62/10_personal/z59900495/workspace/alpagym` (commit f77031d)
- **venv**: `/tmp/alpagym_venv` (ephemeral, `/tmp` 涓娿伀妲嬬瘔)
- **Cosmos-RL**: commit d2a2c57c4, extras 銇仐銇с偆銉炽偣銉堛兗銉?- **Python**: 3.12.13
- **transformers**: 5.14.1
- **safetensors**: 0.8.0
- **VLM model**: `/tmp/qd_model/` (Qwen3.5 VLM, 11GB, 723 shards)
- **Planning Expert**: `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/model.safetensors` (4.1GB)
- **AlPaSim Runtime**: localhost:5011 (SSH tunnel, 170 scenes, capacity=4)
- **GPU**: GPU 0 (CUDA_VISIBLE_DEVICES=0)

## Git 灞ユ

- `f77031d` feat(qwen_drive): Cosmos-RL closed-loop integration with QwenDriveCosmos wrapper
- `1fe42c0` fix: driving_command 4-element one-hot (ego_status_dim=8), inference verified
- `71943d0` fix: update qwen_drive experiment config for colocated 1gpu inference
- `4d639db` feat: Qwen-Drive SFT training pipeline + planning expert inference
- `6e3c6e0` feat: add qwen_drive policy package with bundle, config, and data packer
