# Qwen-Drive-1.0 AlpaGym 閫傞厤鍣?鈥?杩涘害鎶ュ憡

## 鏃ユ湡: 2025-09-17

---

## 涓€銆佹€讳綋鐩爣

灏?Qwen-Drive-1.0 Planning Expert锛圫FT 璁粌瀹屾垚锛夋帴鍏?AlpaGym 闂幆浠跨湡妗嗘灦锛屽疄鐜帮細
1. **Phase 3锛堝綋鍓嶏級**: 绾帹鐞嗗熀绾?鈥?鍔犺浇 SFT 妯″瀷锛屽湪 AlpaSim 浠跨湡鐜涓繍琛岄棴鐜瘎浼?2. **Phase 4锛堟湭鏉ワ級**: GRPO 寮哄寲瀛︿範璁粌 鈥?鍔犲叆 flow-matching logprob 璁＄畻鍜?Cosmos-RL 璁粌妗ユ帴

---

## 浜屻€佸凡瀹屾垚宸ヤ綔

### 2.1 Qwen-Drive SFT 璁粌锛堝凡瀹屾垚锛?- **妯″瀷**: Qwen-Drive-1.0 Planning Expert锛坒low-matching 杞ㄨ抗鐢熸垚锛?- **淇濆畧 10K checkpoint** 涓烘渶缁?SFT 妯″瀷
- 璺緞: `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/`
- 澶囦唤: NFS `qwen-drive-sft/checkpoints/conservative_10k_final/`
- 瀹屾暣妯″瀷锛圴LM+planner锛? `/tmp/qd_model/` (11G)

### 2.2 AlpaGym 閫傞厤鍣ㄥ紑鍙戯紙宸插畬鎴愪唬鐮佺紪鍐欙級
鍒涘缓浜?`alpagym-qwen-drive` 绛栫暐鎻掍欢鍖咃紝浣嶄簬 `packages/policies/qwen_drive/`锛?
#### 鏂囦欢娓呭崟
| 鏂囦欢 | 璇存槑 |
|------|------|
| `pyproject.toml` | 鍖呭畾涔夛紝entry-points: `qwen_drive = alpagym_qwen_drive.bundle:get_bundle` |
| `src/alpagym_qwen_drive/bundle.py` | PolicyBundle 5 涓挬瀛愶紙setup_tokenizer, build_data_packer, install_runtime_bridge, load_inference_model, build_model_inputs锛?|
| `src/alpagym_qwen_drive/inference_model.py` | QwenDriveInferenceModel 鈥?瀹炵幇 InferenceModel 鍗忚锛坰ample_trajectories_from_data, build_policy_replay_data, get_model, set_model锛?|
| `src/alpagym_qwen_drive/configs/policy/qwen_drive.yaml` | 绛栫暐閰嶇疆锛? cameras, 4 context frames, 50 future waypoints, step_dt_us=100000锛?|
| `src/alpagym_qwen_drive/configs/experiment/qwen_drive_a100_1gpu_inference.yaml` | 瀹為獙閰嶇疆锛? GPU inference, rollout_replicas=1, policy_replicas=0锛?|

#### 閫傞厤鍣ㄦ灦鏋?```
AlpaSim 鈫?PolicyInput 鈫?AlpamayoPolicy._preprocess() 鈫?BatchedModelInput
                                                              鈫?                                    QwenDriveInferenceModel.sample_trajectories_from_data()
                                                              鈫?                                    _infer_single() per batch row:
                                      1. _build_camera_views() 鈫?DrivingScene views
                                         (uint8 CHW 鈫?PIL 鈫?CameraFrame, 3 cameras 脳 4 frames)
                                      2. _extract_ego_history() 鈫?history/velocity/acceleration
                                         (ego_history_xyz + ego_history_rot 鈫?[H,3] + heading)
                                      3. route_to_nav_command() 鈫?nav_command (0/1/2)
                                      4. Build DrivingScene 鈫?model.generate_trajectory()
                                      5. trajectories [N,50,3] 鈫?pred_xyz [1,K,T,3] + pred_rot [1,K,T,3,3]
                                                              鈫?                                    BatchedModelOutput 鈫?AlpamayoPolicy._postprocess() 鈫?PolicyOutput 鈫?AlpaSim
```

#### 鍏抽敭璁捐鍐崇瓥
- **logprob = None**: Phase 3 绾帹鐞嗭紝涓嶈绠?flow-matching 瀵嗗害
- **policy_replicas = 0**: 鏃犺缁冭繘绋?- **install_runtime_bridge() = no-op**: 涓嶅畨瑁?Cosmos-RL 妗ユ帴
- **heading_to_rotation_matrix()**: Qwen-Drive 杈撳嚭 (x, y, heading)锛岃浆鎹负 AlpaGym 闇€瑕佺殑 SO(3) 鏃嬭浆鐭╅樀
- **route_to_nav_command()**: 鍚彂寮忓皢杩炵画 route waypoints 杞负绂绘暎瀵艰埅鍛戒护锛圙O STRAIGHT / TURN LEFT / TURN RIGHT锛?- **bundle.py 鐩存帴瀵煎叆**: `from alpagym_runtime.policies.registry import PolicyBundle`锛坴env 淇鍚?cosmos_rl 鍙敤锛屾棤闇€鎳掑姞杞斤級

### 2.3 pyproject.toml 淇敼
- 娣诲姞 `"packages/policies/qwen_drive"` 鍒?`[tool.uv.workspace] members`
- flash-attn/torch 婧愪繚鎸佸師濮?URL锛堟湰鍦?wheel 璺緞浠呭湪鏈湴浣跨敤锛屼笉鎻愪氦锛?
### 2.4 AlpaGym venv 鎭㈠锛堝凡瀹屾垚锛?- **闂**: `uv sync` 涔嬪墠鍥犵綉缁滀腑鏂牬鍧忎簡 venv
- **瑙ｅ喅**: 
  - 璁剧疆浠ｇ悊 `http://z59900495:753951tc-@proxysg.huawei.com:8080`
  - flash-attn/torch 鏀圭敤鏈湴 wheel
  - venv 鍒涘缓鍦?`/tmp/alpagym_venv`锛堟湰鍦版枃浠剁郴缁燂紝閬垮厤 NFS I/O 鐡堕锛?  - 浣跨敤 `UV_PROJECT_ENVIRONMENT=/tmp/alpagym_venv` 鎸囧畾 venv 璺緞
  - `UV_LINK_MODE=copy` 閬垮厤 hardlink 璺ㄦ枃浠剁郴缁熼棶棰?- **楠岃瘉閫氳繃**: torch 2.8.0+cu128, flash_attn 2.8.3, hydra 1.3.2, redis, cosmos_rl, alpagym_runtime 鍏ㄩ儴瀵煎叆鎴愬姛
- **qwen_drive bundle 鍔犺浇鎴愬姛**: `get_policy_bundle('qwen_drive')` 杩斿洖 `PolicyBundle` 瀹炰緥

### 2.5 transformers 鐗堟湰鍗囩骇锛堝凡瀹屾垚锛?- **闂**: Qwen-Drive-1.0 妯″瀷 config 浣跨敤 `model_type: qwen3_5`锛岄渶瑕?`transformers>=5.14.0`锛屼絾 AlpaGym 鍘熷鐜浣跨敤 `transformers==4.57.1`
- **瑙ｅ喅**:
  - 鍗囩骇鍒?`transformers==5.14.1`锛堜粠 PyPI 鐩存帴涓嬭浇 wheel 瀹夎锛屽洜 uv cache 鏈夋崯鍧忕殑 wheel锛?  - 鍗囩骇 `safetensors==0.8.0`锛堝悓鏍蜂粠 PyPI 鐩存帴涓嬭浇 wheel锛?  - 闄嶇骇 `huggingface_hub==1.5.0`锛坱ransformers 5.14.1 闇€瑕?>=1.5.0, 浣?1.31.0 鍒犻櫎浜?`is_offline_mode`锛?  - 淇濇寔 `diffusers==0.37.1`锛?.40.0 闇€瑕?`get_cached_repo_tree` 涓嶅吋瀹?hub 1.5.0锛?  - 淇濇寔 `click==8.3.3`锛?.5.0 鍒犻櫎浜?`click.command` 瑁呴グ鍣級
  - 淇濇寔 `datasets==5.0.1`
- **楠岃瘉閫氳繃**: cosmos_rl 瀵煎叆鎴愬姛锛宷wen_drive bundle 鍔犺浇鎴愬姛

### 2.6 Qwen-Drive 妯″瀷鍔犺浇楠岃瘉锛堝凡瀹屾垚锛?- **鎴愬姛鍔犺浇** `QwenDriveForPlanning.from_pretrained()`:
  - VLM: Qwen3.5 (5579.1M total params)
  - Planning Expert: PlanningExpert (separate SFT checkpoint)
  - 723 weight files loaded in <1s
  - Model moved to GPU (cuda:1), eval mode set
- **妯″瀷缁撴瀯纭**:
  - `model.vlm`: VLM 涓诲共 (AutoModelForImageTextToText)
  - `model.planning_expert`: Planning Expert (flow-matching trajectory generator)
  - `model.processor`: 鎳掑姞杞界殑 QwenDriveProcessor (property)
  - `model.generate_trajectory(scene, mode, num_samples, num_steps, seed)` 鈫?`QwenDriveOutput(trajectories=[N,50,3])`

---

## 涓夈€佸綋鍓嶈繘琛屼腑

### 3.1 闇€瑕佸閮?AI 杈呭姪鐨勯棶棰?
#### 闂 1: pyproject.toml 鏈湴 wheel 璺緞锛堝凡瑙ｅ喅锛?**鍘熷闂**: 浠ｇ悊鐜涓?GitHub releases 鍜?PyTorch 涓嬭浇绔欐湁 SSL 璇佷功闂锛?invalid peer certificate: UnknownIssuer"锛夈€?**褰撳墠鏂规**: pyproject.toml 淇濇寔鍘熷 URL锛屼笉鎻愪氦鏈湴 wheel 璺緞銆傛湰鍦伴€氳繃涓存椂淇敼 pyproject.toml + 鏈湴 wheel 鏂囦欢鏉ュ畬鎴?`uv sync`锛宻ync 瀹屾垚鍚庢仮澶嶅師濮?URL銆?**娉ㄦ剰浜嬮」**: 濡傛灉闇€瑕侀噸鏂?`uv sync`锛岄渶瑕佸啀娆′复鏃舵敼涓烘湰鍦?wheel 璺緞銆傛湰鍦?wheel 鏂囦欢浣嶄簬 alpagym 鏍圭洰褰曘€?
#### 闂 2: venv 浣嶇疆鍦?/tmp 鏄惁鍚堥€傦紵
褰撳墠 venv 鍦?`/tmp/alpagym_venv`锛堟湰鍦?overlay 鏂囦欢绯荤粺锛夛紝鍥犱负 NFS 涓婄殑 venv 鍒涘缓/鍚屾鏋佹參锛堟瘡娆?uv sync 瓒呮椂 5 鍒嗛挓浠ヤ笂锛夈€?**椋庨櫓**: `/tmp` 鏄鍣ㄥ唴涓存椂鏂囦欢绯荤粺锛屽鍣ㄩ噸鍚悗 venv 浼氫涪澶便€?**缂撹В**: 宸插垱寤?`/tmp/activate_alpagym.sh` 婵€娲昏剼鏈紝閲嶅惎鍚庡彧闇€閲嶆柊杩愯 `uv sync` 鍗冲彲鎭㈠銆?**闂**: 鏄惁鏈夋洿濂界殑浣嶇疆锛熸垨鑰?NFS 鐨?I/O 鎬ц兘闂鏄殏鏃舵€х殑锛?
#### 闂 3: bundle.py 瀵煎叆鏂瑰紡锛堝凡瑙ｅ喅锛?**鍘熷闂**: `bundle.py` 鍦ㄦā鍧楅《閮ㄧ洿鎺?`from alpagym_runtime.policies.registry import PolicyBundle`锛屼細瑙﹀彂 `registry.py` 鈫?`cosmos.packer` 鈫?`cosmos_rl` 鏁翠釜渚濊禆閾惧姞杞姐€?**褰撳墠鏂规**: 淇濇寔鐩存帴瀵煎叆锛堝師濮嬫柟寮忥級銆倂env 淇鍚?cosmos_rl 鍙敤锛宍get_policy_bundle('qwen_drive')` 宸查獙璇佹垚鍔熴€?
#### 闂 4: autovla 璁粌浠诲姟鍏辩敤 venv
褰撳墠鏈夊彟涓€涓?autovla 璁粌浠诲姟锛圥ID 2472853锛夋鍦ㄤ娇鐢ㄥ悓涓€涓?`.venv/bin/python3`锛岄€氳繃 `uv run --no-sync` 鍚姩銆?- 璇ヨ繘绋嬪湪 venv 琚牬鍧忓墠宸插惎鍔紝妯″潡宸插姞杞藉埌鍐呭瓨涓?- venv 鎭㈠鍚庯紙鍦?/tmp锛夛紝鏃х殑 `.venv` 璺緞涓嶅啀鎸囧悜鏈夋晥 venv
- **闂**: autovla 杩涚▼鏄惁浼氬洜 venv 璺緞鍙樺寲鑰屽穿婧冿紵濡傛灉瀹冮渶瑕侀噸鏂?import 妯″潡锛堜緥濡?fork 瀛愯繘绋嬶級浼氭€庢牱锛?
#### 闂 5: transformers 鐗堟湰鍐茬獊锛堝凡瑙ｅ喅锛?**鍘熷闂**: Qwen-Drive-1.0 妯″瀷闇€瑕?`transformers>=5.14.0`锛坈onfig 涓娇鐢?`model_type: qwen3_5`锛夛紝浣?AlpaGym 鍘熷鐜浣跨敤 `transformers==4.57.1`銆?**褰撳墠鏂规**: 鍗囩骇鍒?`transformers==5.14.1`銆倁v cache 涓湁鎹熷潖鐨?wheel锛堟爣娉?5.14.1 浣嗗疄闄呭寘鍚?4.57.1 浠ｇ爜锛夛紝閫氳繃鐩存帴浠?PyPI 涓嬭浇 wheel 鏂囦欢瀹夎瑙ｅ喅銆傚悓鏃堕渶瑕佸崌绾?`huggingface_hub` 鍒?1.x銆乣datasets` 鍒?5.x銆乣diffusers` 鍒版渶鏂扮増鏈互淇濇寔鍏煎銆?**椋庨櫓**: transformers 5.x 鏄ぇ鐗堟湰鍗囩骇锛屽彲鑳戒笌 cosmos_rl 鎴栧叾浠?AlpaGym 渚濊禆涓嶅吋瀹广€傞渶瑕侀獙璇佸畬鏁村鍏ラ摼銆?
#### 闂 6: AlpaSim 杩炴帴鍏变韩
- AlpaSim 鍦ㄨ繙绋嬫湇鍔″櫒 `mti@10.174.175.151` 涓婅繍琛?- 褰撳墠閫氳繃 SSH 闅ч亾鏆撮湶绔彛 5011锛坮untime server锛夊拰 5013锛堝弽鍚戦€氶亾锛?- autovla 浠诲姟姝ｅ湪浣跨敤杩欎釜闅ч亾杩涜璁粌
- AlpaSim capacity=4锛岀悊璁轰笂鏀寔 4 涓苟鍙?session
- **闂**: 鏄惁鍙互鍏辩敤鍚屼竴涓毀閬擄紵杩樻槸闇€瑕佸缓绔嬬浜屼釜 SSH 闅ч亾鍒颁笉鍚岀鍙ｏ紵
- **闃荤**: 鐩存帴 SSH 鍒?10.174.175.151 杩斿洖 "Permission denied"锛屽瘑鐮佹湭鐭?
---

## 鍥涖€佷笅涓€姝ヨ鍒?
1. ~~**Git push 褰撳墠淇敼**~~ 鈥?宸插畬鎴?(commit 6e3c6e0)
2. ~~**楠岃瘉 Qwen-Drive 妯″瀷鍔犺浇**~~ 鈥?宸插畬鎴?(5.6B params loaded on GPU)
3. **寤虹珛 AlpaSim 杩炴帴** 鈥?绔彛 5011 宸茬‘璁ゅ紑鏀撅紝闇€娴嬭瘯闂幆鎺ㄧ悊鑳藉惁杩炴帴
4. **杩愯闂幆鎺ㄧ悊** 鈥?鐢?qwen_drive_a100_1gpu_inference 閰嶇疆鍚姩闂幆璇勪及
5. **璋冭瘯 inference_model.py** 鈥?楠岃瘉 camera view 鏋勫缓銆乪go history 鎻愬彇銆乼rajectory 鍚庡鐞嗘槸鍚︽纭?6. **绔埌绔祴璇?* 鈥?浠?AlpaSim 鑾峰彇涓€甯ф暟鎹?鈫?鏋勫缓 DrivingScene 鈫?generate_trajectory 鈫?杩斿洖 PolicyOutput

---

## 浜斻€佺幆澧冧俊鎭?
| 椤圭洰 | 鍊?|
|------|-----|
| AlpaGym 璺緞 | `/data/mnt_m62/10_personal/z59900495/workspace/alpagym` |
| Qwen-Drive 涓婃父婧愮爜 | `/data/mnt_m62/10_personal/z59900495/workspace/a_0914_qwendrive/Qwen-Drive-1.0/src/` |
| Qwen-Drive 瀹屾暣妯″瀷 | `/tmp/qd_model/` (11G) |
| SFT checkpoint | `/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final/` |
| venv 璺緞 | `/tmp/alpagym_venv` (UV_PROJECT_ENVIRONMENT) |
| 婵€娲昏剼鏈?| `/tmp/activate_alpagym.sh` |
| GPU 浣跨敤 | GPU 1 (绌洪棽), GPU 5-7 (autovla 璁粌涓? |
| AlpaSim | `mti@10.174.175.151`, 绔彛 5011/5013 via SSH tunnel |
| 浠ｇ悊 | `http://z59900495:753951tc-@proxysg.huawei.com:8080` |
| Git remote | `https://ikutasama:***@github.com/ikutasama/alpagym.git` |
| Git branch | `main` |

---

## 鍏€佹枃浠跺彉鏇存竻鍗?
### 鏂板鏂囦欢
- `packages/policies/qwen_drive/pyproject.toml`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/__init__.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/inference_model.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/__init__.py`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/policy/qwen_drive.yaml`
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/experiment/qwen_drive_a100_1gpu_inference.yaml`

### 淇敼鏂囦欢
- `pyproject.toml` 鈥?浠呮坊鍔?qwen_drive workspace member锛坒lash-attn/torch 婧愪繚鎸佸師濮?URL锛?- `uv.lock` 鈥?uv sync 鑷姩鏇存柊

### 鏈慨鏀癸紙淇濇寔鍘熺姸锛?- `packages/runtime/src/alpagym_runtime/policies/registry.py` 鈥?鏃犱慨鏀?- `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py` 鈥?鐩存帴瀵煎叆 PolicyBundle锛堟棤鎳掑姞杞斤級

---

## 涓冦€侀渶瑕佸閮?AI 妫€瑙嗙殑閲嶇偣

1. **閫傞厤鍣ㄤ唬鐮佹纭€?*: `inference_model.py` 涓殑 camera view 鏋勫缓銆乪go history 鎻愬彇銆乼rajectory 鍚庡鐞嗛€昏緫鏄惁姝ｇ‘锛?2. **閰嶇疆鏂囦欢瀹屾暣鎬?*: `qwen_drive.yaml` 鍜?`qwen_drive_a100_1gpu_inference.yaml` 鏄惁閬楁紡蹇呰瀛楁锛?3. **pyproject.toml 鍙樻洿**: 浠呮坊鍔?qwen_drive workspace member锛宖lash-attn/torch 婧愪繚鎸佸師濮?URL锛堝凡瑙ｅ喅锛?4. **venv 绛栫暐**: /tmp venv + UV_PROJECT_ENVIRONMENT 鏂规鏄惁鍙锛熸湁鏃犳洿濂界殑鎸佷箙鍖栨柟妗堬紵
5. **AlpaSim 鍏辩敤**: 涓?autovla 璁粌浠诲姟鍏辩敤 AlpaSim 闅ч亾鐨勫彲琛屾€у拰椋庨櫓锛?