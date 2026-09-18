# Qwen-Drive 脳 AlpaGym 闆嗘垚閫傞厤瀹℃煡 鈥?涓庡畼鏂逛粨搴撳鐓?
> 瀵圭収 [NVlabs/alpagym](https://github.com/NVlabs/alpagym/tree/main) 瀹樻柟浠ｇ爜浠擄紝瀹℃煡鏈湴浠撳簱
> `/data/mnt_m62/10_personal/z59900495/workspace/alpagym` 鐨勯€傞厤鍜屼换鍔″惎鍔ㄦ柟娉曟槸鍚﹂渶瑕佷慨姝ｃ€?
## 1. 瀵圭収缁撹

| 椤圭洰 | 瀹樻柟鐘舵€?| 鏈湴鐘舵€?| 闇€瑕佷慨姝? |
|------|---------|---------|----------|
| `docs/ONBOARDING.md` | 233琛?| 瀹屽叏涓€鑷?| 鉂?涓嶉渶瑕?|
| `conf/default.yaml` | 188琛?| 瀹屽叏涓€鑷?| 鉂?涓嶉渶瑕?|
| `packages/policies/` | 浠?`alpamayo_r1` | `alpamayo_r1` + `autovla` + `qwen_drive` | 鉁?鏂板锛堟纭級 |
| `scripts/` | 鍩虹鑴氭湰 | 鏂板 qwen_drive / autovla / dagger 绛夎剼鏈?| 鈿狅笍 閮ㄥ垎闇€淇锛堣涓嬶級 |
| 瀹樻柟 `run_alpamayo_smoke.sh` | 鐢?wizard 鍚姩鏈湴 Docker AlPaSim | 鈥?| 鈥?|
| 鏈湴 `run_qwen_drive_smoke.sh` | 鈥?| 3闃舵缁曡繃 wizard锛岃繛杩滅▼ AlPaSim | 鈿狅笍 闇€淇 |
| `qwen_drive_a100_1gpu_inference.yaml` | 鈥?| `expected_valid_steps: 8` | 鉁?**蹇呴』淇涓?22** |
| Qwen-Drive 鏂囨。 | 鈥?| 鏃狅紙PROGRESS_REPORT.md 鍦ㄦ牴鐩綍锛?| 鉁?闇€琛ュ厖 |

## 2. 瀹樻柟鏋舵瀯 vs 鏈湴閫傞厤

### 2.1 瀹樻柟鍚姩娴佺▼锛堝崟鏈?Docker锛?
```
alpagym_host.cli
  鈫?Hydra 缁勮 config
  鈫?鍐荤粨 resolved_config.yaml
  鈫?缂撳瓨 AlPaSim checkout
  鈫?鍚姩 AlPaSim Wizard锛圖ocker Compose锛?  鈫?鍙戝竷 gRPC endpoint
  鈫?鍚姩 Cosmos-RL runtime
  鈫?闂幆 rollout + GRPO 璁粌
```

鍏抽敭鍓嶆彁锛?*鏈湴鏈?Docker + Docker Compose**锛學izard 鑷姩鎷夎捣 AlPaSim 瀹瑰櫒銆?
### 2.2 鏈湴 Qwen-Drive 閫傞厤娴佺▼锛堟棤 Docker锛岃繙绋?AlPaSim锛?
鎴戜滑鐨勭幆澧冩病鏈?Docker 闀滃儚锛孉lPaSim 杩愯鍦ㄨ繙绋嬫満鍣?(10.174.175.151)銆傚洜姝ら噰鐢?3 闃舵缁曡繃 wizard锛?
```
Phase 1: 杩愯 CLI 鈫?绛夊緟 run_dir 鍒涘缓 鈫?鍦?wizard 鍚姩 Docker 鍓嶆潃鎺?Phase 2: 鎵嬪姩濉厖 topology_registry锛堣繛鎺?localhost:5011 杩滅▼ AlPaSim锛?Phase 3: ALPAGYM_COSMOS_ONLY=1 + resolved_config_path 閲嶆柊鍚姩 Cosmos
```

**杩欎釜閫傞厤鏂瑰悜鏄纭殑**鈥斺€斿畼鏂逛粨搴撴湰韬篃鎻愪緵浜?`run_wizard_only.sh` +
`run_cosmos_only.sh` 鐨勪袱鏈哄垎绂绘ā寮忋€傚尯鍒湪浜庯細

| 缁村害 | 瀹樻柟涓ゆ満妯″紡 | 鎴戜滑鐨?Qwen-Drive 妯″紡 |
|------|------------|---------------------|
| Wizard 鏈哄櫒 | 杩愯 `run_wizard_only.sh`锛孌ocker 鍚姩 AlPaSim | 涓嶈繍琛?wizard锛堟棤 Docker锛?|
| Cosmos 鏈哄櫒 | `run_cosmos_only.sh`锛屼娇鐢ㄤ粠 wizard 鏈烘嫹璐濈殑 run_dir | 鑷繁鍒涘缓 run_dir锛圥hase 1 鏉€杩涚▼鏂瑰紡锛?|
| AlPaSim 鏉ユ簮 | Wizard 鏈虹殑 Docker 瀹瑰櫒 | 杩滅▼棰勯儴缃茬殑 AlPaSim (10.174.175.151) |
| 閫氫俊 | SSH tunnel forward | SSH tunnel forward + reverse |
| run_dir 鍒涘缓 | wizard 姝ｅ父鍒涘缓 | Phase 1 鏆村姏鍒涘缓锛坘ill before docker锛?|

## 3. 闇€瑕佷慨姝ｇ殑闂

### 3.1 鉂?`expected_valid_steps` 閰嶇疆涓嶄竴鑷达紙蹇呴』淇锛?
**鏂囦欢**: `packages/policies/qwen_drive/src/alpagym_qwen_drive/configs/experiment/qwen_drive_a100_1gpu_inference.yaml`

**鐜扮姸**:
```yaml
expected_valid_steps: 8
alpasim:
  wizard_args:
    control_timestep_us: 100_000
    force_gt_duration_us: 8_000_000
    n_sim_steps: 88
    extra_overrides: "runtime.simulation_config.pose_reporting_interval_us=500000"
```

**闂**: 閰嶇疆楠岃瘉閫昏緫 (`config_validation.py:307-317`) 瑕佹眰锛?```
n_sim_steps - (force_gt_duration_us // control_timestep_us) == expected_valid_steps
88 - 80 = 8  鈫?閰嶇疆鑷韩涓€鑷?```

浣嗗疄闄呰繍琛屾椂锛宺ollout 浜х敓浜?**22 涓?policy outputs**锛堜笉鏄?8锛夛紝瀵艰嚧 packer 鎶ラ敊锛?```
ValueError: produced 22 policy outputs, exceeding expected_valid_steps=8
```

**鏍瑰洜**: `pose_reporting_interval_us=500000` 瀵艰嚧瀹為檯 policy step 鏁颁笌鍏紡璁＄畻涓嶇銆?杩滅▼ AlPaSim 鐨勫疄闄呰涓轰笌 `n_sim_steps` 鐨勮涔夊彲鑳戒笉涓€鑷淬€?
**淇**: 鏇存柊涓虹粡 Phase 3aa 楠岃瘉鐨勫€硷細
```yaml
expected_valid_steps: 22
alpasim:
  wizard_args:
    n_sim_steps: 102   # 22 + 80 warmup
```

### 3.2 鈿狅笍 `run_qwen_drive_smoke.sh` 纭紪鐮佽矾寰?
**鏂囦欢**: `scripts/run_qwen_drive_smoke.sh`

**闂**: 澶ч噺璺緞纭紪鐮侊紝鏃犳硶鍦ㄥ叾浠栫幆澧冨鐢細
```bash
ALPAGYM_ROOT=/data/mnt_m62/10_personal/z59900495/workspace/alpagym   # 纭紪鐮?MODEL_PATH="/tmp/qd_model"                                           # ephemeral /tmp
PLANNER_PATH="/tmp/qd_runs/sft_continued_conservative_b4/..."        # ephemeral /tmp
CUDA_VISIBLE_DEVICES=3                                               # 纭紪鐮?GPU
SCENE_ID="clipgt-01d503d4-..."                                       # 纭紪鐮佸満鏅?```

**瀵规瘮**: 瀹樻柟 `run_alpamayo_smoke.sh` 浣跨敤 `ALPAGYM_ROOT` 鐜鍙橀噺 + 榛樿鍊硷細
```bash
cd "${ALPAGYM_ROOT:-$HOME/alpagym}"
MODEL_PATH="${MODEL_PATH:-/mnt/.../Alpamayo-1.5-10B}"
EXPERIMENT="${EXPERIMENT:-alpamayo_1_5_local_2gpu_smoke}"
```

**淇寤鸿**: 灏嗙‖缂栫爜鏀逛负鐜鍙橀噺 + 榛樿鍊硷紝涓庡畼鏂硅剼鏈鏍间竴鑷淬€?
### 3.3 鈿狅笍 SSH tunnel 鏈泦鎴愬埌鑴氭湰涓?
**鐜扮姸**: `run_qwen_drive_smoke.sh` 鍋囪 SSH tunnel 宸插湪杩愯锛屼絾涓嶆鏌ヤ篃涓嶅惎鍔ㄣ€?tunnel 璁剧疆鏁ｈ惤鍦?`/tmp/dagger_supervisor.sh`锛坋phemeral锛夊拰 PROGRESS_REPORT.md 涓€?
**瀵规瘮**: 瀹樻柟鏈?`scripts/start_ssh_tunnel.sh` 鑷姩绠＄悊 tunnel锛堝惈鏂嚎閲嶈繛锛夈€?浣嗚鑴氭湰闈㈠悜 5090鈫擜100 (10.50.121.187:8040, password root)锛屼笉鏄垜浠殑
10.174.175.151 (user mti, password mauto)銆?
**淇寤鸿**: 鏂板 `scripts/start_qwen_drive_tunnel.sh` 鎴栧弬鏁板寲 `start_ssh_tunnel.sh`锛?鏀寔涓嶅悓鐨勮繙绋嬩富鏈哄拰绔彛閰嶇疆銆?
### 3.4 鈿狅笍 Phase 1 "鏉€杩涚▼"鏂瑰紡鍒涘缓 run_dir 杈冭剢寮?
**鐜扮姸**: `run_qwen_drive_smoke.sh` Phase 1 閫氳繃杩愯 CLI 骞跺湪 wizard 鍚姩 Docker 鍓?kill 鏉ュ垱寤?run_dir銆備緷璧栨棩蹇椾腑鍑虹幇 "Wizard process" 瀛楃涓叉潵鍒ゆ柇鏃舵満銆?
**椋庨櫓**: 
- 濡傛灉 CLI 鍚姩杈冩參鎴栨棩蹇楁牸寮忓彉鍖栵紝鍙兘鏃犳硶姝ｇ‘鎹曡幏 run_dir
- kill 鏃舵満涓嶅綋鍙兘瀵艰嚧 Docker 瀹瑰櫒宸插惎鍔紙鎴戜滑鐜娌℃湁 Docker 闀滃儚锛?
**瀵规瘮**: 瀹樻柟 `run_wizard_only.sh` 璁剧疆 `ALPAGYM_WIZARD_ONLY=1` 姝ｅ父鍒涘缓 run_dir锛?涓嶉渶瑕佹潃杩涚▼銆備絾鍓嶆彁鏄湁 Docker銆?
**淇寤鸿**: 鑰冭檻浣跨敤 `ALPAGYM_WIZARD_ONLY=1` + `alpasim.repo_url=null` + 
`alpasim.repo_ref=null` 鏉ュ垱寤?run_dir 鑰屼笉鍚姩 Docker锛屾垨鑰呯爺绌舵槸鍚︽湁
`--dry-run` 绫婚€夐」銆傚鏋滀笉琛岋紝鑷冲皯鍔犲己 Phase 1 鐨勯敊璇鐞嗗拰瓒呮椂鏈哄埗銆?
### 3.5 鉁?缂哄皯 Qwen-Drive 鏂囨。

**鐜扮姸**: `docs/` 涓嬫湁 5 涓枃妗ｏ紝鍏ㄩ儴鍏充簬 Alpamayo 鍜?AutoVLA锛屾棤 Qwen-Drive銆?`PROGRESS_REPORT.md` 鍜?`ISSUES.md` 鏀惧湪浠撳簱鏍圭洰褰曘€?
**淇寤鸿**: 
1. 鍦?`docs/` 涓嬫柊澧?`QWEN_DRIVE_ALPAGYM_INTEGRATION.md`
2. 灏?`PROGRESS_REPORT.md` 鍜?`ISSUES.md` 绉诲叆 `docs/` 鎴栨暣鍚堣繘鏂版枃妗?3. 鏂囨。搴旇鐩栵細妯″瀷缁撴瀯銆侀€傞厤鍣ㄨ璁°€佸惎鍔ㄦ柟娉曘€佸凡鐭ラ檺鍒?
## 4. 涓嶉渶瑕佷慨姝ｇ殑閮ㄥ垎

### 4.1 ONBOARDING.md 鈥?瀹屽叏涓€鑷?瀹樻柟 `docs/ONBOARDING.md` (233琛? 涓庢湰鍦板畬鍏ㄤ竴鑷淬€傛棤闇€淇敼銆?
### 4.2 default.yaml 鈥?瀹屽叏涓€鑷?瀹樻柟 `conf/default.yaml` (188琛? 涓庢湰鍦板畬鍏ㄤ竴鑷淬€傛垜浠殑 Qwen-Drive 瀹為獙閰嶇疆
閫氳繃 `defaults: - override /policy: qwen_drive` 瑕嗙洊锛屼笉淇敼 default.yaml 鏈韩銆?杩欐槸姝ｇ‘鐨勫仛娉曘€?
### 4.3 鏂板 policy 鍖?鈥?姝ｇ‘鐨勬墿灞曟柟寮?瀹樻柟浠撳簱璁捐涓?workspace 鍖呯粨鏋勶紝`packages/policies/` 涓嬬殑姣忎釜瀛愮洰褰曟槸涓€涓嫭绔?policy銆?鎴戜滑鏂板 `qwen_drive` 鍜?`autovla` 鍖呯鍚堣繖涓璁°€?
### 4.4 streaming_worker.py 琛ヤ竵 鈥?鍚堢悊鐨勫寮?鎴戜滑鍦?`streaming_worker.py` 娣诲姞鐨勯敊璇棩蹇楀寮猴紙`getattr(rollout_return, 'error_code', 'N/A')`锛?鏄畨鍏ㄧ殑鍚戝悗鍏煎澧炲己锛屼笉褰卞搷瀹樻柟浠ｇ爜璺緞銆?
## 5. 寤鸿鐨勪慨姝ｆ竻鍗曪紙浼樺厛绾ф帓搴忥級

| 浼樺厛绾?| 淇椤?| 褰卞搷 |
|--------|--------|------|
| P0 | 淇 `expected_valid_steps: 22, n_sim_steps: 102` | 涓嶄慨鍒欎换浣曚汉閮芥棤娉曠洿鎺ヨ繍琛屽疄楠?|
| P1 | `run_qwen_drive_smoke.sh` 璺緞鍙傛暟鍖?| 鍙鐜版€?|
| P1 | 鏂板 Qwen-Drive 鏂囨。鍒?`docs/` | 鍙淮鎶ゆ€?|
| P2 | SSH tunnel 鑴氭湰鍖?| 鍙鐜版€?|
| P2 | Phase 1 鍒涘缓 run_dir 鏂瑰紡鏀硅繘 | 鍋ュ．鎬?|
| P3 | 灏?PROGRESS_REPORT.md / ISSUES.md 绉诲叆 docs/ | 浠撳簱鏁存磥搴?|

## 6. 瀹樻柟鏂囨。涓庢湰鍦版枃妗ｇ殑瀵瑰簲鍏崇郴

```
docs/ONBOARDING.md                          鈫?瀹樻柟鍘熺増锛堜竴鑷达級
docs/SERVER_ALPAMAYO15_CLRL.md              鈫?鏈湴鏂板锛孉lpamayo 1.5 鍙屽崱鏈嶅姟鍣ㄦ寚鍗?docs/AUTOVLA_ALPAGYM_FIXES_2026_07_16.md    鈫?鏈湴鏂板锛孉utoVLA 闆嗘垚淇璁板綍
docs/AUTOVLA_A100_4GPU_GRPO_2026_07_16.md   鈫?鏈湴鏂板锛孉utoVLA 4GPU GRPO 杩愯鎸囧崡
docs/AUTOVLA_A100_PROFILES_2026_07_17.md    鈫?鏈湴鏂板锛孉utoVLA A100 profile 璇存槑
docs/COT_TRAJECTORY_ALIGNMENT.md            鈫?鏈湴鏂板锛孋oT-杞ㄨ抗瀵归綈鐮旂┒绗旇
docs/QWEN_DRIVE_ALPAGYM_INTEGRATION.md      鈫?闇€瑕佹柊澧?```

## 7. 鍏充簬 `expected_valid_steps` 鍏紡鐨勬繁鍏ュ垎鏋?
瀹樻柟 `config_validation.py` 鐨勯獙璇佸叕寮忥細
```python
warmup_steps = force_gt_duration_us // control_timestep_us
closed_loop_steps = n_sim_steps - warmup_steps
assert closed_loop_steps == expected_valid_steps
```

杩欎釜鍏紡鍋囪 **1 涓?control tick = 1 涓?policy step**锛屽嵆 `pose_reporting_interval_us == control_timestep_us`銆?
鎴戜滑鐨?Qwen-Drive 閰嶇疆璁剧疆浜?`pose_reporting_interval_us=500000`锛?00ms锛変絾
`control_timestep_us=100000`锛?00ms锛夛紝姣斾緥涓?5:1銆傝繖瀵艰嚧瀹為檯 policy step 鏁?涓庡叕寮忚绠椾笉涓€鑷淬€?
**瀵规瘮 Alpamayo 閰嶇疆**: `pose_reporting_interval_us=100000` == `control_timestep_us=200000`锛?杩欓噷涔熶笉鏄?1:1锛屼絾 Alpamayo 鐢?`n_sim_steps=30, force_gt_duration_us=1600000`锛?```
warmup = 1600000 // 200000 = 8
closed_loop = 30 - 8 = 22
expected_valid_steps = 22  鈫?涓€鑷?```
Alpamayo 鐨?`pose_reporting_interval_us=100000 < control_timestep_us=200000`锛屾剰鍛崇潃
姣忎釜 control tick 鎶ュ憡涓€娆?pose锛屼絾 policy 鍙湪姣?2 涓?control tick 鏃舵墠琚皟鐢ㄣ€?鎵€浠ュ疄闄?policy steps = (30 - 8) / 2 = 11锛熶絾 expected_valid_steps=22...

杩欒鏄?`expected_valid_steps` 鍙兘涓嶆槸 policy step 鏁帮紝鑰屾槸 **control tick 鏁?*锛?鍗?`n_sim_steps - warmup`銆傝€?policy 瀹為檯琚皟鐢ㄧ殑娆℃暟鍙兘鏇村皯锛堢敱 pose_reporting_interval 鎺у埗锛夛紝
浣?packer 鏀跺埌鐨?replay rows 鏁扮瓑浜?control tick 鏁帮紙姣忎釜 tick 閮芥湁涓€涓?replay row锛夈€?
**缁撹**: 闇€瑕佽繘涓€姝ョ‘璁?`expected_valid_steps` 鐨勭‘鍒囪涔夈€傚鏋滃畠绛変簬 control tick 鏁帮紝
閭ｄ箞鎴戜滑鐨勪慨姝?(`n_sim_steps=102, expected_valid_steps=22`) 鏄鐨勨€斺€斿洜涓?102 - 80 = 22銆備絾瀹為檯 rollout 浜х敓浜?22 涓?policy outputs锛岃鏄庡湪杩欎釜閰嶇疆涓?policy step 鏁扮‘瀹炵瓑浜?control tick 鏁帮紝鍙兘鍥犱负杩滅▼ AlPaSim 鐨勫疄闄呰涓轰笌
閰嶇疆涓嶅畬鍏ㄤ竴鑷淬€?