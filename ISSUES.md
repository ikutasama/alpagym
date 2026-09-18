# Qwen-Drive 脳 AlpaGym 绲卞悎 鈥?娈嬭椤屻兓Phase 4 瑷▓銉°儮

澶栭儴AI銉儞銉ャ兗鐢ㄣ€傜従鍦ㄣ伄鐘舵硜銇≒hase 4锛圙RPO鏈牸瀛︾繏锛夈伀鍚戙亼銇熻椤屻倰鏁寸悊銆?
## 鐝剧姸: Phase 3 瀹屼簡 鉁?
銈ㄣ兂銉夈儎銉笺偍銉炽儔銇甤losed-loop rollout + GRPO training step 銇屾垚鍔燂細
- 22銈广儐銉冦儣銇瓵lPaSim闁夈儷銉笺儣銈枫儫銉ャ儸銉笺偡銉с兂
- QwenDriveInferenceModel 銇倛銈嬭粚璺′簣娓?(generate_trajectory)
- 22 GRPO minibatches 瀹熻 (loss=0.0, lr=0)
- 銉椼儹銈汇偣姝ｅ父绲備簡

## Phase 4 銇繀瑕併仾瀹熻

### Issue 1: forward() 銇偣銈裤儢瀹熻 鈫?flow-matching logprob 瑷堢畻

**鐝剧姸**: `QwenDriveCosmos.forward()` 銇互涓嬨倰杩斻仚锛?```python
log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)
log_probs = log_probs + 0.0 * trainable_params[0].sum()  # grad_fn浣滄垚鐢?return {"log_probs": log_probs, "kl_div": None}
```

**蹇呰**: Planning Expert 銇?flow-matching 銉儑銉仹銆佽閷层仌銈屻仧杌岃贰銇?log-probability 銈掕▓绠椼仚銈嬨€?
**瑷▓妗?*:
1. `build_trainer_model_inputs()` 銇?`replay_data.payload` 銇嬨倝 model inputs 銈掑啀妲嬬瘔
2. `forward()` 銇т互涓嬨倰瀹熻锛?   - `model.planning_expert` 銇Τ娓儑銉笺偪锛堛偒銉°儵鐢诲儚銆乪go history銆乺oute锛夈倰鍏ュ姏
   - flow-matching 銇潯浠朵粯銇戙儥銈儓銉倰鍙栧緱
   - 瑷橀尣銇曘倢銇熻粚璺★紙`samples_list`锛夈伀瀵俱仚銈?flow-matching density 銈掕▓绠?   - `log_probs = log_density` 銈掕繑銇?
**娉ㄦ剰鐐?*:
- VLM 銇噸绲愶紙`requires_grad=False`锛夈€丳lanning Expert 銇伩瀛︾繏
- Flow-matching logprob 銇▓绠椼伅 Planning Expert 銇?`PlanningExpert.compute_logprob()` 绛夈伄銉°偨銉冦儔銇屽繀瑕?- `samples_list` / `timesteps` 銈?replay_data 銇繚瀛樸仚銈嬪繀瑕併亴銇傘倠锛堢従鍦ㄣ伅淇濆瓨銇椼仸銇勩仾銇勶級

### Issue 2: replay_data payload 銇笉鍌?
**鐝剧姸**: `build_policy_replay_data()` 銇互涓嬨倰淇濆瓨锛?```python
payload = {
    "ego_history_xyz": ...,
    "ego_history_rot": ...,
    "camera_frames": ...,
    "camera_indices": ...,
    "relative_timestamps": ...,
    "route_xy": ...,
    "pred_xyz": ...,
    "pred_rot": ...,
}
```

**涓嶈冻**: flow-matching logprob 瑷堢畻銇繀瑕併仾浠ヤ笅銇屾湭淇濆瓨锛?- `samples_list` 鈥?flow-matching SDE 銇偟銉炽儣銉粚璺?- `timesteps` 鈥?flow-matching 銇檪闁撱偣銉嗐儍銉?- `noise_level` 鈥?銉庛偆銈恒儸銉欍儷锛堜娇鐢ㄦ檪锛?- `vlm_generated_ids` 鈥?VLM 銇敓鎴愩儓銉笺偗銉筹紙浣跨敤鏅傦級

**鍙傝€?*: AlpamayoR1 銇?`build_policy_replay_data()` 銇с伅浠ヤ笅銈掍繚瀛橈細
```python
payload = {
    "model_input": asdict(model_input),  # 鍏ㄥ叆鍔涖倰dict鍖?    "samples_list": samples_list,        # 閬告姙銇曘倢銇烻DE銈点兂銉椼儷
    "timesteps": timesteps,              # 鏅傞枔銈广儐銉冦儣
}
```

### Issue 3: VLM visual tower 銈︺偋銈ゃ儓鍚屾湡銇?WARNING

**鐝剧姸**: Policy 鈫?Rollout 銈︺偋銈ゃ儓鍚屾湡鏅傘伀浠ヤ笅銇甒ARNING锛?```
No send instructions generated for parameter model.vlm.model.visual.blocks.*.attn.{q,k,v}.{weight,bias}
```

**鍘熷洜**: `rollout_map_local_key_to_hf_key` 銇?visual tower 銇偊銈с偆銉堝悕銉炪儍銉斻兂銈般亴鏈疅瑁呫€?
**褰遍熆**: VLM 銇噸绲愩仌銈屻仸銇勩倠銇熴倎瀛︾繏銇伅褰遍熆銇椼仾銇勩€傘仐銇嬨仐銆乺ollout 鏅傘伄 VLM 鎺ㄨ珫銇奖闊裤仚銈嬪彲鑳芥€с亴銇傘倠锛坮ollout engine 銇?visual tower 銈︺偋銈ゃ儓銇屾銇椼亸鍚屾湡銇曘倢銇亜锛夈€?
**蹇呰銇蹇?*: `QwenDriveWeightMapper.rollout_map_local_key_to_hf_key()` 銇?visual tower 銉栥儹銉冦偗銇悕鍓嶃優銉冦償銉炽偘銈掕拷鍔犮€?
### Issue 4: build_trainer_model_inputs 銇疅瑁?
**鐝剧姸**: 
```python
@staticmethod
def build_trainer_model_inputs(replay_data, **kwargs):
    model_inputs: dict[str, Any] = {}
    old_logprob = torch.tensor(
        float(replay_data.old_logprob) if replay_data.old_logprob is not None else 0.0,
        dtype=torch.float32,
    ).reshape(())
    return model_inputs, old_logprob
```

**蹇呰**: `replay_data.payload` 銇嬨倝 Planning Expert 銇?forward 鍏ュ姏銈掑啀妲嬬瘔锛?- `camera_frames`, `camera_indices` 鈫?VLM 鍏ュ姏
- `ego_history_xyz`, `ego_history_rot` 鈫?Planning Expert 鍏ュ姏
- `route_xy` 鈫?銉娿儞銈层兗銈枫儳銉冲叆鍔?- `samples_list`, `timesteps` 鈫?flow-matching logprob 瑷堢畻鐢?
**鍙傝€?*: AlpamayoR1 銇?`build_trainer_model_inputs()` 銇с伅锛?```python
@classmethod
def build_trainer_model_inputs(cls, replay_data, num_context_frames):
    payload = replay_data.payload
    model_input = ModelInput(**payload["model_input"])
    forward_kwargs = build_alpamayo_r1_forward_inputs(model_input, num_context_frames)
    old_logprob = torch.as_tensor(replay_data.old_logprob, ...)
    return forward_kwargs, old_logprob
```

### Issue 5: expected_valid_steps 銇?n_sim_steps 銇笉涓€鑷?
**鐝剧姸**: `expected_valid_steps=22`, `n_sim_steps=102` (22 + 80 warmup)

**鍟忛**: 瀹熼殯銇儹銉笺儷銈偊銉堛亴22銈广儐銉冦儣銈掔敓鎴愩仚銈嬨亴銆併亾銈屻伅 `pose_reporting_interval_us=500000` 銇?`n_sim_steps=102` 銇祫銇垮悎銈忋仜銇倛銈嬨€侫lPaSim 鍋淬伄瀹熼殯銇偣銉嗐儍銉楁暟銇?config 銇ㄥ畬鍏ㄣ伀銇竴鑷淬仐銇亜鍙兘鎬с亴銇傘倠銆?
**鎺ㄥエ**: 瑜囨暟銈枫兗銉炽仹銉嗐偣銉堛仐銆併偣銉嗐儍銉楁暟銇伆銈夈仱銇嶃倰纰鸿獚銆俙expected_valid_steps` 銇綑瑁曘倰鎸併仧銇涖倠锛堜緥: 30锛夈亱銆併儜銉囥偅銉炽偘姗熻兘銈掓椿鐢ㄣ€?
## 銉曘偂銈ゃ儷妲嬫垚

### 澶夋洿娓堛伩銉曘偂銈ゃ儷 (git push娓堛伩)
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/cosmos_wrapper.py` 鈥?BaseModel wrapper, WeightMapper, forward()
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/inference_model.py` 鈥?鎺ㄨ珫銉儑銉? replay_data妲嬬瘔
- `packages/policies/qwen_drive/src/alpagym_qwen_drive/bundle.py` 鈥?PolicyBundle瀹氱京
- `packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py` 鈥?銈ㄣ儵銉笺儹銈板挤鍖?- `PROGRESS_REPORT.md` 鈥?閫叉崡鍫卞憡鏇?
### 鍙傜収銉曘偂銈ゃ儷 (澶夋洿銇仐銆佺悊瑙ｇ敤)
- `packages/runtime/src/alpagym_runtime/cosmos/packer.py` 鈥?DataPacker, build_trainer_model_inputs鍛笺伋鍑恒仐
- `packages/runtime/src/alpagym_runtime/replay.py` 鈥?PolicyReplayData瀹氱京
- `packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py` 鈥?AlpamayoPolicy.step()
- `packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py` 鈥?AlpamayoR1鍙傝€冨疅瑁?- `packages/policies/qwen_drive/src/alpagym_qwen_drive/cosmos_wrapper.py` 鈥?QwenDriveCosmos.forward()

## Phase 4 瀹熻銇劒鍏堥爢浣?
1. **(楂? build_policy_replay_data 銇?samples_list/timesteps 銈掕拷鍔?* 鈥?flow-matching logprob 瑷堢畻銇繀闋?2. **(楂? forward() 銇?flow-matching logprob 瑷堢畻銈掑疅瑁?* 鈥?GRPO瀛︾繏銇偝銈?3. **(楂? build_trainer_model_inputs 銇?model_inputs 銈掑啀妲嬬瘔** 鈥?forward() 銇搞伄鍏ュ姏鐢熸垚
4. **(涓? VLM visual tower 銈︺偋銈ゃ儓鍚嶃優銉冦償銉炽偘** 鈥?rollout绮惧害銇悜涓?5. **(浣? expected_valid_steps 銇嫊鐨勮鏁?* 鈥?瑜囨暟銈枫兗銉冲蹇?
## 澶栭儴AI銇搞伄璩晱

1. Qwen-Drive 銇?`PlanningExpert` 銇?flow-matching logprob 銈掕▓绠椼仚銈嬨儭銈姐儍銉夈伅瀛樺湪銇欍倠銇嬶紵锛坄compute_logprob`, `score_fn`, `ode_solve` 绛夛級
2. `generate_trajectory()` 銇?`mode="direct_planning"` 銇?`mode="cfm"` 銇仌銇勩伅浣曘亱锛焞ogprob 瑷堢畻銇仼銇椼仧銉兗銉夈伅锛?3. VLM 銇嚭鍔涳紙`vlm_generated_ids`锛夈伅 Planning Expert 銇?logprob 瑷堢畻銇繀瑕併亱锛?4. `samples_list` 銇?flow-matching 銇仼銇闅庛伄銈点兂銉椼儷銇嬶紵锛堝垵鏈熴儙銈ゃ偤銆佷腑闁撶姸鎱嬨€佹渶绲傝粚璺★紵锛?