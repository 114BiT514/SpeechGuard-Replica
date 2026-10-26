# M5 结论：权限组 + 端到端管线 + 主结果表

> 主表数据：`experiments/results/main_table.json` / `main_table.md`
> 基线对比：`experiments/results/exp4_attack_rates.json`、`exp4_compare.json`
> 端到端演示：`python -m src.pipeline demo <wav>`（产物+summary.json）

## 1. 主结果表（论文 Table 5/6/7 同款，LibriSpeech test-clean 全集 + VCTK 40人×65句）

| 数据集 | EER-Ori | EER-Pro | EER-Att | WER-Ori | WER-Pro | WER-Att | 恢复EER | 恢复WER |
|---|---|---|---|---|---|---|---|---|
| LS test-clean（我们） | 1.61 | **40.01** | **40.29** | 2.08 | **5.44** | **17.80** | **2.95** | **2.23** |
| LS test-clean（论文） | 1.47 | 34.50 | 36.77 | 2.46 | 4.80 | 18.39 | 7.07 | 9.90 |
| VCTK（我们） | 1.59 | **39.87** | **41.36** | 4.01 | **10.06** | **26.11** | **3.16** | **4.81** |
| VCTK（论文） | 1.46 | 35.66 | 37.22 | 6.58 | 14.81 | 32.69 | 7.07 | 9.90 |
| Volunteers（论文） | 2.21 | 30.29 | 33.66 | 6.35 | 10.97 | 29.99 | — | — |

- **Volunteers 列 hold**：自录集 7 人已入库（发音不标准、内容未经人工审听，
  委托方要求谨慎使用），审听后可补第三列。
- 关键规律全部复现：EER 1.6% → 40%（声纹被抹）且攻击不回升；WER 仅温和上升；
  攻击后 WER 大涨（17.8/26.1%）→ 攻击无效化；恢复态全面回到原始水平且优于论文
  （正向偏差来源：single 模式平稳映射 + OLA，见 M3 findings §5）。
- RTC（Table 6 风格，P0-2 统一口径）：主口径用**与基线同一次运行、无预生成池**的值
  ——SG protect 0.0634 / recover 0.06（论文 0.32/0.26，约快 5×）；池化值
  0.0081/0.0075（`m3_complete.json`）作实现优化注脚，不用于跨方法对比。
  四方法同源排序：单参数 0.0463 < McAdams 0.0504 < Overo 0.0515 < SG 0.0634
  （SG 最慢=多参数代价，与论文 Table 6 一致）。
- FNR（Table 7 风格）：WAV 层 Enc margin=8 **98.57%**（论文 95.88%）；Dec=Ori
  （逐字节还原）；MP3 层（P0-5 补测，公式(5) 单变量、WAV 层不加密）enc **100%** /
  quiet 98.4% / clean-mp3 基线 9.7%。见 `main_table.json` `table7_style_fnr`。

## 2. CP-ABE 权限组（任务1/2）

- charm-crypto 在 Windows 编译失败（pip 需 C 扩展，已留尝试记录），按计划 §3-M5
  降级预案实现**方法等价**：`src/auth/cp_abe.py` 对称密钥派生属性加密
  （Naccache94/KD-ABE 谱系，HKDF+AES-GCM，零新增依赖，Windows/云端同构）。
- 提供与论文一致的 **Setup/Encrypt/KeyGen/Decrypt** 四算法 API 与单调访问结构
  （DNF：S ⊇ 某合取项才能解）；属性秘密由主密钥 HKDF 派生，缺失属性在随机预言
  模型下不可推出。**报告需如实标注与配对 CP-ABE 的三点差异**（模块 docstring 已写明：
  无短密文/仅 DNF/撤销需重加密）。
- `src/auth/permission.py`：G/Mi={Si,Wi} 数据结构 + SegmentParams↔base64 JSON 序列化
  （密钥材料可入库/传输）+ 三组典型权限（admin=Si+全部Wi；collab=仅Si，敏感词保持
  加密态；public=无资产只拿静音分发版）。

## 3. 端到端管线（任务3）

`src/pipeline.py` 四阶段：protect（warping→CTC 词流→检测→帧加密）→ publish
（人工增删后重加密 → MP3 公式(5) 帧体加密 + quiet 静音分发版 + CP-ABE 保险库）→
listen（按角色）→ recover（解 Wi + 逆 Si）。`python -m src.pipeline demo` 一键走完，
产出 summary.json（敏感项清单/加密对象数/RTC/恢复 SNR）。
测试：`tests/test_pipeline.py`（模型替身，纯接线验证）+ 全部 57 项单测。

## 4. 基线对比（任务5，论文 Fig.10/11）

`src/baselines/methods.py`：公式(2) 单参数分段线性 / McAdams 正弦频谱移位 warp /
Overo 双参数（独立均匀无斜率约束）——均为整句一段映射 + OLA，与 SG single 同框架公平对比。

**Fig.10（10,000 次 reducing 攻击成功率，双方参数都按 dist∈(0.5,0.6) 校准）**：

| 方法 | 空间自由度 | 攻击成功率 |
|---|---|---|
| single 公式(2) | 1 维 | **57.34%** |
| Overo | 2→1 维 | **49.66%** |
| McAdams | 3→2 维 | **40.24%** |
| **SpeechGuard（raw 论文抽样）** | 2n 维 n∈[1,10] | **14.74%**（论文总基准 15.87%） |
| SpeechGuard（本实现斜率约束） | — | 46.22%（exp2 已知代价：可逆性换抗攻击性，如实报告） |

方向与论文一致：**SG(raw) 显著最低**（比 Overo 低 34.9pp，论文口径低 6.8pp——
差异源于我们的参数空间自由度设定与论文不同，报告中如实标注）。

**Fig.11（test-clean 12人×≤12句小协议，只做方法间相对比较——小子集 EER 绝对值
有偏，见 M3 findings §2）**：SG prot EER 39.6% 且 WER 仅 2.44%（基线同 EER 水平
WER 50–60%，说明其匿名化以毁可懂度为代价）；att 后 SG WER 升至 12.6%（攻击无效化）；
rec EER 10.7%/WER 1.05%（恢复干净）。

## 5. DoD 对照

- `python -m src.pipeline demo assets/demo.wav` 四阶段 ✓（demo 用 test-clean 句实测）
- 数据集数字进表 ✓（LS/VCTK；Volunteers=hold 有说明）
- 与论文偏差全部有解释 ✓（模型差异/协议差异/实现差异三类，均已归档）
- 权限组三组典型场景 ✓（测试固化：collab 打不开 admin 密文、public 无资产）
